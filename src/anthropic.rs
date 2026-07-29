use std::time::Duration;

use async_trait::async_trait;
use reqwest::StatusCode;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::time::sleep;
use tracing::{info, instrument, warn};

use crate::{
    FluxError,
    provider::{CompletionRequest, CompletionResponse, LlmProvider},
};

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 8096;
const MAX_RETRIES: u32 = 6;
const MAX_RETRY_DELAY_SECS: u64 = 90;
// pause_turn means the server-side tool loop hit its iteration limit and
// wants the conversation re-sent to resume where it left off. Cap how many
// times we oblige so a pathological step can't loop forever.
const MAX_CONTINUATIONS: u32 = 3;

pub struct AnthropicClient {
    client: reqwest::Client,
    api_key: String,
}

impl AnthropicClient {
    pub fn from_env() -> crate::Result<Self> {
        let api_key = std::env::var("ANTHROPIC_API_KEY").map_err(|_| {
            FluxError::InvalidArgument("ANTHROPIC_API_KEY environment variable not set".into())
        })?;
        Ok(Self {
            client: reqwest::Client::new(),
            api_key,
        })
    }
}

#[async_trait]
impl LlmProvider for AnthropicClient {
    #[instrument(skip(self, request), fields(model = %request.model))]
    async fn complete(&self, request: CompletionRequest) -> crate::Result<CompletionResponse> {
        let max_tokens = request.max_tokens.unwrap_or(DEFAULT_MAX_TOKENS);

        let tools: Vec<Value> = request
            .tools
            .iter()
            .map(|t| make_tool_def(t, request.max_tool_calls))
            .collect();

        let tool_names = request.tools.join(", ");
        if tool_names.is_empty() {
            info!(model = %request.model, "→ calling API");
        } else {
            info!(model = %request.model, tools = %tool_names, "→ calling API (web search may take 20-40s)");
        }

        // The conversation only grows when the server pauses a tool turn:
        // we echo the assistant's content back verbatim and re-send, and the
        // API resumes its server-side tool loop where it stopped.
        let mut messages = vec![json!({"role": "user", "content": request.prompt})];
        let mut content = String::new();
        let mut sources: Vec<String> = Vec::new();
        let mut input_tokens = 0u32;
        let mut output_tokens = 0u32;
        let mut web_search_requests = 0u32;
        let mut continuations = 0u32;

        loop {
            let mut body = json!({
                "model": request.model,
                "max_tokens": max_tokens,
                "messages": messages.clone(),
            });
            if !tools.is_empty() {
                body["tools"] = Value::Array(tools.clone());
            }

            let response = send_with_retries(&self.client, &self.api_key, &body).await?;

            // Each continuation round bills its own usage; report the sum.
            input_tokens += response.usage.input_tokens;
            output_tokens += response.usage.output_tokens;
            web_search_requests += response
                .usage
                .server_tool_use
                .as_ref()
                .map_or(0, |t| t.web_search_requests);

            let text = extract_text(&response.content);
            if !text.is_empty() {
                if !content.is_empty() {
                    content.push('\n');
                }
                content.push_str(&text);
            }

            // Dedupe across continuation rounds too; linear scan is fine for
            // the handful of sources a step cites.
            for source in extract_sources(&response.content) {
                if !sources.contains(&source) {
                    sources.push(source);
                }
            }

            match response.stop_reason.as_deref() {
                Some("pause_turn") if continuations < MAX_CONTINUATIONS => {
                    continuations += 1;
                    info!(
                        continuation = continuations,
                        "server paused tool turn; resuming"
                    );
                    // Echo the content blocks back untouched — the API detects
                    // the trailing server tool state and continues on its own.
                    // No extra user message.
                    messages.push(json!({"role": "assistant", "content": response.content}));
                    continue;
                }
                Some("pause_turn") => {
                    warn!("continuation limit reached; returning partial result");
                    content.push_str(
                        "\n\n[NOTE: web tool loop cut off after repeated continuations — \
                         results may be incomplete]",
                    );
                }
                Some("max_tokens") => {
                    warn!(
                        max_tokens,
                        "response truncated; raise max_tokens on this step"
                    );
                    content.push_str(
                        "\n\n[NOTE: output truncated at max_tokens limit — \
                         raise max_tokens on this step]",
                    );
                }
                _ => {}
            }

            info!(input_tokens, output_tokens, "← response received");
            return Ok(CompletionResponse {
                content,
                sources,
                input_tokens,
                output_tokens,
                web_search_requests,
            });
        }
    }
}

// Maps the tool names used in workflow TOML to the Anthropic API's server-tool type strings.
// These are managed server-side by Anthropic — no client-side execution loop needed.
fn make_tool_def(name: &str, max_uses: u32) -> Value {
    match name {
        "web_search" => json!({
            "type": "web_search_20260209",
            "name": "web_search",
            "max_uses": max_uses,
            "allowed_callers": ["direct"],
        }),
        "web_fetch" => json!({
            "type": "web_fetch_20260209",
            "name": "web_fetch",
            "max_uses": max_uses,
            "allowed_callers": ["direct"],
        }),
        _ => json!({
            "name": name,
            "description": "A custom tool.",
            "input_schema": {"type": "object", "properties": {}},
        }),
    }
}

async fn send_with_retries(
    client: &reqwest::Client,
    api_key: &str,
    body: &Value,
) -> crate::Result<ApiResponse> {
    let mut attempt = 0u32;
    loop {
        let result = client
            .post(API_URL)
            .header("x-api-key", api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(body)
            .send()
            .await;

        let resp = match result {
            Ok(resp) => resp,
            // Connect and timeout failures are usually transient, and mid-fan-out they
            // would otherwise cost a whole research branch.
            Err(e) if attempt < MAX_RETRIES && (e.is_connect() || e.is_timeout()) => {
                let delay = backoff_delay(attempt, None);
                warn!(attempt, delay_secs = delay, error = %e, "transport error, retrying");
                sleep(Duration::from_secs(delay)).await;
                attempt += 1;
                continue;
            }
            Err(e) => return Err(FluxError::Api(e.to_string())),
        };

        let status = resp.status();
        if is_retryable(status) && attempt < MAX_RETRIES {
            // The server's retry-after knows exactly when the window resets, so it wins
            // over our own guess.
            let retry_after = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok());
            let delay = backoff_delay(attempt, retry_after);
            warn!(attempt, %status, delay_secs = delay, "retryable response, retrying");
            sleep(Duration::from_secs(delay)).await;
            attempt += 1;
            continue;
        }

        if !status.is_success() {
            let text = resp.text().await.unwrap_or_default();
            return Err(FluxError::Api(format!("{status}: {text}")));
        }

        return resp
            .json::<ApiResponse>()
            .await
            .map_err(|e| FluxError::Api(format!("failed to parse API response: {e}")));
    }
}

// 429 plus every 5xx, which covers Anthropic's 529 overloaded_error and the transient
// 500/502/503 responses that a long fan-out reliably runs into.
fn is_retryable(status: StatusCode) -> bool {
    status == StatusCode::TOO_MANY_REQUESTS || status.is_server_error()
}

// Exponential backoff, jittered to keep a concurrent fan-out from retrying in lockstep.
// A server-supplied retry_after is used as-is: it is authoritative, not a guess.
fn backoff_delay(attempt: u32, retry_after_secs: Option<u64>) -> u64 {
    if let Some(secs) = retry_after_secs {
        return secs.min(MAX_RETRY_DELAY_SECS);
    }
    let base = 2u64.saturating_pow(attempt).min(MAX_RETRY_DELAY_SECS);
    base + pseudo_jitter_secs(base)
}

// Dependency-free jitter: clock nanoseconds are more than random enough to decorrelate
// retries, and this avoids pulling in `rand` for one call site.
fn pseudo_jitter_secs(base: u64) -> u64 {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| u64::from(d.subsec_nanos()))
        .unwrap_or(0);
    // Up to half the base delay, so total stays within 1.5x and under the cap.
    let spread = (base / 2).max(1);
    (nanos % spread).min(MAX_RETRY_DELAY_SECS - base.min(MAX_RETRY_DELAY_SECS))
}

// Collects cited sources from web_search citations attached to text blocks,
// formatted "Title — URL" and deduplicated by URL in first-seen order.
// Citation fields don't count toward token usage, so keeping them is free.
fn extract_sources(content: &Value) -> Vec<String> {
    let Some(blocks) = content.as_array() else {
        return Vec::new();
    };
    let mut seen = std::collections::HashSet::new();
    let mut sources = Vec::new();
    for block in blocks.iter().filter(|b| b["type"] == "text") {
        let Some(citations) = block["citations"].as_array() else {
            continue;
        };
        for citation in citations {
            let Some(url) = citation["url"].as_str() else {
                continue;
            };
            if !seen.insert(url.to_string()) {
                continue;
            }
            let title = citation["title"].as_str().unwrap_or(url);
            sources.push(format!("{title} — {url}"));
        }
    }
    sources
}

// Concatenates the text blocks from a raw content array. Non-text blocks
// (server_tool_use, web_search_tool_result, ...) are skipped.
fn extract_text(content: &Value) -> String {
    let Some(blocks) = content.as_array() else {
        return String::new();
    };
    blocks
        .iter()
        .filter(|b| b["type"] == "text")
        .filter_map(|b| b["text"].as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

#[derive(Deserialize)]
struct ApiResponse {
    // Raw JSON rather than typed structs: a pause_turn continuation must echo
    // the content back byte-faithfully, including block types we don't model
    // (server_tool_use, web_search_tool_result, ...). Same typed-vs-dynamic
    // tradeoff as parse_agent_output in agent.rs.
    content: Value,
    usage: Usage,
    stop_reason: Option<String>,
}

#[derive(Deserialize)]
struct Usage {
    input_tokens: u32,
    output_tokens: u32,
    // Only present when server tools (web_search, web_fetch) ran during the
    // request. Option + #[serde(default)] keeps deserialization tolerant:
    // a missing field becomes None instead of an error.
    #[serde(default)]
    server_tool_use: Option<ServerToolUse>,
}

#[derive(Deserialize)]
struct ServerToolUse {
    // default covers responses where only web_fetch ran and this key is absent.
    #[serde(default)]
    web_search_requests: u32,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_text_concatenates_text_blocks_only() {
        let content = json!([
            {"type": "text", "text": "first"},
            {"type": "server_tool_use", "id": "x", "name": "web_search", "input": {}},
            {"type": "web_search_tool_result", "tool_use_id": "x", "content": []},
            {"type": "text", "text": "second"},
        ]);
        assert_eq!(extract_text(&content), "first\nsecond");
    }

    #[test]
    fn extract_text_tolerates_unexpected_shapes() {
        assert_eq!(extract_text(&json!(null)), "");
        assert_eq!(extract_text(&json!([{"type": "text"}])), "");
    }

    #[test]
    fn extract_sources_collects_and_dedupes_citations() {
        let content = json!([
            {"type": "text", "text": "a", "citations": [
                {"type": "web_search_result_location", "url": "https://x.com/1", "title": "One"},
                {"type": "web_search_result_location", "url": "https://x.com/2", "title": "Two"},
            ]},
            {"type": "server_tool_use", "id": "t", "name": "web_search", "input": {}},
            {"type": "text", "text": "b", "citations": [
                // Duplicate URL: must not repeat.
                {"type": "web_search_result_location", "url": "https://x.com/1", "title": "One"},
            ]},
            // Text block without citations: fine.
            {"type": "text", "text": "c"},
        ]);
        assert_eq!(
            extract_sources(&content),
            vec!["One — https://x.com/1", "Two — https://x.com/2"]
        );
    }

    #[test]
    fn retryable_covers_rate_limit_and_all_server_errors() {
        assert!(is_retryable(StatusCode::TOO_MANY_REQUESTS));
        // 529 overloaded_error is Anthropic-specific and has no StatusCode constant.
        assert!(is_retryable(StatusCode::from_u16(529).unwrap()));
        assert!(is_retryable(StatusCode::INTERNAL_SERVER_ERROR));
        assert!(is_retryable(StatusCode::SERVICE_UNAVAILABLE));
        // Client errors are the caller's fault; retrying just wastes the budget.
        assert!(!is_retryable(StatusCode::BAD_REQUEST));
        assert!(!is_retryable(StatusCode::UNAUTHORIZED));
        assert!(!is_retryable(StatusCode::OK));
    }

    #[test]
    fn backoff_prefers_retry_after_and_caps_it() {
        assert_eq!(backoff_delay(0, Some(12)), 12);
        assert_eq!(backoff_delay(5, Some(9_999)), MAX_RETRY_DELAY_SECS);
    }

    #[test]
    fn backoff_grows_and_stays_within_the_cap() {
        // Jitter makes this a band rather than an exact value: base 2^attempt, plus up
        // to half of it again.
        for attempt in 0..MAX_RETRIES {
            let base = 2u64.pow(attempt).min(MAX_RETRY_DELAY_SECS);
            let delay = backoff_delay(attempt, None);
            assert!(delay >= base, "attempt {attempt}: {delay} < {base}");
            assert!(
                delay <= MAX_RETRY_DELAY_SECS,
                "attempt {attempt}: {delay} exceeds cap"
            );
        }
    }

    #[test]
    fn extract_sources_falls_back_to_url_when_title_missing() {
        let content = json!([
            {"type": "text", "text": "a", "citations": [{"url": "https://x.com/only"}]},
        ]);
        assert_eq!(
            extract_sources(&content),
            vec!["https://x.com/only — https://x.com/only"]
        );
    }
}
