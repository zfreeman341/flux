use std::time::Duration;

use async_trait::async_trait;
use reqwest::StatusCode;
use serde::Deserialize;
use tokio::time::sleep;
use tracing::{debug, info, instrument, warn};

use crate::{
    FluxError,
    provider::{CompletionRequest, CompletionResponse, LlmProvider},
};

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VERSION: &str = "2023-06-01";
const DEFAULT_MAX_TOKENS: u32 = 8096;
const MAX_RETRIES: u32 = 6;
const MAX_RETRY_DELAY_SECS: u64 = 90;

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
        let max_tokens = if request.max_tokens > 0 {
            request.max_tokens
        } else {
            DEFAULT_MAX_TOKENS
        };

        let tools: Vec<serde_json::Value> = request
            .tools
            .iter()
            .map(|t| make_tool_def(t, request.max_tool_calls))
            .collect();

        let body = if tools.is_empty() {
            serde_json::json!({
                "model": request.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": request.prompt}],
            })
        } else {
            serde_json::json!({
                "model": request.model,
                "max_tokens": max_tokens,
                "messages": [{"role": "user", "content": request.prompt}],
                "tools": tools,
            })
        };

        let tool_names = request.tools.join(", ");
        if tool_names.is_empty() {
            info!(model = %request.model, "→ calling API");
        } else {
            info!(model = %request.model, tools = %tool_names, "→ calling API (web search may take 20-40s)");
        }
        let response = send_with_retries(&self.client, &self.api_key, &body).await?;

        info!(
            input_tokens = response.usage.input_tokens,
            output_tokens = response.usage.output_tokens,
            "← response received"
        );
        debug!(
            input_tokens = response.usage.input_tokens,
            output_tokens = response.usage.output_tokens,
            "received response"
        );

        Ok(CompletionResponse {
            content: extract_text(&response.content),
            input_tokens: response.usage.input_tokens,
            output_tokens: response.usage.output_tokens,
        })
    }
}

// Maps the tool names used in workflow TOML to the Anthropic API's server-tool type strings.
// These are managed server-side by Anthropic — no client-side execution loop needed.
fn make_tool_def(name: &str, max_uses: u32) -> serde_json::Value {
    match name {
        "web_search" => serde_json::json!({
            "type": "web_search_20260209",
            "name": "web_search",
            "max_uses": max_uses,
            "allowed_callers": ["direct"],
        }),
        "web_fetch" => serde_json::json!({
            "type": "web_fetch_20260209",
            "name": "web_fetch",
            "max_uses": max_uses,
            "allowed_callers": ["direct"],
        }),
        _ => serde_json::json!({
            "name": name,
            "description": "A custom tool.",
            "input_schema": {"type": "object", "properties": {}},
        }),
    }
}

async fn send_with_retries(
    client: &reqwest::Client,
    api_key: &str,
    body: &serde_json::Value,
) -> crate::Result<ApiResponse> {
    let mut attempt = 0u32;
    loop {
        let resp = client
            .post(API_URL)
            .header("x-api-key", api_key)
            .header("anthropic-version", ANTHROPIC_VERSION)
            .json(body)
            .send()
            .await
            .map_err(|e| FluxError::Api(e.to_string()))?;

        if resp.status() == StatusCode::TOO_MANY_REQUESTS && attempt < MAX_RETRIES {
            // Prefer the server's retry-after over our own backoff — it knows
            // exactly when the rate limit window resets.
            let retry_after = resp
                .headers()
                .get("retry-after")
                .and_then(|v| v.to_str().ok())
                .and_then(|s| s.parse::<u64>().ok());

            let delay_secs = retry_after
                .unwrap_or_else(|| 2u64.pow(attempt))
                .min(MAX_RETRY_DELAY_SECS);

            warn!(attempt, delay_secs, "rate limited, retrying");
            sleep(Duration::from_secs(delay_secs)).await;
            attempt += 1;
            continue;
        }

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(FluxError::Api(format!("{status}: {text}")));
        }

        return resp
            .json::<ApiResponse>()
            .await
            .map_err(|e| FluxError::Api(format!("failed to parse API response: {e}")));
    }
}

fn extract_text(blocks: &[ContentBlock]) -> String {
    blocks
        .iter()
        .filter_map(|b| {
            if b.kind == "text" {
                b.text.as_deref()
            } else {
                None
            }
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[derive(Deserialize)]
struct ApiResponse {
    content: Vec<ContentBlock>,
    usage: Usage,
}

// Flat struct rather than enum: content blocks have a `type` field plus optional
// type-specific fields. Using a struct with Option<String> for `text` handles
// both text blocks and server_tool_use blocks without needing to enumerate every type.
#[derive(Deserialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    kind: String,
    text: Option<String>,
}

#[derive(Deserialize)]
struct Usage {
    input_tokens: u32,
    output_tokens: u32,
}
