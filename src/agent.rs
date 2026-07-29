use std::time::{Duration, Instant};

use async_trait::async_trait;
use tracing::{debug, instrument};

use crate::{
    FluxError,
    provider::{AgentProvider, AgentRequest, AgentResponse, Billing},
};

// A generic agent provider that shells out to any CLI tool.
//
// The invocation is: `binary [args...] "<task>"`.
// stdout is captured as the response; stderr is logged at debug level.
//
// This single struct covers both claude-code and hermes — they differ only
// in which binary they call and which flags precede the prompt.
pub struct CliAgentProvider {
    binary: String,
    // Args inserted between the binary and the task string.
    // e.g., ["-p"] for claude, ["-z"] for hermes.
    args: Vec<String>,
    scrub_api_key: bool,
}

impl CliAgentProvider {
    pub fn new(binary: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            binary: binary.into(),
            args,
            scrub_api_key: false,
        }
    }

    /// Hides ANTHROPIC_API_KEY from the subprocess so the CLI falls back to its own
    /// interactive login. Claude Code prefers the env key when present, which would
    /// bill metered API credits instead of the subscription already signed in.
    pub fn without_api_key(mut self) -> Self {
        self.scrub_api_key = true;
        self
    }

    // Factory: `claude -p --allowedTools=WebFetch,WebSearch "<task>"`
    //
    // --allowedTools pre-approves WebFetch and WebSearch so agents running
    // as subprocesses don't hit permission prompts. Without this, agents
    // trying to fetch web pages block or time out waiting for approval
    // that can never come in non-interactive mode.
    //
    // The flag MUST use the --flag=value form (not --flag value). The claude
    // CLI defines --allowedTools as variadic (<tools...>), so a bare
    // `--allowedTools WebFetch,WebSearch "<task>"` consumes the task string
    // as a second tool name and leaves the prompt argument missing.
    // --output-format=json wraps the response in a result envelope that
    // includes total_cost_usd — the actual API spend of the call — so budget
    // tracking can use real numbers instead of the cost_per_call_usd guess.
    pub fn claude_code() -> Self {
        Self::new(
            "claude",
            vec![
                "-p".to_string(),
                "--allowedTools=WebFetch,WebSearch".to_string(),
                "--output-format=json".to_string(),
            ],
        )
        .without_api_key()
    }

    // Factory: `hermes -z "<task>"`
    pub fn hermes() -> Self {
        Self::new("hermes", vec!["-z".to_string()])
    }
}

#[async_trait]
impl AgentProvider for CliAgentProvider {
    // Hiding the API key forces the CLI onto its own login, so whatever it reports is an
    // API-equivalent figure rather than money spent.
    fn billing(&self) -> Billing {
        if self.scrub_api_key {
            Billing::Notional
        } else {
            Billing::Metered
        }
    }

    // #[instrument] attaches a tracing span to this function.
    // skip(self, request) keeps the span fields tidy — we log what matters explicitly.
    #[instrument(skip(self, request), fields(binary = %self.binary))]
    async fn run(&self, request: AgentRequest) -> crate::Result<AgentResponse> {
        let start = Instant::now();

        // Prepend an efficiency note: if a page is unreachable, move on rather
        // than retrying indefinitely. This avoids agents spending their entire
        // budget stuck on one unresponsive site. Deliberately avoids mentioning
        // specific time values — telling an LLM "stop at 240 seconds" causes it
        // to over-research trying to fill a perceived window, then get cut off.
        let task = format!(
            "[Be efficient: if a page is slow or unreachable, skip it and move on. \
             Prioritise producing complete output over exhaustive coverage.]\n\n{}",
            request.task,
        );

        // tokio::process::Command is the async version of std::process::Command.
        // It doesn't block the thread while the subprocess runs — the runtime
        // polls it like any other future, so other tasks can make progress.
        //
        // tokio::time::timeout wraps any future with a deadline. If the future
        // doesn't complete within the duration, it returns Err(Elapsed).
        // We map that into our own FluxError::Agent before surfacing it.
        let mut command = tokio::process::Command::new(&self.binary);
        if self.scrub_api_key {
            command.env_remove("ANTHROPIC_API_KEY");
        }
        command.args(&self.args).arg(&task);

        let result =
            tokio::time::timeout(Duration::from_secs(request.timeout_secs), command.output()).await;

        // Two layers of Result to unwrap:
        //   outer: Err(Elapsed) if timed out
        //   inner: Err(io::Error) if spawn failed (e.g. binary not on PATH)
        let output = match result {
            Err(_elapsed) => {
                return Err(FluxError::Agent(format!(
                    "{} timed out after {}s",
                    self.binary, request.timeout_secs,
                )));
            }
            Ok(Err(e)) => {
                // std::io::ErrorKind::NotFound means the binary doesn't exist.
                // Give a clear error instead of a raw OS message.
                let msg = if e.kind() == std::io::ErrorKind::NotFound {
                    format!(
                        "'{}' not found — is it installed and on your PATH?",
                        self.binary
                    )
                } else {
                    format!("failed to spawn '{}': {e}", self.binary)
                };
                return Err(FluxError::Agent(msg));
            }
            Ok(Ok(o)) => o,
        };

        let stderr = String::from_utf8_lossy(&output.stderr);
        if !stderr.trim().is_empty() {
            debug!(stderr = %stderr.trim(), "agent stderr");
        }

        if !output.status.success() {
            return Err(FluxError::Agent(format!(
                "'{}' exited with {}: {}",
                self.binary,
                output.status,
                stderr.trim()
            )));
        }

        let stdout = String::from_utf8_lossy(&output.stdout);
        let (text, cost_usd) = parse_agent_output(&stdout);

        Ok(AgentResponse {
            output: text,
            duration: start.elapsed(),
            cost_usd,
        })
    }
}

// Extracts the response text and actual cost from agent CLI stdout.
//
// claude-code with --output-format=json emits a single JSON envelope with
// `result` (the response text) and `total_cost_usd`. Other agents (hermes)
// emit plain text. We parse into serde_json::Value rather than a typed
// struct because we only want two fields and must tolerate schema drift:
// anything unexpected falls back to raw stdout with no cost, which keeps
// existing workflows running even if the CLI's envelope changes.
fn parse_agent_output(stdout: &str) -> (String, Option<f64>) {
    let trimmed = stdout.trim();
    let Ok(parsed) = serde_json::from_str::<serde_json::Value>(trimmed) else {
        return (trimmed.to_string(), None);
    };

    let cost = parsed.get("total_cost_usd").and_then(|v| v.as_f64());
    match parsed.get("result").and_then(|v| v.as_str()) {
        Some(text) => (text.to_string(), cost),
        // Valid JSON without a result field: keep the raw envelope so the
        // output isn't silently lost, but still report the cost if present.
        None => (trimmed.to_string(), cost),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_claude_json_envelope() {
        let stdout = r#"{"type":"result","subtype":"success","result":"the answer","total_cost_usd":0.123,"num_turns":3}"#;
        let (text, cost) = parse_agent_output(stdout);
        assert_eq!(text, "the answer");
        assert_eq!(cost, Some(0.123));
    }

    #[test]
    fn json_without_result_field_keeps_envelope_and_cost() {
        let stdout = r#"{"total_cost_usd":0.5,"weird":"shape"}"#;
        let (text, cost) = parse_agent_output(stdout);
        assert!(text.contains("weird"));
        assert_eq!(cost, Some(0.5));
    }

    #[test]
    fn plain_text_falls_back_with_no_cost() {
        let (text, cost) = parse_agent_output("  just some text output\n");
        assert_eq!(text, "just some text output");
        assert_eq!(cost, None);
    }
}
