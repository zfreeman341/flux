use std::process::Stdio;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use tokio::io::AsyncWriteExt;
use tracing::{debug, instrument};

use crate::{
    FluxError,
    provider::{AgentProvider, AgentRequest, AgentResponse, Billing},
};

/// Shells out to any agent CLI as `binary [args...] "<task>"`, capturing stdout.
pub struct CliAgentProvider {
    binary: String,
    // Inserted between the binary and the task: ["-p"] for claude, ["-z"] for hermes.
    args: Vec<String>,
    scrub_api_key: bool,
    task_via_stdin: bool,
}

impl CliAgentProvider {
    pub fn new(binary: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            binary: binary.into(),
            args,
            scrub_api_key: false,
            task_via_stdin: false,
        }
    }

    /// Hides ANTHROPIC_API_KEY so the CLI uses its own login instead of metered credits.
    pub fn without_api_key(mut self) -> Self {
        self.scrub_api_key = true;
        self
    }

    /// Pipes the task to the CLI's stdin instead of passing it as an argument.
    ///
    /// cursor-agent silently exits 0 with empty output when an argv prompt nears
    /// 500KB, so any CLI that reads stdin in print mode should use this.
    pub fn task_via_stdin(mut self) -> Self {
        self.task_via_stdin = true;
        self
    }

    pub fn claude_code() -> Self {
        Self::new(
            "claude",
            vec![
                "-p".to_string(),
                // Must be --flag=value; the variadic space form swallows the task string.
                "--allowedTools=WebFetch,WebSearch".to_string(),
                // Wraps the reply in an envelope carrying total_cost_usd.
                "--output-format=json".to_string(),
            ],
        )
        .without_api_key()
        .task_via_stdin()
    }

    pub fn hermes() -> Self {
        Self::new("hermes", vec!["-z".to_string()])
    }

    // Bills the Cursor login (notional); the key scrub keeps ANTHROPIC_API_KEY out entirely.
    pub fn cursor() -> Self {
        Self::new(
            "cursor-agent",
            vec![
                "-p".to_string(),
                // Pinned: the CLI's sticky default silently follows the last interactive model.
                "--model=composer-2.5".to_string(),
                "--output-format=text".to_string(),
                // Print mode refuses to start in an untrusted working directory.
                "--trust".to_string(),
                // Print mode denies tool calls needing approval; this allows web lookups.
                "--force".to_string(),
            ],
        )
        .without_api_key()
        .task_via_stdin()
    }
}

#[async_trait]
impl AgentProvider for CliAgentProvider {
    // Without the key the CLI uses its own login, so reported cost is not money spent.
    fn billing(&self) -> Billing {
        if self.scrub_api_key {
            Billing::Notional
        } else {
            Billing::Metered
        }
    }

    #[instrument(skip(self, request), fields(binary = %self.binary))]
    async fn run(&self, request: AgentRequest) -> crate::Result<AgentResponse> {
        let start = Instant::now();

        // No time values on purpose: naming a deadline makes agents pad to fill it.
        let task = format!(
            "[Be efficient: if a page is slow or unreachable, skip it and move on. \
             Prioritise producing complete output over exhaustive coverage.]\n\n{}",
            request.task,
        );

        let mut command = tokio::process::Command::new(&self.binary);
        if self.scrub_api_key {
            command.env_remove("ANTHROPIC_API_KEY");
        }
        command.args(&self.args);
        if self.task_via_stdin {
            command.stdin(Stdio::piped());
        } else {
            command.arg(&task).stdin(Stdio::null());
        }
        command.stdout(Stdio::piped()).stderr(Stdio::piped());

        let via_stdin = self.task_via_stdin;
        let run_child = async {
            let mut child = command.spawn()?;
            if via_stdin {
                // Taking stdin moves it out of the child; dropping it closes the
                // pipe so the CLI sees EOF and knows the prompt is complete.
                let mut stdin = child.stdin.take().expect("stdin was configured as piped");
                stdin.write_all(task.as_bytes()).await?;
            }
            child.wait_with_output().await
        };

        let result =
            tokio::time::timeout(Duration::from_secs(request.timeout_secs), run_child).await;

        // Nested Results: the outer is the timeout, the inner is the spawn.
        let output = match result {
            Err(_elapsed) => {
                return Err(FluxError::Agent(format!(
                    "{} timed out after {}s",
                    self.binary, request.timeout_secs,
                )));
            }
            Ok(Err(e)) => {
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

        // A zero-length answer is never a real answer; cursor-agent produces one
        // (exit 0, silent) when it chokes on input, and accepting it poisons
        // every step downstream.
        if text.trim().is_empty() {
            return Err(FluxError::Agent(format!(
                "'{}' exited 0 but produced no output ({} byte task)",
                self.binary,
                task.len(),
            )));
        }

        Ok(AgentResponse {
            output: text,
            duration: start.elapsed(),
            cost_usd,
        })
    }
}

/// Reads text and cost from a claude-code JSON envelope, falling back to plain stdout.
///
/// Untyped on purpose: only two fields matter, and an envelope change should not break
/// a workflow.
fn parse_agent_output(stdout: &str) -> (String, Option<f64>) {
    let trimmed = stdout.trim();
    let Ok(parsed) = serde_json::from_str::<serde_json::Value>(trimmed) else {
        return (trimmed.to_string(), None);
    };

    let cost = parsed.get("total_cost_usd").and_then(|v| v.as_f64());
    match parsed.get("result").and_then(|v| v.as_str()) {
        Some(text) => (text.to_string(), cost),
        // Keep the raw envelope rather than losing the output silently.
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
