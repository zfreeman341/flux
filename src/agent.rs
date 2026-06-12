use std::time::{Duration, Instant};

use async_trait::async_trait;
use tracing::{debug, instrument};

use crate::{
    FluxError,
    provider::{AgentProvider, AgentRequest, AgentResponse},
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
}

impl CliAgentProvider {
    pub fn new(binary: impl Into<String>, args: Vec<String>) -> Self {
        Self {
            binary: binary.into(),
            args,
        }
    }

    // Factory: `claude -p --allowedTools=WebFetch,WebSearch "<task>"`
    //
    // --allowedTools pre-approves WebFetch and WebSearch so agents running
    // as subprocesses don't hit permission prompts. Without this, agents
    // trying to fetch career pages block or time out waiting for approval
    // that can never come in non-interactive mode.
    //
    // The flag MUST use the --flag=value form (not --flag value). The claude
    // CLI defines --allowedTools as variadic (<tools...>), so a bare
    // `--allowedTools WebFetch,WebSearch "<task>"` consumes the task string
    // as a second tool name and leaves the prompt argument missing.
    pub fn claude_code() -> Self {
        Self::new(
            "claude",
            vec![
                "-p".to_string(),
                "--allowedTools=WebFetch,WebSearch".to_string(),
            ],
        )
    }

    // Factory: `hermes -z "<task>"`
    pub fn hermes() -> Self {
        Self::new("hermes", vec!["-z".to_string()])
    }
}

#[async_trait]
impl AgentProvider for CliAgentProvider {
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
        let result = tokio::time::timeout(
            Duration::from_secs(request.timeout_secs),
            tokio::process::Command::new(&self.binary)
                .args(&self.args)
                .arg(&task)
                .output(),
        )
        .await;

        // Two layers of Result to unwrap:
        //   outer: Err(Elapsed) if timed out
        //   inner: Err(io::Error) if spawn failed (e.g. binary not on PATH)
        let output = match result {
            Err(_elapsed) => {
                return Err(FluxError::Agent(format!(
                    "{} timed out after {}s",
                    self.binary,
                    request.timeout_secs,
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

        Ok(AgentResponse {
            output: String::from_utf8_lossy(&output.stdout).trim().to_string(),
            duration: start.elapsed(),
        })
    }
}
