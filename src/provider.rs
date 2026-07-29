use std::time::Duration;

use async_trait::async_trait;

// --- LLM provider (direct API calls) ---

#[async_trait]
pub trait LlmProvider: Send + Sync {
    async fn complete(&self, request: CompletionRequest) -> crate::Result<CompletionResponse>;
}

pub struct CompletionRequest {
    pub model: String,
    pub prompt: String,
    pub tools: Vec<String>,
    pub max_tool_calls: u32,
    /// None means the provider's default. Option instead of a 0-sentinel:
    /// the type says "maybe absent" so callers can't confuse absent with zero.
    pub max_tokens: Option<u32>,
}

pub struct CompletionResponse {
    pub content: String,
    /// Cited sources ("Title — URL") from web_search results, deduplicated.
    /// Empty when the step used no web tools or cited nothing.
    pub sources: Vec<String>,
    pub input_tokens: u32,
    pub output_tokens: u32,
    /// Web searches the server executed during this request. Billed per call
    /// on top of tokens, so cost tracking needs the count.
    pub web_search_requests: u32,
}

// --- Agent provider (shells out to an external agent CLI) ---

/// Whether the spend a provider reports is real money or an equivalent figure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Billing {
    /// Metered API credits. Enforced against the workflow's budget ceiling.
    Metered,
    /// Charged to a subscription the CLI is signed in to. The reported cost is what the
    /// work *would* have cost on the API, so it is tracked for visibility but never
    /// enforced — a cap on it would abort a run that costs nothing.
    Notional,
}

#[async_trait]
pub trait AgentProvider: Send + Sync {
    async fn run(&self, request: AgentRequest) -> crate::Result<AgentResponse>;

    /// Providers opt in to notional accounting; anything unknown is assumed to cost money.
    fn billing(&self) -> Billing {
        Billing::Metered
    }
}

pub struct AgentRequest {
    pub task: String,
    pub timeout_secs: u64,
}

pub struct AgentResponse {
    pub output: String,
    pub duration: Duration,
    /// Actual API spend reported by the agent CLI, when it can report one
    /// (claude-code's JSON output does; hermes doesn't). None means the
    /// engine falls back to the step's cost_per_call_usd estimate.
    pub cost_usd: Option<f64>,
}
