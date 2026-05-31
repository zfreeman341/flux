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
    pub max_tokens: u32,
}

pub struct CompletionResponse {
    pub content: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
}

// --- Agent provider (shells out to an external agent CLI) ---

#[async_trait]
pub trait AgentProvider: Send + Sync {
    async fn run(&self, request: AgentRequest) -> crate::Result<AgentResponse>;
}

pub struct AgentRequest {
    pub task: String,
    pub timeout_secs: u64,
}

pub struct AgentResponse {
    pub output: String,
    pub duration: Duration,
}
