use async_trait::async_trait;

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
