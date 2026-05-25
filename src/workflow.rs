use std::collections::HashMap;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct WorkflowFile {
    pub workflow: WorkflowMeta,
    pub budget: Budget,
    #[serde(default)]
    pub inputs: HashMap<String, InputDef>,
    pub steps: Vec<Step>,
    pub output: Output,
}

#[derive(Debug, Deserialize)]
pub struct WorkflowMeta {
    pub name: String,
    pub description: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Budget {
    pub max_usd: f64,
}

#[derive(Debug, Deserialize)]
pub struct InputDef {
    pub description: String,
}

#[derive(Debug, Deserialize)]
pub struct Step {
    pub id: String,
    #[serde(default = "default_model")]
    pub model: String,
    pub prompt: String,
    #[serde(default)]
    pub depends_on: Vec<String>,
    #[serde(default)]
    pub tools: Vec<String>,
    pub max_tool_calls: Option<u32>,
    pub parallel_over: Option<String>,
    pub max_concurrent: Option<usize>,
}

#[derive(Debug, Deserialize)]
pub struct Output {
    pub step: String,
}

fn default_model() -> String {
    "claude-sonnet-4-6".to_string()
}
