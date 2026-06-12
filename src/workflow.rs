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
    /// Subdirectory under outputs/ where this workflow's results are written.
    /// Plain name ("legal-ai") → outputs/legal-ai/
    /// Absolute or ~/ path used as-is.
    /// Defaults to outputs/ if not set.
    pub output_dir: Option<String>,
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
    /// Static list of items to fan out over. Alternative to parallel_over when
    /// the items are known at workflow-authoring time and don't need a model step
    /// to generate them. Cannot be combined with parallel_over.
    #[serde(default)]
    pub parallel_items: Vec<String>,
    pub max_concurrent: Option<usize>,
    pub max_items: Option<usize>,
    pub provider: Option<String>,
    pub cost_per_call_usd: Option<f64>,
    /// Files to read before this step runs. Keys become template variables.
    /// E.g. `reads_from = {profile = "~/.flux-private/data/profile.md"}` lets
    /// you write `{{ profile }}` in the prompt.
    #[serde(default)]
    pub reads_from: HashMap<String, String>,
    /// File to append this step's output to after it succeeds.
    /// The file is created if it doesn't exist. Parent directories are created too.
    pub appends_to: Option<String>,
    /// Seconds before an agent call is killed and reported as failed.
    /// Only applies to agent provider steps (claude-code, hermes).
    /// Defaults to 60 if not set.
    pub timeout_secs: Option<u64>,
}

#[derive(Debug, Deserialize)]
pub struct Output {
    pub step: String,
}

fn default_model() -> String {
    "claude-sonnet-4-6".to_string()
}
