use std::collections::{HashMap, HashSet};
use std::time::{Duration, Instant};

use tracing::{info, instrument};

use crate::{
    FluxError,
    provider::{CompletionRequest, LlmProvider},
    workflow::{Step, WorkflowFile},
};

pub struct Engine {
    provider: Box<dyn LlmProvider>,
}

pub struct RunResult {
    pub output: String,
    pub step_results: Vec<StepResult>,
    pub spent_usd: f64,
    pub duration: Duration,
}

pub struct StepResult {
    pub id: String,
    pub model: String,
    pub prompt: String,
    pub output: String,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cost_usd: f64,
    pub duration: Duration,
}

impl Engine {
    pub fn new(provider: Box<dyn LlmProvider>) -> Self {
        Self { provider }
    }

    #[instrument(skip_all, fields(workflow = %workflow.workflow.name))]
    pub async fn run(
        &self,
        workflow: &WorkflowFile,
        inputs: &HashMap<String, String>,
        budget: &mut BudgetTracker,
    ) -> crate::Result<RunResult> {
        let run_start = Instant::now();
        let sorted = topological_sort(&workflow.steps);
        let mut step_outputs: HashMap<String, String> = HashMap::new();
        let mut step_results: Vec<StepResult> = Vec::new();

        for step in sorted {
            info!(step = %step.id, model = %step.model, "executing step");
            let step_start = Instant::now();

            let prompt = crate::template::render_prompt(&step.prompt, inputs, &step_outputs)?;

            let response = self
                .provider
                .complete(CompletionRequest {
                    model: step.model.clone(),
                    prompt: prompt.clone(),
                    tools: step.tools.clone(),
                    max_tool_calls: step.max_tool_calls.unwrap_or(5),
                    max_tokens: 0,
                })
                .await?;

            let cost = step_cost_usd(&step.model, response.input_tokens, response.output_tokens);
            budget.add(cost)?;

            let step_duration = step_start.elapsed();
            info!(
                step = %step.id,
                input_tokens = response.input_tokens,
                output_tokens = response.output_tokens,
                cost_usd = format!("{:.4}", cost),
                duration_secs = format!("{:.1}", step_duration.as_secs_f64()),
                "step complete"
            );

            step_results.push(StepResult {
                id: step.id.clone(),
                model: step.model.clone(),
                prompt,
                output: response.content.clone(),
                input_tokens: response.input_tokens,
                output_tokens: response.output_tokens,
                cost_usd: cost,
                duration: step_duration,
            });

            step_outputs.insert(step.id.clone(), response.content);
        }

        let output = step_outputs.remove(&workflow.output.step).ok_or_else(|| {
            FluxError::Config(format!(
                "output step '{}' did not produce any output",
                workflow.output.step
            ))
        })?;

        Ok(RunResult {
            output,
            step_results,
            spent_usd: budget.spent_usd,
            duration: run_start.elapsed(),
        })
    }
}

pub struct BudgetTracker {
    pub limit_usd: f64,
    pub spent_usd: f64,
}

impl BudgetTracker {
    pub fn new(limit_usd: f64) -> Self {
        Self {
            limit_usd,
            spent_usd: 0.0,
        }
    }

    pub fn add(&mut self, cost: f64) -> crate::Result<()> {
        self.spent_usd += cost;
        if self.spent_usd > self.limit_usd {
            return Err(FluxError::BudgetExceeded {
                spent: self.spent_usd,
                limit: self.limit_usd,
            });
        }
        Ok(())
    }
}

// Exposed so main.rs can show per-step cost without re-implementing the formula.
pub fn step_cost_usd(model: &str, input_tokens: u32, output_tokens: u32) -> f64 {
    let (input_price, output_price) = model_pricing(model);
    (input_tokens as f64 * input_price + output_tokens as f64 * output_price) / 1_000_000.0
}

// Approximate pricing per million tokens (input, output) as of 2026.
fn model_pricing(model: &str) -> (f64, f64) {
    if model.contains("opus") {
        (15.0, 75.0)
    } else if model.contains("sonnet") {
        (3.0, 15.0)
    } else if model.contains("haiku") {
        (0.80, 4.0)
    } else {
        (3.0, 15.0)
    }
}

// Returns steps in execution order: every step appears after all its dependencies.
// DFS post-order naturally satisfies this: we recurse into deps before appending self.
fn topological_sort(steps: &[Step]) -> Vec<&Step> {
    let by_id: HashMap<&str, &Step> = steps.iter().map(|s| (s.id.as_str(), s)).collect();
    let mut visited: HashSet<&str> = HashSet::new();
    let mut result: Vec<&Step> = Vec::new();

    for step in steps {
        if !visited.contains(step.id.as_str()) {
            dfs_topo(step.id.as_str(), &by_id, &mut visited, &mut result);
        }
    }

    result
}

fn dfs_topo<'a>(
    id: &'a str,
    by_id: &HashMap<&'a str, &'a Step>,
    visited: &mut HashSet<&'a str>,
    result: &mut Vec<&'a Step>,
) {
    visited.insert(id);
    for dep in &by_id[id].depends_on {
        let dep = dep.as_str();
        if !visited.contains(dep) {
            dfs_topo(dep, by_id, visited, result);
        }
    }
    result.push(by_id[id]);
}
