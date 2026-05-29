use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::{Duration, Instant};

use futures::future::join_all;
use tokio::sync::Semaphore;
use tracing::{info, instrument, warn};

use crate::{
    FluxError,
    provider::{AgentProvider, AgentRequest, CompletionRequest, LlmProvider},
    workflow::{Step, WorkflowFile},
};

pub struct Engine {
    llm: Box<dyn LlmProvider>,
    // Arc (atomic reference count) instead of Box because we clone this handle
    // for each concurrent task in a parallel fan-out. Box gives unique ownership
    // (one owner); Arc gives shared ownership (many owners, freed when all drop).
    // The type must be Send + Sync because it crosses task/thread boundaries.
    agent: Arc<dyn AgentProvider>,
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
    pub fn new(llm: Box<dyn LlmProvider>, agent: Arc<dyn AgentProvider>) -> Self {
        Self { llm, agent }
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
            let step_start = Instant::now();

            let attempt = if let Some(ref upstream_id) = step.parallel_over {
                self.run_parallel(step, upstream_id, inputs, &step_outputs, budget, step_start)
                    .await
            } else {
                let provider = step.provider.as_deref().unwrap_or("anthropic");
                match provider {
                    "hermes" | "claude-code" => {
                        self.run_single_agent(step, inputs, &step_outputs, budget, step_start)
                            .await
                    }
                    _ => {
                        self.run_llm(step, inputs, &step_outputs, budget, step_start)
                            .await
                    }
                }
            };

            // On BudgetExceeded: skip this step, inject a placeholder so downstream
            // steps (the synthesizer) can still run on whatever results exist.
            // enable_emergency() disables further budget checks so the output step
            // can spend a small amount to synthesize partial results — better than
            // returning nothing at all.
            let step_result = match attempt {
                Ok(r) => r,
                Err(FluxError::BudgetExceeded { spent, limit }) => {
                    warn!(
                        step = %step.id,
                        spent = format!("{:.4}", spent),
                        limit = format!("{:.4}", limit),
                        "budget exceeded; skipping step and attempting partial synthesis"
                    );
                    budget.enable_emergency();
                    StepResult {
                        id: step.id.clone(),
                        model: step.provider.clone().unwrap_or_else(|| step.model.clone()),
                        prompt: String::new(),
                        output: format!(
                            "[SKIPPED — budget limit of ${limit:.4} reached before this step could run. \
                             Synthesis below is based on steps that completed successfully.]"
                        ),
                        input_tokens: 0,
                        output_tokens: 0,
                        cost_usd: 0.0,
                        duration: step_start.elapsed(),
                    }
                }
                Err(e) => return Err(e),
            };

            step_outputs.insert(step.id.clone(), step_result.output.clone());
            step_results.push(step_result);
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

    // Executes a step via the Anthropic API (the original path).
    async fn run_llm(
        &self,
        step: &Step,
        inputs: &HashMap<String, String>,
        step_outputs: &HashMap<String, String>,
        budget: &mut BudgetTracker,
        step_start: Instant,
    ) -> crate::Result<StepResult> {
        info!(step = %step.id, model = %step.model, "executing llm step");

        let prompt = crate::template::render_prompt(&step.prompt, inputs, step_outputs)?;
        let response = self
            .llm
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

        let duration = step_start.elapsed();
        info!(
            step = %step.id,
            input_tokens = response.input_tokens,
            output_tokens = response.output_tokens,
            cost_usd = format!("{:.4}", cost),
            duration_secs = format!("{:.1}", duration.as_secs_f64()),
            "step complete"
        );

        Ok(StepResult {
            id: step.id.clone(),
            model: step.model.clone(),
            prompt,
            output: response.content,
            input_tokens: response.input_tokens,
            output_tokens: response.output_tokens,
            cost_usd: cost,
            duration,
        })
    }

    // Executes a single (non-parallel) agent call.
    async fn run_single_agent(
        &self,
        step: &Step,
        inputs: &HashMap<String, String>,
        step_outputs: &HashMap<String, String>,
        budget: &mut BudgetTracker,
        step_start: Instant,
    ) -> crate::Result<StepResult> {
        let provider_name = step.provider.as_deref().unwrap_or("agent");
        info!(step = %step.id, provider = %provider_name, "executing agent step");

        let cost = resolve_cost(step, 1);
        budget.add(cost)?;

        let prompt = crate::template::render_prompt(&step.prompt, inputs, step_outputs)?;
        let response = self
            .agent
            .run(AgentRequest {
                task: prompt.clone(),
            })
            .await?;

        let duration = step_start.elapsed();
        info!(
            step = %step.id,
            cost_usd = format!("{:.4}", cost),
            duration_secs = format!("{:.1}", duration.as_secs_f64()),
            "agent step complete"
        );

        Ok(StepResult {
            id: step.id.clone(),
            model: provider_name.to_string(),
            prompt,
            output: response.output,
            input_tokens: 0,
            output_tokens: 0,
            cost_usd: cost,
            duration,
        })
    }

    // Fans out over every line in the upstream step's output, running one
    // agent call per item with bounded concurrency.
    async fn run_parallel(
        &self,
        step: &Step,
        upstream_id: &str,
        inputs: &HashMap<String, String>,
        step_outputs: &HashMap<String, String>,
        budget: &mut BudgetTracker,
        step_start: Instant,
    ) -> crate::Result<StepResult> {
        let upstream_output = step_outputs.get(upstream_id).ok_or_else(|| {
            FluxError::Config(format!(
                "step '{}' parallel_over '{}' which has not yet run",
                step.id, upstream_id
            ))
        })?;

        // Strip lines that are clearly prose rather than list items.
        // LLMs sometimes output reasoning text before the actual list despite
        // format instructions. Lines over 80 chars are almost certainly sentences,
        // not names. This filter runs before max_items so the cap applies to
        // real items, not to noise lines.
        let mut items: Vec<String> = upstream_output
            .lines()
            .filter(|l| !l.trim().is_empty())
            .filter(|l| l.trim().len() <= 80)
            .map(|l| l.trim().to_string())
            .collect();

        if items.is_empty() {
            return Err(FluxError::Config(format!(
                "step '{}' parallel_over '{}' produced no items to fan out over",
                step.id, upstream_id
            )));
        }

        // max_items caps the fan-out so a chatty planner output doesn't pre-charge
        // the budget for 25 items when you only wanted 6. The planner may output
        // formatted text (descriptions, headers) despite instructions — this is the
        // safety valve. A warning makes the truncation visible in logs.
        if let Some(cap) = step.max_items
            && items.len() > cap
        {
            warn!(
                step = %step.id,
                found = items.len(),
                cap,
                "planner output more lines than max_items; truncating fan-out"
            );
            items.truncate(cap);
        }

        let n = items.len();
        let cost = resolve_cost(step, n);
        budget.add(cost)?;

        info!(step = %step.id, items = n, "starting parallel fan-out");

        // Build (item, rendered_prompt) pairs synchronously before entering any
        // async block. This lets render_prompt borrow step_outputs safely without
        // needing to clone the whole map into every future.
        let item_prompts: Vec<(String, String)> = items
            .iter()
            .map(|item| {
                // Inject {{ item }} as a top-level template variable by cloning
                // inputs (which is small) and adding the key for this iteration.
                let mut inputs_with_item = inputs.clone();
                inputs_with_item.insert("item".to_string(), item.clone());
                let prompt =
                    crate::template::render_prompt(&step.prompt, &inputs_with_item, step_outputs)?;
                Ok((item.clone(), prompt))
            })
            .collect::<crate::Result<_>>()?;

        // Semaphore limits how many agent processes can be in-flight at once.
        // Arc because each async block in the map below takes ownership of a clone.
        // The semaphore itself lives on the heap; Arc is just the handle.
        let semaphore = Arc::new(Semaphore::new(step.max_concurrent.unwrap_or(3)));

        // Clone the Arc for use inside the futures. Cloning an Arc is cheap:
        // it increments an atomic counter, no heap allocation.
        let agent = Arc::clone(&self.agent);
        let step_id = step.id.clone();

        // Build the Vec of futures without spawning threads. join_all polls them
        // concurrently on the current task. Since the actual work (subprocess) is
        // OS-level, true parallelism happens regardless — the semaphore just limits
        // how many we start at once.
        let futures: Vec<_> = item_prompts
            .into_iter()
            .map(|(item, prompt)| {
                let sem = Arc::clone(&semaphore);
                let agent = Arc::clone(&agent);
                let step_id = step_id.clone();
                async move {
                    // acquire() waits until a semaphore permit is available,
                    // then returns a SemaphorePermit that releases on drop.
                    // unwrap() is safe here: the semaphore is never closed.
                    let _permit = sem.acquire().await.unwrap();
                    info!(step = %step_id, item = %item, "agent call start");
                    let result = agent.run(AgentRequest { task: prompt }).await;
                    (item, result)
                }
            })
            .collect();

        let results = join_all(futures).await;

        let mut output_parts: Vec<String> = Vec::new();
        for (item, result) in results {
            match result {
                Ok(resp) => {
                    info!(
                        step = %step.id,
                        item = %item,
                        duration_secs = format!("{:.1}", resp.duration.as_secs_f64()),
                        "agent call complete"
                    );
                    output_parts.push(format!("--- {} ---\n{}", item, resp.output));
                }
                Err(e) => {
                    warn!(step = %step.id, item = %item, error = %e, "agent call failed, skipping");
                    output_parts.push(format!("--- {} ---\n[FAILED: {}]", item, e));
                }
            }
        }

        let combined = output_parts.join("\n\n");
        let duration = step_start.elapsed();
        let provider_name = step.provider.as_deref().unwrap_or("agent");

        info!(
            step = %step.id,
            items = n,
            cost_usd = format!("{:.4}", cost),
            duration_secs = format!("{:.1}", duration.as_secs_f64()),
            "parallel fan-out complete"
        );

        Ok(StepResult {
            id: step.id.clone(),
            model: provider_name.to_string(),
            prompt: step.prompt.clone(),
            output: combined,
            input_tokens: 0,
            output_tokens: 0,
            cost_usd: cost,
            duration,
        })
    }
}

pub struct BudgetTracker {
    pub limit_usd: f64,
    pub spent_usd: f64,
    // When true, add() accumulates cost but never errors. Enabled after a step
    // is skipped due to budget so the output step can still synthesize partial results.
    emergency: bool,
}

impl BudgetTracker {
    pub fn new(limit_usd: f64) -> Self {
        Self {
            limit_usd,
            spent_usd: 0.0,
            emergency: false,
        }
    }

    pub fn add(&mut self, cost: f64) -> crate::Result<()> {
        self.spent_usd += cost;
        if self.spent_usd > self.limit_usd && !self.emergency {
            return Err(FluxError::BudgetExceeded {
                spent: self.spent_usd,
                limit: self.limit_usd,
            });
        }
        Ok(())
    }

    // Called when a step is skipped due to BudgetExceeded so subsequent steps
    // (specifically the output/synthesizer step) can still run.
    pub fn enable_emergency(&mut self) {
        self.emergency = true;
    }
}

// Exposed so main.rs can show per-step cost without re-implementing the formula.
pub fn step_cost_usd(model: &str, input_tokens: u32, output_tokens: u32) -> f64 {
    let (input_price, output_price) = model_pricing(model);
    (input_tokens as f64 * input_price + output_tokens as f64 * output_price) / 1_000_000.0
}

// Resolves cost for agent steps. Warns if not set so authors know it's estimated.
fn resolve_cost(step: &Step, count: usize) -> f64 {
    let per_call = match step.cost_per_call_usd {
        Some(c) => c,
        None => {
            warn!(
                step = %step.id,
                "cost_per_call_usd not set; estimating $0.05 per call"
            );
            0.05
        }
    };
    per_call * count as f64
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
