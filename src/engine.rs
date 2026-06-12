use std::collections::{HashMap, HashSet};
use std::io::Write;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::Utc;
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

            // Augment the base inputs with any state file contents declared in reads_from.
            // Each key in reads_from becomes a template variable for this step's prompt.
            let step_inputs = load_state_for_step(step, inputs)?;

            // Snapshot budget before this step so we can refund a parallel pre-charge
            // if the step gets budget-killed before any agents actually run.
            let pre_step_spent = budget.spent_usd;

            let attempt = if step.parallel_over.is_some() || !step.parallel_items.is_empty() {
                self.run_parallel(step, &step_inputs, &step_outputs, budget, step_start)
                    .await
            } else {
                let provider = step.provider.as_deref().unwrap_or("anthropic");
                match provider {
                    "hermes" | "claude-code" => {
                        self.run_single_agent(step, &step_inputs, &step_outputs, budget, step_start)
                            .await
                    }
                    _ => {
                        self.run_llm(step, &step_inputs, &step_outputs, budget, step_start)
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
                    // Parallel steps pre-charge for all N items before any agent runs.
                    // If that pre-charge itself caused the budget exceeded, no real
                    // money was spent — refund it so reported spend stays honest.
                    if step.parallel_over.is_some() {
                        let phantom = spent - pre_step_spent;
                        if phantom > 0.0 {
                            budget.refund(phantom);
                            warn!(
                                step = %step.id,
                                refunded = format!("{:.4}", phantom),
                                "refunded parallel pre-charge: no agents ran"
                            );
                        }
                    }
                    warn!(
                        step = %step.id,
                        spent = format!("{:.4}", budget.spent_usd),
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
                // Template errors mean a variable was missing or malformed.
                // Rather than aborting the whole run, inject a clear placeholder
                // so downstream steps (the synthesizer) still produce output.
                Err(FluxError::Template(ref msg)) => {
                    warn!(step = %step.id, error = %msg, "template error; skipping step with placeholder");
                    StepResult {
                        id: step.id.clone(),
                        model: step.provider.clone().unwrap_or_else(|| step.model.clone()),
                        prompt: String::new(),
                        output: format!(
                            "[SKIPPED — template error in step '{}': {msg}. \
                             Check that all {{{{ variable }}}} references are defined.]",
                            step.id
                        ),
                        input_tokens: 0,
                        output_tokens: 0,
                        cost_usd: 0.0,
                        duration: step_start.elapsed(),
                    }
                }
                Err(e) => return Err(e),
            };

            // Append the step's output to the configured state file, if any.
            // Failure here is non-fatal — warn loudly but don't abort the run.
            if let Some(ref path) = step.appends_to
                && let Err(e) = append_to_state(path, &step_result.output, &workflow.workflow.name, &step.id)
            {
                warn!(step = %step.id, path = %path, error = %e, "failed to append to state file");
            }

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
                timeout_secs: step.timeout_secs.unwrap_or(300),
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

    // Fans out over a list of items, running one agent call per item with bounded
    // concurrency. Items come from either parallel_items (static TOML list) or by
    // parsing the output of the upstream step named in parallel_over.
    async fn run_parallel(
        &self,
        step: &Step,
        inputs: &HashMap<String, String>,
        step_outputs: &HashMap<String, String>,
        budget: &mut BudgetTracker,
        step_start: Instant,
    ) -> crate::Result<StepResult> {
        // Resolve items from whichever source is configured. parallel_items takes
        // a static list directly from the TOML; parallel_over parses lines from
        // an upstream step's output. The validator guarantees exactly one is set.
        let mut items: Vec<String> = if !step.parallel_items.is_empty() {
            step.parallel_items.clone()
        } else {
            let upstream_id = step.parallel_over.as_deref().unwrap();
            let upstream_output = step_outputs.get(upstream_id).ok_or_else(|| {
                FluxError::Config(format!(
                    "step '{}' parallel_over '{}' which has not yet run",
                    step.id, upstream_id
                ))
            })?;

            // Strip lines that are clearly decorative rather than list items:
            // blank lines, markdown headers, and lines with no alphanumeric content
            // (e.g. "---", "==="). We also strip leading list markers ("- ", "* ",
            // "1. ") so planners that add bullets still produce clean items.
            // The old heuristic (len <= 80) was wrong for structured output like
            // "Company | Role | https://..." which can be well over 80 chars.
            let parsed: Vec<String> = upstream_output
                .lines()
                .filter(|l| !l.trim().is_empty())
                .filter(|l| !l.trim_start().starts_with('#'))
                .filter(|l| l.trim().chars().any(|c| c.is_alphanumeric()))
                .map(|l| strip_list_marker(l.trim()).to_string())
                .collect();

            if parsed.is_empty() {
                return Err(FluxError::Config(format!(
                    "step '{}' parallel_over '{}' produced no items to fan out over",
                    step.id, upstream_id
                )));
            }
            parsed
        };

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
        // Capture as a plain u64 so it can be copied into each async move block
        // without needing a clone or borrow. u64 is Copy.
        let timeout_secs = step.timeout_secs.unwrap_or(300);

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
                    let result = agent.run(AgentRequest { task: prompt, timeout_secs }).await;
                    (item, result)
                }
            })
            .collect();

        let results = join_all(futures).await;

        // Refund pre-charged cost for every call that failed. The pre-charge
        // prevents over-committing the budget before agents fire, but a failed
        // call (bad invocation, immediate exit, permission error) incurs no real
        // API spend — don't report it as money spent.
        let failed_count = results.iter().filter(|(_, r)| r.is_err()).count();
        if failed_count > 0 {
            let refund = resolve_cost(step, failed_count);
            budget.refund(refund);
            warn!(
                step = %step.id,
                failed = failed_count,
                refunded_usd = format!("{:.4}", refund),
                "refunding pre-charged cost for failed agent calls"
            );
        }

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

    // Rolls back a phantom pre-charge. Used when a parallel step pre-charges
    // for N agent calls but gets budget-killed before any agent actually runs —
    // the reported spend should reflect real API calls, not a failed estimate.
    pub fn refund(&mut self, amount: f64) {
        self.spent_usd -= amount;
    }
}

// Loads state files declared in a step's reads_from map and merges them into
// a copy of base_inputs. Missing files produce a warning and inject an empty
// string so first-run workflows work before any state has been accumulated.
fn load_state_for_step(
    step: &crate::workflow::Step,
    base_inputs: &HashMap<String, String>,
) -> crate::Result<HashMap<String, String>> {
    if step.reads_from.is_empty() {
        return Ok(base_inputs.clone());
    }
    let mut inputs = base_inputs.clone();
    for (var_name, raw_path) in &step.reads_from {
        let path = crate::expand_tilde(raw_path);
        match read_state_path(&path) {
            Ok(Some(content)) => {
                inputs.insert(var_name.clone(), content);
            }
            Ok(None) => {
                warn!(
                    step = %step.id,
                    var = %var_name,
                    path = %path,
                    "state path not found or empty; injecting empty string"
                );
                inputs.insert(var_name.clone(), String::new());
            }
            Err(e) => return Err(e),
        }
    }
    Ok(inputs)
}

// Reads a state path that may be either a file or a directory.
// For a file: returns its contents, or None if it doesn't exist.
// For a directory: returns the contents of the most recently modified file
// in the directory, or None if the directory is empty or doesn't exist.
fn read_state_path(path: &str) -> crate::Result<Option<String>> {
    let p = std::path::Path::new(path);

    if p.is_dir() {
        let mut entries: Vec<_> = std::fs::read_dir(p)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_file())
            .collect();

        // Sort by modification time so the most recent file is last.
        entries.sort_by_key(|e| {
            e.metadata()
                .and_then(|m| m.modified())
                .unwrap_or(std::time::SystemTime::UNIX_EPOCH)
        });

        match entries.last() {
            Some(entry) => {
                let file_path = entry.path();
                info!(path = %file_path.display(), "reading most recent file from state directory");
                Ok(Some(std::fs::read_to_string(&file_path)?))
            }
            None => Ok(None),
        }
    } else {
        match std::fs::read_to_string(p) {
            Ok(content) => Ok(Some(content)),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(e) => Err(crate::FluxError::Io(e)),
        }
    }
}

// Appends a step's output to a state file with a timestamped header.
// Creates the file (and any parent directories) if it doesn't exist.
fn append_to_state(
    raw_path: &str,
    output: &str,
    workflow_name: &str,
    step_id: &str,
) -> crate::Result<()> {
    let path = crate::expand_tilde(raw_path);
    if let Some(parent) = std::path::Path::new(&path).parent() {
        std::fs::create_dir_all(parent)?;
    }
    let timestamp = Utc::now().format("%Y-%m-%d %H:%M UTC").to_string();
    let entry = format!(
        "\n\n---\n\n_Added {timestamp} by `{workflow_name}` / step `{step_id}`_\n\n{output}\n"
    );
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)?;
    file.write_all(entry.as_bytes())?;
    info!(step = %step_id, path = %path, "appended output to state file");
    Ok(())
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

// Strips common markdown list markers from the start of a line so planners
// that add bullets ("- Harvey", "1. Harvey") still produce clean item strings.
fn strip_list_marker(s: &str) -> &str {
    // Numbered: "1. ", "12. "
    if let Some(rest) = s.split_once(". ")
        && rest.0.chars().all(|c| c.is_ascii_digit())
        && !rest.0.is_empty()
    {
        return rest.1;
    }
    // Unordered: "- ", "* ", "• "
    for prefix in &["- ", "* ", "• "] {
        if let Some(rest) = s.strip_prefix(prefix) {
            return rest;
        }
    }
    s
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
