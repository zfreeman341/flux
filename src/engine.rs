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
    cost::{self, BudgetTracker},
    provider::{AgentProvider, AgentRequest, CompletionRequest, LlmProvider},
    workflow::{Step, WorkflowFile},
};

pub struct Engine {
    // Arc (atomic reference count) instead of Box because we clone these handles
    // for each concurrent task in a parallel fan-out. Box gives unique ownership
    // (one owner); Arc gives shared ownership (many owners, freed when all drop).
    // The types must be Send + Sync because they cross task/thread boundaries.
    llm: Arc<dyn LlmProvider>,
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
    pub fn new(llm: Arc<dyn LlmProvider>, agent: Arc<dyn AgentProvider>) -> Self {
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

        // Reserve budget for the output step so earlier steps can't starve the
        // synthesis that is the whole point of the run. A research fan-out that
        // finishes just under the cap used to leave nothing for synthesis, which
        // then got billed and discarded. Estimates are heuristic (see
        // estimate.rs), so this is a guard rail, not a guarantee — and the output
        // step itself is still allowed to overshoot rather than be discarded
        // (see run_llm). Map lookups default to 0.0, which simply disables the
        // guard for steps the estimator can't price.
        let estimates: HashMap<String, f64> = crate::estimate::estimate_workflow(workflow)
            .steps
            .into_iter()
            .map(|s| (s.id, s.cost_usd))
            .collect();
        let output_reserve = estimates.get(&workflow.output.step).copied().unwrap_or(0.0);

        for step in sorted {
            let step_start = Instant::now();

            // Pre-flight reservation guard. For non-output steps, skip if running
            // this step's estimated cost would dip into the budget reserved for
            // the output step — better to drop a researcher and synthesize on
            // partial results than to spend everything and have nothing left to
            // synthesize with. The output step is exempt: it gets whatever
            // remains. Disabled in emergency mode, where the cap is already off.
            let is_output_step = step.id == workflow.output.step;
            if !is_output_step && !budget.in_emergency() {
                let step_estimate = estimates.get(&step.id).copied().unwrap_or(0.0);
                let headroom = budget.limit_usd - budget.spent_usd - output_reserve;
                if step_estimate > headroom {
                    warn!(
                        step = %step.id,
                        estimate = format!("{:.4}", step_estimate),
                        reserved_for_output = format!("{:.4}", output_reserve),
                        "skipping step to preserve budget for the output step"
                    );
                    let output = format!(
                        "[SKIPPED — estimated cost ${step_estimate:.2} would not leave enough \
                         budget for the '{}' output step. Synthesis below uses the steps that ran.]",
                        workflow.output.step
                    );
                    step_outputs.insert(step.id.clone(), output.clone());
                    step_results.push(StepResult {
                        id: step.id.clone(),
                        model: step.provider.clone().unwrap_or_else(|| step.model.clone()),
                        prompt: String::new(),
                        output,
                        input_tokens: 0,
                        output_tokens: 0,
                        cost_usd: 0.0,
                        duration: step_start.elapsed(),
                    });
                    continue;
                }
            }

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
                && let Err(e) =
                    append_to_state(path, &step_result.output, &workflow.workflow.name, &step.id)
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
                max_tokens: step.max_tokens,
            })
            .await?;

        let cost = cost::call_cost_usd(
            &step.model,
            response.input_tokens,
            response.output_tokens,
            response.web_search_requests,
        );
        // The call already completed and was paid for; never discard a result
        // we bought. If this charge trips the budget, record the overshoot with
        // a warning and return the output anyway — same contract as
        // run_parallel_llm and reconcile_budget. Without this, the synthesis
        // step could run, get billed, and then be thrown away for a placeholder.
        if let Err(e) = budget.add(cost) {
            warn!(
                step = %step.id,
                error = %e,
                "llm step exceeded budget; output kept, downstream steps will be skipped"
            );
        }

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
            output: append_sources(response.content, &response.sources),
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

        let estimated = cost::resolve_cost(step, 1);
        budget.add(estimated)?;

        let prompt = crate::template::render_prompt(&step.prompt, inputs, step_outputs)?;
        let response = self
            .agent
            .run(AgentRequest {
                task: prompt.clone(),
                timeout_secs: step.timeout_secs.unwrap_or(300),
            })
            .await?;

        // Prefer the agent's reported spend over the pre-charged estimate.
        let cost = response.cost_usd.unwrap_or(estimated);
        reconcile_budget(budget, &step.id, cost - estimated);

        let duration = step_start.elapsed();
        info!(
            step = %step.id,
            cost_usd = format!("{:.4}", cost),
            actual = response.cost_usd.is_some(),
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

        // Build (item, rendered_prompt) pairs synchronously before entering any
        // async block — and before any budget charge, so a template error can't
        // leave a phantom pre-charge behind. render_prompt borrows step_outputs
        // safely without cloning the whole map into every future.
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

        info!(step = %step.id, items = item_prompts.len(), "starting parallel fan-out");

        // Dispatch on provider: agent CLIs fan out as subprocesses; everything
        // else fans out as direct API calls, honoring the step's model and tools.
        match step.provider.as_deref() {
            Some("hermes") | Some("claude-code") => {
                self.run_parallel_agent(step, item_prompts, budget, step_start)
                    .await
            }
            _ => {
                self.run_parallel_llm(step, item_prompts, budget, step_start)
                    .await
            }
        }
    }

    // Agent fan-out: one subprocess per item. Cost is pre-charged from
    // cost_per_call_usd before any agent fires — agents don't report usage
    // up front, and an N-item fan-out must not over-commit the budget.
    // Failed calls are refunded after the fact.
    async fn run_parallel_agent(
        &self,
        step: &Step,
        item_prompts: Vec<(String, String)>,
        budget: &mut BudgetTracker,
        step_start: Instant,
    ) -> crate::Result<StepResult> {
        let n = item_prompts.len();
        // Resolve the per-call estimate once: it's both the pre-charge unit
        // and the fallback for agents that don't report actual cost.
        let per_call = cost::resolve_cost(step, 1);
        budget.add(per_call * n as f64)?;

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
                    let result = agent
                        .run(AgentRequest {
                            task: prompt,
                            timeout_secs,
                        })
                        .await;
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
            let refund = per_call * failed_count as f64;
            budget.refund(refund);
            warn!(
                step = %step.id,
                failed = failed_count,
                refunded_usd = format!("{:.4}", refund),
                "refunding pre-charged cost for failed agent calls"
            );
        }

        let mut output_parts: Vec<String> = Vec::new();
        let mut actual_cost = 0.0f64;
        for (item, result) in results {
            match result {
                Ok(resp) => {
                    actual_cost += resp.cost_usd.unwrap_or(per_call);
                    info!(
                        step = %step.id,
                        item = %item,
                        cost_usd = ?resp.cost_usd,
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

        // Replace the estimated charge for the calls that ran with what the
        // agents actually reported spending.
        let charged = per_call * (n - failed_count) as f64;
        reconcile_budget(budget, &step.id, actual_cost - charged);

        let combined = output_parts.join("\n\n");
        let duration = step_start.elapsed();
        let provider_name = step.provider.as_deref().unwrap_or("agent");

        info!(
            step = %step.id,
            items = n,
            cost_usd = format!("{:.4}", actual_cost),
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
            cost_usd: actual_cost,
            duration,
        })
    }

    // API fan-out: one direct Anthropic call per item, honoring the step's
    // model and tools. Roughly an order of magnitude cheaper per item than an
    // agent subprocess for search-and-summarize work, and rate limits are
    // handled by the client's own 429 backoff.
    //
    // Budgeting differs from the agent path: there is no pre-charge, because
    // actual token costs are only known after each call returns, and updating
    // the budget from inside concurrent futures would need a Mutex. Instead
    // the summed actual cost is charged once after join_all. The overshoot
    // window is bounded by (items × per-item cost) — small for API calls,
    // which is exactly why the agent path keeps its pre-charge and this one
    // doesn't.
    async fn run_parallel_llm(
        &self,
        step: &Step,
        item_prompts: Vec<(String, String)>,
        budget: &mut BudgetTracker,
        step_start: Instant,
    ) -> crate::Result<StepResult> {
        let semaphore = Arc::new(Semaphore::new(step.max_concurrent.unwrap_or(3)));
        let llm = Arc::clone(&self.llm);
        let step_id = step.id.clone();
        let max_tool_calls = step.max_tool_calls.unwrap_or(5);
        // Option<u32> is Copy, so each async move block gets its own copy.
        let max_tokens = step.max_tokens;

        let futures: Vec<_> = item_prompts
            .into_iter()
            .map(|(item, prompt)| {
                let sem = Arc::clone(&semaphore);
                let llm = Arc::clone(&llm);
                let step_id = step_id.clone();
                // Clone per future: each async move block takes ownership of
                // its own copies, so nothing borrows `step` across an await.
                let model = step.model.clone();
                let tools = step.tools.clone();
                async move {
                    let _permit = sem.acquire().await.unwrap();
                    info!(step = %step_id, item = %item, "api call start");
                    let result = llm
                        .complete(CompletionRequest {
                            model,
                            prompt,
                            tools,
                            max_tool_calls,
                            max_tokens,
                        })
                        .await;
                    (item, result)
                }
            })
            .collect();

        let results = join_all(futures).await;

        let mut output_parts: Vec<String> = Vec::new();
        let mut input_tokens = 0u32;
        let mut output_tokens = 0u32;
        let mut total_cost = 0.0f64;
        for (item, result) in results {
            match result {
                Ok(resp) => {
                    total_cost += cost::call_cost_usd(
                        &step.model,
                        resp.input_tokens,
                        resp.output_tokens,
                        resp.web_search_requests,
                    );
                    input_tokens += resp.input_tokens;
                    output_tokens += resp.output_tokens;
                    output_parts.push(format!(
                        "--- {} ---\n{}",
                        item,
                        append_sources(resp.content, &resp.sources)
                    ));
                }
                Err(e) => {
                    warn!(step = %step.id, item = %item, error = %e, "api call failed, skipping");
                    output_parts.push(format!("--- {} ---\n[FAILED: {}]", item, e));
                }
            }
        }

        // Charge actual spend. If this trips the budget the money is already
        // spent and the outputs exist — keep them, record the spend, and let
        // the next step's charge trigger the graceful-skip path.
        if let Err(e) = budget.add(total_cost) {
            warn!(
                step = %step.id,
                error = %e,
                "parallel api step exceeded budget; downstream steps will be skipped"
            );
        }

        let combined = output_parts.join("\n\n");
        let duration = step_start.elapsed();
        info!(
            step = %step.id,
            input_tokens,
            output_tokens,
            cost_usd = format!("{:.4}", total_cost),
            duration_secs = format!("{:.1}", duration.as_secs_f64()),
            "parallel api fan-out complete"
        );

        Ok(StepResult {
            id: step.id.clone(),
            model: step.model.clone(),
            prompt: step.prompt.clone(),
            output: combined,
            input_tokens,
            output_tokens,
            cost_usd: total_cost,
            duration,
        })
    }
}

// Appends a Sources section to a step's output so cited URLs flow into
// downstream prompts, the final result, and run artifacts. Verifiability is
// the point of capturing citations — keep them attached to the text they
// support.
fn append_sources(mut output: String, sources: &[String]) -> String {
    if sources.is_empty() {
        return output;
    }
    output.push_str("\n\nSources:\n");
    for source in sources {
        output.push_str("- ");
        output.push_str(source);
        output.push('\n');
    }
    output
}

// Adjusts the budget after actual costs come in, replacing what was
// pre-charged from estimates. A positive delta can trip the limit — but at
// that point the money is already spent, so record it with a warning and let
// the next step's charge trigger the graceful-skip path, rather than
// discarding work that already happened (and was already paid for).
fn reconcile_budget(budget: &mut BudgetTracker, step_id: &str, delta: f64) {
    if delta > 0.0 {
        if let Err(e) = budget.add(delta) {
            warn!(
                step = %step_id,
                error = %e,
                "actual agent cost exceeded budget; downstream steps will be skipped"
            );
        }
    } else if delta < 0.0 {
        budget.refund(-delta);
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

// Strips common markdown list markers from the start of a line so planners
// that add bullets ("- Acme", "1. Acme") still produce clean item strings.
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

#[cfg(test)]
mod tests {
    use super::*;

    // Deserialize from TOML instead of writing 15-field struct literals —
    // exercises the same parse path real workflows use.
    fn step(src: &str) -> Step {
        toml::from_str(src).expect("test step TOML should parse")
    }

    fn position(sorted: &[&Step], id: &str) -> usize {
        sorted
            .iter()
            .position(|s| s.id == id)
            .unwrap_or_else(|| panic!("step '{id}' missing from sort output"))
    }

    #[test]
    fn topo_sort_puts_dependencies_before_dependents() {
        let steps = vec![
            step("id = \"synth\"\nprompt = \"p\"\ndepends_on = [\"research\"]"),
            step("id = \"research\"\nprompt = \"p\"\ndepends_on = [\"plan\"]"),
            step("id = \"plan\"\nprompt = \"p\""),
        ];
        let sorted = topological_sort(&steps);
        assert_eq!(sorted.len(), 3);
        assert!(position(&sorted, "plan") < position(&sorted, "research"));
        assert!(position(&sorted, "research") < position(&sorted, "synth"));
    }

    #[test]
    fn topo_sort_handles_diamond() {
        // a -> (b, c) -> d, declared in scrambled order.
        let steps = vec![
            step("id = \"d\"\nprompt = \"p\"\ndepends_on = [\"b\", \"c\"]"),
            step("id = \"b\"\nprompt = \"p\"\ndepends_on = [\"a\"]"),
            step("id = \"c\"\nprompt = \"p\"\ndepends_on = [\"a\"]"),
            step("id = \"a\"\nprompt = \"p\""),
        ];
        let sorted = topological_sort(&steps);
        assert_eq!(sorted.len(), 4);
        assert!(position(&sorted, "a") < position(&sorted, "b"));
        assert!(position(&sorted, "a") < position(&sorted, "c"));
        assert!(position(&sorted, "b") < position(&sorted, "d"));
        assert!(position(&sorted, "c") < position(&sorted, "d"));
    }

    #[test]
    fn topo_sort_keeps_independent_steps() {
        let steps = vec![
            step("id = \"x\"\nprompt = \"p\""),
            step("id = \"y\"\nprompt = \"p\""),
        ];
        let sorted = topological_sort(&steps);
        assert_eq!(sorted.len(), 2);
    }

    #[test]
    fn strip_list_marker_handles_bullets_and_numbers() {
        assert_eq!(strip_list_marker("- Acme"), "Acme");
        assert_eq!(strip_list_marker("* Acme"), "Acme");
        assert_eq!(strip_list_marker("• Acme"), "Acme");
        assert_eq!(strip_list_marker("1. Acme"), "Acme");
        assert_eq!(strip_list_marker("12. Acme"), "Acme");
    }

    #[test]
    fn strip_list_marker_leaves_plain_text_alone() {
        assert_eq!(strip_list_marker("Acme"), "Acme");
        // No space after the dot: not a list marker.
        assert_eq!(strip_list_marker("1.Acme"), "1.Acme");
        // Non-numeric prefix before ". " is content, not a marker.
        assert_eq!(strip_list_marker("Inc. Acme"), "Inc. Acme");
    }

    // --- end-to-end engine tests with mock providers ---
    //
    // Implementing the provider traits on test doubles is the payoff of the
    // trait-object design: the whole engine runs against canned responses
    // with no network, no API key, and no money.

    use crate::provider::{AgentResponse, CompletionResponse};
    use async_trait::async_trait;

    struct MockLlm;

    #[async_trait]
    impl LlmProvider for MockLlm {
        async fn complete(&self, request: CompletionRequest) -> crate::Result<CompletionResponse> {
            // A planner prompt yields two fan-out items; anything else echoes
            // its prompt so tests can see exactly which call produced what.
            let (content, sources) = if request.prompt.contains("PLAN") {
                ("alpha\nbeta".to_string(), vec![])
            } else {
                (
                    format!("researched[{}]", request.prompt),
                    vec!["Mock Source — https://example.com/mock".to_string()],
                )
            };
            Ok(CompletionResponse {
                content,
                sources,
                input_tokens: 100,
                output_tokens: 50,
                web_search_requests: 0,
            })
        }
    }

    struct MockAgent;

    #[async_trait]
    impl AgentProvider for MockAgent {
        async fn run(&self, request: AgentRequest) -> crate::Result<AgentResponse> {
            Ok(AgentResponse {
                output: format!("agent[{}]", request.task),
                duration: Duration::from_millis(1),
                // No reported cost: the engine falls back to cost_per_call_usd.
                cost_usd: None,
            })
        }
    }

    fn workflow(fan_step_extra: &str) -> WorkflowFile {
        toml::from_str(&format!(
            r#"
[workflow]
name = "t"
[budget]
max_usd = 5.0
[[steps]]
id = "plan"
prompt = "PLAN"
[[steps]]
id = "fan"
prompt = "look into {{{{ item }}}}"
parallel_over = "plan"
depends_on = ["plan"]
{fan_step_extra}
[[steps]]
id = "synth"
prompt = "combine {{{{ fan.output }}}}"
depends_on = ["fan"]
[output]
step = "synth"
"#
        ))
        .expect("test workflow TOML should parse")
    }

    #[tokio::test]
    async fn parallel_fan_out_without_provider_uses_the_api() {
        let wf = workflow("");
        let engine = Engine::new(Arc::new(MockLlm), Arc::new(MockAgent));
        let mut budget = BudgetTracker::new(5.0);

        let result = engine
            .run(&wf, &HashMap::new(), &mut budget)
            .await
            .expect("workflow should succeed");

        let fan = result.step_results.iter().find(|s| s.id == "fan").unwrap();
        assert!(fan.output.contains("--- alpha ---"), "{}", fan.output);
        assert!(fan.output.contains("--- beta ---"), "{}", fan.output);
        assert!(
            fan.output.contains("researched[look into alpha]"),
            "fan-out should hit the API mock, not the agent: {}",
            fan.output
        );
        // Two items × the mock's 100/50 tokens, actually charged to the budget.
        assert_eq!(fan.input_tokens, 200);
        assert_eq!(fan.output_tokens, 100);
        assert!(fan.cost_usd > 0.0);
        assert!(budget.spent_usd >= fan.cost_usd);
        // Cited sources are appended per item so they flow downstream.
        assert!(
            fan.output
                .contains("Sources:\n- Mock Source — https://example.com/mock"),
            "{}",
            fan.output
        );
    }

    #[tokio::test]
    async fn parallel_fan_out_with_agent_provider_uses_the_agent() {
        let wf = workflow("provider = \"claude-code\"\ncost_per_call_usd = 0.1");
        let engine = Engine::new(Arc::new(MockLlm), Arc::new(MockAgent));
        let mut budget = BudgetTracker::new(5.0);

        let result = engine
            .run(&wf, &HashMap::new(), &mut budget)
            .await
            .expect("workflow should succeed");

        let fan = result.step_results.iter().find(|s| s.id == "fan").unwrap();
        assert!(
            fan.output.contains("agent[look into alpha]"),
            "agent provider should route to the agent mock: {}",
            fan.output
        );
        // Two items pre-charged at the declared cost_per_call_usd.
        assert!((fan.cost_usd - 0.2).abs() < 1e-9);
    }

    // An agent that reports its real spend, like claude-code with
    // --output-format=json.
    struct MockAgentWithCost;

    #[async_trait]
    impl AgentProvider for MockAgentWithCost {
        async fn run(&self, request: AgentRequest) -> crate::Result<AgentResponse> {
            Ok(AgentResponse {
                output: format!("agent[{}]", request.task),
                duration: Duration::from_millis(1),
                cost_usd: Some(0.07),
            })
        }
    }

    #[tokio::test]
    async fn agent_fan_out_reconciles_to_reported_actual_cost() {
        // Estimated $0.10/call, but the agent reports $0.07/call: the step
        // and the budget should both settle on 2 × 0.07, not 2 × 0.10.
        let wf = workflow("provider = \"claude-code\"\ncost_per_call_usd = 0.1");
        let engine = Engine::new(Arc::new(MockLlm), Arc::new(MockAgentWithCost));
        let mut budget = BudgetTracker::new(5.0);

        let result = engine
            .run(&wf, &HashMap::new(), &mut budget)
            .await
            .expect("workflow should succeed");

        let fan = result.step_results.iter().find(|s| s.id == "fan").unwrap();
        assert!((fan.cost_usd - 0.14).abs() < 1e-9, "got {}", fan.cost_usd);

        // Whole-run spend = fan actuals + the two mock LLM steps' token cost.
        let llm_cost: f64 = result
            .step_results
            .iter()
            .filter(|s| s.id != "fan")
            .map(|s| s.cost_usd)
            .sum();
        assert!((budget.spent_usd - (0.14 + llm_cost)).abs() < 1e-9);
    }

    // Fix #1: a completed output step is never discarded, even if its cost
    // pushes spend past the cap. The money is already spent; throwing the
    // result away would mean paying for a synthesis and getting a placeholder.
    #[tokio::test]
    async fn output_step_is_kept_even_when_it_overshoots_budget() {
        let wf: WorkflowFile = toml::from_str(
            r#"
[workflow]
name = "t"
[budget]
max_usd = 1.0
[[steps]]
id = "only"
prompt = "summarize the thing"
[output]
step = "only"
"#,
        )
        .unwrap();
        let engine = Engine::new(Arc::new(MockLlm), Arc::new(MockAgent));
        // A budget far below any real call cost: the single step must overshoot.
        let mut budget = BudgetTracker::new(1e-9);

        let result = engine
            .run(&wf, &HashMap::new(), &mut budget)
            .await
            .expect("workflow should succeed even when the output step overshoots");

        // The real output is returned, not the skip placeholder.
        assert!(result.output.contains("researched"), "{}", result.output);
        assert!(!result.output.contains("SKIPPED"), "{}", result.output);
        // The overshoot is recorded honestly rather than hidden.
        assert!(budget.spent_usd > budget.limit_usd);
    }

    // Fix #2: a non-output step is skipped before it runs when its estimated
    // cost would eat into the budget reserved for the output step. The
    // synthesis still runs, on whatever partial results exist.
    #[tokio::test]
    async fn expensive_step_is_skipped_to_preserve_budget_for_output() {
        // `plan` carries a heavy web_search estimate; `synth` is cheap. With a
        // budget that fits synth but not plan-plus-synth, plan must give way.
        let wf: WorkflowFile = toml::from_str(
            r#"
[workflow]
name = "t"
[budget]
max_usd = 0.10
[[steps]]
id = "plan"
prompt = "PLAN"
tools = ["web_search"]
max_tool_calls = 10
[[steps]]
id = "synth"
prompt = "combine {{ plan.output }}"
depends_on = ["plan"]
[output]
step = "synth"
"#,
        )
        .unwrap();
        let engine = Engine::new(Arc::new(MockLlm), Arc::new(MockAgent));
        let mut budget = BudgetTracker::new(0.10);

        let result = engine
            .run(&wf, &HashMap::new(), &mut budget)
            .await
            .expect("workflow should succeed with the expensive step skipped");

        let plan = result.step_results.iter().find(|s| s.id == "plan").unwrap();
        assert!(plan.output.contains("SKIPPED"), "{}", plan.output);
        assert_eq!(plan.cost_usd, 0.0);

        // The output step still ran for real on the partial input.
        let synth = result
            .step_results
            .iter()
            .find(|s| s.id == "synth")
            .unwrap();
        assert!(synth.output.contains("researched"), "{}", synth.output);
        // Only synth spent money, and it stayed within budget.
        assert!(budget.spent_usd > 0.0 && budget.spent_usd <= budget.limit_usd);
    }
}
