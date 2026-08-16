//! Workflow execution engine.
//!
//! This is the heart of the runtime: it takes a validated workflow and drives
//! it to completion. Steps run in dependency waves computed from each step's
//! `depends_on` edges: every step in a wave depends only on earlier waves, so a
//! wave's steps all run concurrently. Within a step, a `parallel_over` fan-out
//! runs its items concurrently too, bounded by a [`tokio::sync::Semaphore`]
//! (`max_concurrent`, default 3). Note that the number of steps in a wave is
//! *not* bounded — a wide graph fires every step in the wave at once.
//!
//! Failures are handled by graceful skip: if a branch fails after retries, the
//! engine records the failure and continues so that downstream synthesis steps
//! can still run over the surviving outputs. Cost is charged to the shared
//! [`crate::cost::BudgetTracker`] as work completes, and the run aborts once the
//! budget ceiling is crossed.

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use chrono::Utc;
use futures::future::join_all;
use tokio::sync::{Mutex, Semaphore};
use tracing::{info, instrument, warn};

use crate::{
    FluxError,
    artifact::{Artifact, Schema, Source},
    cost::{self, BudgetTracker},
    http::HttpExecutor,
    mcp::{McpExecutor, McpRunContext},
    provider::{AgentProvider, AgentRequest, Billing, CompletionRequest, LlmProvider},
    registry::Registry,
    template::FanItem,
    workflow::{Step, StepKind, WorkflowFile},
};

// Wraps the caller's borrow so steps in a wave can share it without a 'static bound.
type BudgetLock<'a> = Mutex<&'a mut BudgetTracker>;

struct StepContext<'a> {
    workflow: &'a WorkflowFile,
    inputs: &'a HashMap<String, String>,
    step_outputs: &'a HashMap<String, Artifact>,
    /// Pre-run cost estimates by step id, used by the reservation guard below.
    estimates: &'a HashMap<String, f64>,
    output_reserve: f64,
}

pub struct Engine {
    // None when no API key was configured, which is valid if every step is an agent step.
    llm: Option<Arc<dyn LlmProvider>>,
    agent: Arc<dyn AgentProvider>,
    // Named overrides; a step's `provider` selects one, anything else uses `agent`.
    agents: HashMap<String, Arc<dyn AgentProvider>>,
    http: Arc<HttpExecutor>,
    mcp: Arc<McpExecutor>,
    // Where schema, transform, and probe names in a workflow become behavior.
    registry: Arc<Registry>,
    // When set, step results are written here as they complete, surviving a failed run.
    checkpoint_dir: Option<PathBuf>,
}

#[derive(Debug)]
pub struct RunResult {
    pub output: String,
    pub step_results: Vec<StepResult>,
    pub spent_usd: f64,
    /// Subscription-billed spend: what the agent steps would have cost on the API.
    pub notional_usd: f64,
    pub duration: Duration,
}

#[derive(Debug)]
pub struct StepResult {
    pub id: String,
    pub model: String,
    pub prompt: String,
    pub output: Artifact,
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub cost_usd: f64,
    pub duration: Duration,
}

impl StepResult {
    /// The prose form of this step's output, for run artifacts and assertions.
    pub fn text(&self) -> String {
        self.output.as_prompt_text()
    }

    /// The per-step debug artifact: prompt, output, and cost/timing metadata.
    pub fn to_markdown(&self) -> String {
        format!(
            "# {id}\n\nModel: `{model}`  \nTokens: {input_t} in / {output_t} out  \nCost: ${cost:.4}  \nDuration: {dur:.1}s\n\n## Prompt\n\n{prompt}\n\n## Output\n\n{output}\n",
            id = self.id,
            model = self.model,
            input_t = self.input_tokens,
            output_t = self.output_tokens,
            cost = self.cost_usd,
            dur = self.duration.as_secs_f64(),
            prompt = self.prompt,
            output = self.text(),
        )
    }
}

impl Engine {
    pub fn new(llm: Arc<dyn LlmProvider>, agent: Arc<dyn AgentProvider>) -> crate::Result<Self> {
        Ok(Self {
            llm: Some(llm),
            agent,
            agents: HashMap::new(),
            http: Arc::new(HttpExecutor::new(crate::http::client()?)),
            mcp: Arc::new(McpExecutor),
            registry: Arc::new(Registry::builtin()),
            checkpoint_dir: None,
        })
    }

    /// For workflows whose every step runs through an agent CLI, so no API key is needed.
    pub fn agents_only(agent: Arc<dyn AgentProvider>) -> crate::Result<Self> {
        Ok(Self {
            llm: None,
            agent,
            agents: HashMap::new(),
            http: Arc::new(HttpExecutor::new(crate::http::client()?)),
            mcp: Arc::new(McpExecutor),
            registry: Arc::new(Registry::builtin()),
            checkpoint_dir: None,
        })
    }

    /// Replaces the built-in registry, for callers that register their own
    /// schemas, transforms, or probes.
    pub fn with_registry(mut self, registry: Arc<Registry>) -> Self {
        self.registry = registry;
        self
    }

    /// Registers a provider for steps whose `provider` names it. Builder-style,
    /// so `main` can chain registrations after construction.
    pub fn with_agent(mut self, name: impl Into<String>, provider: Arc<dyn AgentProvider>) -> Self {
        self.agents.insert(name.into(), provider);
        self
    }

    /// Writes each step's result into `dir` as it completes, so a run that
    /// fails partway still leaves its finished steps' outputs on disk.
    pub fn with_checkpoint_dir(mut self, dir: PathBuf) -> Self {
        self.checkpoint_dir = Some(dir);
        self
    }

    // Checkpointing is best-effort: a full disk must not kill a healthy run.
    fn checkpoint(&self, step: &StepResult) {
        let Some(dir) = &self.checkpoint_dir else {
            return;
        };
        let write = std::fs::create_dir_all(dir)
            .and_then(|()| std::fs::write(dir.join(format!("{}.md", step.id)), step.to_markdown()));
        if let Err(e) = write {
            warn!(step = %step.id, error = %e, "failed to checkpoint step result");
        }
    }

    fn agent_for(&self, step: &Step) -> &Arc<dyn AgentProvider> {
        step.provider
            .as_deref()
            .and_then(|name| self.agents.get(name))
            .unwrap_or(&self.agent)
    }

    // Unknown names were already rejected by `Step::resolve`, so a miss here is just "none".
    fn output_schema(&self, step: &Step) -> Option<Schema> {
        step.output_type
            .as_deref()
            .and_then(|name| self.registry.schema(name).ok())
    }

    // Cost is an API-equivalent figure, so it needs no pre-charge and no ceiling check.
    fn is_notional(&self, step: &Step) -> bool {
        step.is_agent() && self.agent_for(step).billing() == Billing::Notional
    }

    fn require_llm(&self) -> crate::Result<&Arc<dyn LlmProvider>> {
        self.llm.as_ref().ok_or_else(|| {
            FluxError::Config(
                "step needs the Anthropic API but no client is configured; set ANTHROPIC_API_KEY \
                 or give the step provider = \"claude-code\""
                    .into(),
            )
        })
    }

    #[instrument(skip_all, fields(workflow = %workflow.workflow.name))]
    pub async fn run(
        &self,
        workflow: &WorkflowFile,
        inputs: &HashMap<String, String>,
        budget: &mut BudgetTracker,
    ) -> crate::Result<RunResult> {
        let run_start = Instant::now();
        let mut step_outputs: HashMap<String, Artifact> = HashMap::new();
        let mut step_results: Vec<StepResult> = Vec::new();

        // Reserved so a fan-out finishing just under the cap cannot starve the synthesis.
        let estimates: HashMap<String, f64> = crate::estimate::estimate_workflow(workflow)
            .steps
            .into_iter()
            .map(|s| (s.id, s.cost_usd))
            .collect();
        // Nothing to reserve when the output step spends no real money.
        let output_step_is_notional = workflow
            .steps
            .iter()
            .find(|s| s.id == workflow.output.step)
            .is_some_and(|s| self.is_notional(s));
        let output_reserve = if output_step_is_notional {
            0.0
        } else {
            estimates.get(&workflow.output.step).copied().unwrap_or(0.0)
        };

        let budget = BudgetLock::new(budget);

        for (index, wave) in dependency_waves(&workflow.steps).into_iter().enumerate() {
            info!(wave = index, steps = wave.len(), "starting wave");

            // step_outputs stays immutable until the wave finishes, so no clone is needed.
            let wave_results = {
                let ctx = StepContext {
                    workflow,
                    inputs,
                    step_outputs: &step_outputs,
                    estimates: &estimates,
                    output_reserve,
                };
                let futures = wave
                    .into_iter()
                    .map(|step| self.execute_step(step, &ctx, &budget));
                join_all(futures).await
            };

            // join_all preserves input order, so results stay deterministic run to run.
            // Every success is checkpointed before an error propagates, so a failed wave keeps finished work.
            let mut first_err: Option<FluxError> = None;
            for result in wave_results {
                match result {
                    Ok(step_result) => {
                        self.checkpoint(&step_result);
                        step_outputs.insert(step_result.id.clone(), step_result.output.clone());
                        step_results.push(step_result);
                    }
                    Err(e) => first_err = first_err.or(Some(e)),
                }
            }
            if let Some(e) = first_err {
                return Err(e);
            }
        }

        let output = step_outputs.remove(&workflow.output.step).ok_or_else(|| {
            FluxError::Config(format!(
                "output step '{}' did not produce any output",
                workflow.output.step
            ))
        })?;

        let budget = budget.into_inner();
        Ok(RunResult {
            output: output.as_prompt_text(),
            step_results,
            spent_usd: budget.spent_usd,
            notional_usd: budget.notional_usd,
            duration: run_start.elapsed(),
        })
    }

    // Recoverable failures become a placeholder result so the synthesizer still has input.
    async fn execute_step(
        &self,
        step: &Step,
        ctx: &StepContext<'_>,
        budget: &BudgetLock<'_>,
    ) -> crate::Result<StepResult> {
        let step_start = Instant::now();

        // Steps in a wave check the same snapshot, so a wave can overshoot by one estimate.
        // A free step is exempt: an HTTP fan-out draws nothing, and gating it on
        // headroom would let cheap deterministic work be skipped for no saving.
        let is_output_step = step.id == ctx.workflow.output.step;
        if !is_output_step && !self.is_notional(step) && !step.is_free() {
            let step_estimate = ctx.estimates.get(&step.id).copied().unwrap_or(0.0);
            let tracker = budget.lock().await;
            let headroom = tracker.limit_usd - tracker.spent_usd - ctx.output_reserve;
            let blocked = !tracker.in_emergency() && step_estimate > headroom;
            drop(tracker);

            if blocked {
                warn!(
                    step = %step.id,
                    estimate = format!("{:.4}", step_estimate),
                    reserved_for_output = format!("{:.4}", ctx.output_reserve),
                    "skipping step to preserve budget for the output step"
                );
                return Ok(skipped_step(
                    step,
                    format!(
                        "[SKIPPED — estimated cost ${step_estimate:.2} would not leave enough \
                         budget for the '{}' output step. Synthesis below uses the steps that ran.]",
                        ctx.workflow.output.step
                    ),
                    step_start,
                ));
            }
        }

        let step_inputs = load_state_for_step(step, ctx.inputs)?;

        // Snapshot so a pre-charge can be refunded if the step dies before any agent runs.
        let pre_step_spent = budget.lock().await.spent_usd;

        // Exhaustive: adding a step kind will not compile until it is handled here.
        let attempt = match step.resolve(&self.registry)? {
            StepKind::Transform(t) => {
                self.run_transform(step, &t, ctx.step_outputs, step_start)
                    .await
            }
            StepKind::Http(h) => {
                self.run_http(step, &h, &step_inputs, ctx.step_outputs, step_start)
                    .await
            }
            StepKind::Mcp(m) => {
                self.run_mcp(
                    step,
                    &m,
                    &ctx.workflow.mcp,
                    &step_inputs,
                    ctx.step_outputs,
                    step_start,
                )
                .await
            }
            StepKind::Agent(_) if step.fans_out() => {
                self.run_parallel(step, &step_inputs, ctx.step_outputs, budget, step_start)
                    .await
            }
            StepKind::Agent(_) if step.is_agent() => {
                self.run_single_agent(step, &step_inputs, ctx.step_outputs, budget, step_start)
                    .await
            }
            StepKind::Agent(_) => {
                self.run_llm(step, &step_inputs, ctx.step_outputs, budget, step_start)
                    .await
            }
        };

        let step_result = match attempt {
            Ok(r) => r,
            // enable_emergency below lifts the cap so the output step can still synthesize.
            Err(FluxError::BudgetExceeded { spent, limit }) => {
                let mut tracker = budget.lock().await;
                // A pre-charge that tripped the cap spent no real money, so refund it.
                if step.parallel_over.is_some() {
                    let phantom = spent - pre_step_spent;
                    if phantom > 0.0 {
                        tracker.refund(phantom);
                        warn!(
                            step = %step.id,
                            refunded = format!("{:.4}", phantom),
                            "refunded parallel pre-charge: no agents ran"
                        );
                    }
                }
                warn!(
                    step = %step.id,
                    spent = format!("{:.4}", tracker.spent_usd),
                    limit = format!("{:.4}", limit),
                    "budget exceeded; skipping step and attempting partial synthesis"
                );
                tracker.enable_emergency();
                skipped_step(
                    step,
                    format!(
                        "[SKIPPED — budget limit of ${limit:.4} reached before this step could run. \
                         Synthesis below is based on steps that completed successfully.]"
                    ),
                    step_start,
                )
            }
            Err(FluxError::Template(ref msg)) => {
                warn!(step = %step.id, error = %msg, "template error; skipping step with placeholder");
                skipped_step(
                    step,
                    format!(
                        "[SKIPPED — template error in step '{}': {msg}. \
                         Check that all {{{{ variable }}}} references are defined.]",
                        step.id
                    ),
                    step_start,
                )
            }
            // A dead agent on a non-output step degrades the run, it does not
            // sink it — the same posture as a failed fan-out item. The output
            // step is exempt: with no synthesis there is nothing to salvage.
            Err(FluxError::Agent(ref msg)) if !is_output_step => {
                let mut tracker = budget.lock().await;
                let phantom = tracker.spent_usd - pre_step_spent;
                if phantom > 0.0 {
                    tracker.refund(phantom);
                }
                drop(tracker);
                warn!(step = %step.id, error = %msg, "agent failed; skipping step with placeholder");
                skipped_step(
                    step,
                    format!(
                        "[SKIPPED — agent call in step '{}' failed: {msg}. \
                         Synthesis below uses the steps that ran.]",
                        step.id
                    ),
                    step_start,
                )
            }
            Err(e) => return Err(e),
        };

        // Non-fatal: a state file that cannot be written must not sink the run.
        if let Some(ref path) = step.appends_to
            && let Err(e) = append_to_state(
                path,
                &step_result.output.as_prompt_text(),
                &ctx.workflow.workflow.name,
                &step.id,
            )
        {
            warn!(step = %step.id, path = %path, error = %e, "failed to append to state file");
        }

        Ok(step_result)
    }

    // Deterministic and free: no budget interaction at all.
    async fn run_transform(
        &self,
        step: &Step,
        transform: &crate::workflow::TransformStep<'_>,
        step_outputs: &HashMap<String, Artifact>,
        step_start: Instant,
    ) -> crate::Result<StepResult> {
        info!(step = %step.id, transform = %transform.name, "executing transform step");

        // Dependency order, so a transform over two upstreams concatenates predictably.
        let upstream: Vec<&Artifact> = step
            .depends_on
            .iter()
            .filter_map(|id| step_outputs.get(id))
            .collect();

        let rows = self
            .registry
            .transform(transform.name)?
            .apply(&upstream, transform.args);
        info!(
            transform = transform.name,
            rows = rows.len(),
            "transform complete"
        );
        let output = Artifact::records(transform.output_type.clone(), rows);

        Ok(free_step(step, transform.name, output, step_start))
    }

    async fn run_http(
        &self,
        step: &Step,
        http: &crate::workflow::HttpStep<'_>,
        inputs: &HashMap<String, String>,
        step_outputs: &HashMap<String, Artifact>,
        step_start: Instant,
    ) -> crate::Result<StepResult> {
        info!(step = %step.id, "executing http step");

        let items = if step.fans_out() {
            let items = fan_items(step, step_outputs)?;
            if items.is_empty() {
                info!(step = %step.id, "fan-out over zero items; emitting empty output");
                return Ok(free_step(
                    step,
                    "http",
                    Artifact::records(http.output_type.clone(), Vec::new()),
                    step_start,
                ));
            }
            items
        } else {
            Vec::new()
        };

        let output = self
            .http
            .run(
                http,
                &items,
                inputs,
                step_outputs,
                step.max_concurrent.unwrap_or(DEFAULT_CONCURRENCY),
                &self.registry,
            )
            .await?;

        Ok(free_step(step, "http", output, step_start))
    }

    async fn run_mcp(
        &self,
        step: &Step,
        mcp: &crate::workflow::McpStep<'_>,
        config: &crate::workflow::McpConfig,
        inputs: &HashMap<String, String>,
        step_outputs: &HashMap<String, Artifact>,
        step_start: Instant,
    ) -> crate::Result<StepResult> {
        info!(step = %step.id, server = mcp.server, tool = mcp.tool, "executing MCP step");

        let server = config.servers.get(mcp.server).ok_or_else(|| {
            FluxError::Config(format!(
                "step '{}' references unknown MCP server '{}'",
                step.id, mcp.server
            ))
        })?;
        let items = if step.fans_out() {
            let items = fan_items(step, step_outputs)?;
            if items.is_empty() {
                info!(step = %step.id, "fan-out over zero items; emitting empty output");
                return Ok(free_step(
                    step,
                    "mcp",
                    Artifact::records(Schema::mcp_response(), Vec::new()),
                    step_start,
                ));
            }
            items
        } else {
            Vec::new()
        };

        let output = self
            .mcp
            .run(
                mcp.server,
                server,
                mcp,
                McpRunContext {
                    items: &items,
                    inputs,
                    step_outputs,
                    max_concurrent: step.max_concurrent.unwrap_or(DEFAULT_CONCURRENCY),
                    timeout_secs: step.timeout_secs,
                },
            )
            .await?;
        Ok(free_step(step, "mcp", output, step_start))
    }

    // Executes a step via the Anthropic API (the original path).
    async fn run_llm(
        &self,
        step: &Step,
        inputs: &HashMap<String, String>,
        step_outputs: &HashMap<String, Artifact>,
        budget: &BudgetLock<'_>,
        step_start: Instant,
    ) -> crate::Result<StepResult> {
        info!(step = %step.id, model = %step.model, "executing llm step");

        let prompt = crate::template::render_prompt(&step.prompt, inputs, step_outputs)?;
        let response = self
            .require_llm()?
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
        // Already paid for, so never discard it: record the overshoot and return the output.
        if let Err(e) = budget.lock().await.add(cost) {
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
            output: model_output(
                response.content,
                &response.sources,
                self.output_schema(step),
            ),
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
        step_outputs: &HashMap<String, Artifact>,
        budget: &BudgetLock<'_>,
        step_start: Instant,
    ) -> crate::Result<StepResult> {
        let provider_name = step.provider.as_deref().unwrap_or("agent");
        info!(step = %step.id, provider = %provider_name, "executing agent step");

        // Skipped when notional so it cannot warn about an unused cost_per_call_usd.
        let notional = self.is_notional(step);
        let estimated = if notional {
            0.0
        } else {
            cost::resolve_cost(step, 1)
        };
        if !notional {
            budget.lock().await.add(estimated)?;
        }

        let prompt = crate::template::render_prompt(&step.prompt, inputs, step_outputs)?;
        let response = self
            .agent_for(step)
            .run(AgentRequest {
                task: prompt.clone(),
                timeout_secs: step.timeout_secs.unwrap_or(300),
            })
            .await?;

        // Prefer the agent's reported spend over the pre-charged estimate.
        let cost = response.cost_usd.unwrap_or(estimated);
        if notional {
            budget.lock().await.add_notional(cost);
        } else {
            reconcile_budget(budget, &step.id, cost - estimated).await;
        }

        let duration = step_start.elapsed();
        info!(
            step = %step.id,
            cost_usd = format!("{:.4}", cost),
            notional,
            actual = response.cost_usd.is_some(),
            duration_secs = format!("{:.1}", duration.as_secs_f64()),
            "agent step complete"
        );

        Ok(StepResult {
            id: step.id.clone(),
            model: provider_name.to_string(),
            prompt,
            output: model_output(response.output, &[], self.output_schema(step)),
            input_tokens: 0,
            output_tokens: 0,
            cost_usd: cost,
            duration,
        })
    }

    async fn run_parallel(
        &self,
        step: &Step,
        inputs: &HashMap<String, String>,
        step_outputs: &HashMap<String, Artifact>,
        budget: &BudgetLock<'_>,
        step_start: Instant,
    ) -> crate::Result<StepResult> {
        let items = fan_items(step, step_outputs)?;

        // Only reachable with allow_empty; produce a well-typed empty output so
        // downstream consumers (a union, a template) see zero rows, not a hole.
        if items.is_empty() {
            info!(step = %step.id, "fan-out over zero items; emitting empty output");
            return Ok(empty_fan_out(step, self.output_schema(step), step_start));
        }

        // Rendered before any budget charge, so a template error leaves no phantom charge.
        let item_prompts: Vec<(String, String)> = items
            .iter()
            .map(|item| {
                let prompt =
                    crate::template::render(&step.prompt, inputs, step_outputs, Some(item))?;
                Ok((item.label.clone(), prompt))
            })
            .collect::<crate::Result<_>>()?;

        info!(step = %step.id, items = item_prompts.len(), "starting parallel fan-out");

        if step.is_agent() {
            self.run_parallel_agent(step, item_prompts, budget, step_start)
                .await
        } else {
            self.run_parallel_llm(step, item_prompts, budget, step_start)
                .await
        }
    }

    // One subprocess per item, pre-charged because agents do not report usage up front.
    async fn run_parallel_agent(
        &self,
        step: &Step,
        item_prompts: Vec<(String, String)>,
        budget: &BudgetLock<'_>,
        step_start: Instant,
    ) -> crate::Result<StepResult> {
        let n = item_prompts.len();
        // Both the pre-charge unit and the fallback for agents that report no cost.
        let notional = self.is_notional(step);
        let per_call = if notional {
            0.0
        } else {
            cost::resolve_cost(step, 1)
        };
        if !notional {
            budget.lock().await.add(per_call * n as f64)?;
        }

        let semaphore = Arc::new(Semaphore::new(step.max_concurrent.unwrap_or(3)));
        let agent = Arc::clone(self.agent_for(step));
        let step_id = step.id.clone();
        let timeout_secs = step.timeout_secs.unwrap_or(300);

        let futures: Vec<_> = item_prompts
            .into_iter()
            .map(|(item, prompt)| {
                let sem = Arc::clone(&semaphore);
                let agent = Arc::clone(&agent);
                let step_id = step_id.clone();
                async move {
                    // Bound to a name so the permit lives for the whole block, not one line.
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

        // A failed call incurs no real spend, so do not report it as money spent.
        let failed_count = results.iter().filter(|(_, r)| r.is_err()).count();
        if failed_count > 0 && !notional {
            let refund = per_call * failed_count as f64;
            budget.lock().await.refund(refund);
            warn!(
                step = %step.id,
                failed = failed_count,
                refunded_usd = format!("{:.4}", refund),
                "refunding pre-charged cost for failed agent calls"
            );
        }

        let mut output_parts: Vec<String> = Vec::new();
        let mut row_parts: Vec<serde_json::Value> = Vec::new();
        let mut actual_cost = 0.0f64;
        for (item, result) in results {
            match result {
                Ok(resp) => {
                    actual_cost += resp.cost_usd.unwrap_or(per_call);
                    if step.output_type.is_some()
                        && let Some(rows) = extract_json_rows(&resp.output)
                    {
                        row_parts.extend(rows);
                    }
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

        // Replace the estimate for the calls that ran with what the agents reported.
        if notional {
            budget.lock().await.add_notional(actual_cost);
        } else {
            let charged = per_call * (n - failed_count) as f64;
            reconcile_budget(budget, &step.id, actual_cost - charged).await;
        }

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
            output: fan_out_output(combined, &row_parts, self.output_schema(step)),
            input_tokens: 0,
            output_tokens: 0,
            cost_usd: actual_cost,
            duration,
        })
    }

    // No pre-charge: token costs are only known afterwards, and the overshoot is small.
    async fn run_parallel_llm(
        &self,
        step: &Step,
        item_prompts: Vec<(String, String)>,
        budget: &BudgetLock<'_>,
        step_start: Instant,
    ) -> crate::Result<StepResult> {
        let semaphore = Arc::new(Semaphore::new(step.max_concurrent.unwrap_or(3)));
        let llm = Arc::clone(self.require_llm()?);
        let step_id = step.id.clone();
        let max_tool_calls = step.max_tool_calls.unwrap_or(5);
        let max_tokens = step.max_tokens;

        let futures: Vec<_> = item_prompts
            .into_iter()
            .map(|(item, prompt)| {
                let sem = Arc::clone(&semaphore);
                let llm = Arc::clone(&llm);
                let step_id = step_id.clone();
                // Cloned per future so nothing borrows `step` across an await.
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
        let mut row_parts: Vec<serde_json::Value> = Vec::new();
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
                    if step.output_type.is_some()
                        && let Some(rows) = extract_json_rows(&resp.content)
                    {
                        row_parts.extend(rows);
                    }
                    output_parts.push(format!(
                        "--- {} ---\n{}",
                        item,
                        model_output(resp.content, &resp.sources, None).as_prompt_text()
                    ));
                }
                Err(e) => {
                    warn!(step = %step.id, item = %item, error = %e, "api call failed, skipping");
                    output_parts.push(format!("--- {} ---\n[FAILED: {}]", item, e));
                }
            }
        }

        // Already spent, so keep the outputs and let the next step trip the skip path.
        if let Err(e) = budget.lock().await.add(total_cost) {
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
            output: fan_out_output(combined, &row_parts, self.output_schema(step)),
            input_tokens,
            output_tokens,
            cost_usd: total_cost,
            duration,
        })
    }
}

/// Records if the fan-out declared a schema and any branch produced rows, else prose.
///
/// Falling back to prose matters: one branch returning garbage should not erase
/// the branches that worked, and the synthesizer can still read the text form.
fn fan_out_output(
    combined: String,
    rows: &[serde_json::Value],
    output_type: Option<Schema>,
) -> Artifact {
    match output_type {
        Some(schema) if !rows.is_empty() => Artifact::records(schema, rows.to_vec()),
        _ => Artifact::text(combined),
    }
}

/// Default in-flight requests for a fan-out, matching the LLM path.
const DEFAULT_CONCURRENCY: usize = 3;

// A typed empty result for an allow_empty fan-out that had nothing to do.
fn empty_fan_out(step: &Step, schema: Option<Schema>, step_start: Instant) -> StepResult {
    let output = match schema {
        Some(schema) => Artifact::records(schema, Vec::new()),
        None => Artifact::text(format!("[step '{}' had no items to process]", step.id)),
    };
    StepResult {
        id: step.id.clone(),
        model: step.provider.clone().unwrap_or_else(|| step.model.clone()),
        prompt: String::new(),
        output,
        input_tokens: 0,
        output_tokens: 0,
        cost_usd: 0.0,
        duration: step_start.elapsed(),
    }
}

// The explanation is ordinary prose so the synthesizer can read it as prompt text.
fn skipped_step(step: &Step, output: String, step_start: Instant) -> StepResult {
    StepResult {
        id: step.id.clone(),
        model: step.provider.clone().unwrap_or_else(|| step.model.clone()),
        prompt: String::new(),
        output: Artifact::text(output),
        input_tokens: 0,
        output_tokens: 0,
        cost_usd: 0.0,
        duration: step_start.elapsed(),
    }
}

// Http and transform steps report no tokens and no cost by construction.
fn free_step(step: &Step, model: &str, output: Artifact, step_start: Instant) -> StepResult {
    let duration = step_start.elapsed();
    info!(
        step = %step.id,
        rows = output.rows().len(),
        duration_secs = format!("{:.1}", duration.as_secs_f64()),
        "free step complete"
    );
    StepResult {
        id: step.id.clone(),
        model: model.to_string(),
        prompt: String::new(),
        output,
        input_tokens: 0,
        output_tokens: 0,
        cost_usd: 0.0,
        duration,
    }
}

/// Wraps a model's reply, promoting it to records when the step declared a schema.
///
/// A model told to emit JSON still sometimes wraps it in prose or fences, so the
/// JSON is located rather than parsed from position zero. If nothing parses, the
/// reply degrades to text instead of failing the step.
fn model_output(content: String, sources: &[String], output_type: Option<Schema>) -> Artifact {
    let sources: Vec<Source> = sources.iter().map(|s| Source::parse(s)).collect();
    let Some(schema) = output_type else {
        return Artifact::Text {
            body: content,
            sources,
        };
    };

    match extract_json_rows(&content) {
        Some(rows) => Artifact::records(schema, rows),
        None => {
            warn!(
                schema = schema.name(),
                "step declared output_type but produced no parsable JSON; keeping it as text"
            );
            Artifact::Text {
                body: content,
                sources,
            }
        }
    }
}

// Accepts a bare array, a bare object, or either wrapped in a fenced block.
fn extract_json_rows(content: &str) -> Option<Vec<serde_json::Value>> {
    let candidates = last_fenced_block(content)
        .into_iter()
        .chain(std::iter::once(content.to_string()));

    for candidate in candidates {
        let trimmed = candidate.trim();
        let start = trimmed.find(['[', '{'])?;
        let end = trimmed.rfind([']', '}'])?;
        if end <= start {
            continue;
        }
        let slice = &trimmed[start..=end];
        if let Ok(value) = serde_json::from_str::<serde_json::Value>(slice) {
            return Some(match value {
                serde_json::Value::Array(rows) => rows,
                other => vec![other],
            });
        }
    }
    None
}

/// Resolves the units a fan-out step runs over.
///
/// Records upstream fan out row-wise, so `{{ item.token }}` works; prose upstream
/// falls back to the fenced-list parsing that planner steps rely on.
fn fan_items(step: &Step, step_outputs: &HashMap<String, Artifact>) -> crate::Result<Vec<FanItem>> {
    let mut items: Vec<FanItem> = if !step.parallel_items.is_empty() {
        step.parallel_items.iter().map(FanItem::from_text).collect()
    } else {
        let upstream_id = step.parallel_over.as_deref().ok_or_else(|| {
            FluxError::Config(format!("step '{}' fans out over nothing", step.id))
        })?;
        let upstream = step_outputs.get(upstream_id).ok_or_else(|| {
            FluxError::Config(format!(
                "step '{}' parallel_over '{}' which has not yet run",
                step.id, upstream_id
            ))
        })?;

        match upstream {
            Artifact::Records { rows, .. } => {
                rows.iter().cloned().map(FanItem::from_record).collect()
            }
            Artifact::Text { body, .. } => {
                let (parsed, fenced) = parse_fan_out_items(body);
                if !fenced {
                    warn!(
                        step = %step.id,
                        upstream = %upstream_id,
                        items = parsed.len(),
                        "upstream output had no fenced block; every non-blank line became a \
                         fan-out item. Have the upstream step wrap its list in ``` fences."
                    );
                }
                parsed.into_iter().map(FanItem::from_text).collect()
            }
        }
    };

    if items.is_empty() {
        if step.allow_empty {
            return Ok(items);
        }
        return Err(FluxError::Config(format!(
            "step '{}' produced no items to fan out over",
            step.id
        )));
    }

    // Safety valve for a chatty planner: cap the fan-out rather than trust the count.
    if let Some(cap) = step.max_items
        && items.len() > cap
    {
        warn!(
            step = %step.id,
            found = items.len(),
            cap,
            "more fan-out items than max_items; truncating"
        );
        items.truncate(cap);
    }
    Ok(items)
}

// Replaces what was pre-charged from estimates once the actual cost is known.
async fn reconcile_budget(budget: &BudgetLock<'_>, step_id: &str, delta: f64) {
    if delta > 0.0 {
        if let Err(e) = budget.lock().await.add(delta) {
            warn!(
                step = %step_id,
                error = %e,
                "actual agent cost exceeded budget; downstream steps will be skipped"
            );
        }
    } else if delta < 0.0 {
        budget.lock().await.refund(-delta);
    }
}

// A missing file injects an empty string so a first run works before state exists.
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

/// Reads a file, or the most recently modified file when given a directory.
fn read_state_path(path: &str) -> crate::Result<Option<String>> {
    let p = std::path::Path::new(path);

    if p.is_dir() {
        let mut entries: Vec<_> = std::fs::read_dir(p)?
            .filter_map(|e| e.ok())
            .filter(|e| e.path().is_file())
            .collect();

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

// Creates the file and any parent directories if they do not exist.
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

/// Returns the fan-out items and whether they came from a fenced block.
///
/// A fenced block wins because it is the only part of the output a planner explicitly
/// marked as data; line-splitting alone turns a sentence of preamble into a research target.
fn parse_fan_out_items(output: &str) -> (Vec<String>, bool) {
    match last_fenced_block(output) {
        Some(block) => (parse_item_lines(&block), true),
        None => (parse_item_lines(output), false),
    }
}

// Last rather than first, because a planner that reasons first answers at the end.
fn last_fenced_block(output: &str) -> Option<String> {
    let mut blocks: Vec<String> = Vec::new();
    let mut current: Option<Vec<&str>> = None;
    for line in output.lines() {
        // Only the prefix matters: anything after it is a language tag.
        if line.trim_start().starts_with("```") {
            match current.take() {
                Some(lines) => blocks.push(lines.join("\n")),
                None => current = Some(Vec::new()),
            }
            continue;
        }
        if let Some(lines) = current.as_mut() {
            lines.push(line);
        }
    }
    // An unterminated fence still carries the items the planner meant to emit.
    if let Some(lines) = current
        && !lines.is_empty()
    {
        blocks.push(lines.join("\n"));
    }
    blocks.into_iter().rev().find(|b| !b.trim().is_empty())
}

// Drops decorative lines: blank, markdown headers, and rules like "---" or "===".
fn parse_item_lines(text: &str) -> Vec<String> {
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .filter(|l| !l.trim_start().starts_with('#'))
        .filter(|l| l.trim().chars().any(|c| c.is_alphanumeric()))
        .map(|l| strip_list_marker(l.trim()).to_string())
        .collect()
}

// So a planner that adds bullets ("- Acme", "1. Acme") still yields clean item strings.
fn strip_list_marker(s: &str) -> &str {
    if let Some(rest) = s.split_once(". ")
        && rest.0.chars().all(|c| c.is_ascii_digit())
        && !rest.0.is_empty()
    {
        return rest.1;
    }
    for prefix in &["- ", "* ", "• "] {
        if let Some(rest) = s.strip_prefix(prefix) {
            return rest;
        }
    }
    s
}

/// Groups steps so that everything in wave N depends only on earlier waves.
///
/// Declaration order is preserved within each wave, keeping a run's step order stable.
fn dependency_waves(steps: &[Step]) -> Vec<Vec<&Step>> {
    let by_id: HashMap<&str, &Step> = steps.iter().map(|s| (s.id.as_str(), s)).collect();
    let mut depths: HashMap<&str, usize> = HashMap::new();
    for step in steps {
        step_depth(step.id.as_str(), &by_id, &mut depths);
    }

    let Some(&deepest) = depths.values().max() else {
        return Vec::new();
    };
    // No wave can be empty: a step at depth d must have a dependency at depth d-1.
    let mut waves: Vec<Vec<&Step>> = vec![Vec::new(); deepest + 1];
    for step in steps {
        waves[depths[step.id.as_str()]].push(step);
    }
    waves
}

// One past the deepest dependency, memoized so each step resolves once.
fn step_depth<'a>(
    id: &'a str,
    by_id: &HashMap<&'a str, &'a Step>,
    depths: &mut HashMap<&'a str, usize>,
) -> usize {
    if let Some(&depth) = depths.get(id) {
        return depth;
    }
    // Provisional entry so a cycle terminates here instead of overflowing the stack.
    depths.insert(id, 0);
    let depth = by_id[id]
        .depends_on
        .iter()
        .map(|dep| step_depth(dep.as_str(), by_id, depths) + 1)
        .max()
        .unwrap_or(0);
    depths.insert(id, depth);
    depth
}

#[cfg(test)]
mod tests {
    use super::*;

    fn new_engine(llm: Arc<dyn LlmProvider>, agent: Arc<dyn AgentProvider>) -> Engine {
        Engine::new(llm, agent).unwrap()
    }

    fn new_agents_only(agent: Arc<dyn AgentProvider>) -> Engine {
        Engine::agents_only(agent).unwrap()
    }

    // Via TOML rather than struct literals, so tests use the same parse path as real runs.
    fn step(src: &str) -> Step {
        toml::from_str(src).expect("test step TOML should parse")
    }

    fn wave_ids<'a>(waves: &'a [Vec<&'a Step>]) -> Vec<Vec<&'a str>> {
        waves
            .iter()
            .map(|wave| wave.iter().map(|s| s.id.as_str()).collect())
            .collect()
    }

    #[test]
    fn waves_put_each_step_after_its_dependencies() {
        let steps = vec![
            step("id = \"synth\"\nprompt = \"p\"\ndepends_on = [\"research\"]"),
            step("id = \"research\"\nprompt = \"p\"\ndepends_on = [\"plan\"]"),
            step("id = \"plan\"\nprompt = \"p\""),
        ];
        // A fully sequential chain is one step per wave.
        assert_eq!(
            wave_ids(&dependency_waves(&steps)),
            vec![vec!["plan"], vec!["research"], vec!["synth"]]
        );
    }

    // The payoff over a flat sort: b and c are independent, so they share a wave.
    #[test]
    fn independent_branches_of_a_diamond_share_a_wave() {
        let steps = vec![
            step("id = \"d\"\nprompt = \"p\"\ndepends_on = [\"b\", \"c\"]"),
            step("id = \"b\"\nprompt = \"p\"\ndepends_on = [\"a\"]"),
            step("id = \"c\"\nprompt = \"p\"\ndepends_on = [\"a\"]"),
            step("id = \"a\"\nprompt = \"p\""),
        ];
        assert_eq!(
            wave_ids(&dependency_waves(&steps)),
            vec![vec!["a"], vec!["b", "c"], vec!["d"]]
        );
    }

    #[test]
    fn wholly_independent_steps_all_share_the_first_wave() {
        let steps = vec![
            step("id = \"x\"\nprompt = \"p\""),
            step("id = \"y\"\nprompt = \"p\""),
        ];
        assert_eq!(wave_ids(&dependency_waves(&steps)), vec![vec!["x", "y"]]);
    }

    // A step's depth is its deepest path, not its shortest: d must wait for c, not just b.
    #[test]
    fn depth_follows_the_longest_dependency_path() {
        let steps = vec![
            step("id = \"a\"\nprompt = \"p\""),
            step("id = \"b\"\nprompt = \"p\"\ndepends_on = [\"a\"]"),
            step("id = \"c\"\nprompt = \"p\"\ndepends_on = [\"b\"]"),
            step("id = \"d\"\nprompt = \"p\"\ndepends_on = [\"a\", \"c\"]"),
        ];
        assert_eq!(
            wave_ids(&dependency_waves(&steps)),
            vec![vec!["a"], vec!["b"], vec!["c"], vec!["d"]]
        );
    }

    #[test]
    fn strip_list_marker_handles_bullets_and_numbers() {
        assert_eq!(strip_list_marker("- Acme"), "Acme");
        assert_eq!(strip_list_marker("* Acme"), "Acme");
        assert_eq!(strip_list_marker("• Acme"), "Acme");
        assert_eq!(strip_list_marker("1. Acme"), "Acme");
        assert_eq!(strip_list_marker("12. Acme"), "Acme");
    }

    // The exact failure that cost a research slot on a real run.
    #[test]
    fn fenced_block_wins_over_planner_preamble() {
        let output = "After cross-referencing the exclusion list, here are 6 companies:\n\n\
                      ```\nAcme\nBorealis\nCyclotron\n```\n\nLet me know if you want more.";
        let (items, fenced) = parse_fan_out_items(output);
        assert!(fenced);
        assert_eq!(items, vec!["Acme", "Borealis", "Cyclotron"]);
    }

    #[test]
    fn fenced_block_tolerates_a_language_tag_and_bullets() {
        let output = "```text\n- Acme\n- Borealis\n```";
        let (items, fenced) = parse_fan_out_items(output);
        assert!(fenced);
        assert_eq!(items, vec!["Acme", "Borealis"]);
    }

    #[test]
    fn last_fenced_block_wins() {
        let output = "```\nscratch work\n```\nokay, final:\n```\nAcme\nBorealis\n```";
        let (items, _) = parse_fan_out_items(output);
        assert_eq!(items, vec!["Acme", "Borealis"]);
    }

    #[test]
    fn unfenced_output_falls_back_to_lines_and_reports_it() {
        let (items, fenced) = parse_fan_out_items("Acme\nBorealis\n\n## header\n---\n");
        assert!(!fenced, "caller needs to know the items are unvetted");
        assert_eq!(items, vec!["Acme", "Borealis"]);
    }

    #[test]
    fn unterminated_fence_still_yields_items() {
        let (items, fenced) = parse_fan_out_items("here you go:\n```\nAcme\nBorealis");
        assert!(fenced);
        assert_eq!(items, vec!["Acme", "Borealis"]);
    }

    #[test]
    fn empty_fenced_block_is_ignored_in_favour_of_a_real_one() {
        let (items, _) = parse_fan_out_items("```\nAcme\n```\ntrailing\n```\n```");
        assert_eq!(items, vec!["Acme"]);
    }

    #[test]
    fn strip_list_marker_leaves_plain_text_alone() {
        assert_eq!(strip_list_marker("Acme"), "Acme");
        // No space after the dot: not a list marker.
        assert_eq!(strip_list_marker("1.Acme"), "1.Acme");
        // Non-numeric prefix before ". " is content, not a marker.
        assert_eq!(strip_list_marker("Inc. Acme"), "Inc. Acme");
    }

    // End-to-end tests: mock providers let the whole engine run with no network or money.

    use crate::provider::{AgentResponse, CompletionResponse};
    use async_trait::async_trait;

    struct MockLlm;

    #[async_trait]
    impl LlmProvider for MockLlm {
        async fn complete(&self, request: CompletionRequest) -> crate::Result<CompletionResponse> {
            // Non-planner prompts echo back, so a test can see which call produced what.
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
        let engine = new_engine(Arc::new(MockLlm), Arc::new(MockAgent));
        let mut budget = BudgetTracker::new(5.0);

        let result = engine
            .run(&wf, &HashMap::new(), &mut budget)
            .await
            .expect("workflow should succeed");

        let fan = result.step_results.iter().find(|s| s.id == "fan").unwrap();
        assert!(fan.text().contains("--- alpha ---"), "{}", fan.text());
        assert!(fan.text().contains("--- beta ---"), "{}", fan.text());
        assert!(
            fan.text().contains("researched[look into alpha]"),
            "fan-out should hit the API mock, not the agent: {}",
            fan.text()
        );
        // Two items × the mock's 100/50 tokens, actually charged to the budget.
        assert_eq!(fan.input_tokens, 200);
        assert_eq!(fan.output_tokens, 100);
        assert!(fan.cost_usd > 0.0);
        assert!(budget.spent_usd >= fan.cost_usd);
        // Cited sources are appended per item so they flow downstream.
        assert!(
            fan.text()
                .contains("Sources:\n- Mock Source — https://example.com/mock"),
            "{}",
            fan.text()
        );
    }

    struct FailingAgent;

    #[async_trait]
    impl AgentProvider for FailingAgent {
        async fn run(&self, _request: AgentRequest) -> crate::Result<AgentResponse> {
            Err(FluxError::Agent("agent timed out after 300s".into()))
        }
    }

    // The failure that cost a 40-minute run: one single-call agent step timing
    // out. It must degrade to a placeholder, not abort the workflow.
    #[tokio::test]
    async fn failed_single_agent_step_skips_instead_of_sinking_the_run() {
        let wf: WorkflowFile = toml::from_str(
            r#"
[workflow]
name = "t"
[budget]
max_usd = 5.0
[[steps]]
id = "flaky"
provider = "claude-code"
prompt = "research"
cost_per_call_usd = 0.1
[[steps]]
id = "synth"
prompt = "combine {{ flaky.output }}"
depends_on = ["flaky"]
[output]
step = "synth"
"#,
        )
        .unwrap();

        let engine = new_engine(Arc::new(MockLlm), Arc::new(FailingAgent));
        let mut budget = BudgetTracker::new(5.0);
        let result = engine
            .run(&wf, &HashMap::new(), &mut budget)
            .await
            .expect("a failed non-output agent step must not fail the run");

        // The synthesizer saw the placeholder.
        assert!(result.output.contains("SKIPPED"), "{}", result.output);
        // The $0.10 pre-charge was refunded; only the synth step's tokens remain.
        let flaky = result
            .step_results
            .iter()
            .find(|s| s.id == "flaky")
            .unwrap();
        assert_eq!(flaky.cost_usd, 0.0);
        assert!(budget.spent_usd < 0.01, "{}", budget.spent_usd);
    }

    #[tokio::test]
    async fn failed_agent_output_step_still_fails_the_run() {
        let wf: WorkflowFile = toml::from_str(
            r#"
[workflow]
name = "t"
[budget]
max_usd = 5.0
[[steps]]
id = "only"
provider = "claude-code"
prompt = "research"
[output]
step = "only"
"#,
        )
        .unwrap();

        let engine = new_engine(Arc::new(MockLlm), Arc::new(FailingAgent));
        let mut budget = BudgetTracker::new(5.0);
        let err = engine.run(&wf, &HashMap::new(), &mut budget).await;
        assert!(err.is_err(), "no output step means nothing to salvage");
    }

    // The 503 outage that lost a 49-minute run: an output step failing must
    // leave the earlier steps' checkpoints on disk, not evaporate them.
    #[tokio::test]
    async fn failed_run_checkpoints_completed_steps() {
        let wf: WorkflowFile = toml::from_str(
            r#"
[workflow]
name = "t"
[budget]
max_usd = 5.0
[[steps]]
id = "plan"
prompt = "PLAN"
[[steps]]
id = "out"
provider = "claude-code"
prompt = "synthesize {{ plan.output }}"
depends_on = ["plan"]
[output]
step = "out"
"#,
        )
        .unwrap();

        let dir = std::env::temp_dir().join(format!("flux-ckpt-test-{}", std::process::id()));
        std::fs::remove_dir_all(&dir).ok();

        let engine =
            new_engine(Arc::new(MockLlm), Arc::new(FailingAgent)).with_checkpoint_dir(dir.clone());
        let mut budget = BudgetTracker::new(5.0);
        let err = engine.run(&wf, &HashMap::new(), &mut budget).await;
        assert!(err.is_err());

        let plan_md = std::fs::read_to_string(dir.join("plan.md")).expect("plan.md checkpoint");
        assert!(plan_md.contains("alpha\nbeta"), "{plan_md}");
        assert!(
            !dir.join("out.md").exists(),
            "failed step must not checkpoint"
        );
        std::fs::remove_dir_all(&dir).ok();
    }

    // A fallback step's steady state: nothing upstream is left to do, so the
    // fan-out has zero items. allow_empty must yield typed empty output and
    // let the rest of the workflow proceed.
    #[tokio::test]
    async fn allow_empty_fan_out_over_zero_items_succeeds() {
        let wf: WorkflowFile = toml::from_str(
            r#"
[workflow]
name = "t"
[budget]
max_usd = 5.0
[[steps]]
id = "plan"
prompt = "PLAN"
[[steps]]
id = "todo"
kind = "transform"
transform = "dedupe"
output_type = "record"
depends_on = ["plan"]
[[steps]]
id = "fallback"
provider = "claude-code"
prompt = "find {{ item.name }}"
parallel_over = "todo"
depends_on = ["todo"]
allow_empty = true
output_type = "record"
[[steps]]
id = "synth"
prompt = "combine {{ fallback.count }}"
depends_on = ["fallback"]
[output]
step = "synth"
"#,
        )
        .unwrap();

        let engine = new_engine(Arc::new(MockLlm), Arc::new(MockAgent));
        let mut budget = BudgetTracker::new(5.0);
        let result = engine
            .run(&wf, &HashMap::new(), &mut budget)
            .await
            .expect("empty fan-out with allow_empty should not fail the run");

        let fallback = result
            .step_results
            .iter()
            .find(|s| s.id == "fallback")
            .unwrap();
        assert!(fallback.output.rows().is_empty());
        assert_eq!(fallback.cost_usd, 0.0);
        // The synthesizer saw a zero count, not a template error.
        assert!(result.output.contains("combine 0"), "{}", result.output);
    }

    #[tokio::test]
    async fn parallel_fan_out_with_agent_provider_uses_the_agent() {
        let wf = workflow("provider = \"claude-code\"\ncost_per_call_usd = 0.1");
        let engine = new_engine(Arc::new(MockLlm), Arc::new(MockAgent));
        let mut budget = BudgetTracker::new(5.0);

        let result = engine
            .run(&wf, &HashMap::new(), &mut budget)
            .await
            .expect("workflow should succeed");

        let fan = result.step_results.iter().find(|s| s.id == "fan").unwrap();
        assert!(
            fan.text().contains("agent[look into alpha]"),
            "agent provider should route to the agent mock: {}",
            fan.text()
        );
        // Two items pre-charged at the declared cost_per_call_usd.
        assert!((fan.cost_usd - 0.2).abs() < 1e-9);
    }

    // Reports real spend, like claude-code with --output-format=json.
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

    // Records peak in-flight calls, so a test can prove steps overlap rather than queue.
    struct ConcurrencyProbe {
        in_flight: std::sync::atomic::AtomicUsize,
        peak: std::sync::atomic::AtomicUsize,
    }

    #[async_trait]
    impl LlmProvider for ConcurrencyProbe {
        async fn complete(&self, request: CompletionRequest) -> crate::Result<CompletionResponse> {
            use std::sync::atomic::Ordering::SeqCst;
            let now = self.in_flight.fetch_add(1, SeqCst) + 1;
            self.peak.fetch_max(now, SeqCst);
            // Long enough that sequential execution could not overlap by accident.
            tokio::time::sleep(Duration::from_millis(50)).await;
            self.in_flight.fetch_sub(1, SeqCst);
            Ok(CompletionResponse {
                content: format!("did[{}]", request.prompt),
                sources: vec![],
                input_tokens: 10,
                output_tokens: 5,
                web_search_requests: 0,
            })
        }
    }

    #[tokio::test]
    async fn independent_steps_in_a_wave_run_at_the_same_time() {
        let wf: WorkflowFile = toml::from_str(
            r#"
[workflow]
name = "t"
[budget]
max_usd = 5.0
[[steps]]
id = "left"
prompt = "left branch"
[[steps]]
id = "right"
prompt = "right branch"
[[steps]]
id = "synth"
depends_on = ["left", "right"]
prompt = "{{ left.output }} {{ right.output }}"
[output]
step = "synth"
"#,
        )
        .unwrap();

        let probe = Arc::new(ConcurrencyProbe {
            in_flight: std::sync::atomic::AtomicUsize::new(0),
            peak: std::sync::atomic::AtomicUsize::new(0),
        });
        let engine = new_engine(
            Arc::clone(&probe) as Arc<dyn LlmProvider>,
            Arc::new(MockAgent),
        );
        let mut budget = BudgetTracker::new(5.0);

        let result = engine
            .run(&wf, &HashMap::new(), &mut budget)
            .await
            .expect("workflow should succeed");

        assert_eq!(
            probe.peak.load(std::sync::atomic::Ordering::SeqCst),
            2,
            "left and right are independent and must overlap"
        );
        // The synthesizer still saw both upstream outputs.
        assert!(result.output.contains("left branch"), "{}", result.output);
        assert!(result.output.contains("right branch"), "{}", result.output);
    }

    // Reports an API-equivalent cost, like claude-code with the API key hidden from it.
    struct MockSubscriptionAgent;

    #[async_trait]
    impl AgentProvider for MockSubscriptionAgent {
        async fn run(&self, request: AgentRequest) -> crate::Result<AgentResponse> {
            Ok(AgentResponse {
                output: format!("agent[{}]", request.task),
                duration: Duration::from_millis(1),
                cost_usd: Some(0.90),
            })
        }

        fn billing(&self) -> Billing {
            Billing::Notional
        }
    }

    #[tokio::test]
    async fn agent_fan_out_reconciles_to_reported_actual_cost() {
        // Estimated $0.10/call but reported $0.07, so both should settle on 2 x 0.07.
        let wf = workflow("provider = \"claude-code\"\ncost_per_call_usd = 0.1");
        let engine = new_engine(Arc::new(MockLlm), Arc::new(MockAgentWithCost));
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

    // Discarding it would mean paying for a synthesis and getting a placeholder.
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
        let engine = new_engine(Arc::new(MockLlm), Arc::new(MockAgent));
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

    // A budget that fits synth but not plan-plus-synth: plan must give way.
    #[tokio::test]
    async fn expensive_step_is_skipped_to_preserve_budget_for_output() {
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
        let engine = new_engine(Arc::new(MockLlm), Arc::new(MockAgent));
        let mut budget = BudgetTracker::new(0.10);

        let result = engine
            .run(&wf, &HashMap::new(), &mut budget)
            .await
            .expect("workflow should succeed with the expensive step skipped");

        let plan = result.step_results.iter().find(|s| s.id == "plan").unwrap();
        assert!(plan.text().contains("SKIPPED"), "{}", plan.text());
        assert_eq!(plan.cost_usd, 0.0);

        // The output step still ran for real on the partial input.
        let synth = result
            .step_results
            .iter()
            .find(|s| s.id == "synth")
            .unwrap();
        assert!(synth.text().contains("researched"), "{}", synth.text());
        // Only synth spent money, and it stayed within budget.
        assert!(budget.spent_usd > 0.0 && budget.spent_usd <= budget.limit_usd);
    }

    // Two items reporting $0.90 each would blow a $1.00 cap that costs nothing to spend.
    #[tokio::test]
    async fn subscription_agent_spend_is_notional_and_never_trips_the_budget() {
        let wf = workflow("provider = \"claude-code\"\ncost_per_call_usd = 0.30");
        let engine = new_engine(Arc::new(MockLlm), Arc::new(MockSubscriptionAgent));
        let mut budget = BudgetTracker::new(1.00);

        let result = engine
            .run(&wf, &HashMap::new(), &mut budget)
            .await
            .expect("notional spend must not abort the run");

        let fan = result.step_results.iter().find(|s| s.id == "fan").unwrap();
        // The step still reports what it would have cost, for visibility.
        assert!((fan.cost_usd - 1.80).abs() < 1e-9, "got {}", fan.cost_usd);
        assert!((result.notional_usd - 1.80).abs() < 1e-9);
        // ...but none of it counted against the ceiling. Only the two API steps did.
        assert!(budget.spent_usd < budget.limit_usd);
        let api_cost: f64 = result
            .step_results
            .iter()
            .filter(|s| s.id != "fan")
            .map(|s| s.cost_usd)
            .sum();
        assert!((budget.spent_usd - api_cost).abs() < 1e-9);
    }

    // The same reported figure must still be enforced when the agent bills real credits.
    #[tokio::test]
    async fn metered_agent_spend_still_counts_against_the_budget() {
        let wf = workflow("provider = \"claude-code\"\ncost_per_call_usd = 0.30");
        let engine = new_engine(Arc::new(MockLlm), Arc::new(MockAgentWithCost));
        let mut budget = BudgetTracker::new(5.0);

        let result = engine
            .run(&wf, &HashMap::new(), &mut budget)
            .await
            .expect("workflow should succeed");

        assert_eq!(result.notional_usd, 0.0);
        let fan = result.step_results.iter().find(|s| s.id == "fan").unwrap();
        assert!((fan.cost_usd - 0.14).abs() < 1e-9, "got {}", fan.cost_usd);
        assert!(budget.spent_usd >= fan.cost_usd);
    }

    fn agent_only_workflow(step_extra: &str) -> WorkflowFile {
        toml::from_str(&format!(
            r#"
[workflow]
name = "t"
[budget]
max_usd = 5.0
[[steps]]
id = "only"
prompt = "do the thing"
{step_extra}
[output]
step = "only"
"#
        ))
        .expect("test workflow TOML should parse")
    }

    // Distinguishable from MockAgent output, so routing tests can tell who ran.
    struct MockNamedAgent(&'static str);

    #[async_trait]
    impl AgentProvider for MockNamedAgent {
        async fn run(&self, _request: AgentRequest) -> crate::Result<AgentResponse> {
            Ok(AgentResponse {
                output: format!("via[{}]", self.0),
                duration: Duration::from_millis(1),
                cost_usd: None,
            })
        }
    }

    #[tokio::test]
    async fn steps_route_to_the_provider_they_name() {
        let wf = agent_only_workflow("provider = \"cursor\"\ncost_per_call_usd = 0.01");
        let engine = new_agents_only(Arc::new(MockAgent))
            .with_agent("cursor", Arc::new(MockNamedAgent("cursor")));
        let mut budget = BudgetTracker::new(5.0);

        let result = engine.run(&wf, &HashMap::new(), &mut budget).await.unwrap();
        assert!(result.output.contains("via[cursor]"), "{}", result.output);
    }

    #[tokio::test]
    async fn unregistered_provider_names_fall_back_to_the_default_agent() {
        let wf = agent_only_workflow("provider = \"claude-code\"\ncost_per_call_usd = 0.01");
        let engine = new_agents_only(Arc::new(MockAgent))
            .with_agent("cursor", Arc::new(MockNamedAgent("cursor")));
        let mut budget = BudgetTracker::new(5.0);

        let result = engine.run(&wf, &HashMap::new(), &mut budget).await.unwrap();
        assert!(result.output.contains("agent["), "{}", result.output);
    }

    // What lets `flux run` work with ANTHROPIC_API_KEY unset and billing via the CLI login.
    #[tokio::test]
    async fn agents_only_engine_runs_without_an_llm_client() {
        let wf = agent_only_workflow("provider = \"claude-code\"\ncost_per_call_usd = 0.01");
        let engine = new_agents_only(Arc::new(MockAgent));
        let mut budget = BudgetTracker::new(5.0);

        let result = engine
            .run(&wf, &HashMap::new(), &mut budget)
            .await
            .expect("an all-agent workflow should not need an API client");

        assert!(
            result.output.contains("agent[do the thing]"),
            "{}",
            result.output
        );
    }

    #[tokio::test]
    async fn agents_only_engine_rejects_a_step_that_needs_the_api() {
        // No provider means the direct API, which is unavailable here.
        let wf = agent_only_workflow("");
        let engine = new_agents_only(Arc::new(MockAgent));
        let mut budget = BudgetTracker::new(5.0);

        let err = engine
            .run(&wf, &HashMap::new(), &mut budget)
            .await
            .expect_err("an API step without a client must fail");

        assert!(err.to_string().contains("ANTHROPIC_API_KEY"), "{err}");
        assert_eq!(budget.spent_usd, 0.0);
    }
}
