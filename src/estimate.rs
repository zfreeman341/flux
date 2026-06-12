//! Pre-run cost estimation for `flux explain`.
//!
//! Estimation is heuristic by necessity: template variables, upstream step
//! outputs, and fan-out item counts don't exist until a run happens. Each
//! guess is surfaced as an explicit assumption string so the user can judge
//! how much to trust the number. No API calls are made — explain stays free.

use crate::cost::{WEB_SEARCH_COST_PER_CALL_USD, step_cost_usd};
use crate::workflow::{Step, WorkflowFile};

// Rough average for English prose and markdown.
const CHARS_PER_TOKEN: f64 = 4.0;
// Stand-in for an upstream step output or state file injected into a prompt.
const ASSUMED_INPUT_REF_TOKENS: u32 = 2_000;
// Typical response length for a research/synthesis step.
const ASSUMED_OUTPUT_TOKENS: u32 = 1_500;
// Search results are billed as input tokens on top of the per-search fee.
const ASSUMED_TOKENS_PER_SEARCH: u32 = 2_000;
// Must match the engine's default when a step omits max_tool_calls.
const DEFAULT_MAX_TOOL_CALLS: u32 = 5;
// Fan-out size guess when neither parallel_items nor max_items bounds it.
const DEFAULT_FAN_OUT_ITEMS: usize = 6;
// Must match cost::resolve_cost's fallback for agent steps.
const DEFAULT_AGENT_COST_PER_CALL_USD: f64 = 0.05;

pub struct WorkflowEstimate {
    /// One entry per step, in workflow file order.
    pub steps: Vec<StepEstimate>,
    pub total_usd: f64,
}

pub struct StepEstimate {
    pub id: String,
    pub cost_usd: f64,
    /// Human-readable guesses this estimate rests on, for display in explain.
    pub assumptions: Vec<String>,
}

pub fn estimate_workflow(wf: &WorkflowFile) -> WorkflowEstimate {
    let steps: Vec<StepEstimate> = wf.steps.iter().map(estimate_step).collect();
    let total_usd = steps.iter().map(|s| s.cost_usd).sum();
    WorkflowEstimate { steps, total_usd }
}

fn estimate_step(step: &Step) -> StepEstimate {
    // Mirrors the engine's dispatch: any parallel step runs through the agent
    // provider, as does a non-parallel step with an agent provider name.
    let is_parallel = step.parallel_over.is_some() || !step.parallel_items.is_empty();
    let is_agent = matches!(
        step.provider.as_deref(),
        Some("hermes") | Some("claude-code")
    );

    if is_parallel || is_agent {
        estimate_agent_step(step, is_parallel)
    } else {
        estimate_llm_step(step)
    }
}

fn estimate_agent_step(step: &Step, is_parallel: bool) -> StepEstimate {
    let mut assumptions = Vec::new();

    let count = if !is_parallel {
        1
    } else if !step.parallel_items.is_empty() {
        step.parallel_items.len()
    } else if let Some(cap) = step.max_items {
        assumptions.push(format!("fan-out at the max_items cap of {cap}"));
        cap
    } else {
        assumptions.push(format!(
            "fan-out of {DEFAULT_FAN_OUT_ITEMS} items (set max_items to bound this)"
        ));
        DEFAULT_FAN_OUT_ITEMS
    };

    let per_call = match step.cost_per_call_usd {
        Some(c) => c,
        None => {
            assumptions.push(format!(
                "cost_per_call_usd not set; assuming ${DEFAULT_AGENT_COST_PER_CALL_USD:.2}/call"
            ));
            DEFAULT_AGENT_COST_PER_CALL_USD
        }
    };

    StepEstimate {
        id: step.id.clone(),
        cost_usd: per_call * count as f64,
        assumptions,
    }
}

fn estimate_llm_step(step: &Step) -> StepEstimate {
    let mut assumptions = Vec::new();

    let prompt_tokens = (step.prompt.chars().count() as f64 / CHARS_PER_TOKEN) as u32;

    // Every upstream dependency and state file is a template variable whose
    // size is unknown until run time — charge a flat stand-in for each.
    let input_refs = (step.depends_on.len() + step.reads_from.len()) as u32;
    if input_refs > 0 {
        assumptions.push(format!(
            "~{ASSUMED_INPUT_REF_TOKENS} tokens per upstream/state input ({input_refs})"
        ));
    }

    let mut input_tokens = prompt_tokens + input_refs * ASSUMED_INPUT_REF_TOKENS;
    let mut search_surcharge = 0.0;

    if step.tools.iter().any(|t| t == "web_search") {
        let searches = step.max_tool_calls.unwrap_or(DEFAULT_MAX_TOOL_CALLS);
        search_surcharge = searches as f64 * WEB_SEARCH_COST_PER_CALL_USD;
        input_tokens += searches * ASSUMED_TOKENS_PER_SEARCH;
        assumptions.push(format!(
            "{searches} web searches at ${WEB_SEARCH_COST_PER_CALL_USD}/search \
             (+~{ASSUMED_TOKENS_PER_SEARCH} result tokens each)"
        ));
    }

    assumptions.push(format!("~{ASSUMED_OUTPUT_TOKENS} output tokens"));

    StepEstimate {
        id: step.id.clone(),
        cost_usd: step_cost_usd(&step.model, input_tokens, ASSUMED_OUTPUT_TOKENS)
            + search_surcharge,
        assumptions,
    }
}
