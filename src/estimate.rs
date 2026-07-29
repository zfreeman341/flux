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
const ASSUMED_OUTPUT_TOKENS: u32 = 2_000;
// Input tokens a single web_search result adds to the conversation. Calibrated
// against real runs (~3,500 tokens/result reproduced two web-grounded steps
// within ~3%). This is per result; the compounding from re-sending it across
// the tool-use loop is modeled separately in estimate_llm_step.
const ASSUMED_TOKENS_PER_SEARCH: u32 = 3_500;
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
    // Mirrors the engine's dispatch: agent providers fan out as subprocesses
    // priced per call; everything else is direct API calls priced by tokens —
    // including parallel steps, which fan out one API call per item.
    let is_parallel = step.parallel_over.is_some() || !step.parallel_items.is_empty();

    if step.is_agent() {
        estimate_agent_step(step, is_parallel)
    } else if is_parallel {
        estimate_parallel_llm_step(step)
    } else {
        estimate_llm_step(step)
    }
}

// Resolves how many items a parallel step will fan out over, recording an
// assumption whenever the answer is a guess rather than a known count.
fn resolve_fan_out_count(step: &Step, assumptions: &mut Vec<String>) -> usize {
    if !step.parallel_items.is_empty() {
        step.parallel_items.len()
    } else if let Some(cap) = step.max_items {
        assumptions.push(format!("fan-out at the max_items cap of {cap}"));
        cap
    } else {
        assumptions.push(format!(
            "fan-out of {DEFAULT_FAN_OUT_ITEMS} items (set max_items to bound this)"
        ));
        DEFAULT_FAN_OUT_ITEMS
    }
}

fn estimate_parallel_llm_step(step: &Step) -> StepEstimate {
    let mut assumptions = Vec::new();
    let count = resolve_fan_out_count(step, &mut assumptions);
    // Each item is an independent API call with the same prompt shape;
    // estimate one and multiply.
    let per_item = estimate_llm_step(step);
    assumptions.push(format!("~${:.2} per item × {count}", per_item.cost_usd));
    assumptions.extend(per_item.assumptions);

    StepEstimate {
        id: step.id.clone(),
        cost_usd: per_item.cost_usd * count as f64,
        assumptions,
    }
}

fn estimate_agent_step(step: &Step, is_parallel: bool) -> StepEstimate {
    let mut assumptions = Vec::new();

    let count = if is_parallel {
        resolve_fan_out_count(step, &mut assumptions)
    } else {
        1
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

    // Static context re-sent on every model call: the prompt plus any upstream
    // outputs or state files injected as template variables.
    let static_context = prompt_tokens + input_refs * ASSUMED_INPUT_REF_TOKENS;
    let mut input_tokens = static_context;
    let mut search_surcharge = 0.0;

    if step.tools.iter().any(|t| t == "web_search") {
        let searches = step.max_tool_calls.unwrap_or(DEFAULT_MAX_TOOL_CALLS);
        search_surcharge = searches as f64 * WEB_SEARCH_COST_PER_CALL_USD;

        // A web_search step runs a tool-use loop: searches + 1 model calls, and
        // the conversation grows each turn. The static context is re-sent on
        // every call, and each search result is re-sent on every later call —
        // so result tokens compound as the triangular number S(S+1)/2, not S.
        // This compounding is why a flat per-search estimate ran ~5x low.
        let turns = searches + 1;
        let compounded_results = ASSUMED_TOKENS_PER_SEARCH * searches * (searches + 1) / 2;
        input_tokens = static_context * turns + compounded_results;
        assumptions.push(format!(
            "{searches} web searches at ${WEB_SEARCH_COST_PER_CALL_USD}/search; \
             ~{ASSUMED_TOKENS_PER_SEARCH} tokens/result compounded across the \
             {turns}-turn tool loop"
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

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-9;

    fn step(src: &str) -> Step {
        toml::from_str(src).expect("test step TOML should parse")
    }

    #[test]
    fn static_parallel_items_use_exact_count() {
        let est = estimate_step(&step(
            "id = \"fan\"\nprompt = \"p\"\nprovider = \"claude-code\"\n\
             parallel_items = [\"a\", \"b\", \"c\"]\ncost_per_call_usd = 0.5",
        ));
        assert!((est.cost_usd - 1.5).abs() < EPS);
        // Count is known exactly, cost is explicit: nothing was assumed.
        assert!(est.assumptions.is_empty());
    }

    #[test]
    fn parallel_over_uses_max_items_cap() {
        let est = estimate_step(&step(
            "id = \"fan\"\nprompt = \"p\"\nprovider = \"claude-code\"\n\
             parallel_over = \"plan\"\ndepends_on = [\"plan\"]\n\
             max_items = 4\ncost_per_call_usd = 0.5",
        ));
        assert!((est.cost_usd - 2.0).abs() < EPS);
        assert!(est.assumptions.iter().any(|a| a.contains("max_items")));
    }

    #[test]
    fn unbounded_parallel_over_assumes_default_fan_out() {
        let est = estimate_step(&step(
            "id = \"fan\"\nprompt = \"p\"\nprovider = \"claude-code\"\n\
             parallel_over = \"plan\"\ndepends_on = [\"plan\"]\ncost_per_call_usd = 1.0",
        ));
        assert!((est.cost_usd - DEFAULT_FAN_OUT_ITEMS as f64).abs() < EPS);
        assert!(est.assumptions.iter().any(|a| a.contains("max_items")));
    }

    #[test]
    fn missing_agent_cost_falls_back_and_is_flagged() {
        let est = estimate_step(&step(
            "id = \"solo\"\nprompt = \"p\"\nprovider = \"claude-code\"",
        ));
        assert!((est.cost_usd - DEFAULT_AGENT_COST_PER_CALL_USD).abs() < EPS);
        assert!(
            est.assumptions
                .iter()
                .any(|a| a.contains("cost_per_call_usd"))
        );
    }

    #[test]
    fn parallel_step_without_agent_provider_estimates_as_api_fan_out() {
        // No provider: this now fans out as direct API calls, so the estimate
        // is per-item token cost × count, not cost_per_call_usd × count.
        let parallel = step(
            "id = \"fan\"\nprompt = \"research {{ item }}\"\n\
             parallel_over = \"plan\"\ndepends_on = [\"plan\"]\nmax_items = 4\n\
             tools = [\"web_search\"]\nmax_tool_calls = 3",
        );
        let single = step(
            "id = \"one\"\nprompt = \"research {{ item }}\"\ndepends_on = [\"plan\"]\n\
             tools = [\"web_search\"]\nmax_tool_calls = 3",
        );
        let est = estimate_step(&parallel);
        let per_item = estimate_step(&single);
        assert!((est.cost_usd - 4.0 * per_item.cost_usd).abs() < EPS);
        assert!(est.assumptions.iter().any(|a| a.contains("per item")));
    }

    #[test]
    fn web_search_adds_per_call_surcharge() {
        let plain = estimate_step(&step("id = \"a\"\nprompt = \"find papers\""));
        let searching = estimate_step(&step(
            "id = \"a\"\nprompt = \"find papers\"\ntools = [\"web_search\"]\nmax_tool_calls = 10",
        ));
        // 10 searches at $0.01 plus their result tokens — strictly more than
        // the surcharge alone.
        assert!(searching.cost_usd - plain.cost_usd > 10.0 * WEB_SEARCH_COST_PER_CALL_USD);
        assert!(
            searching
                .assumptions
                .iter()
                .any(|a| a.contains("web searches"))
        );
    }

    #[test]
    fn web_search_input_compounds_across_the_tool_loop() {
        // Each search result is re-sent on every later turn, so input grows as
        // the triangular number S(S+1)/2 — doubling the search count more than
        // doubles the cost. A flat per-search model (the old bug) would have
        // made these scale linearly.
        let two = estimate_step(&step(
            "id = \"a\"\nprompt = \"research it\"\ntools = [\"web_search\"]\nmax_tool_calls = 2",
        ));
        let four = estimate_step(&step(
            "id = \"a\"\nprompt = \"research it\"\ntools = [\"web_search\"]\nmax_tool_calls = 4",
        ));
        assert!(
            four.cost_usd > 2.0 * two.cost_usd,
            "4 searches ({}) should exceed 2x 2 searches ({})",
            four.cost_usd,
            two.cost_usd
        );
    }

    #[test]
    fn web_search_step_lands_in_the_calibrated_band() {
        // A 4-search research step mirroring the resort/villa fan-out. Real runs
        // put this near $0.20/item; the old flat model put it near $0.09. Guard
        // against regressing back to the optimistic estimate.
        let est = estimate_step(&step(
            "id = \"r\"\nprompt = \"Research this stay; cover layout, rates, vibe, food, \
             location, and booking reality.\"\ndepends_on = [\"scout\"]\n\
             tools = [\"web_search\"]\nmax_tool_calls = 4",
        ));
        assert!(
            est.cost_usd > 0.15 && est.cost_usd < 0.30,
            "expected ~$0.20/item, got {}",
            est.cost_usd
        );
    }

    #[test]
    fn workflow_total_is_sum_of_steps() {
        let wf: WorkflowFile = toml::from_str(
            r#"
[workflow]
name = "t"
[budget]
max_usd = 1.0
[[steps]]
id = "a"
prompt = "p"
[[steps]]
id = "b"
prompt = "p"
depends_on = ["a"]
[output]
step = "b"
"#,
        )
        .unwrap();
        let est = estimate_workflow(&wf);
        assert_eq!(est.steps.len(), 2);
        let sum: f64 = est.steps.iter().map(|s| s.cost_usd).sum();
        assert!((est.total_usd - sum).abs() < EPS);
        assert!(est.total_usd > 0.0);
    }
}
