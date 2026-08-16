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
// Per result, calibrated against real runs to within ~3%; compounding is applied below.
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
    // Mirrors the engine's dispatch: agents priced per call, everything else by tokens.
    let is_parallel = step.fans_out();

    // Deterministic steps cost nothing, however wide they fan out.
    if step.is_free() {
        return StepEstimate {
            id: step.id.clone(),
            cost_usd: 0.0,
            assumptions: vec!["deterministic step — no model calls".to_string()],
        };
    }

    if step.is_agent() {
        estimate_agent_step(step, is_parallel)
    } else if is_parallel {
        estimate_parallel_llm_step(step)
    } else {
        estimate_llm_step(step)
    }
}

/// Records an assumption whenever the item count is a guess rather than a known number.
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
    // Every item has the same prompt shape, so estimate one and multiply.
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

    // Each upstream output and state file has unknown size until run time.
    let input_refs = (step.depends_on.len() + step.reads_from.len()) as u32;
    if input_refs > 0 {
        assumptions.push(format!(
            "~{ASSUMED_INPUT_REF_TOKENS} tokens per upstream/state input ({input_refs})"
        ));
    }

    // Re-sent on every model call in a tool loop, so it is charged per turn below.
    let static_context = prompt_tokens + input_refs * ASSUMED_INPUT_REF_TOKENS;
    let mut input_tokens = static_context;
    let mut search_surcharge = 0.0;

    if step.tools.iter().any(|t| t == "web_search") {
        let searches = step.max_tool_calls.unwrap_or(DEFAULT_MAX_TOOL_CALLS);
        search_surcharge = searches as f64 * WEB_SEARCH_COST_PER_CALL_USD;

        // Results are re-sent each later turn, so they compound as S(S+1)/2, not S.
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
        // No provider means per-item token cost x count, not cost_per_call_usd x count.
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
        // Result tokens make this strictly more than the per-search surcharge alone.
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
        // Superlinear by S(S+1)/2, so a flat per-search model would fail this.
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
        // Real runs put a 4-search research step near $0.20/item, not the old $0.09.
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
