//! The `flux run` pipeline as a library call.
//!
//! `main.rs` is a thin wrapper over this module. It lives in the library so a
//! binary that registers its own schemas, transforms, or probes can run the
//! same pipeline with its own [`Registry`], then act on the [`RunOutcome`].

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::FluxError;
use crate::cli::RunArgs;
use crate::engine::RunResult;
use crate::registry::Registry;
use crate::workflow::WorkflowFile;

/// Everything a finished run produced, for a caller that wants to post-process it.
pub struct RunOutcome {
    pub workflow: WorkflowFile,
    pub result: RunResult,
    /// The final report, or `None` if writing run artifacts failed.
    pub output_path: Option<PathBuf>,
}

/// Parses and statically checks a workflow against `registry`.
pub fn load(path: &Path, registry: &Registry) -> crate::Result<WorkflowFile> {
    let wf = crate::parser::parse_workflow(path)?;
    crate::validator::validate_workflow_with(&wf, registry)?;
    Ok(wf)
}

/// `flux validate`: parse and check, spending nothing.
pub fn validate(path: &Path, registry: &Registry) -> crate::Result<()> {
    load(path, registry)?;
    println!("✓ {} is valid", path.display());
    Ok(())
}

/// Executes a workflow and writes its run artifacts.
pub async fn run(args: &RunArgs, registry: Arc<Registry>) -> crate::Result<RunOutcome> {
    let wf = load(&args.workflow, &registry)?;

    let inputs = build_inputs(args)?;
    // Lenient templates would otherwise spend the budget on a prompt full of blanks.
    crate::validator::check_required_inputs(&wf, &inputs)?;

    // Agent steps route by their `provider` name; claude-code is the default.
    let agent = Arc::new(crate::agent::CliAgentProvider::claude_code());
    // Skipped entirely for an all-agent workflow, which needs no ANTHROPIC_API_KEY.
    let needs_api = wf.steps.iter().any(|s| !s.is_agent() && !s.is_free());
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let artifact_dir = PathBuf::from(format!(".flux-runs/{ts}"));

    let engine = if needs_api {
        let client = crate::anthropic::AnthropicClient::from_env()?;
        crate::engine::Engine::new(Arc::new(client), agent)?
    } else {
        crate::engine::Engine::agents_only(agent)?
    }
    .with_agent("cursor", Arc::new(crate::agent::CliAgentProvider::cursor()))
    .with_agent("hermes", Arc::new(crate::agent::CliAgentProvider::hermes()))
    .with_registry(registry)
    .with_checkpoint_dir(artifact_dir.clone());
    let mut budget = crate::cost::BudgetTracker::new(wf.budget.max_usd);

    let result = engine.run(&wf, &inputs, &mut budget).await;
    if result.is_err() && artifact_dir.exists() {
        eprintln!(
            "Partial step outputs from the failed run: {}",
            artifact_dir.display()
        );
    }
    let result = result?;

    let output_path =
        match write_run_artifacts(&args.workflow, &wf, &result, &inputs, &artifact_dir, ts) {
            Ok(path) => Some(path),
            Err(e) => {
                eprintln!("warning: failed to write run artifacts: {e}");
                None
            }
        };

    Ok(RunOutcome {
        workflow: wf,
        result,
        output_path,
    })
}

/// Reports where the output went and what the run cost.
pub fn finish(args: &RunArgs, outcome: &RunOutcome) -> crate::Result<()> {
    let result = &outcome.result;
    if let Some(out_path) = &args.output {
        std::fs::write(out_path, &result.output)?;
        eprintln!("Output: {}", out_path.display());
    } else if let Some(path) = &outcome.output_path {
        eprintln!("Output: {}", path.display());
    } else {
        // Artifact write failed — fall back to stdout so output isn't lost.
        println!("{}", result.output);
    }

    eprintln!(
        "Done  — spent ${:.4} of ${:.2} budget in {:.1}s",
        result.spent_usd,
        outcome.workflow.budget.max_usd,
        result.duration.as_secs_f64()
    );
    if result.notional_usd > 0.0 {
        eprintln!(
            "        plus ${:.4} billed to the agent CLI's own plan (not charged against the budget)",
            result.notional_usd
        );
    }
    Ok(())
}

fn build_inputs(args: &RunArgs) -> crate::Result<HashMap<String, String>> {
    let mut inputs = HashMap::new();

    for kv in &args.inputs {
        let (key, value) = split_key_value(kv);
        inputs.insert(key, value);
    }

    for kf in &args.input_files {
        let (key, raw_path) = split_key_value(kf);
        // So callers can write --input-file notes=~/notes.md and have it resolve.
        let expanded = crate::expand_tilde(&raw_path);
        let content = std::fs::read_to_string(&expanded).map_err(|e| {
            FluxError::InvalidArgument(format!("failed to read input file '{expanded}': {e}"))
        })?;
        inputs.insert(key, content);
    }

    Ok(inputs)
}

// "key=value" splits on first '='; a plain string maps to key "input".
fn split_key_value(s: &str) -> (String, String) {
    match s.split_once('=') {
        Some((k, v)) => (k.to_string(), v.to_string()),
        None => ("input".to_string(), s.to_string()),
    }
}

fn resolve_output_dir(output_dir: &Option<String>) -> PathBuf {
    match output_dir {
        None => PathBuf::from("outputs"),
        Some(dir) => {
            let expanded = crate::expand_tilde(dir);
            let p = std::path::Path::new(&expanded);
            if p.is_absolute() {
                p.to_path_buf()
            } else {
                PathBuf::from("outputs").join(&expanded)
            }
        }
    }
}

fn write_run_artifacts(
    source_path: &std::path::Path,
    wf: &crate::workflow::WorkflowFile,
    result: &crate::engine::RunResult,
    inputs: &HashMap<String, String>,
    dir: &std::path::Path,
    ts: u64,
) -> crate::Result<PathBuf> {
    let input_slug = inputs
        .values()
        .next()
        .map(|v| {
            let truncated: String = v.chars().take(40).collect();
            format!("-{}", slugify(&truncated))
        })
        .unwrap_or_default();
    let output_filename = format!("{}{}-{}.md", slugify(&wf.workflow.name), input_slug, ts);

    let out_dir = resolve_output_dir(&wf.workflow.output_dir);
    std::fs::create_dir_all(&out_dir)?;
    let output_path = out_dir.join(&output_filename);

    let output_md = format!(
        "# {name}\n\n{desc}**Cost:** ${cost:.4}{notional}  \n**Duration:** {dur:.1}s\n\n---\n\n{output}\n",
        name = wf.workflow.name,
        desc = wf
            .workflow
            .description
            .as_deref()
            .map(|d| format!("{d}\n\n"))
            .unwrap_or_default(),
        cost = result.spent_usd,
        notional = notional_note(result.notional_usd),
        dur = result.duration.as_secs_f64(),
        output = result.output,
    );
    std::fs::write(&output_path, &output_md)?;

    // .flux-runs/{ts}/ — debug artifacts (prompts, step outputs, summary).
    // The engine already checkpointed per-step files here as steps completed.
    std::fs::create_dir_all(dir)?;
    std::fs::copy(source_path, dir.join("workflow.toml"))?;

    for step in &result.step_results {
        std::fs::write(dir.join(format!("{}.md", step.id)), step.to_markdown())?;
    }

    let rows: String = result
        .step_results
        .iter()
        .map(|s| {
            format!(
                "| {} | {} | {} | {} | ${:.4} | {:.1}s |\n",
                s.id,
                s.model,
                s.input_tokens,
                s.output_tokens,
                s.cost_usd,
                s.duration.as_secs_f64(),
            )
        })
        .collect();

    let summary = format!(
        "# Run Summary\n\nWorkflow: **{name}**  \nTotal cost: **${cost:.4}**{notional}  \nDuration: **{dur:.1}s**\n\n## Steps\n\n| Step | Model | In tokens | Out tokens | Cost | Duration |\n|------|-------|-----------|------------|------|----------|\n{rows}",
        name = wf.workflow.name,
        cost = result.spent_usd,
        notional = notional_note(result.notional_usd),
        dur = result.duration.as_secs_f64(),
    );
    std::fs::write(dir.join("summary.md"), summary)?;

    Ok(output_path)
}

// Labelled so an artifact read months later cannot be mistaken for a bill.
fn notional_note(notional_usd: f64) -> String {
    if notional_usd > 0.0 {
        format!(" (+ ${notional_usd:.4} notional, billed to the agent CLI's own plan)")
    } else {
        String::new()
    }
}

/// Lowercases, collapses non-alphanumeric runs to one hyphen, and trims hyphens.
fn slugify(s: &str) -> String {
    let mut slug = String::with_capacity(s.len());
    let mut last_was_hyphen = true; // start true to strip leading hyphens
    for c in s.chars() {
        if c.is_ascii_alphanumeric() {
            slug.push(c.to_ascii_lowercase());
            last_was_hyphen = false;
        } else if !last_was_hyphen {
            slug.push('-');
            last_was_hyphen = true;
        }
    }
    // Strip trailing hyphen
    if slug.ends_with('-') {
        slug.pop();
    }
    slug
}

/// Prints the execution plan and a per-step cost estimate. Makes no API calls.
pub fn explain(path: &Path, registry: &Registry) -> crate::Result<()> {
    let wf = load(path, registry)?;

    let estimate = crate::estimate::estimate_workflow(&wf);

    println!("Workflow: {}", wf.workflow.name);
    if let Some(desc) = &wf.workflow.description {
        println!("Description: {desc}");
    }
    println!("Budget: ${:.2}", wf.budget.max_usd);
    if !wf.inputs.is_empty() {
        println!("Inputs:");
        for (k, v) in &wf.inputs {
            println!("  {k}: {}", v.description);
        }
    }
    println!();

    for ((i, step), est) in wf.steps.iter().enumerate().zip(&estimate.steps) {
        println!("Step {}: {}", i + 1, step.id);
        match step.kind {
            crate::workflow::StepKindTag::Mcp => {
                println!("  Kind:     MCP");
                println!("  Server:   {}", step.server.as_deref().unwrap_or(""));
                println!("  Tool:     {}", step.tool.as_deref().unwrap_or(""));
            }
            crate::workflow::StepKindTag::Http => println!("  Kind:     HTTP"),
            crate::workflow::StepKindTag::Transform => {
                println!("  Kind:     transform");
                println!("  Transform: {}", step.transform.as_deref().unwrap_or(""));
            }
            crate::workflow::StepKindTag::Agent => match step.provider.as_deref() {
                Some(p) if p != "anthropic" => println!("  Provider: {p}"),
                _ => println!("  Model:    {}", step.model),
            },
        }
        if let Some(ref upstream) = step.parallel_over {
            let concurrency = step.max_concurrent.unwrap_or(3);
            let cap = step
                .max_items
                .map(|n| format!(", max {n} items"))
                .unwrap_or_default();
            println!("  Fan-out:  over '{upstream}' (max {concurrency} concurrent{cap})");
        }
        if !step.depends_on.is_empty() {
            println!("  Deps:     {}", step.depends_on.join(", "));
        }
        if !step.tools.is_empty() {
            let cap = step
                .max_tool_calls
                .map(|n| format!(" (max {n})"))
                .unwrap_or_default();
            println!("  Tools:    {}{cap}", step.tools.join(", "));
        }
        if step.kind == crate::workflow::StepKindTag::Agent {
            println!("  Prompt:   {} chars", step.prompt.len());
        }
        println!("  Est cost: ~${:.2}", est.cost_usd);
        if !est.assumptions.is_empty() {
            println!("            ({})", est.assumptions.join("; "));
        }
        println!();
    }

    println!("Output step: {}", wf.output.step);
    println!(
        "Estimated total: ~${:.2} of ${:.2} budget",
        estimate.total_usd, wf.budget.max_usd
    );
    if estimate.total_usd > wf.budget.max_usd {
        println!(
            "WARNING: estimate exceeds budget — `flux run` would likely abort mid-workflow. \
             Raise budget.max_usd or shrink the workflow."
        );
    }
    println!("(Estimates are heuristic: pre-run token counts and fan-out sizes are unknown.)");
    if wf.steps.iter().any(|s| s.is_agent()) {
        println!(
            "(Agent steps bill the CLI's own login, so their cost is notional — what the \
             work would cost on the API — and is not charged against the budget.)"
        );
    }
    Ok(())
}

/// `flux list-models`: the supported model shorthand.
pub fn list_models() -> crate::Result<()> {
    println!("{:<35} DESCRIPTION", "MODEL");
    println!("{}", "-".repeat(75));
    println!(
        "{:<35} Fast and cheap — good for simple, high-volume steps",
        "claude-haiku-4-5-20251001"
    );
    println!(
        "{:<35} Balanced — default choice for most steps",
        "claude-sonnet-4-6"
    );
    println!(
        "{:<35} Most capable — use for complex synthesis and reasoning",
        "claude-opus-4-7"
    );
    Ok(())
}
