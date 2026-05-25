use std::collections::HashMap;
use std::path::PathBuf;

use anyhow::{Context, Result};
use clap::Parser;
use flux::cli::{Cli, Command};

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Check verbose before init so we can set the log level once.
    let default_filter = match &cli.command {
        Command::Run(args) if args.verbose => "flux=debug",
        _ => "flux=info",
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(default_filter)),
        )
        .with_writer(std::io::stderr)
        .without_time()
        .with_target(false)
        .init();

    match cli.command {
        Command::Run(args) => run(args).await,
        Command::Validate(args) => validate(args),
        Command::Explain(args) => explain(args),
        Command::ListModels => list_models(),
    }
}

async fn run(args: flux::cli::RunArgs) -> Result<()> {
    let wf = flux::parser::parse_workflow(&args.workflow)
        .with_context(|| format!("failed to parse {}", args.workflow.display()))?;
    flux::validator::validate_workflow(&wf).context("workflow validation failed")?;

    let inputs = build_inputs(&args)?;

    let client =
        flux::anthropic::AnthropicClient::from_env().context("failed to init Anthropic client")?;
    let engine = flux::engine::Engine::new(Box::new(client));
    let mut budget = flux::engine::BudgetTracker::new(wf.budget.max_usd);

    let result = engine
        .run(&wf, &inputs, &mut budget)
        .await
        .context("workflow execution failed")?;

    let run_dir = match write_run_artifacts(&args.workflow, &wf, &result, &inputs) {
        Ok(dir) => Some(dir),
        Err(e) => {
            eprintln!("warning: failed to write run artifacts: {e}");
            None
        }
    };

    if let Some(out_path) = args.output {
        std::fs::write(&out_path, &result.output)
            .with_context(|| format!("failed to write to {}", out_path.display()))?;
        eprintln!("Output: {}", out_path.display());
    } else if let Some(path) = &run_dir {
        eprintln!("Output: {}", path.display());
    } else {
        // Artifact write failed — fall back to stdout so output isn't lost.
        println!("{}", result.output);
    }

    eprintln!(
        "Done  — spent ${:.4} of ${:.2} budget in {:.1}s",
        result.spent_usd,
        wf.budget.max_usd,
        result.duration.as_secs_f64()
    );

    Ok(())
}

fn build_inputs(args: &flux::cli::RunArgs) -> Result<HashMap<String, String>> {
    let mut inputs = HashMap::new();

    if let Some(text) = &args.input {
        let (key, value) = split_key_value(text);
        inputs.insert(key, value);
    } else if let Some(path) = &args.input_file {
        let content = std::fs::read_to_string(path)
            .with_context(|| format!("failed to read {}", path.display()))?;
        inputs.insert("input".to_string(), content);
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

fn write_run_artifacts(
    source_path: &std::path::Path,
    wf: &flux::workflow::WorkflowFile,
    result: &flux::engine::RunResult,
    inputs: &HashMap<String, String>,
) -> Result<PathBuf> {
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();

    // outputs/{workflow-name}-{input-slug}-{ts}.md
    // This gives each file a meaningful, human-readable name rather than
    // just a timestamp, so you can tell at a glance what a run was about.
    let input_slug = inputs
        .values()
        .next()
        .map(|v| {
            // Take first 40 chars of the first input value, slugify.
            let truncated: String = v.chars().take(40).collect();
            format!("-{}", slugify(&truncated))
        })
        .unwrap_or_default();
    let output_filename = format!("{}{}-{}.md", slugify(&wf.workflow.name), input_slug, ts);

    std::fs::create_dir_all("outputs")?;
    let output_path = PathBuf::from("outputs").join(&output_filename);

    let output_md = format!(
        "# {name}\n\n{desc}**Cost:** ${cost:.4}  \n**Duration:** {dur:.1}s\n\n---\n\n{output}\n",
        name = wf.workflow.name,
        desc = wf
            .workflow
            .description
            .as_deref()
            .map(|d| format!("{d}\n\n"))
            .unwrap_or_default(),
        cost = result.spent_usd,
        dur = result.duration.as_secs_f64(),
        output = result.output,
    );
    std::fs::write(&output_path, &output_md)?;

    // .flux-runs/{ts}/ — debug artifacts (prompts, step outputs, summary).
    let dir = PathBuf::from(format!(".flux-runs/{ts}"));
    std::fs::create_dir_all(&dir)?;
    std::fs::copy(source_path, dir.join("workflow.toml"))?;

    // Per-step debug files.
    for step in &result.step_results {
        let md = format!(
            "# {id}\n\nModel: `{model}`  \nTokens: {input_t} in / {output_t} out  \nCost: ${cost:.4}  \nDuration: {dur:.1}s\n\n## Prompt\n\n{prompt}\n\n## Output\n\n{output}\n",
            id = step.id,
            model = step.model,
            input_t = step.input_tokens,
            output_t = step.output_tokens,
            cost = step.cost_usd,
            dur = step.duration.as_secs_f64(),
            prompt = step.prompt,
            output = step.output,
        );
        std::fs::write(dir.join(format!("{}.md", step.id)), md)?;
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
        "# Run Summary\n\nWorkflow: **{name}**  \nTotal cost: **${cost:.4}**  \nDuration: **{dur:.1}s**\n\n## Steps\n\n| Step | Model | In tokens | Out tokens | Cost | Duration |\n|------|-------|-----------|------------|------|----------|\n{rows}",
        name = wf.workflow.name,
        cost = result.spent_usd,
        dur = result.duration.as_secs_f64(),
    );
    std::fs::write(dir.join("summary.md"), summary)?;

    Ok(output_path)
}

// Converts a string to a filename-safe slug: lowercase, runs of
// non-alphanumeric chars collapsed to a single hyphen, leading/trailing
// hyphens stripped.
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

fn validate(args: flux::cli::ValidateArgs) -> Result<()> {
    let wf = flux::parser::parse_workflow(&args.workflow)?;
    flux::validator::validate_workflow(&wf)?;
    println!("✓ {} is valid", args.workflow.display());
    Ok(())
}

fn explain(args: flux::cli::ExplainArgs) -> Result<()> {
    let wf = flux::parser::parse_workflow(&args.workflow)
        .with_context(|| format!("failed to parse {}", args.workflow.display()))?;
    flux::validator::validate_workflow(&wf).context("workflow validation failed")?;

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

    for (i, step) in wf.steps.iter().enumerate() {
        println!("Step {}: {}", i + 1, step.id);
        println!("  Model:  {}", step.model);
        if !step.depends_on.is_empty() {
            println!("  Deps:   {}", step.depends_on.join(", "));
        }
        if !step.tools.is_empty() {
            let cap = step
                .max_tool_calls
                .map(|n| format!(" (max {n})"))
                .unwrap_or_default();
            println!("  Tools:  {}{cap}", step.tools.join(", "));
        }
        println!("  Prompt: {} chars", step.prompt.len());
        println!();
    }

    println!("Output step: {}", wf.output.step);
    Ok(())
}

fn list_models() -> Result<()> {
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
