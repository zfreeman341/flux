use anyhow::Result;
use clap::Parser;
use flux::cli::{Cli, Command};

fn main() -> Result<()> {
    let cli = Cli::parse();

    match cli.command {
        Command::Run(args) => run(args),
        Command::Validate(args) => validate(args),
        Command::Explain(args) => explain(args),
        Command::ListModels => list_models(),
    }
}

fn run(args: flux::cli::RunArgs) -> Result<()> {
    println!("Would run workflow: {}", args.workflow.display());
    if let Some(input) = args.input {
        println!("  input: {input}");
    }
    if let Some(path) = args.input_file {
        println!("  input-file: {}", path.display());
    }
    if let Some(out) = args.output {
        println!("  output: {}", out.display());
    }
    if args.verbose {
        println!("  verbose: true");
    }
    Ok(())
}

fn validate(args: flux::cli::ValidateArgs) -> Result<()> {
    println!("Would validate workflow: {}", args.workflow.display());
    Ok(())
}

fn explain(args: flux::cli::ExplainArgs) -> Result<()> {
    println!("Would explain workflow: {}", args.workflow.display());
    if let Some(input) = args.input {
        println!("  input: {input}");
    }
    if let Some(path) = args.input_file {
        println!("  input-file: {}", path.display());
    }
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
