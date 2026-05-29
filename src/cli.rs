use clap::{Args, Parser, Subcommand};
use std::path::PathBuf;

#[derive(Debug, Parser)]
#[command(name = "flux", about = "Multi-agent LLM workflow runner", version)]
pub struct Cli {
    #[command(subcommand)]
    pub command: Command,
}

#[derive(Debug, Subcommand)]
pub enum Command {
    /// Execute a workflow
    Run(RunArgs),
    /// Validate a workflow file without running it
    Validate(ValidateArgs),
    /// Show what a workflow would do and estimate its cost
    Explain(ExplainArgs),
    /// List supported models
    ListModels,
}

#[derive(Debug, Args)]
pub struct RunArgs {
    /// Path to the workflow TOML file
    pub workflow: PathBuf,

    /// Input key=value pair; may be repeated for multiple inputs.
    /// A bare value (no '=') maps to key "input".
    /// Example: --input sector="legal AI" --input role="engineer"
    #[arg(long = "input")]
    pub inputs: Vec<String>,

    /// Read an input from a file as key=path; may be repeated.
    /// A bare path (no '=') maps to key "input".
    /// Example: --input-file resume=~/.flux-private/data/resume.md
    #[arg(long = "input-file")]
    pub input_files: Vec<String>,

    /// Write output to a file instead of the default outputs/ directory
    #[arg(long)]
    pub output: Option<PathBuf>,

    /// Increase logging verbosity
    #[arg(short, long)]
    pub verbose: bool,
}

#[derive(Debug, Args)]
pub struct ValidateArgs {
    /// Path to the workflow TOML file
    pub workflow: PathBuf,
}

#[derive(Debug, Args)]
pub struct ExplainArgs {
    /// Path to the workflow TOML file
    pub workflow: PathBuf,
}
