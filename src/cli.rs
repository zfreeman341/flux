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

    /// Input text passed directly to the workflow
    #[arg(long, conflicts_with = "input_file")]
    pub input: Option<String>,

    /// Read workflow input from a file
    #[arg(long, conflicts_with = "input")]
    pub input_file: Option<PathBuf>,

    /// Write output to a file instead of stdout
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

    /// Input text passed directly to the workflow
    #[arg(long, conflicts_with = "input_file")]
    pub input: Option<String>,

    /// Read workflow input from a file
    #[arg(long, conflicts_with = "input")]
    pub input_file: Option<PathBuf>,
}
