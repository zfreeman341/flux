use std::sync::Arc;

use anyhow::{Context, Result};
use clap::Parser;
use flux::cli::{Cli, Command};
use flux::registry::Registry;
use flux::runner;

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

    let registry = Registry::builtin();
    match cli.command {
        Command::Run(args) => {
            let outcome = runner::run(&args, Arc::new(registry))
                .await
                .context("workflow execution failed")?;
            runner::finish(&args, &outcome)?;
        }
        Command::Validate(args) => runner::validate(&args.workflow, &registry)?,
        Command::Explain(args) => runner::explain(&args.workflow, &registry)?,
        Command::ListModels => runner::list_models()?,
    }
    Ok(())
}
