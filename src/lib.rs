//! # flux
//!
//! `flux` runs multi-agent LLM workflows defined in TOML. A workflow is a DAG
//! of steps, where each step is an agent — an LLM call with its own prompt,
//! model, and optional web-search tools. The runtime resolves dependencies
//! between steps, fans work out in parallel where the graph allows, and
//! synthesizes the results.
//!
//! ## Design goals
//!
//! - **Bounded concurrency.** Fan-out steps run in parallel under a semaphore
//!   so a workflow never oversubscribes the provider's rate limits.
//! - **Budget enforcement.** Every run has a hard USD ceiling. Cost is tracked
//!   per call and the run aborts before it overspends, rather than after.
//! - **Graceful degradation.** A single failed branch does not sink the whole
//!   workflow — surviving outputs still flow downstream to the synthesizer.
//! - **Observability.** Structured `tracing` spans and on-disk run artifacts
//!   make each step's input, output, and cost inspectable after the fact.
//!
//! ## Module map
//!
//! - [`workflow`] / [`parser`] / [`validator`] — the TOML schema, how it is
//!   parsed, and the checks that reject a malformed workflow before it runs.
//! - [`engine`] — the async executor: dependency resolution, parallel fan-out,
//!   and partial-failure handling.
//! - [`provider`] — the [`provider::LlmProvider`] abstraction that decouples
//!   the runtime from any single backend, plus its Anthropic implementation in
//!   [`anthropic`].
//! - [`cost`] / [`estimate`] — pricing, the budget tracker, and the cost
//!   estimate surfaced by `flux explain`.
//! - [`template`] — prompt rendering with cross-step variable substitution.
//!
//! ## Example
//!
//! A minimal two-step workflow — plan, then answer — looks like:
//!
//! ```toml
//! [workflow]
//! name = "example"
//!
//! [[steps]]
//! id = "plan"
//! prompt = "List three angles to research: {{ input }}"
//!
//! [[steps]]
//! id = "answer"
//! depends_on = ["plan"]
//! prompt = "Using this plan, write the answer:\n{{ plan.output }}"
//! ```

pub mod agent;
pub mod anthropic;
pub mod cli;
pub mod cost;
pub mod engine;
pub mod error;
pub mod estimate;
pub mod parser;
pub mod provider;
pub mod template;
pub mod validator;
pub mod workflow;

pub use error::{FluxError, Result};

/// Expands a leading `~/` to the user's home directory.
/// Paths without a leading `~/` are returned unchanged.
pub fn expand_tilde(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("~/")
        && let Some(home) = std::env::var_os("HOME")
    {
        return format!("{}/{}", home.to_string_lossy(), rest);
    }
    path.to_string()
}
