//! Cost tracking and pricing.
//!
//! Everything that knows what a dollar is lives here: per-model token pricing,
//! per-call server-tool pricing, and the budget tracker that enforces the
//! workflow's spending limit.

use tracing::warn;

use crate::FluxError;
use crate::workflow::Step;

/// Anthropic charges $10 per 1,000 web searches on top of token costs.
/// web_fetch has no per-call charge — only the fetched content's tokens.
pub const WEB_SEARCH_COST_PER_CALL_USD: f64 = 0.01;

pub struct BudgetTracker {
    pub limit_usd: f64,
    pub spent_usd: f64,
    // When true, add() accumulates cost but never errors. Enabled after a step
    // is skipped due to budget so the output step can still synthesize partial results.
    emergency: bool,
}

impl BudgetTracker {
    pub fn new(limit_usd: f64) -> Self {
        Self {
            limit_usd,
            spent_usd: 0.0,
            emergency: false,
        }
    }

    pub fn add(&mut self, cost: f64) -> crate::Result<()> {
        self.spent_usd += cost;
        if self.spent_usd > self.limit_usd && !self.emergency {
            return Err(FluxError::BudgetExceeded {
                spent: self.spent_usd,
                limit: self.limit_usd,
            });
        }
        Ok(())
    }

    // Called when a step is skipped due to BudgetExceeded so subsequent steps
    // (specifically the output/synthesizer step) can still run.
    pub fn enable_emergency(&mut self) {
        self.emergency = true;
    }

    // Rolls back a phantom pre-charge. Used when a parallel step pre-charges
    // for N agent calls but gets budget-killed before any agent actually runs —
    // the reported spend should reflect real API calls, not a failed estimate.
    pub fn refund(&mut self, amount: f64) {
        self.spent_usd -= amount;
    }
}

/// Cost of one API call: token cost plus the per-call surcharge for any
/// web searches the server executed during the request.
pub fn call_cost_usd(
    model: &str,
    input_tokens: u32,
    output_tokens: u32,
    web_search_requests: u32,
) -> f64 {
    step_cost_usd(model, input_tokens, output_tokens)
        + web_search_requests as f64 * WEB_SEARCH_COST_PER_CALL_USD
}

// Exposed so main.rs can show per-step cost without re-implementing the formula.
pub fn step_cost_usd(model: &str, input_tokens: u32, output_tokens: u32) -> f64 {
    let (input_price, output_price) = model_pricing(model);
    (input_tokens as f64 * input_price + output_tokens as f64 * output_price) / 1_000_000.0
}

// Resolves cost for agent steps. Warns if not set so authors know it's estimated.
pub fn resolve_cost(step: &Step, count: usize) -> f64 {
    let per_call = match step.cost_per_call_usd {
        Some(c) => c,
        None => {
            warn!(
                step = %step.id,
                "cost_per_call_usd not set; estimating $0.05 per call"
            );
            0.05
        }
    };
    per_call * count as f64
}

/// Pricing per million tokens (input, output), verified against
/// platform.claude.com June 2026: opus 4.7 $5/$25, sonnet 4.6 $3/$15,
/// haiku 4.5 $1/$5. Unknown models fall back to sonnet pricing.
pub fn model_pricing(model: &str) -> (f64, f64) {
    if model.contains("opus") {
        (5.0, 25.0)
    } else if model.contains("haiku") {
        (1.0, 5.0)
    } else {
        (3.0, 15.0)
    }
}
