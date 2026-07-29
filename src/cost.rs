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
    /// Subscription-billed spend, reported for visibility but never enforced.
    pub notional_usd: f64,
    // While set, add() accumulates cost but never errors.
    emergency: bool,
}

impl BudgetTracker {
    pub fn new(limit_usd: f64) -> Self {
        Self {
            limit_usd,
            spent_usd: 0.0,
            notional_usd: 0.0,
            emergency: false,
        }
    }

    /// Records spend that isn't real money. Never fails: there is no ceiling to breach.
    pub fn add_notional(&mut self, cost: f64) {
        self.notional_usd += cost;
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

    /// Lifts the cap so the output step can still synthesize after a step was skipped.
    pub fn enable_emergency(&mut self) {
        self.emergency = true;
    }

    pub fn in_emergency(&self) -> bool {
        self.emergency
    }

    /// Rolls back a pre-charge for calls that never happened.
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

#[cfg(test)]
mod tests {
    use super::*;

    // Direct float equality is safe: these inputs are exact in f64.
    const EPS: f64 = 1e-9;

    #[test]
    fn add_under_limit_succeeds() {
        let mut budget = BudgetTracker::new(1.0);
        assert!(budget.add(0.5).is_ok());
        assert!((budget.spent_usd - 0.5).abs() < EPS);
    }

    #[test]
    fn add_exactly_at_limit_succeeds() {
        // The check is strictly greater-than: spending the full budget is allowed.
        let mut budget = BudgetTracker::new(1.0);
        assert!(budget.add(1.0).is_ok());
    }

    #[test]
    fn add_over_limit_errors_but_still_records_spend() {
        let mut budget = BudgetTracker::new(1.0);
        let err = budget.add(1.5).unwrap_err();
        assert!(matches!(err, FluxError::BudgetExceeded { .. }));
        // Recorded even when the limit trips; callers refund if no real spend happened.
        assert!((budget.spent_usd - 1.5).abs() < EPS);
    }

    #[test]
    fn emergency_mode_accumulates_without_erroring() {
        let mut budget = BudgetTracker::new(1.0);
        budget.enable_emergency();
        assert!(budget.add(5.0).is_ok());
        assert!((budget.spent_usd - 5.0).abs() < EPS);
    }

    #[test]
    fn refund_rolls_back_spend() {
        let mut budget = BudgetTracker::new(10.0);
        budget.add(3.0).unwrap();
        budget.refund(2.0);
        assert!((budget.spent_usd - 1.0).abs() < EPS);
        // Refund frees headroom: this would have exceeded without it.
        assert!(budget.add(9.0).is_ok());
    }

    #[test]
    fn token_pricing_matches_published_rates() {
        // 1M input + 1M output tokens at haiku rates = $1 + $5.
        assert!((step_cost_usd("claude-haiku-4-5", 1_000_000, 1_000_000) - 6.0).abs() < EPS);
        // Unknown model falls back to sonnet pricing ($3 + $15).
        assert!((step_cost_usd("mystery-model", 1_000_000, 1_000_000) - 18.0).abs() < EPS);
    }

    #[test]
    fn call_cost_adds_web_search_surcharge() {
        let without = call_cost_usd("claude-sonnet-4-6", 10_000, 1_000, 0);
        let with = call_cost_usd("claude-sonnet-4-6", 10_000, 1_000, 3);
        assert!((with - without - 3.0 * WEB_SEARCH_COST_PER_CALL_USD).abs() < EPS);
    }
}
