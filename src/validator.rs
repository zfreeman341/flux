use std::collections::{HashMap, HashSet};

use crate::{FluxError, workflow::WorkflowFile};

pub fn validate_workflow(wf: &WorkflowFile) -> crate::Result<()> {
    check_budget(wf)?;
    check_unique_ids(wf)?;
    check_depends_on_refs(wf)?;
    check_no_cycles(wf)?;
    check_providers(wf)?;
    check_parallel_over(wf)?;
    Ok(())
}

fn check_budget(wf: &WorkflowFile) -> crate::Result<()> {
    if wf.budget.max_usd <= 0.0 {
        return Err(FluxError::Config(
            "budget.max_usd must be greater than zero".into(),
        ));
    }
    Ok(())
}

fn check_unique_ids(wf: &WorkflowFile) -> crate::Result<()> {
    let mut seen = HashSet::new();
    for step in &wf.steps {
        if !seen.insert(step.id.as_str()) {
            return Err(FluxError::Config(format!(
                "duplicate step id: '{}'",
                step.id
            )));
        }
    }
    Ok(())
}

fn check_depends_on_refs(wf: &WorkflowFile) -> crate::Result<()> {
    let ids: HashSet<&str> = wf.steps.iter().map(|s| s.id.as_str()).collect();
    for step in &wf.steps {
        for dep in &step.depends_on {
            if !ids.contains(dep.as_str()) {
                return Err(FluxError::Config(format!(
                    "step '{}' depends_on '{}', which does not exist",
                    step.id, dep
                )));
            }
        }
    }
    Ok(())
}

const VALID_PROVIDERS: &[&str] = &["anthropic", "hermes", "claude-code"];

fn check_providers(wf: &WorkflowFile) -> crate::Result<()> {
    for step in &wf.steps {
        if let Some(ref p) = step.provider
            && !VALID_PROVIDERS.contains(&p.as_str())
        {
            return Err(FluxError::Config(format!(
                "step '{}' has unknown provider '{}' (valid: {})",
                step.id,
                p,
                VALID_PROVIDERS.join(", ")
            )));
        }
    }
    Ok(())
}

fn check_parallel_over(wf: &WorkflowFile) -> crate::Result<()> {
    let ids: HashSet<&str> = wf.steps.iter().map(|s| s.id.as_str()).collect();
    for step in &wf.steps {
        // parallel_over and parallel_items are mutually exclusive.
        if step.parallel_over.is_some() && !step.parallel_items.is_empty() {
            return Err(FluxError::Config(format!(
                "step '{}' cannot set both parallel_over and parallel_items",
                step.id
            )));
        }

        if let Some(ref upstream) = step.parallel_over {
            if !ids.contains(upstream.as_str()) {
                return Err(FluxError::Config(format!(
                    "step '{}' parallel_over '{}', which does not exist",
                    step.id, upstream
                )));
            }
            if !step.depends_on.iter().any(|d| d == upstream) {
                return Err(FluxError::Config(format!(
                    "step '{}' parallel_over '{}' but '{}' is not listed in depends_on",
                    step.id, upstream, upstream
                )));
            }
        }
    }
    Ok(())
}

fn check_no_cycles(wf: &WorkflowFile) -> crate::Result<()> {
    let steps_by_id: HashMap<&str, _> = wf.steps.iter().map(|s| (s.id.as_str(), s)).collect();
    let mut visited: HashSet<&str> = HashSet::new();
    let mut in_stack: HashSet<&str> = HashSet::new();

    for step in &wf.steps {
        if !visited.contains(step.id.as_str()) {
            dfs(step.id.as_str(), &steps_by_id, &mut visited, &mut in_stack)?;
        }
    }
    Ok(())
}

fn dfs<'a>(
    id: &'a str,
    steps_by_id: &HashMap<&'a str, &'a crate::workflow::Step>,
    visited: &mut HashSet<&'a str>,
    in_stack: &mut HashSet<&'a str>,
) -> crate::Result<()> {
    in_stack.insert(id);

    for dep in &steps_by_id[id].depends_on {
        let dep = dep.as_str();
        if in_stack.contains(dep) {
            return Err(FluxError::Config(format!(
                "circular dependency: step '{}' is part of a cycle (reached '{}' again)",
                id, dep
            )));
        }
        if !visited.contains(dep) {
            dfs(dep, steps_by_id, visited, in_stack)?;
        }
    }

    in_stack.remove(id);
    visited.insert(id);
    Ok(())
}
