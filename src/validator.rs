use std::collections::{HashMap, HashSet};

use crate::{FluxError, registry::Registry, workflow::WorkflowFile};

/// Static checks against the built-in registry. Use [`validate_workflow_with`]
/// when extra schemas, transforms, or probes are registered.
pub fn validate_workflow(wf: &WorkflowFile) -> crate::Result<()> {
    validate_workflow_with(wf, &Registry::builtin())
}

pub fn validate_workflow_with(wf: &WorkflowFile, registry: &Registry) -> crate::Result<()> {
    check_budget(wf)?;
    check_mcp_servers(wf)?;
    check_unique_ids(wf)?;
    check_depends_on_refs(wf)?;
    check_no_cycles(wf)?;
    check_providers(wf)?;
    check_parallel_over(wf)?;
    check_output_step(wf)?;
    check_step_kinds(wf, registry)?;
    Ok(())
}

/// Runs each step's kind-specific checks so a misconfigured HTTP or transform
/// step is caught by `flux validate`, not halfway through a paid run.
fn check_step_kinds(wf: &WorkflowFile, registry: &Registry) -> crate::Result<()> {
    for step in &wf.steps {
        let resolved = step.resolve(registry)?;
        match resolved {
            crate::workflow::StepKind::Mcp(mcp) => {
                if !wf.mcp.servers.contains_key(mcp.server) {
                    return Err(FluxError::Config(format!(
                        "step '{}' references unknown MCP server '{}'",
                        step.id, mcp.server
                    )));
                }
                if !step.prompt.is_empty()
                    || step.provider.is_some()
                    || !step.tools.is_empty()
                    || step.max_tool_calls.is_some()
                    || step.url.is_some()
                    || step.body.is_some()
                    || step.probe.is_some()
                    || step.transform.is_some()
                    || !step.args.is_empty()
                    || step.output_type.is_some()
                {
                    return Err(FluxError::Config(format!(
                        "step '{}': kind = \"mcp\" contains fields for another step kind",
                        step.id
                    )));
                }
            }
            crate::workflow::StepKind::Transform(_) => {
                if step.depends_on.is_empty() {
                    return Err(FluxError::Config(format!(
                        "step '{}' is a transform but depends_on nothing, so it has no input",
                        step.id
                    )));
                }
            }
            crate::workflow::StepKind::Agent(_) | crate::workflow::StepKind::Http(_) => {}
        }
    }
    Ok(())
}

fn check_mcp_servers(wf: &WorkflowFile) -> crate::Result<()> {
    for (name, server) in &wf.mcp.servers {
        if name.trim().is_empty() {
            return Err(FluxError::Config(
                "MCP server names must not be empty".to_string(),
            ));
        }
        if server.command.trim().is_empty() {
            return Err(FluxError::Config(format!(
                "MCP server '{name}' has an empty command"
            )));
        }
        let mut seen = HashSet::new();
        for variable in &server.pass_env {
            if variable.is_empty() || variable.contains('=') || variable.contains('\0') {
                return Err(FluxError::Config(format!(
                    "MCP server '{name}' has invalid environment variable name '{variable}'"
                )));
            }
            if !seen.insert(variable) {
                return Err(FluxError::Config(format!(
                    "MCP server '{name}' lists environment variable '{variable}' more than once"
                )));
            }
        }
    }
    Ok(())
}

// Checked up front because a typo here would otherwise surface after the budget is spent.
fn check_output_step(wf: &WorkflowFile) -> crate::Result<()> {
    if !wf.steps.iter().any(|s| s.id == wf.output.step) {
        return Err(FluxError::Config(format!(
            "output step '{}' does not match any step id",
            wf.output.step
        )));
    }
    Ok(())
}

/// Checks that every input declared in `[inputs]` was actually provided on
/// the command line. Called by `flux run` before any API client is built:
/// a missing input would otherwise render as an empty template variable
/// (lenient mode) and silently burn the whole budget on a garbage prompt.
pub fn check_required_inputs(
    wf: &WorkflowFile,
    provided: &HashMap<String, String>,
) -> crate::Result<()> {
    let mut missing: Vec<String> = wf
        .inputs
        .iter()
        .filter(|(name, _)| !provided.contains_key(*name))
        .map(|(name, def)| format!("  --input {name}=...  ({})", def.description))
        .collect();
    // HashMap iteration order is nondeterministic; sort for a stable message.
    missing.sort();

    if missing.is_empty() {
        return Ok(());
    }
    Err(FluxError::Config(format!(
        "missing required input(s):\n{}",
        missing.join("\n")
    )))
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

const VALID_PROVIDERS: &[&str] = &["anthropic", "hermes", "claude-code", "cursor"];

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

#[cfg(test)]
mod tests {
    use super::*;

    fn wf(src: &str) -> WorkflowFile {
        toml::from_str(src).expect("test workflow TOML should parse")
    }

    // Asserts on the message too, so a refactor cannot silently swap which check fired.
    fn assert_rejects(src: &str, expected_in_message: &str) {
        let err = validate_workflow(&wf(src)).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains(expected_in_message),
            "expected error containing '{expected_in_message}', got: {msg}"
        );
    }

    const VALID: &str = r#"
[workflow]
name = "test"
[budget]
max_usd = 1.0
[[steps]]
id = "a"
prompt = "p"
[[steps]]
id = "b"
prompt = "p"
depends_on = ["a"]
[output]
step = "b"
"#;

    #[test]
    fn accepts_valid_workflow() {
        assert!(validate_workflow(&wf(VALID)).is_ok());
    }

    #[test]
    fn rejects_zero_budget() {
        assert_rejects(
            r#"
[workflow]
name = "t"
[budget]
max_usd = 0.0
[[steps]]
id = "a"
prompt = "p"
[output]
step = "a"
"#,
            "max_usd",
        );
    }

    #[test]
    fn rejects_duplicate_step_ids() {
        assert_rejects(
            r#"
[workflow]
name = "t"
[budget]
max_usd = 1.0
[[steps]]
id = "a"
prompt = "p"
[[steps]]
id = "a"
prompt = "p"
[output]
step = "a"
"#,
            "duplicate step id",
        );
    }

    #[test]
    fn rejects_missing_dependency() {
        assert_rejects(
            r#"
[workflow]
name = "t"
[budget]
max_usd = 1.0
[[steps]]
id = "a"
prompt = "p"
depends_on = ["ghost"]
[output]
step = "a"
"#,
            "does not exist",
        );
    }

    #[test]
    fn rejects_cycle() {
        assert_rejects(
            r#"
[workflow]
name = "t"
[budget]
max_usd = 1.0
[[steps]]
id = "a"
prompt = "p"
depends_on = ["b"]
[[steps]]
id = "b"
prompt = "p"
depends_on = ["a"]
[output]
step = "b"
"#,
            "circular dependency",
        );
    }

    #[test]
    fn rejects_unknown_provider() {
        assert_rejects(
            r#"
[workflow]
name = "t"
[budget]
max_usd = 1.0
[[steps]]
id = "a"
prompt = "p"
provider = "gpt-4"
[output]
step = "a"
"#,
            "unknown provider",
        );
    }

    #[test]
    fn rejects_parallel_over_combined_with_parallel_items() {
        assert_rejects(
            r#"
[workflow]
name = "t"
[budget]
max_usd = 1.0
[[steps]]
id = "plan"
prompt = "p"
[[steps]]
id = "fan"
prompt = "p"
parallel_over = "plan"
parallel_items = ["x"]
depends_on = ["plan"]
[output]
step = "fan"
"#,
            "cannot set both",
        );
    }

    #[test]
    fn rejects_output_step_that_does_not_exist() {
        assert_rejects(
            r#"
[workflow]
name = "t"
[budget]
max_usd = 1.0
[[steps]]
id = "a"
prompt = "p"
[output]
step = "synthesisze"
"#,
            "does not match any step id",
        );
    }

    #[test]
    fn required_inputs_pass_when_provided() {
        let wf = wf(r#"
[workflow]
name = "t"
[budget]
max_usd = 1.0
[inputs]
topic = { description = "what to research" }
[[steps]]
id = "a"
prompt = "p"
[output]
step = "a"
"#);
        let provided = HashMap::from([("topic".to_string(), "rust".to_string())]);
        assert!(check_required_inputs(&wf, &provided).is_ok());
    }

    #[test]
    fn required_inputs_fail_listing_missing_keys() {
        let wf = wf(r#"
[workflow]
name = "t"
[budget]
max_usd = 1.0
[inputs]
topic = { description = "what to research" }
region = { description = "where" }
[[steps]]
id = "a"
prompt = "p"
[output]
step = "a"
"#);
        let provided = HashMap::from([("topic".to_string(), "rust".to_string())]);
        let msg = check_required_inputs(&wf, &provided)
            .unwrap_err()
            .to_string();
        assert!(
            msg.contains("region"),
            "should name the missing input: {msg}"
        );
        assert!(
            msg.contains("where"),
            "should include the description: {msg}"
        );
        assert!(
            !msg.contains("topic ="),
            "provided input should not be listed: {msg}"
        );
    }

    #[test]
    fn rejects_parallel_over_missing_from_depends_on() {
        assert_rejects(
            r#"
[workflow]
name = "t"
[budget]
max_usd = 1.0
[[steps]]
id = "plan"
prompt = "p"
[[steps]]
id = "fan"
prompt = "p"
parallel_over = "plan"
[output]
step = "fan"
"#,
            "not listed in depends_on",
        );
    }

    #[test]
    fn accepts_mcp_step_with_declared_server() {
        let workflow = wf(r#"
[workflow]
name = "mcp"
[budget]
max_usd = 1.0
[mcp.servers.local]
command = "mock-server"
args = ["--stdio"]
pass_env = ["TOKEN"]
[[steps]]
id = "call"
kind = "mcp"
server = "local"
tool = "echo"
arguments = { value = 42 }
[output]
step = "call"
"#);
        assert!(validate_workflow(&workflow).is_ok());
    }

    #[test]
    fn rejects_unknown_mcp_server() {
        assert_rejects(
            r#"
[workflow]
name = "mcp"
[budget]
max_usd = 1.0
[[steps]]
id = "call"
kind = "mcp"
server = "missing"
tool = "echo"
[output]
step = "call"
"#,
            "unknown MCP server",
        );
    }

    #[test]
    fn rejects_mcp_fields_from_other_step_kinds() {
        assert_rejects(
            r#"
[workflow]
name = "mcp"
[budget]
max_usd = 1.0
[mcp.servers.local]
command = "mock-server"
[[steps]]
id = "call"
kind = "mcp"
server = "local"
tool = "echo"
prompt = "not used"
[output]
step = "call"
"#,
            "fields for another step kind",
        );
    }
}
