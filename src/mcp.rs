//! Deterministic MCP tool execution over local stdio child processes.

use std::collections::{HashMap, HashSet};
use std::time::Duration;

use futures::future::join_all;
use rmcp::{
    ServiceExt,
    model::{CallToolRequestParams, CallToolResult, ContentBlock},
    transport::TokioChildProcess,
};
use serde_json::{Map, Value, json};
use tokio::{process::Command, sync::Semaphore, time::timeout};
use tracing::{info, warn};

use crate::{
    artifact::{Artifact, Schema},
    template::FanItem,
    workflow::{McpServerConfig, McpStep},
};

const DEFAULT_TIMEOUT: Duration = Duration::from_secs(60);
const SHUTDOWN_TIMEOUT: Duration = Duration::from_secs(3);
const BASE_ENV: &[&str] = &["HOME", "PATH", "TMPDIR"];

#[derive(Debug, Default)]
pub struct McpExecutor;

pub struct McpRunContext<'a> {
    pub items: &'a [FanItem],
    pub inputs: &'a HashMap<String, String>,
    pub step_outputs: &'a HashMap<String, Artifact>,
    pub max_concurrent: usize,
    pub timeout_secs: Option<u64>,
}

impl McpExecutor {
    /// Starts one server session, calls one named tool, then closes the session.
    pub async fn run(
        &self,
        server_name: &str,
        server: &McpServerConfig,
        step: &McpStep<'_>,
        context: McpRunContext<'_>,
    ) -> crate::Result<Artifact> {
        let plan = plan_calls(
            step.arguments,
            context.items,
            context.inputs,
            context.step_outputs,
        )?;
        let transport = TokioChildProcess::new(configure_command(server))
            .map_err(|e| crate::FluxError::Mcp(format!("failed to start '{server_name}': {e}")))?;
        let mut client = ().serve(transport).await.map_err(|e| {
            crate::FluxError::Mcp(format!("failed to initialize '{server_name}': {e}"))
        })?;

        let outcome = async {
            let tools = client.list_all_tools().await.map_err(|e| {
                crate::FluxError::Mcp(format!(
                    "failed to list tools from MCP server '{server_name}': {e}"
                ))
            })?;
            if !tools.iter().any(|candidate| candidate.name == step.tool) {
                return Err(crate::FluxError::Mcp(format!(
                    "MCP server '{server_name}' does not expose tool '{}'",
                    step.tool
                )));
            }

            let call_timeout = Duration::from_secs(
                context
                    .timeout_secs
                    .unwrap_or(DEFAULT_TIMEOUT.as_secs())
                    .max(1),
            );
            let semaphore = Semaphore::new(context.max_concurrent.max(1));
            let calls = plan.into_iter().map(|(item, arguments)| {
                let client = &client;
                let semaphore = &semaphore;
                async move {
                    let _permit = semaphore
                        .acquire()
                        .await
                        .expect("semaphore is never closed");
                    let params =
                        CallToolRequestParams::new(step.tool.to_string()).with_arguments(arguments);
                    let response = timeout(call_timeout, client.call_tool(params)).await;
                    response_row(server_name, step.tool, item, response)
                }
            });
            let rows = join_all(calls).await;
            let errors = rows
                .iter()
                .filter(|row| row["is_error"].as_bool() == Some(true))
                .count();
            info!(
                server = server_name,
                tool = step.tool,
                calls = rows.len(),
                errors,
                "MCP step complete"
            );
            Ok(Artifact::records(Schema::mcp_response(), rows))
        }
        .await;

        match client.close_with_timeout(SHUTDOWN_TIMEOUT).await {
            Ok(Some(_)) => {}
            Ok(None) => warn!(server = server_name, "MCP server shutdown timed out"),
            Err(e) => warn!(server = server_name, error = %e, "MCP server shutdown failed"),
        }
        outcome
    }
}

fn configure_command(server: &McpServerConfig) -> Command {
    let mut command = Command::new(&server.command);
    command.args(&server.args).env_clear();

    let variables: HashSet<&str> = BASE_ENV
        .iter()
        .copied()
        .chain(server.pass_env.iter().map(String::as_str))
        .collect();
    for name in variables {
        if let Some(value) = std::env::var_os(name) {
            command.env(name, value);
        }
    }
    command
}

fn plan_calls(
    arguments: &Map<String, Value>,
    items: &[FanItem],
    inputs: &HashMap<String, String>,
    step_outputs: &HashMap<String, Artifact>,
) -> crate::Result<Vec<(Value, Map<String, Value>)>> {
    let render_one = |item: Option<&FanItem>| {
        let rendered = render_value(
            &Value::Object(arguments.clone()),
            inputs,
            step_outputs,
            item,
        )?;
        let Value::Object(arguments) = rendered else {
            unreachable!("MCP arguments begin as an object");
        };
        let tag = item.map(|item| item.value.clone()).unwrap_or(Value::Null);
        Ok((tag, arguments))
    };

    if items.is_empty() {
        return Ok(vec![render_one(None)?]);
    }
    items.iter().map(|item| render_one(Some(item))).collect()
}

fn render_value(
    value: &Value,
    inputs: &HashMap<String, String>,
    step_outputs: &HashMap<String, Artifact>,
    item: Option<&FanItem>,
) -> crate::Result<Value> {
    match value {
        Value::String(template) => {
            crate::template::render(template, inputs, step_outputs, item).map(Value::String)
        }
        Value::Array(values) => values
            .iter()
            .map(|value| render_value(value, inputs, step_outputs, item))
            .collect::<crate::Result<Vec<_>>>()
            .map(Value::Array),
        Value::Object(values) => values
            .iter()
            .map(|(key, value)| {
                render_value(value, inputs, step_outputs, item)
                    .map(|rendered| (key.clone(), rendered))
            })
            .collect::<crate::Result<Map<_, _>>>()
            .map(Value::Object),
        other => Ok(other.clone()),
    }
}

fn response_row(
    server: &str,
    tool: &str,
    item: Value,
    response: Result<
        Result<CallToolResult, rmcp::service::ServiceError>,
        tokio::time::error::Elapsed,
    >,
) -> Value {
    let (is_error, result) = match response {
        Ok(Ok(result)) => (result.is_error.unwrap_or(false), normalize_result(&result)),
        Ok(Err(error)) => {
            warn!(server, tool, error = %error, "MCP tool call failed");
            (true, json!({"error": error.to_string()}))
        }
        Err(_) => {
            warn!(server, tool, "MCP tool call timed out");
            (true, json!({"error": "tool call timed out"}))
        }
    };
    json!({
        "server": server,
        "tool": tool,
        "item": item,
        "is_error": is_error,
        "result": result,
    })
}

fn normalize_result(result: &CallToolResult) -> Value {
    if let Some(structured) = &result.structured_content {
        return structured.clone();
    }

    let mut values = Vec::with_capacity(result.content.len());
    for content in &result.content {
        match content {
            ContentBlock::Text(text) => values.push(Value::String(text.text.clone())),
            other => values.push(serde_json::to_value(other).unwrap_or(Value::Null)),
        }
    }
    match values.len() {
        0 => Value::Null,
        1 => values.pop().unwrap_or(Value::Null),
        _ => Value::Array(values),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fixture_server(pass_env: Vec<String>) -> McpServerConfig {
        McpServerConfig {
            command: "python3".to_string(),
            args: vec![format!(
                "{}/tests/fixtures/mock_mcp_server.py",
                env!("CARGO_MANIFEST_DIR")
            )],
            pass_env,
        }
    }

    #[test]
    fn recursive_render_preserves_non_string_types() {
        let value = json!({
            "query": "{{ topic }}",
            "limit": 3,
            "nested": {"enabled": true, "label": "{{ topic }}"}
        });
        let inputs = HashMap::from([("topic".to_string(), "Rust".to_string())]);
        let rendered = render_value(&value, &inputs, &HashMap::new(), None).unwrap();
        assert_eq!(rendered["query"], "Rust");
        assert_eq!(rendered["limit"], 3);
        assert_eq!(rendered["nested"]["enabled"], true);
        assert_eq!(rendered["nested"]["label"], "Rust");
    }

    #[tokio::test]
    async fn stdio_calls_preserve_fan_out_order_and_close_the_server() {
        let temp = tempfile::tempdir().unwrap();
        let marker = temp.path().join("closed");
        let arguments = json!({
            "value": "{{ item }}",
            "shutdown_marker": marker.to_string_lossy()
        })
        .as_object()
        .unwrap()
        .clone();
        let step = McpStep {
            server: "test",
            tool: "echo",
            arguments: &arguments,
        };
        let items = vec![FanItem::from_text("first"), FanItem::from_text("second")];

        let output = McpExecutor
            .run(
                "test",
                &fixture_server(Vec::new()),
                &step,
                McpRunContext {
                    items: &items,
                    inputs: &HashMap::new(),
                    step_outputs: &HashMap::new(),
                    max_concurrent: 2,
                    timeout_secs: Some(5),
                },
            )
            .await
            .unwrap();

        assert_eq!(output.rows().len(), 2);
        assert_eq!(output.rows()[0]["item"], "first");
        assert_eq!(output.rows()[0]["result"]["arguments"]["value"], "first");
        assert_eq!(output.rows()[1]["item"], "second");
        assert_eq!(output.rows()[1]["result"]["arguments"]["value"], "second");
        assert!(marker.exists());
    }

    #[tokio::test]
    async fn text_results_and_tool_errors_are_normalized() {
        let text_arguments = json!({"mode": "text", "text": "hello"})
            .as_object()
            .unwrap()
            .clone();
        let text_step = McpStep {
            server: "test",
            tool: "echo",
            arguments: &text_arguments,
        };
        let output = McpExecutor
            .run(
                "test",
                &fixture_server(Vec::new()),
                &text_step,
                McpRunContext {
                    items: &[],
                    inputs: &HashMap::new(),
                    step_outputs: &HashMap::new(),
                    max_concurrent: 1,
                    timeout_secs: Some(5),
                },
            )
            .await
            .unwrap();
        assert_eq!(output.rows()[0]["result"], "hello");
        assert_eq!(output.rows()[0]["is_error"], false);

        let error_arguments = json!({"fail": true}).as_object().unwrap().clone();
        let error_step = McpStep {
            server: "test",
            tool: "echo",
            arguments: &error_arguments,
        };
        let output = McpExecutor
            .run(
                "test",
                &fixture_server(Vec::new()),
                &error_step,
                McpRunContext {
                    items: &[],
                    inputs: &HashMap::new(),
                    step_outputs: &HashMap::new(),
                    max_concurrent: 1,
                    timeout_secs: Some(5),
                },
            )
            .await
            .unwrap();
        assert_eq!(output.rows()[0]["is_error"], true);
    }

    #[tokio::test]
    async fn child_receives_only_baseline_and_allowlisted_environment() {
        let mut candidates = std::env::vars()
            .filter(|(name, _)| !BASE_ENV.contains(&name.as_str()))
            .take(2);
        let (allowed_name, allowed_value) = candidates.next().expect("test needs one env variable");
        let (blocked_name, _) = candidates.next().expect("test needs two env variables");
        let arguments = json!({
            "allowed_env_name": allowed_name,
            "blocked_env_name": blocked_name
        })
        .as_object()
        .unwrap()
        .clone();
        let step = McpStep {
            server: "test",
            tool: "echo",
            arguments: &arguments,
        };

        let output = McpExecutor
            .run(
                "test",
                &fixture_server(vec![allowed_name]),
                &step,
                McpRunContext {
                    items: &[],
                    inputs: &HashMap::new(),
                    step_outputs: &HashMap::new(),
                    max_concurrent: 1,
                    timeout_secs: Some(5),
                },
            )
            .await
            .unwrap();

        assert_eq!(output.rows()[0]["result"]["allowed_env"], allowed_value);
        assert_eq!(output.rows()[0]["result"]["blocked_env"], Value::Null);
    }

    #[tokio::test]
    async fn missing_tool_is_a_step_error() {
        let arguments = Map::new();
        let step = McpStep {
            server: "test",
            tool: "missing",
            arguments: &arguments,
        };
        let error = McpExecutor
            .run(
                "test",
                &fixture_server(Vec::new()),
                &step,
                McpRunContext {
                    items: &[],
                    inputs: &HashMap::new(),
                    step_outputs: &HashMap::new(),
                    max_concurrent: 1,
                    timeout_secs: Some(5),
                },
            )
            .await
            .unwrap_err();
        assert!(error.to_string().contains("does not expose tool"));
    }

    #[tokio::test]
    async fn timeout_becomes_an_error_row() {
        let arguments = json!({"delay_ms": 1_500}).as_object().unwrap().clone();
        let step = McpStep {
            server: "test",
            tool: "echo",
            arguments: &arguments,
        };
        let output = McpExecutor
            .run(
                "test",
                &fixture_server(Vec::new()),
                &step,
                McpRunContext {
                    items: &[],
                    inputs: &HashMap::new(),
                    step_outputs: &HashMap::new(),
                    max_concurrent: 1,
                    timeout_secs: Some(1),
                },
            )
            .await
            .unwrap();
        assert_eq!(output.rows()[0]["is_error"], true);
        assert_eq!(output.rows()[0]["result"]["error"], "tool call timed out");
    }
}
