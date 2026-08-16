use std::collections::HashMap;

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::artifact::Schema;
use crate::registry::Registry;

#[derive(Debug, Deserialize)]
pub struct WorkflowFile {
    pub workflow: WorkflowMeta,
    pub budget: Budget,
    #[serde(default)]
    pub inputs: HashMap<String, InputDef>,
    #[serde(default)]
    pub mcp: McpConfig,
    pub steps: Vec<Step>,
    pub output: Output,
}

#[derive(Debug, Default, Deserialize)]
pub struct McpConfig {
    #[serde(default)]
    pub servers: HashMap<String, McpServerConfig>,
}

/// A local MCP server started as a child process over stdio.
#[derive(Debug, Deserialize)]
pub struct McpServerConfig {
    pub command: String,
    #[serde(default)]
    pub args: Vec<String>,
    /// Environment variable names copied from Flux into the child process.
    #[serde(default)]
    pub pass_env: Vec<String>,
}

#[derive(Debug, Deserialize)]
pub struct WorkflowMeta {
    pub name: String,
    pub description: Option<String>,
    /// Subdirectory under outputs/ where this workflow's results are written.
    /// Plain name ("legal-ai") → outputs/legal-ai/
    /// Absolute or ~/ path used as-is.
    /// Defaults to outputs/ if not set.
    pub output_dir: Option<String>,
}

#[derive(Debug, Deserialize)]
pub struct Budget {
    pub max_usd: f64,
}

#[derive(Debug, Deserialize)]
pub struct InputDef {
    pub description: String,
}

/// Which executor a step dispatches to.
///
/// Deserialized from a `kind = "..."` key that defaults to `agent`, so every
/// workflow written before heterogeneous steps existed still parses unchanged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StepKindTag {
    #[default]
    Agent,
    Http,
    Mcp,
    Transform,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Deserialize)]
#[serde(rename_all = "UPPERCASE")]
pub enum HttpMethod {
    #[default]
    Get,
    Post,
}

impl HttpMethod {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Get => "GET",
            Self::Post => "POST",
        }
    }
}

/// The TOML wire format for one step.
///
/// This struct is deliberately permissive: it is the shape of untrusted input,
/// where an HTTP step and an agent step are indistinguishable until checked.
/// [`Step::resolve`] turns it into a [`StepKind`], which is the shape the engine
/// actually runs and in which those illegal combinations cannot be represented.
#[derive(Debug, Deserialize)]
pub struct Step {
    pub id: String,
    #[serde(default)]
    pub kind: StepKindTag,
    #[serde(default = "default_model")]
    pub model: String,
    #[serde(default)]
    pub prompt: String,
    #[serde(default)]
    pub depends_on: Vec<String>,
    #[serde(default)]
    pub tools: Vec<String>,
    pub max_tool_calls: Option<u32>,
    /// Output token budget for this step's API calls (default 8096).
    /// Raise on synthesis steps that write long reports — a step that hits
    /// the cap is truncated (flagged with a [NOTE: ...] marker in its output).
    pub max_tokens: Option<u32>,
    pub parallel_over: Option<String>,
    /// Static list of items to fan out over. Alternative to parallel_over when
    /// the items are known at workflow-authoring time and don't need a model step
    /// to generate them. Cannot be combined with parallel_over.
    #[serde(default)]
    pub parallel_items: Vec<String>,
    pub max_concurrent: Option<usize>,
    pub max_items: Option<usize>,
    /// Permits a fan-out over zero items, producing an empty output instead of
    /// an error. For steps whose upstream is legitimately empty in the steady
    /// state — e.g. a fallback lookup once every item is already cached.
    #[serde(default)]
    pub allow_empty: bool,
    pub provider: Option<String>,
    pub cost_per_call_usd: Option<f64>,
    /// Files to read before this step runs. Keys become template variables.
    /// E.g. `reads_from = {notes = "~/research/notes.md"}` lets you write
    /// `{{ notes }}` in the prompt.
    #[serde(default)]
    pub reads_from: HashMap<String, String>,
    /// File to append this step's output to after it succeeds.
    /// The file is created if it doesn't exist. Parent directories are created too.
    pub appends_to: Option<String>,
    /// Seconds before an agent or MCP tool call is reported as failed.
    /// Defaults to 300 for agent providers and 60 for MCP.
    pub timeout_secs: Option<u64>,

    // --- kind = "http" ---
    /// Request URL. Rendered as a template, so `{{ item.url }}` works under fan-out.
    pub url: Option<String>,
    #[serde(default)]
    pub method: HttpMethod,
    /// Request body for POST, rendered as a template.
    pub body: Option<String>,
    #[serde(default)]
    pub headers: HashMap<String, String>,
    /// Selects a registered [`crate::registry::Probe`] instead of a plain request.
    pub probe: Option<String>,

    // --- kind = "mcp" ---
    /// Name of a server declared under `[mcp.servers]`.
    pub server: Option<String>,
    /// Exact MCP tool name to call.
    pub tool: Option<String>,
    /// Tool arguments. String leaves are rendered as templates before the call.
    #[serde(default)]
    pub arguments: Map<String, Value>,

    // --- kind = "transform" ---
    /// Name of the registered transform to apply. See [`crate::transform`].
    pub transform: Option<String>,
    /// Keyword lists consumed by the transform, e.g. `key = ["name"]`.
    #[serde(default)]
    pub args: HashMap<String, Vec<String>>,

    /// Name of the schema this step's rows are validated against.
    /// Required for `transform`; optional for `http`, and for `agent` steps that
    /// are prompted to emit JSON. Resolved to a [`Schema`] by [`Step::resolve`].
    pub output_type: Option<String>,
}

/// A step whose configuration has been checked against its kind.
///
/// Borrows from the [`Step`] it was resolved from rather than cloning: the
/// workflow outlives every execution, so `'a` here costs nothing and avoids
/// copying prompts and URL templates once per fan-out item.
#[derive(Debug)]
pub enum StepKind<'a> {
    Agent(AgentStep<'a>),
    Http(HttpStep<'a>),
    Mcp(McpStep<'a>),
    Transform(TransformStep<'a>),
}

#[derive(Debug)]
pub struct AgentStep<'a> {
    pub prompt: &'a str,
    pub model: &'a str,
    pub tools: &'a [String],
    pub output_type: Option<Schema>,
}

#[derive(Debug)]
pub struct HttpStep<'a> {
    pub mode: HttpMode<'a>,
    pub output_type: Schema,
}

#[derive(Debug)]
pub struct McpStep<'a> {
    pub server: &'a str,
    pub tool: &'a str,
    pub arguments: &'a Map<String, Value>,
}

/// A plain request, or a registered multi-request probe.
#[derive(Debug)]
pub enum HttpMode<'a> {
    Request {
        method: HttpMethod,
        url: &'a str,
        body: Option<&'a str>,
        headers: &'a HashMap<String, String>,
    },
    /// A [`crate::registry::Probe`], by its registered name.
    Probe(&'a str),
}

#[derive(Debug)]
pub struct TransformStep<'a> {
    pub name: &'a str,
    pub args: &'a HashMap<String, Vec<String>>,
    pub output_type: Schema,
}

impl Step {
    /// True when this step runs through an external agent CLI rather than the Anthropic API.
    /// Single source of truth for provider dispatch, shared by the engine and the estimator.
    pub fn is_agent(&self) -> bool {
        self.kind == StepKindTag::Agent
            && matches!(
                self.provider.as_deref(),
                Some("hermes") | Some("claude-code") | Some("cursor")
            )
    }

    /// True when the step spends no money at all, so budget checks can skip it.
    pub fn is_free(&self) -> bool {
        matches!(
            self.kind,
            StepKindTag::Http | StepKindTag::Mcp | StepKindTag::Transform
        )
    }

    pub fn fans_out(&self) -> bool {
        self.parallel_over.is_some() || !self.parallel_items.is_empty()
    }

    /// Checks kind-specific configuration and produces the runtime view of this step.
    ///
    /// The validator calls this before a run so a misconfigured workflow fails at
    /// `flux validate` time rather than midway through a paid execution. Names
    /// (`output_type`, `transform`, `probe`) are looked up in `registry` here, so
    /// the returned [`StepKind`] holds resolved schemas rather than strings.
    pub fn resolve(&self, registry: &Registry) -> crate::Result<StepKind<'_>> {
        let missing = |field: &str, kind: &str| {
            crate::FluxError::Config(format!(
                "step '{}': kind = \"{kind}\" requires `{field}`",
                self.id
            ))
        };

        // `transpose` flips Option<Result<_>> into Result<Option<_>> so `?` applies.
        let output_type = self
            .output_type
            .as_deref()
            .map(|name| registry.schema(name))
            .transpose()?;

        match self.kind {
            StepKindTag::Agent => {
                if self.prompt.trim().is_empty() {
                    return Err(missing("prompt", "agent"));
                }
                Ok(StepKind::Agent(AgentStep {
                    prompt: &self.prompt,
                    model: &self.model,
                    tools: &self.tools,
                    output_type,
                }))
            }
            StepKindTag::Http => match self.probe.as_deref() {
                Some(name) => Ok(StepKind::Http(HttpStep {
                    mode: HttpMode::Probe(name),
                    output_type: registry.probe(name)?.output_schema(),
                })),
                None => {
                    let url = self.url.as_deref().ok_or_else(|| missing("url", "http"))?;
                    if self.method == HttpMethod::Post && self.body.is_none() {
                        return Err(crate::FluxError::Config(format!(
                            "step '{}': method = \"POST\" requires `body`",
                            self.id
                        )));
                    }
                    Ok(StepKind::Http(HttpStep {
                        mode: HttpMode::Request {
                            method: self.method,
                            url,
                            body: self.body.as_deref(),
                            headers: &self.headers,
                        },
                        // Raw responses stay in the upstream API's shape; a transform normalizes them.
                        output_type: output_type.unwrap_or_else(Schema::http_response),
                    }))
                }
            },
            StepKindTag::Mcp => Ok(StepKind::Mcp(McpStep {
                server: self
                    .server
                    .as_deref()
                    .ok_or_else(|| missing("server", "mcp"))?,
                tool: self.tool.as_deref().ok_or_else(|| missing("tool", "mcp"))?,
                arguments: &self.arguments,
            })),
            StepKindTag::Transform => {
                let name = self
                    .transform
                    .as_deref()
                    .ok_or_else(|| missing("transform", "transform"))?;
                // Looked up only to reject an unknown name before the run starts.
                registry.transform(name)?;
                Ok(StepKind::Transform(TransformStep {
                    name,
                    args: &self.args,
                    output_type: output_type.ok_or_else(|| missing("output_type", "transform"))?,
                }))
            }
        }
    }
}

#[derive(Debug, Deserialize)]
pub struct Output {
    pub step: String,
}

fn default_model() -> String {
    "claude-sonnet-4-6".to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn step(src: &str) -> Step {
        toml::from_str(src).expect("test step TOML should parse")
    }

    #[test]
    fn kind_defaults_to_agent_so_old_workflows_still_parse() {
        let s = step(
            r#"id = "a"
prompt = "hi""#,
        );
        assert_eq!(s.kind, StepKindTag::Agent);
        assert!(matches!(
            s.resolve(&Registry::builtin()).unwrap(),
            StepKind::Agent(_)
        ));
    }

    #[test]
    fn http_step_requires_a_url() {
        let s = step(
            r#"id = "f"
kind = "http""#,
        );
        assert!(s.resolve(&Registry::builtin()).is_err());

        let s = step(
            r#"id = "f"
kind = "http"
url = "https://example.com""#,
        );
        assert!(matches!(
            s.resolve(&Registry::builtin()).unwrap(),
            StepKind::Http(_)
        ));
    }

    #[test]
    fn unknown_output_type_is_rejected() {
        let s = step("id = \"a\"\nprompt = \"p\"\noutput_type = \"nope\"");
        assert!(s.resolve(&Registry::builtin()).is_err());
    }

    #[test]
    fn unknown_transform_is_rejected() {
        let s = step(
            r#"id = "t"
kind = "transform"
transform = "nope"
output_type = "record""#,
        );
        assert!(s.resolve(&Registry::builtin()).is_err());
    }

    #[test]
    fn unknown_probe_is_rejected() {
        let s = step(
            r#"id = "f"
kind = "http"
probe = "unregistered""#,
        );
        assert!(s.resolve(&Registry::builtin()).is_err());
    }

    #[test]
    fn post_without_body_is_rejected() {
        let s = step(
            r#"id = "f"
kind = "http"
method = "POST"
url = "https://example.com""#,
        );
        assert!(s.resolve(&Registry::builtin()).is_err());
    }

    #[test]
    fn transform_requires_a_name() {
        let s = step(
            r#"id = "t"
kind = "transform"
output_type = "record""#,
        );
        assert!(s.resolve(&Registry::builtin()).is_err());

        let s = step(
            r#"id = "t"
kind = "transform"
transform = "dedupe"
output_type = "record""#,
        );
        assert!(matches!(
            s.resolve(&Registry::builtin()).unwrap(),
            StepKind::Transform(_)
        ));
    }

    #[test]
    fn mcp_step_requires_a_server_and_tool() {
        assert!(
            step("id = \"m\"\nkind = \"mcp\"")
                .resolve(&Registry::builtin())
                .is_err()
        );
        assert!(
            step("id = \"m\"\nkind = \"mcp\"\nserver = \"local\"")
                .resolve(&Registry::builtin())
                .is_err()
        );

        let s = step(
            r#"id = "m"
kind = "mcp"
server = "local"
tool = "echo"
arguments = { count = 2, enabled = true, text = "{{ input }}" }"#,
        );
        let StepKind::Mcp(mcp) = s.resolve(&Registry::builtin()).unwrap() else {
            panic!("expected an MCP step");
        };
        assert_eq!(mcp.arguments["count"], 2);
        assert_eq!(mcp.arguments["enabled"], true);
        assert_eq!(mcp.arguments["text"], "{{ input }}");
    }

    #[test]
    fn deterministic_steps_are_free() {
        assert!(step("id = \"a\"\nkind = \"http\"").is_free());
        assert!(step("id = \"a\"\nkind = \"mcp\"").is_free());
        assert!(step("id = \"a\"\nkind = \"transform\"").is_free());
        assert!(!step("id = \"a\"\nprompt = \"x\"").is_free());
    }

    #[test]
    fn http_step_is_never_an_agent_step() {
        // provider is meaningless for http; it must not route to the agent CLI.
        let s = step(
            r#"id = "f"
kind = "http"
provider = "claude-code"
url = "https://example.com"
output_type = "record""#,
        );
        assert!(!s.is_agent());
    }

    #[test]
    fn cursor_provider_is_an_agent_step() {
        let s = step("id = \"a\"\nprovider = \"cursor\"\nprompt = \"p\"");
        assert!(s.is_agent());
    }

    #[test]
    fn unknown_kind_is_a_parse_error() {
        let err = toml::from_str::<Step>("id = \"a\"\nkind = \"wat\"");
        assert!(err.is_err());
    }
}
