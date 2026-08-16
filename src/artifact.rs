//! Typed values passed between steps.
//!
//! Steps used to hand each other bare prose. That works when every step is an
//! LLM call, but a deterministic HTTP step produces records, not paragraphs, and
//! turning those into prose just to parse them back out is lossy and expensive.
//!
//! [`Artifact`] is the union of the two shapes the DAG carries. [`Artifact::Records`]
//! carries the [`Schema`] its rows conform to, and rows are revalidated at every step
//! boundary: a row that no longer parses is dropped and logged rather than
//! aborting the run, matching the graceful degradation already used for budget
//! exhaustion.

use std::fmt;
use std::sync::Arc;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::warn;

/// A citation captured alongside prose output.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Source {
    pub title: String,
    pub url: String,
}

impl Source {
    /// Parses the `"Title — URL"` form the Anthropic client emits.
    pub fn parse(raw: &str) -> Self {
        match raw.rsplit_once(" — ") {
            Some((title, url)) => Self {
                title: title.trim().to_string(),
                url: url.trim().to_string(),
            },
            None => Self {
                title: String::new(),
                url: raw.trim().to_string(),
            },
        }
    }
}

/// A named row shape that [`Artifact::Records`] rows are checked against.
///
/// A workflow names a schema as a string (`output_type = "record"`); the
/// [`crate::registry::Registry`] turns that name into one of these before a run
/// starts, so an unknown name is a validation error rather than a silent skip.
/// Cloning is cheap: the definition sits behind an [`Arc`].
#[derive(Clone)]
pub struct Schema(Arc<SchemaDef>);

// A trait-object type: any closure that judges one row. Aliased because clippy
// (rightly) finds the spelled-out form hard to read inline.
type RowCheck = dyn Fn(&Value) -> Result<(), String> + Send + Sync;

struct SchemaDef {
    name: String,
    check: Box<RowCheck>,
}

impl Schema {
    /// A schema with a hand-written row check.
    pub fn new(
        name: impl Into<String>,
        check: impl Fn(&Value) -> Result<(), String> + Send + Sync + 'static,
    ) -> Self {
        Self(Arc::new(SchemaDef {
            name: name.into(),
            check: Box::new(check),
        }))
    }

    /// A schema whose rows must deserialize into `T`.
    pub fn of<T: DeserializeOwned>(name: impl Into<String>) -> Self {
        Self::new(name, |row| {
            serde_json::from_value::<T>(row.clone())
                .map(|_| ())
                .map_err(|e| e.to_string())
        })
    }

    /// Any JSON object: the untyped schema for rows with no fixed shape.
    pub fn record() -> Self {
        Self::new("record", |row| {
            if row.is_object() {
                Ok(())
            } else {
                Err("row is not a JSON object".to_string())
            }
        })
    }

    pub fn http_response() -> Self {
        Self::of::<HttpResponse>("http_response")
    }

    pub fn mcp_response() -> Self {
        Self::of::<McpResponse>("mcp_response")
    }

    pub fn name(&self) -> &str {
        &self.0.name
    }

    /// Drops rows that do not match the schema, keeping the rest.
    ///
    /// The count of dropped rows is logged, not returned as an error: one
    /// malformed row in a 400-row response must not sink the run.
    pub fn validate(&self, rows: Vec<Value>) -> Vec<Value> {
        let total = rows.len();
        let kept: Vec<Value> = rows
            .into_iter()
            .filter(|row| match (self.0.check)(row) {
                Ok(()) => true,
                Err(e) => {
                    warn!(schema = self.name(), error = %e, "dropping malformed row");
                    false
                }
            })
            .collect();
        if kept.len() < total {
            warn!(
                schema = self.name(),
                dropped = total - kept.len(),
                kept = kept.len(),
                "records failed schema validation"
            );
        }
        kept
    }
}

// Manual impl: a boxed closure has no Debug, and the name is what a log needs.
impl fmt::Debug for Schema {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Schema({})", self.name())
    }
}

/// A value flowing along a DAG edge.
#[derive(Debug, Clone)]
pub enum Artifact {
    Text { body: String, sources: Vec<Source> },
    Records { schema: Schema, rows: Vec<Value> },
}

impl Artifact {
    pub fn text(body: impl Into<String>) -> Self {
        Self::Text {
            body: body.into(),
            sources: Vec::new(),
        }
    }

    /// Builds a validated record set, dropping rows that fail the schema.
    pub fn records(schema: Schema, rows: Vec<Value>) -> Self {
        let rows = schema.validate(rows);
        Self::Records { schema, rows }
    }

    /// The prose form, used for prompt substitution and run artifacts.
    ///
    /// Records render as pretty JSON so a model can read them directly, which is
    /// what makes a `Records` artifact usable as input to an ordinary agent step.
    pub fn as_prompt_text(&self) -> String {
        match self {
            Self::Text { body, sources } => {
                if sources.is_empty() {
                    return body.clone();
                }
                let mut out = body.clone();
                out.push_str("\n\nSources:\n");
                for s in sources {
                    out.push_str(&format!("- {} — {}\n", s.title, s.url));
                }
                out
            }
            Self::Records { rows, .. } => {
                serde_json::to_string_pretty(rows).unwrap_or_else(|_| "[]".to_string())
            }
        }
    }

    /// Rows if this is a record set, otherwise empty.
    pub fn rows(&self) -> &[Value] {
        match self {
            Self::Records { rows, .. } => rows,
            Self::Text { .. } => &[],
        }
    }

    pub fn sources(&self) -> &[Source] {
        match self {
            Self::Text { sources, .. } => sources,
            Self::Records { .. } => &[],
        }
    }

    pub fn is_empty(&self) -> bool {
        match self {
            Self::Text { body, .. } => body.trim().is_empty(),
            Self::Records { rows, .. } => rows.is_empty(),
        }
    }
}

/// The normalized result of one MCP tool invocation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResponse {
    pub server: String,
    pub tool: String,
    pub item: Value,
    pub is_error: bool,
    pub result: Value,
}

/// One HTTP response, kept in the upstream API's own shape.
///
/// The `item` the request was issued for travels with the response so that a
/// downstream transform still knows which fan-out item a payload belongs to.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpResponse {
    pub url: String,
    pub status: u16,
    #[serde(default)]
    pub item: Value,
    #[serde(default)]
    pub payload: Value,
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[derive(Deserialize)]
    struct Point {
        #[allow(dead_code)]
        x: i64,
    }

    #[test]
    fn malformed_rows_are_dropped_not_fatal() {
        let rows = vec![json!({"x": 1}), json!({"y": 2}), json!({"x": 3})];
        let artifact = Artifact::records(Schema::of::<Point>("point"), rows);
        assert_eq!(artifact.rows().len(), 2);
    }

    #[test]
    fn record_schema_accepts_any_object_and_nothing_else() {
        let rows = vec![json!({"anything": true}), json!("a string"), json!(42)];
        assert_eq!(Artifact::records(Schema::record(), rows).rows().len(), 1);
    }

    #[test]
    fn mcp_response_requires_the_normalized_envelope() {
        let valid = json!({
            "server": "local",
            "tool": "echo",
            "item": null,
            "is_error": false,
            "result": {"value": 42}
        });
        let invalid = json!({"tool": "echo", "result": null});
        let artifact = Artifact::records(Schema::mcp_response(), vec![valid, invalid]);
        assert_eq!(artifact.rows().len(), 1);
    }

    #[test]
    fn records_render_as_json_for_prompts() {
        let artifact = Artifact::records(Schema::record(), vec![json!({"name": "Acme"})]);
        assert!(artifact.as_prompt_text().contains("\"name\": \"Acme\""));
    }

    #[test]
    fn text_artifact_appends_sources() {
        let artifact = Artifact::Text {
            body: "Body.".into(),
            sources: vec![Source::parse("Example Docs — https://example.com")],
        };
        let text = artifact.as_prompt_text();
        assert!(text.contains("Sources:"));
        assert!(text.contains("https://example.com"));
    }
}
