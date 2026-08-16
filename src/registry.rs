//! The extension seam: schemas, transforms, and probes looked up by name.
//!
//! A workflow file names these as strings (`output_type = "record"`,
//! `transform = "dedupe"`, `probe = "..."`). The registry is where a name
//! becomes behavior. flux ships a small generic set via [`Registry::builtin`];
//! a downstream crate builds on it by registering its own before constructing
//! the engine, without flux knowing anything about that crate's domain.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use serde_json::Value;

use crate::artifact::{Artifact, Schema};
use crate::template::FanItem;

/// Keyword lists a workflow passes to a transform via `args`.
pub type TransformArgs = HashMap<String, Vec<String>>;

/// A pure function over the outputs of a step's dependencies.
///
/// Implemented for any matching closure or `fn`, so registering a transform is
/// usually just passing a function name.
pub trait Transform: Send + Sync {
    fn apply(&self, inputs: &[&Artifact], args: &TransformArgs) -> Vec<Value>;
}

impl<F> Transform for F
where
    F: Fn(&[&Artifact], &TransformArgs) -> Vec<Value> + Send + Sync,
{
    fn apply(&self, inputs: &[&Artifact], args: &TransformArgs) -> Vec<Value> {
        self(inputs, args)
    }
}

/// A multi-request HTTP routine selected with `probe = "<name>"` on an http step.
///
/// A plain http step is one request per item. A probe is for the cases that are
/// not: trying several candidate URLs per item, or consulting a cache first.
#[async_trait]
pub trait Probe: Send + Sync {
    /// The schema of the rows this probe emits.
    fn output_schema(&self) -> Schema;

    async fn run(
        &self,
        items: &[FanItem],
        client: &reqwest::Client,
        max_concurrent: usize,
    ) -> crate::Result<Artifact>;
}

pub struct Registry {
    schemas: HashMap<String, Schema>,
    transforms: HashMap<String, Arc<dyn Transform>>,
    probes: HashMap<String, Arc<dyn Probe>>,
}

impl Registry {
    /// The schemas and transforms flux itself provides. No probes are built in.
    pub fn builtin() -> Self {
        let mut registry = Self {
            schemas: HashMap::new(),
            transforms: HashMap::new(),
            probes: HashMap::new(),
        };
        registry.register_schema(Schema::record());
        registry.register_schema(Schema::http_response());
        registry.register_schema(Schema::mcp_response());
        crate::transform::register_builtin(&mut registry);
        registry
    }

    /// Adds a schema under its own name, replacing any earlier one of that name.
    pub fn register_schema(&mut self, schema: Schema) {
        self.schemas.insert(schema.name().to_string(), schema);
    }

    pub fn register_transform(
        &mut self,
        name: impl Into<String>,
        transform: impl Transform + 'static,
    ) {
        self.transforms.insert(name.into(), Arc::new(transform));
    }

    pub fn register_probe(&mut self, name: impl Into<String>, probe: Arc<dyn Probe>) {
        self.probes.insert(name.into(), probe);
    }

    pub fn schema(&self, name: &str) -> crate::Result<Schema> {
        self.schemas.get(name).cloned().ok_or_else(|| {
            crate::FluxError::Config(format!(
                "unknown output_type '{name}' (available: {})",
                sorted_names(&self.schemas)
            ))
        })
    }

    pub fn transform(&self, name: &str) -> crate::Result<Arc<dyn Transform>> {
        self.transforms.get(name).cloned().ok_or_else(|| {
            crate::FluxError::Config(format!(
                "unknown transform '{name}' (available: {})",
                sorted_names(&self.transforms)
            ))
        })
    }

    pub fn probe(&self, name: &str) -> crate::Result<Arc<dyn Probe>> {
        self.probes.get(name).cloned().ok_or_else(|| {
            let available = match sorted_names(&self.probes) {
                names if names.is_empty() => "none registered".to_string(),
                names => names,
            };
            crate::FluxError::Config(format!("unknown probe '{name}' (available: {available})"))
        })
    }
}

impl Default for Registry {
    fn default() -> Self {
        Self::builtin()
    }
}

// HashMap order is random; sorted so an error message is stable run to run.
fn sorted_names<V>(map: &HashMap<String, V>) -> String {
    let mut names: Vec<&str> = map.keys().map(String::as_str).collect();
    names.sort_unstable();
    names.join(", ")
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn builtin_knows_the_generic_schemas_and_transforms() {
        let registry = Registry::builtin();
        assert!(registry.schema("record").is_ok());
        assert!(registry.schema("http_response").is_ok());
        assert!(registry.transform("dedupe").is_ok());
        assert!(registry.probe("anything").is_err());
    }

    #[test]
    fn unknown_names_list_what_is_available() {
        // `.err()` rather than `.unwrap_err()`: the Ok side is a trait object with no Debug.
        let err = Registry::builtin().transform("nope").err().unwrap();
        assert!(err.to_string().contains("dedupe"));
    }

    #[test]
    fn a_closure_registers_as_a_transform() {
        let mut registry = Registry::builtin();
        registry.register_transform("count", |inputs: &[&Artifact], _: &TransformArgs| {
            vec![json!({"n": inputs.iter().map(|a| a.rows().len()).sum::<usize>()})]
        });
        let input = Artifact::records(Schema::record(), vec![json!({"a": 1}), json!({"a": 2})]);
        let out = registry
            .transform("count")
            .unwrap()
            .apply(&[&input], &HashMap::new());
        assert_eq!(out, vec![json!({"n": 2})]);
    }
}
