use std::collections::HashMap;

use minijinja::Environment;
use serde_json::{Map, Value, json};

pub fn render_prompt(
    template: &str,
    inputs: &HashMap<String, String>,
    step_outputs: &HashMap<String, String>,
) -> crate::Result<String> {
    let mut ctx: Map<String, Value> = Map::new();

    // Inputs are top-level: {{ profile }} resolves inputs["profile"]
    for (k, v) in inputs {
        ctx.insert(k.clone(), Value::String(v.clone()));
    }

    // Step outputs are nested: {{ scan.output }} resolves step_outputs["scan"]
    for (id, output) in step_outputs {
        ctx.insert(id.clone(), json!({"output": output}));
    }

    Environment::new()
        .render_str(template, Value::Object(ctx))
        .map_err(|e| crate::FluxError::Template(e.to_string()))
}
