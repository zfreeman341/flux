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

    let mut env = Environment::new();
    // Strict mode: referencing an undefined variable is an error, not silent empty string.
    // Without this, {{ item }} in a prompt that wasn't given an item renders as "",
    // producing a confused LLM response rather than a clear template error.
    env.set_undefined_behavior(minijinja::UndefinedBehavior::Strict);
    env.render_str(template, Value::Object(ctx))
        .map_err(|e| crate::FluxError::Template(e.to_string()))
}
