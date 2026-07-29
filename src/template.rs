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
    // Lenient so a first-run empty state file cannot abort a workflow; typos render empty.
    env.set_undefined_behavior(minijinja::UndefinedBehavior::Lenient);
    env.render_str(template, Value::Object(ctx))
        .map_err(|e| crate::FluxError::Template(e.to_string()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn map(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn inputs_resolve_as_top_level_variables() {
        let rendered = render_prompt(
            "Research {{ topic }}",
            &map(&[("topic", "rust async")]),
            &HashMap::new(),
        )
        .unwrap();
        assert_eq!(rendered, "Research rust async");
    }

    #[test]
    fn step_outputs_resolve_under_dot_output() {
        let rendered = render_prompt(
            "Prior findings: {{ find.output }}",
            &HashMap::new(),
            &map(&[("find", "three papers")]),
        )
        .unwrap();
        assert_eq!(rendered, "Prior findings: three papers");
    }

    #[test]
    fn undefined_variables_render_empty_in_lenient_mode() {
        let rendered = render_prompt(
            "before {{ missing }} after",
            &HashMap::new(),
            &HashMap::new(),
        )
        .unwrap();
        assert_eq!(rendered, "before  after");
    }
}
