use std::collections::HashMap;

use minijinja::Environment;
use serde_json::{Map, Value, json};

use crate::artifact::Artifact;

/// One unit of work in a fan-out.
///
/// `label` is what appears in logs and section headers; `value` is what the
/// template sees as `{{ item }}`. They differ when fanning out over records:
/// the label is the row's name while the value is the whole row, so a prompt
/// can reach `{{ item.url }}`.
#[derive(Debug, Clone)]
pub struct FanItem {
    pub label: String,
    pub value: Value,
}

impl FanItem {
    pub fn from_text(s: impl Into<String>) -> Self {
        let s = s.into();
        Self {
            label: s.clone(),
            value: Value::String(s),
        }
    }

    /// Labels a record by its most name-like field, else its first string
    /// field, else compact JSON. Capped, because a label is for logs and section
    /// headers and a row can carry kilobytes of text.
    pub fn from_record(row: Value) -> Self {
        let named = ["label", "name", "title", "id"]
            .iter()
            .find_map(|k| row.get(*k).and_then(Value::as_str));
        let first_string = || {
            row.as_object()
                .and_then(|o| o.values().find_map(Value::as_str))
        };
        let label: String = named
            .or_else(first_string)
            .map(str::to_string)
            .unwrap_or_else(|| row.to_string())
            .chars()
            .take(80)
            .collect();
        Self { label, value: row }
    }
}

pub fn render_prompt(
    template: &str,
    inputs: &HashMap<String, String>,
    step_outputs: &HashMap<String, Artifact>,
) -> crate::Result<String> {
    render(template, inputs, step_outputs, None)
}

/// Renders with a fan-out item bound to `{{ item }}`.
pub fn render(
    template: &str,
    inputs: &HashMap<String, String>,
    step_outputs: &HashMap<String, Artifact>,
    item: Option<&FanItem>,
) -> crate::Result<String> {
    let mut ctx: Map<String, Value> = Map::new();

    // Inputs are top-level: {{ profile }} resolves inputs["profile"]
    for (k, v) in inputs {
        ctx.insert(k.clone(), Value::String(v.clone()));
    }

    // Step outputs are nested: {{ scan.output }} is the prose form, and
    // {{ scan.rows }} exposes records so a template can loop over them.
    for (id, artifact) in step_outputs {
        ctx.insert(
            id.clone(),
            json!({
                "output": artifact.as_prompt_text(),
                "rows": artifact.rows(),
                "count": artifact.rows().len(),
            }),
        );
    }

    if let Some(item) = item {
        ctx.insert("item".to_string(), item.value.clone());
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
    use crate::artifact::Schema;

    fn map(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn outputs(pairs: &[(&str, &str)]) -> HashMap<String, Artifact> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), Artifact::text(*v)))
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
            &outputs(&[("find", "three papers")]),
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

    #[test]
    fn record_items_expose_their_fields() {
        let item = FanItem::from_record(serde_json::json!({
            "name": "Acme", "url": "https://example.com/acme"
        }));
        let rendered = render(
            "GET {{ item.url }}",
            &HashMap::new(),
            &HashMap::new(),
            Some(&item),
        )
        .unwrap();
        assert_eq!(rendered, "GET https://example.com/acme");
        assert_eq!(item.label, "Acme");
    }

    #[test]
    fn record_labels_fall_back_to_a_string_field_and_stay_short() {
        let item = FanItem::from_record(serde_json::json!({"n": 3, "city": "Lisbon"}));
        assert_eq!(item.label, "Lisbon");
        let long = FanItem::from_record(serde_json::json!({"n": "x".repeat(500)}));
        assert_eq!(long.label.chars().count(), 80);
    }

    #[test]
    fn text_items_render_whole() {
        let item = FanItem::from_text("battery storage");
        let rendered = render(
            "Topic: {{ item }}",
            &HashMap::new(),
            &HashMap::new(),
            Some(&item),
        )
        .unwrap();
        assert_eq!(rendered, "Topic: battery storage");
    }

    #[test]
    fn record_step_outputs_expose_rows_and_count() {
        let mut outs = HashMap::new();
        outs.insert(
            "parse".to_string(),
            Artifact::records(
                Schema::record(),
                vec![serde_json::json!({"name": "Acme", "kind": "a"})],
            ),
        );
        let rendered = render_prompt(
            "{{ parse.count }} rows; first is {{ parse.rows[0].name }}",
            &HashMap::new(),
            &outs,
        )
        .unwrap();
        assert_eq!(rendered, "1 rows; first is Acme");
    }
}
