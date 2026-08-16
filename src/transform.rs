//! Pure functions over artifacts.
//!
//! A transform never makes a network call, never draws budget, and given the
//! same input always produces the same output. That is what makes the expensive
//! steps around it worth caching: everything between a fetch and a model call is
//! reproducible, so a failed run can be re-driven from stored rows.
//!
//! Transforms are selected by name from TOML (`transform = "dedupe"`) and looked
//! up in the [`Registry`]. Policy lives in the workflow's `args`, not in this
//! file, so tuning a step does not mean recompiling. The ones here are generic;
//! a downstream crate registers its own domain-specific transforms alongside.

use std::collections::{HashMap, HashSet};

use serde_json::{Map, Value, json};
use tracing::warn;

use crate::artifact::Artifact;
use crate::registry::{Registry, TransformArgs};

/// Registers the generic transforms on a registry.
pub fn register_builtin(registry: &mut Registry) {
    registry.register_transform("dedupe", |i: &[&Artifact], a: &TransformArgs| {
        dedupe(all_rows(i), a)
    });
    registry.register_transform("difference", difference);
    registry.register_transform("top_n", |i: &[&Artifact], a: &TransformArgs| {
        top_n(all_rows(i), a)
    });
    registry.register_transform("group_by", |i: &[&Artifact], a: &TransformArgs| {
        group_by(all_rows(i), a)
    });
}

/// Every row from every input, in dependency order.
pub fn all_rows(inputs: &[&Artifact]) -> Vec<Value> {
    inputs.iter().flat_map(|a| a.rows().to_vec()).collect()
}

/// Drops repeats, keyed by `key` fields if given and by the whole row otherwise.
fn dedupe(rows: Vec<Value>, args: &TransformArgs) -> Vec<Value> {
    let keys = arg(args, "key");
    let mut seen = HashSet::new();
    rows.into_iter()
        .filter(|row| {
            let fingerprint = if keys.is_empty() {
                row.to_string()
            } else {
                keys.iter()
                    .map(|k| {
                        row.get(k)
                            .map(|v| v.to_string().to_lowercase())
                            .unwrap_or_default()
                    })
                    .collect::<Vec<_>>()
                    .join("|")
            };
            seen.insert(fingerprint)
        })
        .collect()
}

/// Rows of the first dependency that appear in none of the later ones, matched
/// on the `key` fields: `difference(all, handled)` is what is still to do.
fn difference(inputs: &[&Artifact], args: &TransformArgs) -> Vec<Value> {
    let keys = arg(args, "key");
    if inputs.len() < 2 {
        warn!("difference transform has fewer than two dependencies; passing rows through");
    }
    let fingerprint = |row: &Value| -> String {
        keys.iter()
            .map(|k| {
                row.get(k)
                    .and_then(Value::as_str)
                    .map(normalize_key)
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>()
            .join("|")
    };
    let seen: HashSet<String> = inputs
        .iter()
        .skip(1)
        .flat_map(|a| a.rows())
        .map(fingerprint)
        .collect();
    inputs
        .first()
        .map(|a| a.rows().to_vec())
        .unwrap_or_default()
        .into_iter()
        .filter(|row| !seen.contains(&fingerprint(row)))
        .collect()
}

/// The `n` rows with the highest numeric `by` field (default `score`).
/// Rows missing the field sort last. The sort is stable, so ties keep their
/// upstream order and reruns stay deterministic.
fn top_n(rows: Vec<Value>, args: &TransformArgs) -> Vec<Value> {
    let by = args
        .get("by")
        .and_then(|v| v.first())
        .map(String::as_str)
        .unwrap_or("score");
    let n = args
        .get("n")
        .and_then(|v| v.first())
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(usize::MAX);
    let mut rows = rows;
    rows.sort_by(|a, b| {
        let field = |r: &Value| r.get(by).and_then(Value::as_f64).unwrap_or(f64::MIN);
        field(b).total_cmp(&field(a))
    });
    rows.truncate(n);
    rows
}

/// One row per distinct `key` value, carrying the group's rows as a JSON
/// string in `items_json`, plus a `count`. Groups keep first-seen order.
///
/// This exists so a fan-out step can process one *group* per call: a model
/// fanning out per group emits a small JSON array each time, where a single
/// call over every row at once gets truncated at the output cap and parses
/// as nothing.
///
/// `items_json` is a pre-serialized string rather than nested rows because a
/// template renders `{{ item.items_json }}` verbatim; nested structures would
/// be at the mercy of the template engine's display format.
fn group_by(rows: Vec<Value>, args: &TransformArgs) -> Vec<Value> {
    let keys = arg(args, "key");
    if keys.is_empty() {
        warn!("group_by has no `key` arg; emitting one group with every row");
    }
    let fingerprint = |row: &Value| -> String {
        keys.iter()
            .map(|k| {
                row.get(k)
                    .and_then(Value::as_str)
                    .map(normalize_key)
                    .unwrap_or_default()
            })
            .collect::<Vec<_>>()
            .join("|")
    };

    let mut order: Vec<String> = Vec::new();
    let mut groups: HashMap<String, Vec<Value>> = HashMap::new();
    for row in rows {
        let f = fingerprint(&row);
        if !groups.contains_key(&f) {
            order.push(f.clone());
        }
        groups.entry(f).or_default().push(row);
    }

    order
        .into_iter()
        .filter_map(|f| {
            let group = groups.remove(&f)?;
            let mut out = Map::new();
            // Key fields copied from the first row, display casing intact.
            for k in &keys {
                if let Some(v) = group[0].get(k) {
                    out.insert(k.clone(), v.clone());
                }
            }
            out.insert("count".into(), json!(group.len()));
            out.insert(
                "items_json".into(),
                json!(serde_json::to_string_pretty(&group).unwrap_or_else(|_| "[]".into())),
            );
            Some(Value::Object(out))
        })
        .collect()
}

/// Case and punctuation are ignored, so 'Nakatomi Ai' matches 'nakatomi-ai'.
pub fn normalize_key(s: &str) -> String {
    s.chars()
        .filter(|c| c.is_alphanumeric())
        .flat_map(|c| c.to_lowercase())
        .collect()
}

// Lowercased once here so every comparison downstream is a plain substring test.
pub fn arg(args: &TransformArgs, key: &str) -> Vec<String> {
    args.get(key)
        .map(|v| v.iter().map(|s| s.to_ascii_lowercase()).collect())
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::artifact::Schema;

    fn args(pairs: &[(&str, &[&str])]) -> TransformArgs {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.iter().map(|s| s.to_string()).collect()))
            .collect()
    }

    #[test]
    fn dedupe_collapses_identical_rows() {
        let row = json!({"name": "A", "url": "u"});
        let out = dedupe(vec![row.clone(), row], &HashMap::new());
        assert_eq!(out.len(), 1);
    }

    #[test]
    fn dedupe_by_key_ignores_other_fields() {
        let rows = vec![json!({"name": "A", "n": 1}), json!({"name": "A", "n": 2})];
        assert_eq!(dedupe(rows, &args(&[("key", &["name"])])).len(), 1);
    }

    #[test]
    fn difference_subtracts_later_inputs_with_normalized_keys() {
        let all = Artifact::records(
            Schema::record(),
            vec![
                json!({"name": "Nakatomi Ai"}),
                json!({"name": "Contoso"}),
                json!({"name": "Tyrell"}),
            ],
        );
        // Resolved under a slug-ish variant; normalization must still match it.
        let resolved = Artifact::records(Schema::record(), vec![json!({"name": "nakatomi-ai"})]);
        let args = args(&[("key", &["name"])]);
        let out = difference(&[&all, &resolved], &args);
        let names: Vec<&str> = out
            .iter()
            .filter_map(|r| r.get("name").and_then(Value::as_str))
            .collect();
        assert_eq!(names, vec!["Contoso", "Tyrell"]);
    }

    #[test]
    fn group_by_buckets_rows_and_serializes_items() {
        let rows = vec![
            json!({"name": "Acme", "kind": "a"}),
            json!({"name": "Beta", "kind": "b"}),
            json!({"name": "acme", "kind": "c"}),
        ];
        let out = group_by(rows, &args(&[("key", &["name"])]));
        assert_eq!(out.len(), 2);
        // Case-insensitive grouping, first-seen order, display casing kept.
        assert_eq!(out[0]["name"], "Acme");
        assert_eq!(out[0]["count"], 2);
        let items: Vec<Value> =
            serde_json::from_str(out[0]["items_json"].as_str().unwrap()).unwrap();
        assert_eq!(items.len(), 2);
        assert_eq!(items[1]["kind"], "c");
        assert_eq!(out[1]["name"], "Beta");
        assert_eq!(out[1]["count"], 1);
    }

    #[test]
    fn top_n_sorts_by_score_and_truncates() {
        let rows = vec![
            json!({"name": "A", "score": 70}),
            json!({"name": "B", "score": 95}),
            json!({"name": "C"}),
            json!({"name": "D", "score": 80}),
        ];
        let out = top_n(rows, &args(&[("n", &["2"])]));
        let names: Vec<&str> = out
            .iter()
            .filter_map(|r| r.get("name").and_then(Value::as_str))
            .collect();
        assert_eq!(names, vec!["B", "D"]);
    }
}
