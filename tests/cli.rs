use assert_cmd::Command;
use predicates::str::contains;
use std::io::Write;

fn flux() -> Command {
    Command::cargo_bin("flux").expect("flux binary not found")
}

#[test]
fn help_shows_subcommands() {
    flux()
        .arg("--help")
        .assert()
        .success()
        .stdout(predicates::str::contains("run"))
        .stdout(predicates::str::contains("validate"))
        .stdout(predicates::str::contains("explain"))
        .stdout(predicates::str::contains("list-models"));
}

#[test]
fn list_models_shows_all_models() {
    flux()
        .arg("list-models")
        .assert()
        .success()
        .stdout(predicates::str::contains("claude-haiku-4-5-20251001"))
        .stdout(predicates::str::contains("claude-sonnet-4-6"))
        .stdout(predicates::str::contains("claude-opus-4-7"));
}

#[test]
fn run_rejects_missing_input_file() {
    flux()
        .args([
            "run",
            "examples/paper-compare.toml",
            "--input-file",
            "resume=no-such-file.md",
        ])
        .assert()
        .failure()
        .stderr(contains("no-such-file.md"));
}

#[test]
fn validate_accepts_valid_workflow() {
    flux()
        .args(["validate", "examples/paper-compare.toml"])
        .assert()
        .success()
        .stdout(contains("is valid"));
}

#[test]
fn validate_rejects_missing_file() {
    flux()
        .args(["validate", "no-such-file.toml"])
        .assert()
        .failure();
}

#[test]
fn explain_shows_workflow_structure() {
    flux()
        .args(["explain", "examples/paper-compare.toml"])
        .assert()
        .success()
        .stdout(contains("paper-compare"))
        .stdout(contains("find"))
        .stdout(contains("compare"));
}

#[test]
fn validate_accepts_agent_workflow() {
    flux()
        .args(["validate", "examples/company-research.toml"])
        .assert()
        .success()
        .stdout(contains("is valid"));
}

#[test]
fn explain_shows_fan_out_for_agent_step() {
    flux()
        .args(["explain", "examples/company-research.toml"])
        .assert()
        .success()
        .stdout(contains("Provider: claude-code"))
        .stdout(contains("Fan-out"))
        .stdout(contains("planner"));
}

#[test]
fn validate_rejects_unknown_provider() {
    let mut wf = tempfile::NamedTempFile::new().unwrap();
    write!(
        wf,
        r#"
[workflow]
name = "test"
[budget]
max_usd = 1.0
[[steps]]
id = "a"
provider = "gpt-4"
prompt = "hello"
[output]
step = "a"
"#
    )
    .unwrap();

    flux()
        .args(["validate", wf.path().to_str().unwrap()])
        .assert()
        .failure()
        .stderr(contains("unknown provider"));
}

#[test]
fn validate_rejects_parallel_over_without_depends_on() {
    let mut wf = tempfile::NamedTempFile::new().unwrap();
    write!(
        wf,
        r#"
[workflow]
name = "test"
[budget]
max_usd = 1.0
[[steps]]
id = "planner"
prompt = "list things"
[[steps]]
id = "researcher"
provider = "claude-code"
parallel_over = "planner"
prompt = "research {{ item }}"
[output]
step = "researcher"
"#
    )
    .unwrap();

    flux()
        .args(["validate", wf.path().to_str().unwrap()])
        .assert()
        .failure()
        .stderr(contains("not listed in depends_on"));
}
