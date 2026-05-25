use assert_cmd::Command;
use predicates::str::contains;

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
fn run_rejects_conflicting_input_flags() {
    flux()
        .args([
            "run",
            "workflow.toml",
            "--input",
            "hello",
            "--input-file",
            "foo.txt",
        ])
        .assert()
        .failure();
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
