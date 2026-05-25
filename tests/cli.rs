use assert_cmd::Command;

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
