use assert_cmd::Command;
use predicates::prelude::*;
use std::env;
use std::fs;
use std::io;

// Going to keep it light: Just test the happy path and a simple
// parser failure.

#[test]
fn test_cli_success() {
    let expected = std::fs::read_to_string("results/example_9x9.txt").unwrap();

    let mut cmd = Command::cargo_bin("hashiwokakero-solver").unwrap();

    cmd.pipe_stdin("examples/example_9x9.txt")
        .unwrap()
        .assert()
        .success()
        .stdout(expected);
}

#[test]
fn test_cli_input_flag() {
    let expected = std::fs::read_to_string("results/example_17x17.txt").unwrap();

    let mut cmd = Command::cargo_bin("hashiwokakero-solver").unwrap();

    cmd.arg("--input-file=examples/example_17x17.txt")
        .assert()
        .success()
        .stdout(expected);
}

#[test]
fn test_cli_output_flag() -> io::Result<()> {
    let expected = std::fs::read_to_string("results/example_17x17.txt").unwrap();

    let mut cmd = Command::cargo_bin("hashiwokakero-solver").unwrap();

    let mut path = env::temp_dir();
    path.push("test-hashiwokakero-solver.txt");

    println!("{}", path.to_str().unwrap());
    cmd.pipe_stdin("examples/example_17x17.txt")
        .unwrap()
        .arg(format!("--output-file={}", path.to_str().unwrap()))
        .assert()
        .success()
        .stdout("");

    let actual = fs::read_to_string(path.clone())?;
    fs::remove_file(path)?;
    assert_eq!(expected, actual);

    Ok(())
}

#[test]
fn test_cli_no_solutions() {
    let mut cmd = Command::cargo_bin("hashiwokakero-solver").unwrap();

    cmd.write_stdin("...1..")
        .assert()
        .success()
        .stdout("")
        .stderr(predicate::str::contains("No solutions"));
}

#[test]
fn test_cli_parse_error() {
    let mut cmd = Command::cargo_bin("hashiwokakero-solver").unwrap();

    cmd.write_stdin("This is not a valid input.")
        .assert()
        .failure()
        .stdout("")
        .stderr(predicate::str::contains("Unexpected character in input"));
}
