use engine::SltDatabase;
use sqllogictest::Runner;
use std::env;
use std::fs;
use std::path::{Path, PathBuf};

fn main() {
    if let Err(error) = run() {
        eprintln!("{error}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), String> {
    let mut args = env::args().skip(1);
    let path = args
        .next()
        .unwrap_or_else(|| "tests/sqllogictest/test/aidb_smoke.test".to_string());
    let db_path = args
        .next()
        .unwrap_or_else(|| "target/aidb_sqllogictest.db".to_string());

    let mut runner =
        Runner::new(SltDatabase::create(db_path).map_err(|err| format!("{:?}", err))?);
    runner.enable_testdir();

    let path = Path::new(&path);
    let files = collect_test_files(path).map_err(|err| err.to_string())?;
    if files.is_empty() {
        return Err(format!("no .test files found under {}", path.display()));
    }
    for file in files {
        if is_select4_test(&file) {
            runner.db_mut().clear_timings();
        }
        runner
            .run_file(&file)
            .map_err(|err| err.display(false).to_string())?;
        if is_select4_test(&file) {
            runner
                .db_mut()
                .emit_slowest_summary("select4.test", 10);
        }
    }
    Ok(())
}

fn collect_test_files(path: &Path) -> Result<Vec<PathBuf>, std::io::Error> {
    if path.is_file() {
        return Ok(vec![path.to_path_buf()]);
    }
    let mut files = Vec::new();
    visit_dir(path, &mut files)?;
    files.sort();
    Ok(files)
}

fn visit_dir(path: &Path, files: &mut Vec<PathBuf>) -> Result<(), std::io::Error> {
    for entry in fs::read_dir(path)? {
        let entry = entry?;
        let entry_path = entry.path();
        if entry_path.is_dir() {
            visit_dir(&entry_path, files)?;
        } else if entry_path.extension().and_then(|s| s.to_str()) == Some("test") {
            files.push(entry_path);
        }
    }
    Ok(())
}

fn is_select4_test(path: &Path) -> bool {
    matches!(
        path.file_name().and_then(|name| name.to_str()),
        Some("select4.test")
    )
}
