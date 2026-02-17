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

    let path = Path::new(&path);
    let files = collect_test_files(path).map_err(|err| err.to_string())?;
    if files.is_empty() {
        return Err(format!("no .test files found under {}", path.display()));
    }
    for file in files {
        let mut runner = make_runner(&db_path)?;
        if is_select4_test(&file) {
            runner.db_mut().clear_timings();
        }
        runner
            .run_file(&file)
            .map_err(|err| format!("{}: {}", file.display(), err.display(false)))?;
        if is_select4_test(&file) {
            runner
                .db_mut()
                .emit_slowest_summary("select4.test", 10);
        }
    }
    Ok(())
}

fn make_runner(db_path: &str) -> Result<Runner<SltDatabase>, String> {
    let mut runner =
        Runner::new(SltDatabase::create(db_path).map_err(|err| format!("{:?}", err))?);
    runner.enable_testdir();
    Ok(runner)
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

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::Path;

    #[test]
    fn collects_random_aggregate_tests() {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let aggregates_dir = manifest
            .join("..")
            .join("tests")
            .join("sqllogictest")
            .join("test")
            .join("random")
            .join("aggregates");
        let files = collect_test_files(&aggregates_dir).expect("collect aggregates files");
        assert!(
            files.iter().any(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name == "slt_good_0.test")
                    .unwrap_or(false)
            }),
            "expected slt_good_0.test under {}",
            aggregates_dir.display()
        );
    }

    #[test]
    fn collects_random_expr_tests() {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let expr_dir = manifest
            .join("..")
            .join("tests")
            .join("sqllogictest")
            .join("test")
            .join("random")
            .join("expr");
        let files = collect_test_files(&expr_dir).expect("collect expr files");
        assert!(
            files.iter().any(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name == "slt_good_0.test")
                    .unwrap_or(false)
            }),
            "expected slt_good_0.test under {}",
            expr_dir.display()
        );
    }

    #[test]
    fn collects_random_groupby_tests() {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let groupby_dir = manifest
            .join("..")
            .join("tests")
            .join("sqllogictest")
            .join("test")
            .join("random")
            .join("groupby");
        let files = collect_test_files(&groupby_dir).expect("collect groupby files");
        assert!(
            files.iter().any(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name == "slt_good_0.test")
                    .unwrap_or(false)
            }),
            "expected slt_good_0.test under {}",
            groupby_dir.display()
        );
    }

    #[test]
    fn collects_random_select_tests() {
        let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
        let select_dir = manifest
            .join("..")
            .join("tests")
            .join("sqllogictest")
            .join("test")
            .join("random")
            .join("select");
        let files = collect_test_files(&select_dir).expect("collect select files");
        assert!(
            files.iter().any(|path| {
                path.file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name == "slt_good_0.test")
                    .unwrap_or(false)
            }),
            "expected slt_good_0.test under {}",
            select_dir.display()
        );
    }
}
