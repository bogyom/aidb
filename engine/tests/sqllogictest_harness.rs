use engine::SltDatabase;
use sqllogictest::Runner;
use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_db_path() -> PathBuf {
    let mut path = std::env::temp_dir();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    path.push(format!("aidb_slt_{nanos}.db"));
    path
}

#[test]
fn sqllogictest_runs_smoke_file() {
    let db_path = temp_db_path();
    let mut runner = Runner::new(SltDatabase::create(db_path.to_string_lossy().as_ref()).unwrap());
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let test_file = manifest
        .join("..")
        .join("tests")
        .join("sqllogictest")
        .join("test")
        .join("aidb_smoke.test");

    runner.run_file(test_file).expect("run smoke test");
}

#[test]
fn sqllogictest_runs_select1_file() {
    let db_path = temp_db_path();
    let mut runner = Runner::new(SltDatabase::create(db_path.to_string_lossy().as_ref()).unwrap());
    let manifest = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let test_file = manifest
        .join("..")
        .join("tests")
        .join("sqllogictest")
        .join("test")
        .join("select1.test");

    runner.run_file(test_file).expect("run select1 test");
}
