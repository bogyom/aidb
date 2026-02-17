use crate::{Database, EngineError, EngineResult};
use sqllogictest::{ColumnType, DBOutput};
use std::error::Error;
use std::fmt;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

#[derive(Debug)]
pub struct SltDatabase {
    db: Database,
    query_counter: u64,
    timing_sink: TimingSink,
    timings: Vec<QueryTiming>,
}

impl SltDatabase {
    pub fn create(path: impl Into<String>) -> EngineResult<Self> {
        Ok(Self {
            db: Database::create(path)?,
            query_counter: 0,
            timing_sink: TimingSink::StdErr,
            timings: Vec::new(),
        })
    }

    pub fn open(path: impl Into<String>) -> EngineResult<Self> {
        Ok(Self {
            db: Database::open(path)?,
            query_counter: 0,
            timing_sink: TimingSink::StdErr,
            timings: Vec::new(),
        })
    }

    #[cfg(test)]
    fn create_with_timing_capture(
        path: impl Into<String>,
    ) -> EngineResult<(Self, Arc<Mutex<Vec<String>>>)> {
        let captured = Arc::new(Mutex::new(Vec::new()));
        let db = Self {
            db: Database::create(path)?,
            query_counter: 0,
            timing_sink: TimingSink::Capture(Arc::clone(&captured)),
            timings: Vec::new(),
        };
        Ok((db, captured))
    }

    fn next_query_id(&mut self) -> u64 {
        self.query_counter += 1;
        self.query_counter
    }

    fn log_timing(&mut self, query_id: u64, duration: Duration) {
        self.timings.push(QueryTiming { query_id, duration });
        let line = format_timing_line(query_id, duration);
        self.timing_sink.log(line);
    }

    pub fn clear_timings(&mut self) {
        self.timings.clear();
    }

    pub fn emit_slowest_summary(&mut self, label: &str, limit: usize) {
        if self.timings.is_empty() || limit == 0 {
            return;
        }
        let mut timings = self.timings.clone();
        timings.sort_by(|a, b| b.duration.cmp(&a.duration));
        let count = limit.min(timings.len());
        self.timing_sink
            .log(format!("slt slowest {label} queries (top {count})"));
        for (rank, timing) in timings.into_iter().take(count).enumerate() {
            let duration_ms = timing.duration.as_secs_f64() * 1000.0;
            self.timing_sink.log(format!(
                "slt slow_query rank={} query_id={} duration_ms={duration_ms:.3}",
                rank + 1,
                timing.query_id
            ));
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct QueryTiming {
    query_id: u64,
    duration: Duration,
}

#[derive(Debug)]
enum TimingSink {
    StdErr,
    Capture(Arc<Mutex<Vec<String>>>),
}

impl TimingSink {
    fn log(&mut self, line: String) {
        match self {
            TimingSink::StdErr => eprintln!("{line}"),
            TimingSink::Capture(captured) => {
                if let Ok(mut logs) = captured.lock() {
                    logs.push(line);
                }
            }
        }
    }
}

#[derive(Debug)]
pub struct SltError {
    message: String,
}

impl SltError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl fmt::Display for SltError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl Error for SltError {}

impl From<EngineError> for SltError {
    fn from(error: EngineError) -> Self {
        SltError::new(format!("{:?}", error))
    }
}

impl sqllogictest::DB for SltDatabase {
    type Error = SltError;

    fn run(&mut self, sql: &str) -> Result<DBOutput, Self::Error> {
        let query_id = self.next_query_id();
        let start = Instant::now();
        let result = self.db.execute(sql);
        let duration = start.elapsed();
        self.log_timing(query_id, duration);
        let result = result.map_err(SltError::from)?;
        let normalized = sql.trim().trim_end_matches(';').trim();
        if normalized.to_ascii_uppercase().starts_with("SELECT") {
            let types = result
                .columns
                .iter()
                .map(|col| sql_type_to_column_type(col.sql_type))
                .collect::<Vec<_>>();
            let rows: Vec<Vec<String>> = result
                .rows
                .into_iter()
                .map(|row| row.into_iter().map(value_to_string).collect())
                .collect();
            return Ok(DBOutput::Rows { types, rows });
        }

        Ok(DBOutput::StatementComplete(0))
    }

    fn engine_name(&self) -> &str {
        "aidb"
    }
}

fn format_timing_line(query_id: u64, duration: Duration) -> String {
    let duration_ms = duration.as_secs_f64() * 1000.0;
    format!("slt query_id={query_id} duration_ms={duration_ms:.3}")
}

fn sql_type_to_column_type(sql_type: crate::SqlType) -> ColumnType {
    match sql_type {
        crate::SqlType::Integer | crate::SqlType::Boolean => ColumnType::Integer,
        crate::SqlType::Real => ColumnType::FloatingPoint,
        crate::SqlType::Text => ColumnType::Text,
    }
}

fn value_to_string(value: crate::Value) -> String {
    match value {
        crate::Value::Null => "NULL".to_string(),
        crate::Value::Integer(value) => value.to_string(),
        crate::Value::Real(value) => {
            if value.fract() == 0.0 {
                (value as i64).to_string()
            } else {
                value.to_string()
            }
        }
        crate::Value::Text(value) => value,
        crate::Value::Boolean(value) => {
            if value {
                "1".to_string()
            } else {
                "0".to_string()
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use sqllogictest::DB;
    use sqllogictest::Runner;

    #[test]
    fn logs_query_timing_with_identifier_and_duration() {
        let temp = tempfile::NamedTempFile::new().expect("temp file");
        let (mut db, captured) =
            SltDatabase::create_with_timing_capture(temp.path().to_string_lossy().as_ref())
                .expect("create slt db");

        db.run("CREATE TABLE t (id INTEGER);")
            .expect("create table");
        db.run("SELECT 1;").expect("select");

        let logs = captured.lock().expect("lock logs").clone();
        assert_eq!(logs.len(), 2);
        assert!(logs[0].contains("query_id=1"));
        assert!(logs[1].contains("query_id=2"));
        for line in logs {
            assert!(line.contains("duration_ms="));
        }
    }

    #[test]
    fn logs_query_timing_when_run_via_sqllogictest_runner() {
        let temp = tempfile::NamedTempFile::new().expect("temp file");
        let (db, captured) =
            SltDatabase::create_with_timing_capture(temp.path().to_string_lossy().as_ref())
                .expect("create slt db");
        let mut runner = Runner::new(db);
        let script = r#"
statement ok
CREATE TABLE t1(id INTEGER)

query I
SELECT 1;
----
1
"#;

        runner.run_script(script).expect("run script");

        let logs = captured.lock().expect("lock logs").clone();
        assert_eq!(logs.len(), 2);
        assert!(logs[0].contains("query_id=1"));
        assert!(logs[1].contains("query_id=2"));
        for line in logs {
            assert!(line.contains("duration_ms="));
        }
    }

    #[test]
    fn emits_slowest_summary_with_query_id_and_duration() {
        let temp = tempfile::NamedTempFile::new().expect("temp file");
        let (mut db, captured) =
            SltDatabase::create_with_timing_capture(temp.path().to_string_lossy().as_ref())
                .expect("create slt db");

        for query_id in 1..=12 {
            db.timings.push(QueryTiming {
                query_id,
                duration: Duration::from_millis(query_id),
            });
        }

        db.emit_slowest_summary("select4.test", 10);

        let logs = captured.lock().expect("lock logs").clone();
        assert_eq!(logs.len(), 11);
        assert!(logs[0].contains("slowest select4.test queries"));
        assert!(logs[0].contains("top 10"));
        assert!(logs[1].contains("rank=1"));
        assert!(logs[1].contains("query_id=12"));
        assert!(logs[1].contains("duration_ms=12.000"));
        assert!(logs[10].contains("rank=10"));
        assert!(logs[10].contains("query_id=3"));
        assert!(logs[10].contains("duration_ms=3.000"));
    }
}
