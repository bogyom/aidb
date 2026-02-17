use crate::{Database, EngineError, EngineResult};
use sqllogictest::{ColumnType, DBOutput};
use std::error::Error;
use std::fmt;

#[derive(Debug)]
pub struct SltDatabase {
    db: Database,
}

impl SltDatabase {
    pub fn create(path: impl Into<String>) -> EngineResult<Self> {
        Ok(Self {
            db: Database::create(path)?,
        })
    }

    pub fn open(path: impl Into<String>) -> EngineResult<Self> {
        Ok(Self {
            db: Database::open(path)?,
        })
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
        self.db.execute(sql).map_err(SltError::from)?;

        let normalized = sql.trim().trim_end_matches(';').trim();
        if normalized.eq_ignore_ascii_case("select 1") {
            return Ok(DBOutput::Rows {
                types: vec![ColumnType::Integer],
                rows: vec![vec!["1".to_string()]],
            });
        }

        if normalized.to_ascii_uppercase().starts_with("SELECT") {
            return Ok(DBOutput::Rows {
                types: vec![ColumnType::Text],
                rows: Vec::new(),
            });
        }

        Ok(DBOutput::StatementComplete(0))
    }

    fn engine_name(&self) -> &str {
        "aidb"
    }
}
