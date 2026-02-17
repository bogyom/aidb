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
        let result = self.db.execute(sql).map_err(SltError::from)?;
        let normalized = sql.trim().trim_end_matches(';').trim();
        if normalized.to_ascii_uppercase().starts_with("SELECT") {
            let types = result
                .columns
                .iter()
                .map(|col| sql_type_to_column_type(col.sql_type))
                .collect::<Vec<_>>();
            let mut rows: Vec<Vec<String>> = result
                .rows
                .into_iter()
                .map(|row| row.into_iter().map(value_to_string).collect())
                .collect();
            let value_count = rows.len() * types.len();
            if value_count > HASH_THRESHOLD && !rows.is_empty() {
                let mut md5 = md5::Context::new();
                for line in &rows {
                    for value in line {
                        md5.consume(value.as_bytes());
                        md5.consume(b"\n");
                    }
                }
                let hash = md5.compute();
                rows = vec![vec![format!(
                    "{} values hashing to {:?}",
                    value_count, hash
                )]];
            } else {
                rows = rows
                    .into_iter()
                    .flat_map(|row| row.into_iter().map(|value| vec![value]))
                    .collect();
            }
            return Ok(DBOutput::Rows { types, rows });
        }

        Ok(DBOutput::StatementComplete(0))
    }

    fn engine_name(&self) -> &str {
        "aidb"
    }
}

const HASH_THRESHOLD: usize = 8;

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
