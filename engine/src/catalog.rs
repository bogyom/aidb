/// Supported SQL types for the MVP SQL subset.
///
/// - `Integer`: signed 64-bit integer values
/// - `Real`: 64-bit floating point values
/// - `Text`: UTF-8 string values
/// - `Boolean`: true/false values
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SqlType {
    Integer,
    Real,
    Text,
    Boolean,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnSchema {
    pub name: String,
    pub sql_type: SqlType,
    pub nullable: bool,
}

impl ColumnSchema {
    pub fn new(name: impl Into<String>, sql_type: SqlType, nullable: bool) -> Self {
        Self {
            name: name.into(),
            sql_type,
            nullable,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableSchema {
    pub name: String,
    pub columns: Vec<ColumnSchema>,
    pub primary_key: Option<String>,
}

impl TableSchema {
    pub fn new(name: impl Into<String>, columns: Vec<ColumnSchema>) -> Self {
        Self {
            name: name.into(),
            columns,
            primary_key: None,
        }
    }

    pub fn with_primary_key(mut self, column: impl Into<String>) -> Self {
        self.primary_key = Some(column.into());
        self
    }

    pub fn column(&self, name: &str) -> Option<&ColumnSchema> {
        self.columns.iter().find(|column| column.name == name)
    }
}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct Catalog {
    tables: Vec<TableSchema>,
}

impl Catalog {
    pub fn new() -> Self {
        Self { tables: Vec::new() }
    }

    pub fn add_table(&mut self, table: TableSchema) {
        self.tables.push(table);
    }

    pub fn remove_table(&mut self, name: &str) -> Option<TableSchema> {
        if let Some(index) = self.tables.iter().position(|table| table.name == name) {
            return Some(self.tables.remove(index));
        }
        None
    }

    pub fn table(&self, name: &str) -> Option<&TableSchema> {
        self.tables.iter().find(|table| table.name == name)
    }

    pub fn tables(&self) -> &[TableSchema] {
        &self.tables
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_schema_lookup_column() {
        let table = TableSchema::new(
            "people",
            vec![
                ColumnSchema::new("id", SqlType::Integer, false),
                ColumnSchema::new("name", SqlType::Text, false),
            ],
        )
        .with_primary_key("id");

        let column = table.column("name").expect("column exists");
        assert_eq!(column.sql_type, SqlType::Text);
        assert!(!column.nullable);
    }

    #[test]
    fn catalog_stores_tables() {
        let mut catalog = Catalog::new();
        catalog.add_table(TableSchema::new(
            "items",
            vec![ColumnSchema::new("sku", SqlType::Text, false)],
        ));
        let table = catalog.table("items").expect("table exists");
        assert_eq!(table.columns.len(), 1);
    }
}
