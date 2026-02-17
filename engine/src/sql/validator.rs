use super::parser::{ColumnDef, Expr, OrderBy, Select, SelectItem, Statement};
use crate::Catalog;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ValidationError {
    TableNotFound(String),
    ColumnNotFound { table: String, column: String },
    TableAlreadyExists(String),
    Unsupported(String),
}

impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ValidationError::TableNotFound(name) => write!(f, "table not found: {}", name),
            ValidationError::ColumnNotFound { table, column } => {
                write!(f, "column not found: {}.{}", table, column)
            }
            ValidationError::TableAlreadyExists(name) => {
                write!(f, "table already exists: {}", name)
            }
            ValidationError::Unsupported(message) => write!(f, "unsupported: {}", message),
        }
    }
}

impl ValidationError {
    fn unsupported(message: impl Into<String>) -> Self {
        ValidationError::Unsupported(message.into())
    }
}

pub fn validate_statement(statement: &Statement, catalog: &Catalog) -> Result<(), ValidationError> {
    match statement {
        Statement::CreateTable(create) => validate_create_table(create, catalog),
        Statement::DropTable(drop) => validate_drop_table(drop, catalog),
        Statement::Insert(insert) => validate_insert(insert, catalog),
        Statement::Select(select) => validate_select(select, catalog),
    }
}

fn validate_create_table(
    create: &super::parser::CreateTable,
    catalog: &Catalog,
) -> Result<(), ValidationError> {
    if catalog.table(&create.name).is_some() {
        return Err(ValidationError::TableAlreadyExists(create.name.clone()));
    }
    if create.columns.is_empty() {
        return Err(ValidationError::unsupported(
            "CREATE TABLE requires at least one column",
        ));
    }

    let mut seen_primary = None;
    let mut seen_columns = std::collections::HashSet::new();
    for ColumnDef {
        name,
        data_type: _,
        primary_key,
    } in &create.columns
    {
        if !seen_columns.insert(name.clone()) {
            return Err(ValidationError::unsupported(
                "duplicate column name in CREATE TABLE",
            ));
        }
        if *primary_key {
            if seen_primary.replace(name).is_some() {
                return Err(ValidationError::unsupported(
                    "multiple primary key columns are not supported",
                ));
            }
        }
    }

    Ok(())
}

fn validate_drop_table(
    drop: &super::parser::DropTable,
    catalog: &Catalog,
) -> Result<(), ValidationError> {
    if catalog.table(&drop.name).is_none() {
        return Err(ValidationError::TableNotFound(drop.name.clone()));
    }
    Ok(())
}

fn validate_insert(
    insert: &super::parser::Insert,
    catalog: &Catalog,
) -> Result<(), ValidationError> {
    let table = catalog
        .table(&insert.table)
        .ok_or_else(|| ValidationError::TableNotFound(insert.table.clone()))?;

    let target_columns: Vec<String> = if let Some(columns) = &insert.columns {
        let mut seen = std::collections::HashSet::new();
        for column in columns {
            if !seen.insert(column) {
                return Err(ValidationError::unsupported(
                    "duplicate column in INSERT",
                ));
            }
            if table.column(column).is_none() {
                return Err(ValidationError::ColumnNotFound {
                    table: table.name.clone(),
                    column: column.clone(),
                });
            }
        }
        columns.clone()
    } else {
        table.columns.iter().map(|column| column.name.clone()).collect()
    };

    if insert.values.len() != target_columns.len() {
        return Err(ValidationError::unsupported(
            "INSERT values count does not match column count",
        ));
    }

    for value in &insert.values {
        if expr_contains_identifier(value) {
            return Err(ValidationError::unsupported(
                "INSERT values must be literals",
            ));
        }
    }

    Ok(())
}

fn validate_select(select: &Select, catalog: &Catalog) -> Result<(), ValidationError> {
    let table = match &select.from {
        Some(name) => Some(
            catalog
                .table(name)
                .ok_or_else(|| ValidationError::TableNotFound(name.clone()))?,
        ),
        None => None,
    };

    for item in &select.items {
        match item {
            SelectItem::Wildcard(Some(prefix)) => {
                let table = table.ok_or_else(|| {
                    ValidationError::unsupported("qualified wildcard requires FROM clause")
                })?;
                if prefix != &table.name {
                    return Err(ValidationError::unsupported(
                        "qualified wildcard does not match FROM table",
                    ));
                }
            }
            SelectItem::Wildcard(None) => {
                if table.is_none() {
                    return Err(ValidationError::unsupported(
                        "wildcard select requires FROM clause",
                    ));
                }
            }
            SelectItem::Expr(expr) => validate_expr(expr, table)?,
        }
    }

    if let Some(filter) = &select.filter {
        validate_expr(filter, table)?;
    }

    if let Some(OrderBy { expr, direction: _ }) = &select.order_by {
        validate_expr(expr, table)?;
    }

    Ok(())
}

fn validate_expr(
    expr: &Expr,
    table: Option<&crate::TableSchema>,
) -> Result<(), ValidationError> {
    match expr {
        Expr::Identifier(parts) => validate_identifier(parts, table),
        Expr::Literal(_) => Ok(()),
        Expr::Unary { expr, .. } => validate_expr(expr, table),
        Expr::Binary { left, right, .. } => {
            validate_expr(left, table)?;
            validate_expr(right, table)
        }
    }
}

fn validate_identifier(
    parts: &[String],
    table: Option<&crate::TableSchema>,
) -> Result<(), ValidationError> {
    let table = table.ok_or_else(|| {
        ValidationError::unsupported("column reference requires FROM clause")
    })?;

    match parts {
        [column] => {
            if table.column(column).is_none() {
                return Err(ValidationError::ColumnNotFound {
                    table: table.name.clone(),
                    column: column.clone(),
                });
            }
            Ok(())
        }
        [table_name, column] => {
            if table_name != &table.name {
                return Err(ValidationError::unsupported(
                    "qualified column does not match FROM table",
                ));
            }
            if table.column(column).is_none() {
                return Err(ValidationError::ColumnNotFound {
                    table: table.name.clone(),
                    column: column.clone(),
                });
            }
            Ok(())
        }
        _ => Err(ValidationError::unsupported(
            "unsupported qualified identifier",
        )),
    }
}

fn expr_contains_identifier(expr: &Expr) -> bool {
    match expr {
        Expr::Identifier(_) => true,
        Expr::Literal(_) => false,
        Expr::Unary { expr, .. } => expr_contains_identifier(expr),
        Expr::Binary { left, right, .. } => {
            expr_contains_identifier(left) || expr_contains_identifier(right)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Catalog, ColumnSchema, SqlType, TableSchema};
    use super::super::parser::{
        BinaryOp, CreateTable, DropTable, Expr, Insert, Select, SelectItem, Statement, TypeName,
    };

    fn catalog_with_users() -> Catalog {
        let mut catalog = Catalog::new();
        catalog.add_table(
            TableSchema::new(
                "users",
                vec![
                    ColumnSchema::new("id", SqlType::Integer, false),
                    ColumnSchema::new("name", SqlType::Text, false),
                ],
            )
            .with_primary_key("id"),
        );
        catalog
    }

    #[test]
    fn rejects_missing_table() {
        let catalog = Catalog::new();
        let stmt = Statement::Select(Select {
            items: vec![SelectItem::Wildcard(None)],
            from: Some("users".to_string()),
            filter: None,
            order_by: None,
            limit: None,
        });
        let err = validate_statement(&stmt, &catalog).expect_err("error");
        assert!(matches!(err, ValidationError::TableNotFound(_)));
    }

    #[test]
    fn rejects_unknown_column() {
        let catalog = catalog_with_users();
        let stmt = Statement::Select(Select {
            items: vec![SelectItem::Expr(Expr::Identifier(vec![
                "age".to_string(),
            ]))],
            from: Some("users".to_string()),
            filter: None,
            order_by: None,
            limit: None,
        });
        let err = validate_statement(&stmt, &catalog).expect_err("error");
        assert!(matches!(err, ValidationError::ColumnNotFound { .. }));
    }

    #[test]
    fn rejects_insert_value_identifier() {
        let catalog = catalog_with_users();
        let stmt = Statement::Insert(Insert {
            table: "users".to_string(),
            columns: None,
            values: vec![Expr::Identifier(vec!["id".to_string()])],
        });
        let err = validate_statement(&stmt, &catalog).expect_err("error");
        assert!(matches!(err, ValidationError::Unsupported(_)));
    }

    #[test]
    fn rejects_insert_column_count_mismatch() {
        let catalog = catalog_with_users();
        let stmt = Statement::Insert(Insert {
            table: "users".to_string(),
            columns: Some(vec!["id".to_string()]),
            values: vec![Expr::Literal(super::super::lexer::Literal::Number("1".to_string())), Expr::Literal(super::super::lexer::Literal::Number("2".to_string()))],
        });
        let err = validate_statement(&stmt, &catalog).expect_err("error");
        assert!(matches!(err, ValidationError::Unsupported(_)));
    }

    #[test]
    fn rejects_select_identifier_without_from() {
        let catalog = catalog_with_users();
        let stmt = Statement::Select(Select {
            items: vec![SelectItem::Expr(Expr::Identifier(vec![
                "id".to_string(),
            ]))],
            from: None,
            filter: None,
            order_by: None,
            limit: None,
        });
        let err = validate_statement(&stmt, &catalog).expect_err("error");
        assert!(matches!(err, ValidationError::Unsupported(_)));
    }

    #[test]
    fn allows_select_literal_without_from() {
        let catalog = catalog_with_users();
        let stmt = Statement::Select(Select {
            items: vec![SelectItem::Expr(Expr::Literal(
                super::super::lexer::Literal::Number("1".to_string()),
            ))],
            from: None,
            filter: Some(Expr::Binary {
                left: Box::new(Expr::Literal(super::super::lexer::Literal::Number(
                    "1".to_string(),
                ))),
                op: BinaryOp::Eq,
                right: Box::new(Expr::Literal(super::super::lexer::Literal::Number(
                    "1".to_string(),
                ))),
            }),
            order_by: None,
            limit: None,
        });
        validate_statement(&stmt, &catalog).expect("valid");
    }

    #[test]
    fn rejects_multiple_primary_keys() {
        let catalog = Catalog::new();
        let stmt = Statement::CreateTable(CreateTable {
            name: "widgets".to_string(),
            columns: vec![
                ColumnDef {
                    name: "id".to_string(),
                    data_type: TypeName::Integer,
                    primary_key: true,
                },
                ColumnDef {
                    name: "code".to_string(),
                    data_type: TypeName::Text,
                    primary_key: true,
                },
            ],
        });
        let err = validate_statement(&stmt, &catalog).expect_err("error");
        assert!(matches!(err, ValidationError::Unsupported(_)));
    }

    #[test]
    fn rejects_drop_missing_table() {
        let catalog = Catalog::new();
        let stmt = Statement::DropTable(DropTable {
            name: "missing".to_string(),
        });
        let err = validate_statement(&stmt, &catalog).expect_err("error");
        assert!(matches!(err, ValidationError::TableNotFound(_)));
    }
}
