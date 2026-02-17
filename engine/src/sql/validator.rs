use super::parser::{
    ColumnDef, CreateIndex, Expr, FunctionArg, OrderBy, Select, SelectItem, Statement,
};
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
        Statement::CreateIndex(create) => validate_create_index(create, catalog),
        Statement::DropTable(drop) => validate_drop_table(drop, catalog),
        Statement::Insert(insert) => validate_insert(insert, catalog),
        Statement::Select(select) => validate_select(select, catalog),
        Statement::SetOperation(expr) => validate_set_expr(expr, catalog),
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

fn validate_create_index(
    create: &CreateIndex,
    catalog: &Catalog,
) -> Result<(), ValidationError> {
    let table = catalog
        .table(&create.table)
        .ok_or_else(|| ValidationError::TableNotFound(create.table.clone()))?;
    for column in &create.columns {
        if table.column(&column.name).is_none() {
            return Err(ValidationError::ColumnNotFound {
                table: table.name.clone(),
                column: column.name.clone(),
            });
        }
    }
    Ok(())
}

fn validate_set_expr(
    expr: &super::parser::SetExpr,
    catalog: &Catalog,
) -> Result<(), ValidationError> {
    match expr {
        super::parser::SetExpr::Select(select) => validate_select(select, catalog),
        super::parser::SetExpr::SetOp { left, right, .. } => {
            validate_set_expr(left, catalog)?;
            validate_set_expr(right, catalog)
        }
    }
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
    let mut tables = Vec::new();
    for from in &select.from {
        let table = catalog
            .table(&from.table)
            .ok_or_else(|| ValidationError::TableNotFound(from.table.clone()))?;
        tables.push((table, from.alias.as_deref()));
    }

    for item in &select.items {
        match item {
            SelectItem::Wildcard(Some(prefix)) => {
                if tables.is_empty() {
                    return Err(ValidationError::unsupported(
                        "qualified wildcard requires FROM clause",
                    ));
                }
                let matches = tables.iter().filter(|(table, alias)| {
                    &table.name == prefix || alias.map(|name| name == prefix).unwrap_or(false)
                });
                if matches.count() != 1 {
                    return Err(ValidationError::unsupported(
                        "qualified wildcard does not match FROM table",
                    ));
                }
            }
            SelectItem::Wildcard(None) => {
                if tables.is_empty() {
                    return Err(ValidationError::unsupported(
                        "wildcard select requires FROM clause",
                    ));
                }
            }
            SelectItem::Expr(expr) => validate_expr(expr, &tables)?,
        }
    }

    if let Some(filter) = &select.filter {
        validate_expr(filter, &tables)?;
    }

    for from in &select.from {
        if let Some(on_expr) = &from.left_join_on {
            if expr_contains_identifier(on_expr) {
                return Err(ValidationError::unsupported(
                    "LEFT JOIN ON must be constant",
                ));
            }
            validate_expr(on_expr, &tables)?;
        }
    }

    for expr in &select.group_by {
        validate_expr(expr, &tables)?;
    }

    if let Some(having) = &select.having {
        validate_expr(having, &tables)?;
    }

    for OrderBy { expr, direction: _ } in &select.order_by {
        validate_expr(expr, &tables)?;
    }

    Ok(())
}

fn validate_expr(
    expr: &Expr,
    tables: &[(&crate::TableSchema, Option<&str>)],
) -> Result<(), ValidationError> {
    match expr {
        Expr::Identifier(parts) => validate_identifier(parts, tables),
        Expr::Literal(_) => Ok(()),
        Expr::Unary { expr, .. } => validate_expr(expr, tables),
        Expr::Cast { expr, .. } => validate_expr(expr, tables),
        Expr::Binary { left, right, .. } => {
            validate_expr(left, tables)?;
            validate_expr(right, tables)
        }
        Expr::IsNull { expr, .. } => validate_expr(expr, tables),
        Expr::Between {
            expr,
            low,
            high,
            ..
        } => {
            validate_expr(expr, tables)?;
            validate_expr(low, tables)?;
            validate_expr(high, tables)
        }
        Expr::InList { expr, list, .. } => {
            validate_expr(expr, tables)?;
            for item in list {
                validate_expr(item, tables)?;
            }
            Ok(())
        }
        Expr::Case {
            operand,
            when_thens,
            else_expr,
        } => {
            if let Some(operand) = operand {
                validate_expr(operand, tables)?;
            }
            for (when_expr, then_expr) in when_thens {
                validate_expr(when_expr, tables)?;
                validate_expr(then_expr, tables)?;
            }
            if let Some(else_expr) = else_expr {
                validate_expr(else_expr, tables)?;
            }
            Ok(())
        }
        Expr::Function { args, .. } => {
            for arg in args {
                match arg {
                    FunctionArg::Expr(expr) => validate_expr(expr, tables)?,
                    FunctionArg::Star => {}
                }
            }
            Ok(())
        }
        Expr::Subquery(_) | Expr::Exists(_) => Ok(()),
    }
}

fn validate_identifier(
    parts: &[String],
    tables: &[(&crate::TableSchema, Option<&str>)],
) -> Result<(), ValidationError> {
    match parts {
        [column] => {
            if tables.is_empty() {
                return Err(ValidationError::unsupported(
                    "column reference requires FROM clause",
                ));
            }
            let mut matches = tables
                .iter()
                .filter(|(table, _)| table.column(column).is_some())
                .map(|(table, _)| table.name.clone());
            let first = matches.next();
            if matches.next().is_some() {
                return Err(ValidationError::unsupported("ambiguous column"));
            }
            if let Some(table_name) = first {
                return Ok(());
            }
            Err(ValidationError::ColumnNotFound {
                table: tables[0].0.name.clone(),
                column: column.clone(),
            })
        }
        [table_name, column] => {
            let table = tables.iter().find(|(table, alias)| {
                table.name == *table_name
                    || alias.map(|alias| alias == table_name).unwrap_or(false)
            });
            let table = match table {
                Some((table, _)) => table,
                None => {
                    return Err(ValidationError::unsupported(
                        "qualified column does not match FROM table",
                    ))
                }
            };
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
        Expr::Cast { expr, .. } => expr_contains_identifier(expr),
        Expr::Binary { left, right, .. } => {
            expr_contains_identifier(left) || expr_contains_identifier(right)
        }
        Expr::IsNull { expr, .. } => expr_contains_identifier(expr),
        Expr::Between { expr, low, high, .. } => {
            expr_contains_identifier(expr)
                || expr_contains_identifier(low)
                || expr_contains_identifier(high)
        }
        Expr::InList { expr, list, .. } => {
            expr_contains_identifier(expr)
                || list.iter().any(expr_contains_identifier)
        }
        Expr::Case {
            operand,
            when_thens,
            else_expr,
        } => {
            operand
                .as_ref()
                .map_or(false, |expr| expr_contains_identifier(expr))
                || when_thens.iter().any(|(when, then)| {
                    expr_contains_identifier(when) || expr_contains_identifier(then)
                })
                || else_expr
                    .as_ref()
                    .map_or(false, |expr| expr_contains_identifier(expr))
        }
        Expr::Function { args, .. } => args.iter().any(|arg| match arg {
            FunctionArg::Expr(expr) => expr_contains_identifier(expr),
            FunctionArg::Star => false,
        }),
        Expr::Subquery(_) | Expr::Exists(_) => true,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Catalog, ColumnSchema, SqlType, TableSchema};
    use crate::sql::parser;
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
            modifier: parser::SelectModifier::All,
            items: vec![SelectItem::Wildcard(None)],
            from: vec![parser::FromClause {
                table: "users".to_string(),
                alias: None,
                left_join_on: None,
            }],
            filter: None,
            group_by: Vec::new(),
            having: None,
            order_by: Vec::new(),
            limit: None,
        });
        let err = validate_statement(&stmt, &catalog).expect_err("error");
        assert!(matches!(err, ValidationError::TableNotFound(_)));
    }

    #[test]
    fn rejects_unknown_column() {
        let catalog = catalog_with_users();
        let stmt = Statement::Select(Select {
            modifier: parser::SelectModifier::All,
            items: vec![SelectItem::Expr(Expr::Identifier(vec![
                "age".to_string(),
            ]))],
            from: vec![parser::FromClause {
                table: "users".to_string(),
                alias: None,
                left_join_on: None,
            }],
            filter: None,
            group_by: Vec::new(),
            having: None,
            order_by: Vec::new(),
            limit: None,
        });
        let err = validate_statement(&stmt, &catalog).expect_err("error");
        assert!(matches!(err, ValidationError::ColumnNotFound { .. }));
    }

    #[test]
    fn rejects_group_by_unknown_column() {
        let catalog = catalog_with_users();
        let stmt = Statement::Select(Select {
            modifier: parser::SelectModifier::All,
            items: vec![SelectItem::Wildcard(None)],
            from: vec![parser::FromClause {
                table: "users".to_string(),
                alias: None,
                left_join_on: None,
            }],
            filter: None,
            group_by: vec![Expr::Identifier(vec!["age".to_string()])],
            having: None,
            order_by: Vec::new(),
            limit: None,
        });
        let err = validate_statement(&stmt, &catalog).expect_err("error");
        assert!(matches!(err, ValidationError::ColumnNotFound { .. }));
    }

    #[test]
    fn rejects_having_unknown_column() {
        let catalog = catalog_with_users();
        let stmt = Statement::Select(Select {
            modifier: parser::SelectModifier::All,
            items: vec![SelectItem::Wildcard(None)],
            from: vec![parser::FromClause {
                table: "users".to_string(),
                alias: None,
                left_join_on: None,
            }],
            filter: None,
            group_by: vec![Expr::Identifier(vec!["id".to_string()])],
            having: Some(Expr::Identifier(vec!["age".to_string()])),
            order_by: Vec::new(),
            limit: None,
        });
        let err = validate_statement(&stmt, &catalog).expect_err("error");
        assert!(matches!(err, ValidationError::ColumnNotFound { .. }));
    }

    #[test]
    fn allows_group_by_table_alias() {
        let catalog = catalog_with_users();
        let stmt = Statement::Select(Select {
            modifier: parser::SelectModifier::All,
            items: vec![SelectItem::Wildcard(None)],
            from: vec![parser::FromClause {
                table: "users".to_string(),
                alias: Some("u".to_string()),
                left_join_on: None,
            }],
            filter: None,
            group_by: vec![Expr::Identifier(vec![
                "u".to_string(),
                "id".to_string(),
            ])],
            having: None,
            order_by: Vec::new(),
            limit: None,
        });
        validate_statement(&stmt, &catalog).expect("valid");
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
            modifier: parser::SelectModifier::All,
            items: vec![SelectItem::Expr(Expr::Identifier(vec![
                "id".to_string(),
            ]))],
            from: vec![],
            filter: None,
            group_by: Vec::new(),
            having: None,
            order_by: Vec::new(),
            limit: None,
        });
        let err = validate_statement(&stmt, &catalog).expect_err("error");
        assert!(matches!(err, ValidationError::Unsupported(_)));
    }

    #[test]
    fn allows_select_literal_without_from() {
        let catalog = catalog_with_users();
        let stmt = Statement::Select(Select {
            modifier: parser::SelectModifier::All,
            items: vec![SelectItem::Expr(Expr::Literal(
                super::super::lexer::Literal::Number("1".to_string()),
            ))],
            from: vec![],
            filter: Some(Expr::Binary {
                left: Box::new(Expr::Literal(super::super::lexer::Literal::Number(
                    "1".to_string(),
                ))),
                op: BinaryOp::Eq,
                right: Box::new(Expr::Literal(super::super::lexer::Literal::Number(
                    "1".to_string(),
                ))),
            }),
            group_by: Vec::new(),
            having: None,
            order_by: Vec::new(),
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
