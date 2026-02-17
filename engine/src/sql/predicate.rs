use super::parser::{BinaryOp, Expr};
use crate::TableSchema;
use std::collections::BTreeMap;

/// Collect all AND-conjuncts from a predicate expression.
pub fn collect_conjuncts(expr: &Expr) -> Vec<Expr> {
    let mut out = Vec::new();
    collect_conjuncts_inner(expr, &mut out);
    out
}

fn collect_conjuncts_inner(expr: &Expr, out: &mut Vec<Expr>) {
    match expr {
        Expr::Binary {
            left,
            op: BinaryOp::And,
            right,
        } => {
            collect_conjuncts_inner(left, out);
            collect_conjuncts_inner(right, out);
        }
        _ => out.push(expr.clone()),
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableRef<'a> {
    pub schema: &'a TableSchema,
    pub alias: Option<&'a str>,
}

impl<'a> TableRef<'a> {
    pub fn new(schema: &'a TableSchema, alias: Option<&'a str>) -> Self {
        Self { schema, alias }
    }

    fn matches_name(&self, name: &str) -> bool {
        self.schema.name == name || self.alias == Some(name)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TableUsageError {
    AmbiguousColumn(String),
    AmbiguousTable(String),
    UnknownColumn(String),
    UnknownTable(String),
    UnsupportedSubquery,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JoinPredicateGroup {
    pub tables: (usize, usize),
    pub predicates: Vec<Expr>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PredicateGroups {
    pub single_table: Vec<(usize, Expr)>,
    pub joins: Vec<JoinPredicateGroup>,
    pub multi_table: Vec<Expr>,
}

pub fn predicate_table_indices(
    expr: &Expr,
    tables: &[TableRef<'_>],
) -> Result<Vec<usize>, TableUsageError> {
    match expr {
        Expr::Identifier(parts) => match parts.as_slice() {
            [column] => Ok(vec![unqualified_column_index(column, tables)?]),
            [table_name, column] => Ok(vec![qualified_column_index(table_name, column, tables)?]),
            _ => Err(TableUsageError::UnknownColumn(parts.join("."))),
        },
        Expr::Literal(_) => Ok(Vec::new()),
        Expr::Unary { expr, .. } => predicate_table_indices(expr, tables),
        Expr::Binary { left, right, .. } => {
            let mut indices = predicate_table_indices(left, tables)?;
            merge_indices(&mut indices, predicate_table_indices(right, tables)?);
            Ok(indices)
        }
        Expr::IsNull { expr, .. } => predicate_table_indices(expr, tables),
        Expr::Between { expr, low, high, .. } => {
            let mut indices = predicate_table_indices(expr, tables)?;
            merge_indices(&mut indices, predicate_table_indices(low, tables)?);
            merge_indices(&mut indices, predicate_table_indices(high, tables)?);
            Ok(indices)
        }
        Expr::InList { expr, list, .. } => {
            let mut indices = predicate_table_indices(expr, tables)?;
            for item in list {
                merge_indices(&mut indices, predicate_table_indices(item, tables)?);
            }
            Ok(indices)
        }
        Expr::Case { operand, when_thens, else_expr } => {
            let mut indices = Vec::new();
            if let Some(operand) = operand {
                indices = predicate_table_indices(operand, tables)?;
            }
            for (when_expr, then_expr) in when_thens {
                merge_indices(&mut indices, predicate_table_indices(when_expr, tables)?);
                merge_indices(&mut indices, predicate_table_indices(then_expr, tables)?);
            }
            if let Some(else_expr) = else_expr {
                merge_indices(&mut indices, predicate_table_indices(else_expr, tables)?);
            }
            Ok(indices)
        }
        Expr::Function { args, .. } => {
            let mut indices = Vec::new();
            for arg in args {
                match arg {
                    super::parser::FunctionArg::Expr(expr) => {
                        merge_indices(&mut indices, predicate_table_indices(expr, tables)?);
                    }
                    super::parser::FunctionArg::Star => {}
                }
            }
            Ok(indices)
        }
        Expr::Subquery(_) | Expr::Exists(_) => Err(TableUsageError::UnsupportedSubquery),
    }
}

fn merge_indices(into: &mut Vec<usize>, other: Vec<usize>) {
    for idx in other {
        if !into.contains(&idx) {
            into.push(idx);
        }
    }
}

fn unqualified_column_index(
    column: &str,
    tables: &[TableRef<'_>],
) -> Result<usize, TableUsageError> {
    let mut found = None;
    for (idx, table) in tables.iter().enumerate() {
        if table.schema.column(column).is_some() {
            if found.is_some() {
                return Err(TableUsageError::AmbiguousColumn(column.to_string()));
            }
            found = Some(idx);
        }
    }
    found.ok_or_else(|| TableUsageError::UnknownColumn(column.to_string()))
}

fn qualified_column_index(
    table_name: &str,
    column: &str,
    tables: &[TableRef<'_>],
) -> Result<usize, TableUsageError> {
    let mut matches = tables
        .iter()
        .enumerate()
        .filter(|(_, table)| table.matches_name(table_name));
    let (idx, table) = matches
        .next()
        .ok_or_else(|| TableUsageError::UnknownTable(table_name.to_string()))?;
    if matches.next().is_some() {
        return Err(TableUsageError::AmbiguousTable(table_name.to_string()));
    }
    if table.schema.column(column).is_none() {
        return Err(TableUsageError::UnknownColumn(column.to_string()));
    }
    Ok(idx)
}

pub fn group_predicates_by_table_pairs(
    predicates: impl IntoIterator<Item = Expr>,
    tables: &[TableRef<'_>],
) -> Result<PredicateGroups, TableUsageError> {
    let mut single_table = Vec::new();
    let mut join_map: BTreeMap<(usize, usize), Vec<Expr>> = BTreeMap::new();
    let mut multi_table = Vec::new();

    for predicate in predicates {
        match predicate_table_indices(&predicate, tables) {
            Ok(indices) if indices.len() == 1 => {
                single_table.push((indices[0], predicate));
            }
            Ok(indices) if indices.len() == 2 => {
                let mut pair = (indices[0], indices[1]);
                if pair.0 > pair.1 {
                    pair = (pair.1, pair.0);
                }
                join_map.entry(pair).or_default().push(predicate);
            }
            Ok(_) => multi_table.push(predicate),
            Err(TableUsageError::UnsupportedSubquery) => multi_table.push(predicate),
            Err(err) => return Err(err),
        }
    }

    let joins = join_map
        .into_iter()
        .map(|(tables, predicates)| JoinPredicateGroup { tables, predicates })
        .collect();

    Ok(PredicateGroups {
        single_table,
        joins,
        multi_table,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ColumnSchema, SqlType, TableSchema};

    fn ident(name: &str) -> Expr {
        Expr::Identifier(vec![name.to_string()])
    }

    #[test]
    fn collects_flat_and_conjuncts() {
        let expr = Expr::Binary {
            left: Box::new(Expr::Binary {
                left: Box::new(ident("a")),
                op: BinaryOp::And,
                right: Box::new(ident("b")),
            }),
            op: BinaryOp::And,
            right: Box::new(Expr::Binary {
                left: Box::new(ident("c")),
                op: BinaryOp::And,
                right: Box::new(ident("d")),
            }),
        };

        let conjuncts = collect_conjuncts(&expr);
        assert_eq!(
            conjuncts,
            vec![ident("a"), ident("b"), ident("c"), ident("d")]
        );
    }

    #[test]
    fn preserves_non_and_predicates() {
        let or_expr = Expr::Binary {
            left: Box::new(ident("b")),
            op: BinaryOp::Or,
            right: Box::new(ident("c")),
        };
        let expr = Expr::Binary {
            left: Box::new(ident("a")),
            op: BinaryOp::And,
            right: Box::new(or_expr.clone()),
        };

        let conjuncts = collect_conjuncts(&expr);
        assert_eq!(conjuncts, vec![ident("a"), or_expr]);
    }

    #[test]
    fn predicate_table_indices_returns_referenced_tables() {
        let users = TableSchema::new(
            "users",
            vec![
                ColumnSchema::new("id", SqlType::Integer, false),
                ColumnSchema::new("name", SqlType::Text, false),
            ],
        );
        let orders = TableSchema::new(
            "orders",
            vec![ColumnSchema::new("user_id", SqlType::Integer, false)],
        );
        let tables = vec![
            TableRef::new(&users, None),
            TableRef::new(&orders, Some("o")),
        ];

        let expr = Expr::Binary {
            left: Box::new(Expr::Binary {
                left: Box::new(Expr::Identifier(vec!["users".to_string(), "id".to_string()])),
                op: BinaryOp::Eq,
                right: Box::new(Expr::Identifier(vec!["o".to_string(), "user_id".to_string()])),
            }),
            op: BinaryOp::And,
            right: Box::new(Expr::Identifier(vec!["name".to_string()])),
        };

        let indices = predicate_table_indices(&expr, &tables).expect("indices");
        assert_eq!(indices, vec![0, 1]);
    }

    #[test]
    fn predicate_table_indices_reports_ambiguous_columns() {
        let left = TableSchema::new(
            "left",
            vec![ColumnSchema::new("id", SqlType::Integer, false)],
        );
        let right = TableSchema::new(
            "right",
            vec![ColumnSchema::new("id", SqlType::Integer, false)],
        );
        let tables = vec![TableRef::new(&left, None), TableRef::new(&right, None)];
        let expr = Expr::Identifier(vec!["id".to_string()]);

        let error = predicate_table_indices(&expr, &tables).expect_err("ambiguous");
        assert!(matches!(error, TableUsageError::AmbiguousColumn(column) if column == "id"));
    }

    #[test]
    fn groups_join_predicates_by_table_pair() {
        let users = TableSchema::new(
            "users",
            vec![
                ColumnSchema::new("id", SqlType::Integer, false),
                ColumnSchema::new("age", SqlType::Integer, false),
            ],
        );
        let orders = TableSchema::new(
            "orders",
            vec![ColumnSchema::new("user_id", SqlType::Integer, false)],
        );
        let tables = vec![TableRef::new(&users, None), TableRef::new(&orders, None)];

        let join_predicate = Expr::Binary {
            left: Box::new(Expr::Identifier(vec!["users".to_string(), "id".to_string()])),
            op: BinaryOp::Eq,
            right: Box::new(Expr::Identifier(vec!["orders".to_string(), "user_id".to_string()])),
        };
        let single_predicate = Expr::Binary {
            left: Box::new(Expr::Identifier(vec!["users".to_string(), "age".to_string()])),
            op: BinaryOp::Gt,
            right: Box::new(Expr::Literal(super::super::lexer::Literal::Number(
                "21".to_string(),
            ))),
        };

        let groups = group_predicates_by_table_pairs(
            vec![join_predicate.clone(), single_predicate.clone()],
            &tables,
        )
        .expect("groups");

        assert_eq!(groups.single_table, vec![(0, single_predicate)]);
        assert_eq!(groups.multi_table.len(), 0);
        assert_eq!(groups.joins.len(), 1);
        assert_eq!(groups.joins[0].tables, (0, 1));
        assert_eq!(groups.joins[0].predicates, vec![join_predicate]);
    }

    #[test]
    fn join_grouping_reports_ambiguous_reference() {
        let left = TableSchema::new(
            "left",
            vec![ColumnSchema::new("id", SqlType::Integer, false)],
        );
        let right = TableSchema::new(
            "right",
            vec![ColumnSchema::new("id", SqlType::Integer, false)],
        );
        let tables = vec![TableRef::new(&left, None), TableRef::new(&right, None)];
        let predicate = Expr::Binary {
            left: Box::new(Expr::Identifier(vec!["id".to_string()])),
            op: BinaryOp::Eq,
            right: Box::new(Expr::Literal(super::super::lexer::Literal::Number(
                "1".to_string(),
            ))),
        };

        let error =
            group_predicates_by_table_pairs(vec![predicate], &tables).expect_err("ambiguous");
        assert!(matches!(error, TableUsageError::AmbiguousColumn(column) if column == "id"));
    }
}
