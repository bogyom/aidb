use super::lexer::Literal as AstLiteral;
use super::parser::{
    Expr as AstExpr, OrderDirection, Select, SelectItem, Statement, UnaryOp as AstUnaryOp,
};
use super::parser::BinaryOp as AstBinaryOp;
use crate::{Catalog, SqlType, TableSchema, Value};

#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    pub root: PlanNode,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PlanNode {
    Values { rows: Vec<Vec<Value>> },
    Scan(TableScan),
    Filter { predicate: ExprPlan, input: Box<PlanNode> },
    Order { by: OrderByPlan, input: Box<PlanNode> },
    Limit { limit: usize, input: Box<PlanNode> },
    Projection { items: Vec<ProjectionItem>, input: Box<PlanNode> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableScan {
    pub table: String,
    pub columns: Vec<ColumnRef>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnRef {
    pub name: String,
    pub index: usize,
    pub sql_type: SqlType,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ProjectionItem {
    pub expr: ExprPlan,
    pub label: String,
    pub sql_type: SqlType,
}

#[derive(Debug, Clone, PartialEq)]
pub struct OrderByPlan {
    pub expr: ExprPlan,
    pub direction: OrderDirection,
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExprPlan {
    Column(ColumnRef),
    Literal(Value),
    Unary { op: UnaryOp, expr: Box<ExprPlan> },
    Binary { left: Box<ExprPlan>, op: BinaryOp, right: Box<ExprPlan> },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Eq,
    And,
    Or,
}

#[derive(Debug, Clone, PartialEq)]
pub struct PlanError {
    pub message: String,
}

impl PlanError {
    fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

pub fn plan_statement(statement: &Statement, catalog: &Catalog) -> Result<Option<Plan>, PlanError> {
    match statement {
        Statement::Select(select) => plan_select(select, catalog).map(Some),
        _ => Ok(None),
    }
}

pub fn plan_select(select: &Select, catalog: &Catalog) -> Result<Plan, PlanError> {
    let table = select
        .from
        .as_ref()
        .and_then(|name| catalog.table(name));

    let input = if let Some(table) = table {
        PlanNode::Scan(TableScan {
            table: table.name.clone(),
            columns: table_columns(table),
        })
    } else {
        PlanNode::Values { rows: vec![Vec::new()] }
    };

    let mut node = input;

    if let Some(filter) = &select.filter {
        let predicate = plan_expr(filter, table)?;
        node = PlanNode::Filter {
            predicate,
            input: Box::new(node),
        };
    }

    if let Some(order_by) = &select.order_by {
        let expr = plan_expr(&order_by.expr, table)?;
        node = PlanNode::Order {
            by: OrderByPlan {
                expr,
                direction: order_by.direction.clone(),
            },
            input: Box::new(node),
        };
    }

    if let Some(limit) = select.limit {
        node = PlanNode::Limit {
            limit: limit as usize,
            input: Box::new(node),
        };
    }

    let items = plan_projection_items(&select.items, table)?;
    node = PlanNode::Projection {
        items,
        input: Box::new(node),
    };

    Ok(Plan { root: node })
}

fn plan_projection_items(
    items: &[SelectItem],
    table: Option<&TableSchema>,
) -> Result<Vec<ProjectionItem>, PlanError> {
    let mut planned = Vec::new();
    for item in items {
        match item {
            SelectItem::Wildcard(prefix) => {
                let table = table.ok_or_else(|| {
                    PlanError::new("wildcard select requires FROM clause")
                })?;
                if let Some(prefix) = prefix {
                    if prefix != &table.name {
                        return Err(PlanError::new(
                            "qualified wildcard does not match FROM table",
                        ));
                    }
                }
                for column in table_columns(table) {
                    planned.push(ProjectionItem {
                        label: column.name.clone(),
                        sql_type: column.sql_type,
                        expr: ExprPlan::Column(column),
                    });
                }
            }
            SelectItem::Expr(expr) => {
                let planned_expr = plan_expr(expr, table)?;
                let sql_type = expr_plan_type(&planned_expr);
                let label = label_for_expr(&planned_expr);
                planned.push(ProjectionItem {
                    label,
                    sql_type,
                    expr: planned_expr,
                });
            }
        }
    }
    Ok(planned)
}

fn label_for_expr(expr: &ExprPlan) -> String {
    match expr {
        ExprPlan::Column(column) => column.name.clone(),
        ExprPlan::Literal(value) => match value {
            Value::Null => "NULL".to_string(),
            Value::Integer(v) => v.to_string(),
            Value::Real(v) => v.to_string(),
            Value::Text(v) => v.clone(),
            Value::Boolean(v) => v.to_string(),
        },
        _ => "expr".to_string(),
    }
}

fn table_columns(table: &TableSchema) -> Vec<ColumnRef> {
    table
        .columns
        .iter()
        .enumerate()
        .map(|(index, column)| ColumnRef {
            name: column.name.clone(),
            index,
            sql_type: column.sql_type,
        })
        .collect()
}

fn plan_expr(expr: &AstExpr, table: Option<&TableSchema>) -> Result<ExprPlan, PlanError> {
    match expr {
        AstExpr::Identifier(parts) => {
            let table = table.ok_or_else(|| {
                PlanError::new("column reference requires FROM clause")
            })?;
            match parts.as_slice() {
                [column] => resolve_column(table, column).map(ExprPlan::Column),
                [table_name, column] => {
                    if table_name != &table.name {
                        return Err(PlanError::new(
                            "qualified column does not match FROM table",
                        ));
                    }
                    resolve_column(table, column).map(ExprPlan::Column)
                }
                _ => Err(PlanError::new("unsupported identifier")),
            }
        }
        AstExpr::Literal(literal) => {
            let value = literal_to_value(literal)?;
            Ok(ExprPlan::Literal(value))
        }
        AstExpr::Unary { op, expr } => {
            let expr = plan_expr(expr, table)?;
            let op = match op {
                AstUnaryOp::Not => UnaryOp::Not,
            };
            Ok(ExprPlan::Unary {
                op,
                expr: Box::new(expr),
            })
        }
        AstExpr::Binary { left, op, right } => {
            let left = plan_expr(left, table)?;
            let right = plan_expr(right, table)?;
            let op = match op {
                AstBinaryOp::Eq => BinaryOp::Eq,
                AstBinaryOp::And => BinaryOp::And,
                AstBinaryOp::Or => BinaryOp::Or,
            };
            Ok(ExprPlan::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            })
        }
    }
}

fn resolve_column(table: &TableSchema, name: &str) -> Result<ColumnRef, PlanError> {
    table
        .columns
        .iter()
        .position(|column| column.name == name)
        .map(|index| ColumnRef {
            name: name.to_string(),
            index,
            sql_type: table.columns[index].sql_type,
        })
        .ok_or_else(|| PlanError::new("unknown column"))
}

fn literal_to_value(literal: &AstLiteral) -> Result<Value, PlanError> {
    match literal {
        AstLiteral::Number(raw) => {
            if raw.contains('.') {
                let value = raw
                    .parse::<f64>()
                    .map_err(|_| PlanError::new("invalid number"))?;
                Ok(Value::Real(value))
            } else {
                let value = raw
                    .parse::<i64>()
                    .map_err(|_| PlanError::new("invalid number"))?;
                Ok(Value::Integer(value))
            }
        }
        AstLiteral::String(value) => Ok(Value::Text(value.clone())),
        AstLiteral::Boolean(value) => Ok(Value::Boolean(*value)),
        AstLiteral::Null => Ok(Value::Null),
    }
}

fn expr_plan_type(expr: &ExprPlan) -> SqlType {
    match expr {
        ExprPlan::Column(column) => column.sql_type,
        ExprPlan::Literal(value) => match value {
            Value::Integer(_) => SqlType::Integer,
            Value::Real(_) => SqlType::Real,
            Value::Text(_) => SqlType::Text,
            Value::Boolean(_) => SqlType::Boolean,
            Value::Null => SqlType::Text,
        },
        ExprPlan::Unary { .. } => SqlType::Boolean,
        ExprPlan::Binary { .. } => SqlType::Boolean,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Catalog, ColumnSchema, SqlType, TableSchema};
    use super::super::parser;

    fn catalog_with_users() -> Catalog {
        let mut catalog = Catalog::new();
        catalog.add_table(TableSchema::new(
            "users",
            vec![
                ColumnSchema::new("id", SqlType::Integer, false),
                ColumnSchema::new("name", SqlType::Text, false),
            ],
        ));
        catalog
    }

    #[test]
    fn planner_builds_scan_and_filter_nodes() {
        let catalog = catalog_with_users();
        let statements = parser::parse("SELECT name FROM users WHERE id = 1")
            .expect("parse");
        let select = match &statements[0] {
            Statement::Select(select) => select,
            _ => panic!("expected select"),
        };
        let plan = plan_select(select, &catalog).expect("plan");
        match plan.root {
            PlanNode::Projection { input, .. } => match *input {
                PlanNode::Filter { input, .. } => match *input {
                    PlanNode::Scan(_) => {}
                    _ => panic!("expected scan"),
                },
                _ => panic!("expected filter"),
            },
            _ => panic!("expected projection"),
        }
    }

    #[test]
    fn planner_resolves_column_indices() {
        let catalog = catalog_with_users();
        let statements = parser::parse("SELECT name FROM users WHERE id = 1")
            .expect("parse");
        let select = match &statements[0] {
            Statement::Select(select) => select,
            _ => panic!("expected select"),
        };
        let plan = plan_select(select, &catalog).expect("plan");
        let projection = match plan.root {
            PlanNode::Projection { items, .. } => items,
            _ => panic!("expected projection"),
        };
        match &projection[0].expr {
            ExprPlan::Column(column) => assert_eq!(column.index, 1),
            _ => panic!("expected column"),
        }
    }
}
