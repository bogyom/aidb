use super::planner::{BinaryOp, ExprPlan, OrderByPlan, PlanNode, ProjectionItem, UnaryOp};
use crate::{ColumnMeta, EngineError, Value};

#[derive(Debug, Clone, PartialEq)]
pub struct ExecResult {
    pub columns: Vec<ColumnMeta>,
    pub rows: Vec<Vec<Value>>,
}

pub fn execute_plan(plan: &PlanNode, db: &crate::Database) -> Result<ExecResult, EngineError> {
    match plan {
        PlanNode::Values { rows } => Ok(ExecResult {
            columns: Vec::new(),
            rows: rows.clone(),
        }),
        PlanNode::Scan(scan) => {
            let columns = scan
                .columns
                .iter()
                .map(|col| ColumnMeta::new(col.name.clone(), col.sql_type))
                .collect();
            let mut rows = Vec::new();
            for row in db.scan_table_rows_unlocked(&scan.table)? {
                rows.push(row.values);
            }
            Ok(ExecResult { columns, rows })
        }
        PlanNode::Filter { predicate, input } => {
            let mut result = execute_plan(input, db)?;
            let mut filtered = Vec::new();
            for row in result.rows.into_iter() {
                if eval_predicate(predicate, &row)? {
                    filtered.push(row);
                }
            }
            result.rows = filtered;
            Ok(result)
        }
        PlanNode::Order { by, input } => {
            let mut result = execute_plan(input, db)?;
            apply_order(&mut result.rows, by)?;
            Ok(result)
        }
        PlanNode::Limit { limit, input } => {
            let mut result = execute_plan(input, db)?;
            if result.rows.len() > *limit {
                result.rows.truncate(*limit);
            }
            Ok(result)
        }
        PlanNode::Projection { items, input } => {
            let result = execute_plan(input, db)?;
            project_rows(result, items)
        }
    }
}

fn project_rows(
    input: ExecResult,
    items: &[ProjectionItem],
) -> Result<ExecResult, EngineError> {
    let mut rows = Vec::with_capacity(input.rows.len());
    for row in &input.rows {
        let mut projected = Vec::with_capacity(items.len());
        for item in items {
            projected.push(eval_expr(&item.expr, row)?);
        }
        rows.push(projected);
    }

    let columns = items
        .iter()
        .map(|item| ColumnMeta::new(item.label.clone(), item.sql_type))
        .collect();
    Ok(ExecResult { columns, rows })
}

fn apply_order(rows: &mut [Vec<Value>], order: &OrderByPlan) -> Result<(), EngineError> {
    let mut keyed: Vec<(Value, Vec<Value>)> = Vec::with_capacity(rows.len());
    for row in rows.iter() {
        let key = eval_expr(&order.expr, row)?;
        keyed.push((key, row.clone()));
    }
    keyed.sort_by(|(left, _), (right, _)| value_cmp(left, right));
    if matches!(order.direction, super::parser::OrderDirection::Desc) {
        keyed.reverse();
    }
    for (slot, (_, row)) in rows.iter_mut().zip(keyed.into_iter()) {
        *slot = row;
    }
    Ok(())
}

fn eval_predicate(expr: &ExprPlan, row: &[Value]) -> Result<bool, EngineError> {
    let value = eval_expr(expr, row)?;
    match value {
        Value::Boolean(value) => Ok(value),
        _ => Err(EngineError::InvalidSql),
    }
}

fn eval_expr(expr: &ExprPlan, row: &[Value]) -> Result<Value, EngineError> {
    match expr {
        ExprPlan::Column(column) => row
            .get(column.index)
            .cloned()
            .ok_or(EngineError::InvalidSql),
        ExprPlan::Literal(value) => Ok(value.clone()),
        ExprPlan::Unary { op, expr } => {
            let value = eval_expr(expr, row)?;
            match op {
                UnaryOp::Not => match value {
                    Value::Boolean(value) => Ok(Value::Boolean(!value)),
                    _ => Err(EngineError::InvalidSql),
                },
            }
        }
        ExprPlan::Binary { left, op, right } => {
            let left_value = eval_expr(left, row)?;
            let right_value = eval_expr(right, row)?;
            match op {
                BinaryOp::Eq => Ok(Value::Boolean(values_equal(&left_value, &right_value))),
                BinaryOp::And => Ok(Value::Boolean(
                    value_to_bool(&left_value)? && value_to_bool(&right_value)?,
                )),
                BinaryOp::Or => Ok(Value::Boolean(
                    value_to_bool(&left_value)? || value_to_bool(&right_value)?,
                )),
            }
        }
    }
}

fn value_to_bool(value: &Value) -> Result<bool, EngineError> {
    match value {
        Value::Boolean(value) => Ok(*value),
        _ => Err(EngineError::InvalidSql),
    }
}

fn values_equal(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Null, _) | (_, Value::Null) => false,
        (Value::Integer(a), Value::Integer(b)) => a == b,
        (Value::Real(a), Value::Real(b)) => a == b,
        (Value::Integer(a), Value::Real(b)) => (*a as f64) == *b,
        (Value::Real(a), Value::Integer(b)) => *a == (*b as f64),
        (Value::Text(a), Value::Text(b)) => a == b,
        (Value::Boolean(a), Value::Boolean(b)) => a == b,
        _ => false,
    }
}

fn value_cmp(left: &Value, right: &Value) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (left, right) {
        (Value::Null, Value::Null) => Ordering::Equal,
        (Value::Null, _) => Ordering::Greater,
        (_, Value::Null) => Ordering::Less,
        (Value::Integer(a), Value::Integer(b)) => a.cmp(b),
        (Value::Real(a), Value::Real(b)) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
        (Value::Integer(a), Value::Real(b)) => (*a as f64)
            .partial_cmp(b)
            .unwrap_or(Ordering::Equal),
        (Value::Real(a), Value::Integer(b)) => a
            .partial_cmp(&(*b as f64))
            .unwrap_or(Ordering::Equal),
        (Value::Text(a), Value::Text(b)) => a.cmp(b),
        (Value::Boolean(a), Value::Boolean(b)) => a.cmp(b),
        _ => value_rank(left).cmp(&value_rank(right)),
    }
}

fn value_rank(value: &Value) -> u8 {
    match value {
        Value::Integer(_) => 0,
        Value::Real(_) => 1,
        Value::Text(_) => 2,
        Value::Boolean(_) => 3,
        Value::Null => 4,
    }
}

pub fn to_query_result(result: ExecResult) -> crate::QueryResult {
    crate::QueryResult {
        columns: result.columns,
        rows: result.rows,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sql::planner::{BinaryOp, ExprPlan, ProjectionItem};
    use crate::sql::parser::OrderDirection;

    #[test]
    fn predicate_evaluates_eq() {
        let expr = ExprPlan::Binary {
            left: Box::new(ExprPlan::Literal(Value::Integer(1))),
            op: BinaryOp::Eq,
            right: Box::new(ExprPlan::Literal(Value::Integer(1))),
        };
        let row = vec![];
        let value = eval_expr(&expr, &row).expect("eval");
        assert_eq!(value, Value::Boolean(true));
    }

    #[test]
    fn projection_uses_labels() {
        let input = ExecResult {
            columns: vec![ColumnMeta::new("id", crate::SqlType::Integer)],
            rows: vec![vec![Value::Integer(1)]],
        };
        let items = vec![ProjectionItem {
            expr: ExprPlan::Column(super::super::planner::ColumnRef {
                name: "id".to_string(),
                index: 0,
                sql_type: crate::SqlType::Integer,
            }),
            label: "id".to_string(),
            sql_type: crate::SqlType::Integer,
        }];
        let result = project_rows(input, &items).expect("project");
        assert_eq!(
            result.columns,
            vec![ColumnMeta::new("id", crate::SqlType::Integer)]
        );
    }

    #[test]
    fn order_applies_descending() {
        let mut rows = vec![
            vec![Value::Integer(1)],
            vec![Value::Integer(3)],
            vec![Value::Integer(2)],
        ];
        let order = OrderByPlan {
            expr: ExprPlan::Column(super::super::planner::ColumnRef {
                name: "id".to_string(),
                index: 0,
                sql_type: crate::SqlType::Integer,
            }),
            direction: OrderDirection::Desc,
        };
        apply_order(&mut rows, &order).expect("order");
        assert_eq!(rows[0], vec![Value::Integer(3)]);
    }
}
