use super::planner::{
    BinaryOp, ExprPlan, FunctionArgPlan, OrderByPlan, PlanNode, ProjectionItem, UnaryOp,
};
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
    let has_aggregate = items.iter().any(|item| item.is_aggregate);
    if has_aggregate {
        if !items.iter().all(|item| item.is_aggregate) {
            return Err(EngineError::InvalidSql);
        }
        let mut row = Vec::with_capacity(items.len());
        for item in items {
            row.push(eval_aggregate(&item.expr, &input.rows)?);
        }
        let columns = items
            .iter()
            .map(|item| ColumnMeta::new(item.label.clone(), item.sql_type))
            .collect();
        return Ok(ExecResult {
            columns,
            rows: vec![row],
        });
    }

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

fn eval_aggregate(expr: &ExprPlan, rows: &[Vec<Value>]) -> Result<Value, EngineError> {
    match expr {
        ExprPlan::Function { name, args } => {
            let name = name.to_ascii_lowercase();
            if name == "count" {
                if args.len() != 1 {
                    return Err(EngineError::InvalidSql);
                }
                match &args[0] {
                    FunctionArgPlan::Star => Ok(Value::Integer(rows.len() as i64)),
                    FunctionArgPlan::Expr(expr) => {
                        let mut count = 0_i64;
                        for row in rows {
                            let value = eval_expr(expr, row)?;
                            if !matches!(value, Value::Null) {
                                count += 1;
                            }
                        }
                        Ok(Value::Integer(count))
                    }
                }
            } else if name == "avg" {
                if args.len() != 1 {
                    return Err(EngineError::InvalidSql);
                }
                match &args[0] {
                    FunctionArgPlan::Expr(expr) => {
                        let mut sum = 0.0;
                        let mut count = 0_i64;
                        for row in rows {
                            let value = eval_expr(expr, row)?;
                            if matches!(value, Value::Null) {
                                continue;
                            }
                            let numeric = value_to_f64(value).ok_or(EngineError::InvalidSql)?;
                            sum += numeric;
                            count += 1;
                        }
                        if count == 0 {
                            Ok(Value::Null)
                        } else {
                            Ok(Value::Real(sum / count as f64))
                        }
                    }
                    FunctionArgPlan::Star => Err(EngineError::InvalidSql),
                }
            } else {
                Err(EngineError::InvalidSql)
            }
        }
        _ => Err(EngineError::InvalidSql),
    }
}

fn apply_order(rows: &mut [Vec<Value>], orders: &[OrderByPlan]) -> Result<(), EngineError> {
    if orders.is_empty() {
        return Ok(());
    }
    let mut keyed: Vec<(Vec<Value>, Vec<Value>)> = Vec::with_capacity(rows.len());
    for row in rows.iter() {
        let mut keys = Vec::with_capacity(orders.len());
        for order in orders {
            keys.push(eval_expr(&order.expr, row)?);
        }
        keyed.push((keys, row.clone()));
    }
    keyed.sort_by(|(left_keys, _), (right_keys, _)| {
        for (idx, order) in orders.iter().enumerate() {
            let ordering = value_cmp(
                left_keys.get(idx).unwrap_or(&Value::Null),
                right_keys.get(idx).unwrap_or(&Value::Null),
            );
            if ordering != std::cmp::Ordering::Equal {
                return if matches!(order.direction, super::parser::OrderDirection::Desc) {
                    ordering.reverse()
                } else {
                    ordering
                };
            }
        }
        std::cmp::Ordering::Equal
    });
    for (slot, (_, row)) in rows.iter_mut().zip(keyed.into_iter()) {
        *slot = row;
    }
    Ok(())
}

fn eval_predicate(expr: &ExprPlan, row: &[Value]) -> Result<bool, EngineError> {
    let value = eval_expr(expr, row)?;
    match value {
        Value::Boolean(value) => Ok(value),
        Value::Null => Ok(false),
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
                    Value::Null => Ok(Value::Null),
                    _ => Err(EngineError::InvalidSql),
                },
                UnaryOp::Neg => numeric_negate(value),
            }
        }
        ExprPlan::Binary { left, op, right } => {
            let left_value = eval_expr(left, row)?;
            let right_value = eval_expr(right, row)?;
            match op {
                BinaryOp::Eq => compare_with_nulls(left_value, right_value, |l, r| {
                    Value::Boolean(values_equal(&l, &r))
                }),
                BinaryOp::NotEq => compare_with_nulls(left_value, right_value, |l, r| {
                    Value::Boolean(!values_equal(&l, &r))
                }),
                BinaryOp::Lt => compare_with_nulls(left_value, right_value, |l, r| {
                    Value::Boolean(value_cmp(&l, &r) == std::cmp::Ordering::Less)
                }),
                BinaryOp::Lte => compare_with_nulls(left_value, right_value, |l, r| {
                    Value::Boolean(value_cmp(&l, &r) != std::cmp::Ordering::Greater)
                }),
                BinaryOp::Gt => compare_with_nulls(left_value, right_value, |l, r| {
                    Value::Boolean(value_cmp(&l, &r) == std::cmp::Ordering::Greater)
                }),
                BinaryOp::Gte => compare_with_nulls(left_value, right_value, |l, r| {
                    Value::Boolean(value_cmp(&l, &r) != std::cmp::Ordering::Less)
                }),
                BinaryOp::And => Ok(three_valued_and(left_value, right_value)?),
                BinaryOp::Or => Ok(three_valued_or(left_value, right_value)?),
                BinaryOp::Add => numeric_add(left_value, right_value),
                BinaryOp::Sub => numeric_sub(left_value, right_value),
                BinaryOp::Mul => numeric_mul(left_value, right_value),
                BinaryOp::Div => numeric_div(left_value, right_value),
            }
        }
        ExprPlan::Between {
            expr,
            low,
            high,
            negated,
        } => {
            let value = eval_expr(expr, row)?;
            let low = eval_expr(low, row)?;
            let high = eval_expr(high, row)?;
            if matches!(value, Value::Null) || matches!(low, Value::Null) || matches!(high, Value::Null) {
                return Ok(Value::Null);
            }
            let between = value_cmp(&value, &low) != std::cmp::Ordering::Less
                && value_cmp(&value, &high) != std::cmp::Ordering::Greater;
            Ok(Value::Boolean(if *negated { !between } else { between }))
        }
        ExprPlan::IsNull { expr, negated } => {
            let value = eval_expr(expr, row)?;
            let is_null = matches!(value, Value::Null);
            Ok(Value::Boolean(if *negated { !is_null } else { is_null }))
        }
        ExprPlan::Case {
            operand,
            when_thens,
            else_expr,
        } => {
            if let Some(operand) = operand {
                let base = eval_expr(operand, row)?;
                for (when_expr, then_expr) in when_thens {
                    let when_value = eval_expr(when_expr, row)?;
                    if values_equal(&base, &when_value) {
                        return eval_expr(then_expr, row);
                    }
                }
            } else {
                for (when_expr, then_expr) in when_thens {
                    let condition = eval_expr(when_expr, row)?;
                    match condition {
                        Value::Boolean(true) => {
                            return eval_expr(then_expr, row);
                        }
                        Value::Boolean(false) | Value::Null => {}
                        _ => return Err(EngineError::InvalidSql),
                    }
                }
            }
            if let Some(else_expr) = else_expr {
                eval_expr(else_expr, row)
            } else {
                Ok(Value::Null)
            }
        }
        ExprPlan::Function { name, args } => {
            let name = name.to_ascii_lowercase();
            if name == "abs" {
                if args.len() != 1 {
                    return Err(EngineError::InvalidSql);
                }
                match &args[0] {
                    FunctionArgPlan::Expr(expr) => {
                        let value = eval_expr(expr, row)?;
                        numeric_abs(value)
                    }
                    FunctionArgPlan::Star => Err(EngineError::InvalidSql),
                }
            } else if name == "coalesce" {
                for arg in args {
                    if let FunctionArgPlan::Expr(expr) = arg {
                        let value = eval_expr(expr, row)?;
                        if !matches!(value, Value::Null) {
                            return Ok(value);
                        }
                    }
                }
                Ok(Value::Null)
            } else {
                Err(EngineError::InvalidSql)
            }
        }
        ExprPlan::Subquery(_) | ExprPlan::Exists(_) => Err(EngineError::InvalidSql),
    }
}

fn value_to_bool(value: &Value) -> Result<bool, EngineError> {
    match value {
        Value::Boolean(value) => Ok(*value),
        _ => Err(EngineError::InvalidSql),
    }
}

fn compare_with_nulls<F>(left: Value, right: Value, op: F) -> Result<Value, EngineError>
where
    F: FnOnce(Value, Value) -> Value,
{
    if matches!(left, Value::Null) || matches!(right, Value::Null) {
        Ok(Value::Null)
    } else {
        Ok(op(left, right))
    }
}

fn bool_or_null(value: Value) -> Result<Option<bool>, EngineError> {
    match value {
        Value::Boolean(value) => Ok(Some(value)),
        Value::Null => Ok(None),
        _ => Err(EngineError::InvalidSql),
    }
}

fn three_valued_and(left: Value, right: Value) -> Result<Value, EngineError> {
    let left = bool_or_null(left)?;
    let right = bool_or_null(right)?;
    if left == Some(false) || right == Some(false) {
        Ok(Value::Boolean(false))
    } else if left == Some(true) && right == Some(true) {
        Ok(Value::Boolean(true))
    } else {
        Ok(Value::Null)
    }
}

fn three_valued_or(left: Value, right: Value) -> Result<Value, EngineError> {
    let left = bool_or_null(left)?;
    let right = bool_or_null(right)?;
    if left == Some(true) || right == Some(true) {
        Ok(Value::Boolean(true))
    } else if left == Some(false) && right == Some(false) {
        Ok(Value::Boolean(false))
    } else {
        Ok(Value::Null)
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

fn numeric_negate(value: Value) -> Result<Value, EngineError> {
    match value {
        Value::Integer(value) => Ok(Value::Integer(-value)),
        Value::Real(value) => Ok(Value::Real(-value)),
        Value::Null => Ok(Value::Null),
        _ => Err(EngineError::InvalidSql),
    }
}

fn numeric_abs(value: Value) -> Result<Value, EngineError> {
    match value {
        Value::Integer(value) => Ok(Value::Integer(value.abs())),
        Value::Real(value) => Ok(Value::Real(value.abs())),
        Value::Null => Ok(Value::Null),
        _ => Err(EngineError::InvalidSql),
    }
}

fn numeric_add(left: Value, right: Value) -> Result<Value, EngineError> {
    numeric_op(left, right, |a, b| a + b)
}

fn numeric_sub(left: Value, right: Value) -> Result<Value, EngineError> {
    numeric_op(left, right, |a, b| a - b)
}

fn numeric_mul(left: Value, right: Value) -> Result<Value, EngineError> {
    numeric_op(left, right, |a, b| a * b)
}

fn numeric_div(left: Value, right: Value) -> Result<Value, EngineError> {
    match (left, right) {
        (Value::Null, _) | (_, Value::Null) => Ok(Value::Null),
        (Value::Integer(a), Value::Integer(b)) => {
            if b == 0 {
                return Err(EngineError::InvalidSql);
            }
            if a % b == 0 {
                Ok(Value::Integer(a / b))
            } else {
                Ok(Value::Real(a as f64 / b as f64))
            }
        }
        (Value::Integer(a), Value::Real(b)) => {
            if b == 0.0 {
                return Err(EngineError::InvalidSql);
            }
            Ok(Value::Real(a as f64 / b))
        }
        (Value::Real(a), Value::Integer(b)) => {
            if b == 0 {
                return Err(EngineError::InvalidSql);
            }
            Ok(Value::Real(a / b as f64))
        }
        (Value::Real(a), Value::Real(b)) => {
            if b == 0.0 {
                return Err(EngineError::InvalidSql);
            }
            Ok(Value::Real(a / b))
        }
        _ => Err(EngineError::InvalidSql),
    }
}

fn numeric_op<F>(left: Value, right: Value, op: F) -> Result<Value, EngineError>
where
    F: FnOnce(f64, f64) -> f64,
{
    match (left, right) {
        (Value::Null, _) | (_, Value::Null) => Ok(Value::Null),
        (Value::Integer(a), Value::Integer(b)) => {
            let result = op(a as f64, b as f64);
            if result.fract() == 0.0 {
                Ok(Value::Integer(result as i64))
            } else {
                Ok(Value::Real(result))
            }
        }
        (Value::Integer(a), Value::Real(b)) => Ok(Value::Real(op(a as f64, b))),
        (Value::Real(a), Value::Integer(b)) => Ok(Value::Real(op(a, b as f64))),
        (Value::Real(a), Value::Real(b)) => Ok(Value::Real(op(a, b))),
        _ => Err(EngineError::InvalidSql),
    }
}

fn value_to_f64(value: Value) -> Option<f64> {
    match value {
        Value::Integer(value) => Some(value as f64),
        Value::Real(value) => Some(value),
        _ => None,
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
    use crate::sql::planner::{BinaryOp, ExprPlan, FunctionArgPlan, ProjectionItem};
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
    fn comparisons_with_null_return_null() {
        let expr = ExprPlan::Binary {
            left: Box::new(ExprPlan::Literal(Value::Null)),
            op: BinaryOp::Eq,
            right: Box::new(ExprPlan::Literal(Value::Integer(1))),
        };
        let row = vec![];
        let value = eval_expr(&expr, &row).expect("eval");
        assert_eq!(value, Value::Null);
    }

    #[test]
    fn predicates_treat_null_as_false() {
        let expr = ExprPlan::Literal(Value::Null);
        let row = vec![];
        let value = eval_predicate(&expr, &row).expect("eval");
        assert!(!value);
    }

    #[test]
    fn boolean_logic_with_nulls() {
        let row = vec![];
        let expr = ExprPlan::Binary {
            left: Box::new(ExprPlan::Literal(Value::Boolean(true))),
            op: BinaryOp::And,
            right: Box::new(ExprPlan::Literal(Value::Null)),
        };
        let value = eval_expr(&expr, &row).expect("eval");
        assert_eq!(value, Value::Null);

        let expr = ExprPlan::Binary {
            left: Box::new(ExprPlan::Literal(Value::Boolean(false))),
            op: BinaryOp::And,
            right: Box::new(ExprPlan::Literal(Value::Null)),
        };
        let value = eval_expr(&expr, &row).expect("eval");
        assert_eq!(value, Value::Boolean(false));

        let expr = ExprPlan::Binary {
            left: Box::new(ExprPlan::Literal(Value::Boolean(true))),
            op: BinaryOp::Or,
            right: Box::new(ExprPlan::Literal(Value::Null)),
        };
        let value = eval_expr(&expr, &row).expect("eval");
        assert_eq!(value, Value::Boolean(true));

        let expr = ExprPlan::Binary {
            left: Box::new(ExprPlan::Literal(Value::Boolean(false))),
            op: BinaryOp::Or,
            right: Box::new(ExprPlan::Literal(Value::Null)),
        };
        let value = eval_expr(&expr, &row).expect("eval");
        assert_eq!(value, Value::Null);
    }

    #[test]
    fn between_respects_nulls_and_is_inclusive() {
        let row = vec![];
        let expr = ExprPlan::Between {
            expr: Box::new(ExprPlan::Literal(Value::Integer(2))),
            low: Box::new(ExprPlan::Literal(Value::Integer(1))),
            high: Box::new(ExprPlan::Literal(Value::Integer(2))),
            negated: false,
        };
        let value = eval_expr(&expr, &row).expect("eval");
        assert_eq!(value, Value::Boolean(true));

        let expr = ExprPlan::Between {
            expr: Box::new(ExprPlan::Literal(Value::Null)),
            low: Box::new(ExprPlan::Literal(Value::Integer(1))),
            high: Box::new(ExprPlan::Literal(Value::Integer(2))),
            negated: false,
        };
        let value = eval_expr(&expr, &row).expect("eval");
        assert_eq!(value, Value::Null);

        let expr = ExprPlan::Between {
            expr: Box::new(ExprPlan::Literal(Value::Integer(2))),
            low: Box::new(ExprPlan::Literal(Value::Null)),
            high: Box::new(ExprPlan::Literal(Value::Integer(3))),
            negated: false,
        };
        let value = eval_expr(&expr, &row).expect("eval");
        assert_eq!(value, Value::Null);
    }

    #[test]
    fn is_null_returns_boolean() {
        let row = vec![];
        let expr = ExprPlan::IsNull {
            expr: Box::new(ExprPlan::Literal(Value::Null)),
            negated: false,
        };
        let value = eval_expr(&expr, &row).expect("eval");
        assert_eq!(value, Value::Boolean(true));

        let expr = ExprPlan::IsNull {
            expr: Box::new(ExprPlan::Literal(Value::Null)),
            negated: true,
        };
        let value = eval_expr(&expr, &row).expect("eval");
        assert_eq!(value, Value::Boolean(false));

        let expr = ExprPlan::IsNull {
            expr: Box::new(ExprPlan::Literal(Value::Integer(1))),
            negated: true,
        };
        let value = eval_expr(&expr, &row).expect("eval");
        assert_eq!(value, Value::Boolean(true));
    }

    #[test]
    fn case_returns_first_match_and_handles_else() {
        let row = vec![];
        let expr = ExprPlan::Case {
            operand: None,
            when_thens: vec![
                (
                    ExprPlan::Literal(Value::Boolean(false)),
                    ExprPlan::Literal(Value::Integer(1)),
                ),
                (
                    ExprPlan::Literal(Value::Boolean(true)),
                    ExprPlan::Literal(Value::Integer(2)),
                ),
                (
                    ExprPlan::Literal(Value::Boolean(true)),
                    ExprPlan::Literal(Value::Integer(3)),
                ),
            ],
            else_expr: Some(Box::new(ExprPlan::Literal(Value::Integer(4)))),
        };
        let value = eval_expr(&expr, &row).expect("eval");
        assert_eq!(value, Value::Integer(2));

        let expr = ExprPlan::Case {
            operand: None,
            when_thens: vec![(
                ExprPlan::Literal(Value::Boolean(false)),
                ExprPlan::Literal(Value::Integer(1)),
            )],
            else_expr: Some(Box::new(ExprPlan::Literal(Value::Integer(9)))),
        };
        let value = eval_expr(&expr, &row).expect("eval");
        assert_eq!(value, Value::Integer(9));
    }

    #[test]
    fn case_handles_null_conditions() {
        let row = vec![];
        let expr = ExprPlan::Case {
            operand: None,
            when_thens: vec![(
                ExprPlan::Literal(Value::Null),
                ExprPlan::Literal(Value::Integer(1)),
            )],
            else_expr: None,
        };
        let value = eval_expr(&expr, &row).expect("eval");
        assert_eq!(value, Value::Null);
    }

    #[test]
    fn simple_case_matches_on_operand() {
        let row = vec![];
        let expr = ExprPlan::Case {
            operand: Some(Box::new(ExprPlan::Literal(Value::Integer(2)))),
            when_thens: vec![
                (
                    ExprPlan::Literal(Value::Integer(1)),
                    ExprPlan::Literal(Value::Text("one".to_string())),
                ),
                (
                    ExprPlan::Literal(Value::Integer(2)),
                    ExprPlan::Literal(Value::Text("two".to_string())),
                ),
            ],
            else_expr: Some(Box::new(ExprPlan::Literal(Value::Text(
                "other".to_string(),
            )))),
        };
        let value = eval_expr(&expr, &row).expect("eval");
        assert_eq!(value, Value::Text("two".to_string()));
    }

    #[test]
    fn abs_returns_absolute_value() {
        let row = vec![];
        let expr = ExprPlan::Function {
            name: "abs".to_string(),
            args: vec![FunctionArgPlan::Expr(ExprPlan::Literal(Value::Integer(-5)))],
        };
        let value = eval_expr(&expr, &row).expect("eval");
        assert_eq!(value, Value::Integer(5));

        let expr = ExprPlan::Function {
            name: "ABS".to_string(),
            args: vec![FunctionArgPlan::Expr(ExprPlan::Literal(Value::Real(-3.5)))],
        };
        let value = eval_expr(&expr, &row).expect("eval");
        assert_eq!(value, Value::Real(3.5));
    }

    #[test]
    fn arithmetic_with_null_returns_null() {
        let row = vec![];
        let expr = ExprPlan::Binary {
            left: Box::new(ExprPlan::Literal(Value::Null)),
            op: BinaryOp::Add,
            right: Box::new(ExprPlan::Literal(Value::Integer(1))),
        };
        let value = eval_expr(&expr, &row).expect("eval");
        assert_eq!(value, Value::Null);

        let expr = ExprPlan::Binary {
            left: Box::new(ExprPlan::Literal(Value::Integer(2))),
            op: BinaryOp::Mul,
            right: Box::new(ExprPlan::Literal(Value::Null)),
        };
        let value = eval_expr(&expr, &row).expect("eval");
        assert_eq!(value, Value::Null);

        let expr = ExprPlan::Unary {
            op: UnaryOp::Neg,
            expr: Box::new(ExprPlan::Literal(Value::Null)),
        };
        let value = eval_expr(&expr, &row).expect("eval");
        assert_eq!(value, Value::Null);
    }

    #[test]
    fn division_by_zero_returns_error() {
        let row = vec![];
        let expr = ExprPlan::Binary {
            left: Box::new(ExprPlan::Literal(Value::Integer(10))),
            op: BinaryOp::Div,
            right: Box::new(ExprPlan::Literal(Value::Integer(0))),
        };
        assert!(eval_expr(&expr, &row).is_err());

        let expr = ExprPlan::Binary {
            left: Box::new(ExprPlan::Literal(Value::Real(10.0))),
            op: BinaryOp::Div,
            right: Box::new(ExprPlan::Literal(Value::Real(0.0))),
        };
        assert!(eval_expr(&expr, &row).is_err());
    }

    #[test]
    fn coalesce_returns_first_non_null() {
        let row = vec![];
        let expr = ExprPlan::Function {
            name: "coalesce".to_string(),
            args: vec![
                FunctionArgPlan::Expr(ExprPlan::Literal(Value::Null)),
                FunctionArgPlan::Expr(ExprPlan::Literal(Value::Integer(7))),
                FunctionArgPlan::Expr(ExprPlan::Literal(Value::Integer(9))),
            ],
        };
        let value = eval_expr(&expr, &row).expect("eval");
        assert_eq!(value, Value::Integer(7));

        let expr = ExprPlan::Function {
            name: "COALESCE".to_string(),
            args: vec![
                FunctionArgPlan::Expr(ExprPlan::Literal(Value::Null)),
                FunctionArgPlan::Expr(ExprPlan::Literal(Value::Null)),
            ],
        };
        let value = eval_expr(&expr, &row).expect("eval");
        assert_eq!(value, Value::Null);
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
            is_aggregate: false,
        }];
        let result = project_rows(input, &items).expect("project");
        assert_eq!(
            result.columns,
            vec![ColumnMeta::new("id", crate::SqlType::Integer)]
        );
    }

    #[test]
    fn count_star_aggregates_rows() {
        let input = ExecResult {
            columns: vec![ColumnMeta::new("id", crate::SqlType::Integer)],
            rows: vec![vec![Value::Integer(1)], vec![Value::Null], vec![Value::Integer(3)]],
        };
        let items = vec![ProjectionItem {
            expr: ExprPlan::Function {
                name: "count".to_string(),
                args: vec![FunctionArgPlan::Star],
            },
            label: "count".to_string(),
            sql_type: crate::SqlType::Integer,
            is_aggregate: true,
        }];

        let result = project_rows(input, &items).expect("project");
        assert_eq!(result.rows, vec![vec![Value::Integer(3)]]);
    }

    #[test]
    fn avg_ignores_nulls() {
        let input = ExecResult {
            columns: vec![ColumnMeta::new("id", crate::SqlType::Integer)],
            rows: vec![vec![Value::Integer(1)], vec![Value::Null], vec![Value::Integer(3)]],
        };
        let items = vec![ProjectionItem {
            expr: ExprPlan::Function {
                name: "avg".to_string(),
                args: vec![FunctionArgPlan::Expr(ExprPlan::Column(
                    super::super::planner::ColumnRef {
                        name: "id".to_string(),
                        index: 0,
                        sql_type: crate::SqlType::Integer,
                    },
                ))],
            },
            label: "avg".to_string(),
            sql_type: crate::SqlType::Real,
            is_aggregate: true,
        }];

        let result = project_rows(input, &items).expect("project");
        assert_eq!(result.rows, vec![vec![Value::Real(2.0)]]);
    }

    #[test]
    fn avg_returns_null_when_all_null() {
        let input = ExecResult {
            columns: vec![ColumnMeta::new("id", crate::SqlType::Integer)],
            rows: vec![vec![Value::Null], vec![Value::Null]],
        };
        let items = vec![ProjectionItem {
            expr: ExprPlan::Function {
                name: "AVG".to_string(),
                args: vec![FunctionArgPlan::Expr(ExprPlan::Column(
                    super::super::planner::ColumnRef {
                        name: "id".to_string(),
                        index: 0,
                        sql_type: crate::SqlType::Integer,
                    },
                ))],
            },
            label: "avg".to_string(),
            sql_type: crate::SqlType::Real,
            is_aggregate: true,
        }];

        let result = project_rows(input, &items).expect("project");
        assert_eq!(result.rows, vec![vec![Value::Null]]);
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
        apply_order(&mut rows, &[order]).expect("order");
        assert_eq!(rows[0], vec![Value::Integer(3)]);
    }

    #[test]
    fn order_applies_multiple_keys_with_mixed_directions() {
        let mut rows = vec![
            vec![Value::Integer(1), Value::Integer(2)],
            vec![Value::Integer(1), Value::Integer(3)],
            vec![Value::Integer(2), Value::Integer(1)],
            vec![Value::Integer(2), Value::Integer(4)],
        ];
        let orders = vec![
            OrderByPlan {
                expr: ExprPlan::Column(super::super::planner::ColumnRef {
                    name: "a".to_string(),
                    index: 0,
                    sql_type: crate::SqlType::Integer,
                }),
                direction: OrderDirection::Asc,
            },
            OrderByPlan {
                expr: ExprPlan::Column(super::super::planner::ColumnRef {
                    name: "b".to_string(),
                    index: 1,
                    sql_type: crate::SqlType::Integer,
                }),
                direction: OrderDirection::Desc,
            },
        ];
        apply_order(&mut rows, &orders).expect("order");
        assert_eq!(
            rows,
            vec![
                vec![Value::Integer(1), Value::Integer(3)],
                vec![Value::Integer(1), Value::Integer(2)],
                vec![Value::Integer(2), Value::Integer(4)],
                vec![Value::Integer(2), Value::Integer(1)],
            ]
        );
    }
}
