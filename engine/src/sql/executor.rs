use super::parser::{CastType, FunctionModifier, SetOp};
use super::planner::{
    BinaryOp, ExprPlan, FunctionArgPlan, OrderByPlan, PlanNode, ProjectionItem, UnaryOp,
};
use crate::{ColumnMeta, EngineError, Value};
use std::cmp::Ordering;

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
        PlanNode::GroupBy { .. } | PlanNode::Having { .. } => Err(EngineError::InvalidSql),
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
        PlanNode::SetOp {
            left,
            op,
            right,
            all,
            columns,
        } => {
            let left_result = execute_plan(left, db)?;
            let right_result = execute_plan(right, db)?;
            match op {
                SetOp::Union => {
                    let rows = if *all {
                        let mut rows = left_result.rows;
                        rows.extend(right_result.rows);
                        rows
                    } else {
                        union_distinct_rows(left_result.rows, right_result.rows)
                    };
                    Ok(ExecResult {
                        columns: columns.clone(),
                        rows,
                    })
                }
                SetOp::Intersect => {
                    if *all {
                        return Err(EngineError::InvalidSql);
                    }
                    let rows = intersect_distinct_rows(left_result.rows, right_result.rows);
                    Ok(ExecResult {
                        columns: columns.clone(),
                        rows,
                    })
                }
                SetOp::Except => {
                    if *all {
                        return Err(EngineError::InvalidSql);
                    }
                    let rows = except_distinct_rows(left_result.rows, right_result.rows);
                    Ok(ExecResult {
                        columns: columns.clone(),
                        rows,
                    })
                }
            }
        }
    }
}

fn union_distinct_rows(
    left_rows: Vec<Vec<Value>>,
    right_rows: Vec<Vec<Value>>,
) -> Vec<Vec<Value>> {
    let mut result: Vec<Vec<Value>> = Vec::new();
    for row in left_rows.into_iter().chain(right_rows) {
        if !result.iter().any(|existing| rows_equal_with_nulls(existing, &row)) {
            result.push(row);
        }
    }
    result
}

fn rows_equal_with_nulls(left: &[Value], right: &[Value]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter().zip(right.iter()).all(|(l, r)| value_equal_with_nulls(l, r))
}

fn value_equal_with_nulls(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Null, Value::Null) => true,
        _ => values_equal(left, right),
    }
}

fn intersect_distinct_rows(
    left_rows: Vec<Vec<Value>>,
    right_rows: Vec<Vec<Value>>,
) -> Vec<Vec<Value>> {
    let mut result: Vec<Vec<Value>> = Vec::new();
    for row in left_rows {
        if result.iter().any(|existing| rows_equal_with_nulls(existing, &row)) {
            continue;
        }
        if right_rows
            .iter()
            .any(|candidate| rows_equal_with_nulls(candidate, &row))
        {
            result.push(row);
        }
    }
    result
}

fn except_distinct_rows(
    left_rows: Vec<Vec<Value>>,
    right_rows: Vec<Vec<Value>>,
) -> Vec<Vec<Value>> {
    let mut result: Vec<Vec<Value>> = Vec::new();
    for row in left_rows {
        if result.iter().any(|existing| rows_equal_with_nulls(existing, &row)) {
            continue;
        }
        if !right_rows
            .iter()
            .any(|candidate| rows_equal_with_nulls(candidate, &row))
        {
            result.push(row);
        }
    }
    result
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
        ExprPlan::Function {
            name,
            modifier,
            args,
        } => {
            let name = name.to_ascii_lowercase();
            let distinct = matches!(modifier, FunctionModifier::Distinct);
            if name == "count" {
                if args.len() != 1 {
                    return Err(EngineError::InvalidSql);
                }
                match &args[0] {
                    FunctionArgPlan::Star => {
                        if distinct {
                            Err(EngineError::InvalidSql)
                        } else {
                            Ok(Value::Integer(rows.len() as i64))
                        }
                    }
                    FunctionArgPlan::Expr(expr) => {
                        let mut count = 0_i64;
                        let mut seen: Vec<Value> = Vec::new();
                        for row in rows {
                            let value = eval_expr(expr, row)?;
                            if matches!(value, Value::Null) {
                                continue;
                            }
                            if distinct && seen.iter().any(|existing| values_equal(existing, &value))
                            {
                                continue;
                            }
                            if distinct {
                                seen.push(value.clone());
                            }
                            count += 1;
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
                        let mut seen: Vec<Value> = Vec::new();
                        for row in rows {
                            let value = eval_expr(expr, row)?;
                            if matches!(value, Value::Null) {
                                continue;
                            }
                            if distinct && seen.iter().any(|existing| values_equal(existing, &value))
                            {
                                continue;
                            }
                            if distinct {
                                seen.push(value.clone());
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
            } else if name == "sum" {
                if args.len() != 1 {
                    return Err(EngineError::InvalidSql);
                }
                match &args[0] {
                    FunctionArgPlan::Expr(expr) => {
                        let mut total: Option<Value> = None;
                        let mut seen: Vec<Value> = Vec::new();
                        for row in rows {
                            let value = eval_expr(expr, row)?;
                            if matches!(value, Value::Null) {
                                continue;
                            }
                            if distinct && seen.iter().any(|existing| values_equal(existing, &value))
                            {
                                continue;
                            }
                            if distinct {
                                seen.push(value.clone());
                            }
                            total = match total {
                                Some(existing) => Some(numeric_add(existing, value)?),
                                None => Some(value),
                            };
                        }
                        Ok(total.unwrap_or(Value::Null))
                    }
                    FunctionArgPlan::Star => Err(EngineError::InvalidSql),
                }
            } else if name == "min" {
                if args.len() != 1 {
                    return Err(EngineError::InvalidSql);
                }
                match &args[0] {
                    FunctionArgPlan::Expr(expr) => {
                        let mut current: Option<Value> = None;
                        let mut seen: Vec<Value> = Vec::new();
                        for row in rows {
                            let value = eval_expr(expr, row)?;
                            if matches!(value, Value::Null) {
                                continue;
                            }
                            if distinct && seen.iter().any(|existing| values_equal(existing, &value))
                            {
                                continue;
                            }
                            if distinct {
                                seen.push(value.clone());
                            }
                            let replace = match current.as_ref() {
                                None => true,
                                Some(existing) => value_cmp(&value, existing) == Ordering::Less,
                            };
                            if replace {
                                current = Some(value);
                            }
                        }
                        Ok(current.unwrap_or(Value::Null))
                    }
                    FunctionArgPlan::Star => Err(EngineError::InvalidSql),
                }
            } else if name == "max" {
                if args.len() != 1 {
                    return Err(EngineError::InvalidSql);
                }
                match &args[0] {
                    FunctionArgPlan::Expr(expr) => {
                        let mut current: Option<Value> = None;
                        let mut seen: Vec<Value> = Vec::new();
                        for row in rows {
                            let value = eval_expr(expr, row)?;
                            if matches!(value, Value::Null) {
                                continue;
                            }
                            if distinct && seen.iter().any(|existing| values_equal(existing, &value))
                            {
                                continue;
                            }
                            if distinct {
                                seen.push(value.clone());
                            }
                            let replace = match current.as_ref() {
                                None => true,
                                Some(existing) => value_cmp(&value, existing) == Ordering::Greater,
                            };
                            if replace {
                                current = Some(value);
                            }
                        }
                        Ok(current.unwrap_or(Value::Null))
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
                UnaryOp::Pos => numeric_positive(value),
            }
        }
        ExprPlan::Cast { expr, ty } => {
            let value = eval_expr(expr, row)?;
            cast_numeric(value, *ty)
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
                BinaryOp::DivInt => numeric_div_int(left_value, right_value),
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
        ExprPlan::InList {
            expr,
            list,
            negated,
        } => {
            let value = eval_expr(expr, row)?;
            if matches!(value, Value::Null) {
                return Ok(Value::Null);
            }
            let mut has_null = false;
            for item in list {
                let item_value = eval_expr(item, row)?;
                if matches!(item_value, Value::Null) {
                    has_null = true;
                    continue;
                }
                if values_equal(&value, &item_value) {
                    return Ok(Value::Boolean(!*negated));
                }
            }
            if has_null {
                Ok(Value::Null)
            } else {
                Ok(Value::Boolean(*negated))
            }
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
        ExprPlan::Function {
            name,
            modifier,
            args,
        } => {
            if matches!(modifier, FunctionModifier::Distinct) {
                return Err(EngineError::InvalidSql);
            }
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
            } else if name == "nullif" {
                if args.len() != 2 {
                    return Err(EngineError::InvalidSql);
                }
                let first = match &args[0] {
                    FunctionArgPlan::Expr(expr) => eval_expr(expr, row)?,
                    FunctionArgPlan::Star => return Err(EngineError::InvalidSql),
                };
                if matches!(first, Value::Null) {
                    return Ok(Value::Null);
                }
                let second = match &args[1] {
                    FunctionArgPlan::Expr(expr) => eval_expr(expr, row)?,
                    FunctionArgPlan::Star => return Err(EngineError::InvalidSql),
                };
                if matches!(second, Value::Null) {
                    return Ok(first);
                }
                if values_equal(&first, &second) {
                    Ok(Value::Null)
                } else {
                    Ok(first)
                }
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

fn numeric_positive(value: Value) -> Result<Value, EngineError> {
    match value {
        Value::Integer(value) => Ok(Value::Integer(value)),
        Value::Real(value) => Ok(Value::Real(value)),
        Value::Null => Ok(Value::Null),
        _ => Err(EngineError::InvalidSql),
    }
}

fn cast_numeric(value: Value, ty: CastType) -> Result<Value, EngineError> {
    match ty {
        CastType::Signed | CastType::Integer => cast_to_integer(value),
        CastType::Decimal | CastType::Real => cast_to_real(value),
    }
}

fn cast_to_integer(value: Value) -> Result<Value, EngineError> {
    match value {
        Value::Null => Ok(Value::Null),
        Value::Integer(value) => Ok(Value::Integer(value)),
        Value::Real(value) => Ok(Value::Integer(value.trunc() as i64)),
        _ => Err(EngineError::InvalidSql),
    }
}

fn cast_to_real(value: Value) -> Result<Value, EngineError> {
    match value {
        Value::Null => Ok(Value::Null),
        Value::Integer(value) => Ok(Value::Real(value as f64)),
        Value::Real(value) => Ok(Value::Real(value)),
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

fn numeric_div_int(left: Value, right: Value) -> Result<Value, EngineError> {
    match (left, right) {
        (Value::Null, _) | (_, Value::Null) => Ok(Value::Null),
        (Value::Integer(a), Value::Integer(b)) => {
            if b == 0 {
                return Err(EngineError::InvalidSql);
            }
            Ok(Value::Integer(a / b))
        }
        (Value::Integer(a), Value::Real(b)) => {
            let b_int = b.trunc() as i64;
            if b_int == 0 {
                return Err(EngineError::InvalidSql);
            }
            Ok(Value::Integer(a / b_int))
        }
        (Value::Real(a), Value::Integer(b)) => {
            if b == 0 {
                return Err(EngineError::InvalidSql);
            }
            Ok(Value::Integer(a.trunc() as i64 / b))
        }
        (Value::Real(a), Value::Real(b)) => {
            let b_int = b.trunc() as i64;
            if b_int == 0 {
                return Err(EngineError::InvalidSql);
            }
            Ok(Value::Integer(a.trunc() as i64 / b_int))
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
    use crate::{Database, SqlType};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn eval_in_list(value: Value, list: Vec<Value>, negated: bool) -> Value {
        let expr = ExprPlan::InList {
            expr: Box::new(ExprPlan::Literal(value)),
            list: list.into_iter().map(ExprPlan::Literal).collect(),
            negated,
        };
        eval_expr(&expr, &[]).expect("eval")
    }

    fn temp_db_path(label: &str) -> String {
        let mut path = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        path.push(format!("aidb_{label}_{nanos}.db"));
        path.to_string_lossy().to_string()
    }

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
    fn in_list_semantics_with_nulls() {
        let matched = eval_in_list(
            Value::Integer(1),
            vec![Value::Integer(1), Value::Null],
            false,
        );
        assert_eq!(matched, Value::Boolean(true));

        let no_match_no_null = eval_in_list(
            Value::Integer(2),
            vec![Value::Integer(1), Value::Integer(3)],
            false,
        );
        assert_eq!(no_match_no_null, Value::Boolean(false));

        let no_match_with_null = eval_in_list(
            Value::Integer(2),
            vec![Value::Integer(1), Value::Null],
            false,
        );
        assert_eq!(no_match_with_null, Value::Null);
    }

    #[test]
    fn not_in_list_semantics_with_nulls() {
        let matched = eval_in_list(
            Value::Integer(1),
            vec![Value::Integer(1), Value::Null],
            true,
        );
        assert_eq!(matched, Value::Boolean(false));

        let no_match_no_null = eval_in_list(
            Value::Integer(2),
            vec![Value::Integer(1), Value::Integer(3)],
            true,
        );
        assert_eq!(no_match_no_null, Value::Boolean(true));

        let no_match_with_null = eval_in_list(
            Value::Integer(2),
            vec![Value::Integer(1), Value::Null],
            true,
        );
        assert_eq!(no_match_with_null, Value::Null);
    }

    #[test]
    fn union_distinct_deduplicates_with_nulls() {
        let db = Database::create(temp_db_path("union_distinct")).expect("create");
        let plan = PlanNode::SetOp {
            left: Box::new(PlanNode::Values {
                rows: vec![
                    vec![Value::Integer(1)],
                    vec![Value::Null],
                    vec![Value::Null],
                ],
            }),
            op: SetOp::Union,
            right: Box::new(PlanNode::Values {
                rows: vec![
                    vec![Value::Integer(1)],
                    vec![Value::Null],
                    vec![Value::Integer(2)],
                ],
            }),
            all: false,
            columns: vec![ColumnMeta::new("col", SqlType::Integer)],
        };

        let result = execute_plan(&plan, &db).expect("execute");
        assert_eq!(
            result.columns,
            vec![ColumnMeta::new("col", SqlType::Integer)]
        );
        assert_eq!(
            result.rows,
            vec![
                vec![Value::Integer(1)],
                vec![Value::Null],
                vec![Value::Integer(2)],
            ]
        );
    }

    #[test]
    fn union_all_preserves_order_and_duplicates() {
        let db = Database::create(temp_db_path("union_all")).expect("create");
        let plan = PlanNode::SetOp {
            left: Box::new(PlanNode::Values {
                rows: vec![vec![Value::Integer(1)], vec![Value::Integer(1)]],
            }),
            op: SetOp::Union,
            right: Box::new(PlanNode::Values {
                rows: vec![vec![Value::Integer(1)], vec![Value::Integer(2)]],
            }),
            all: true,
            columns: vec![ColumnMeta::new("col", SqlType::Integer)],
        };

        let result = execute_plan(&plan, &db).expect("execute");
        assert_eq!(
            result.rows,
            vec![
                vec![Value::Integer(1)],
                vec![Value::Integer(1)],
                vec![Value::Integer(1)],
                vec![Value::Integer(2)],
            ]
        );
    }

    #[test]
    fn intersect_distinct_deduplicates_with_nulls() {
        let db = Database::create(temp_db_path("intersect_distinct")).expect("create");
        let plan = PlanNode::SetOp {
            left: Box::new(PlanNode::Values {
                rows: vec![
                    vec![Value::Integer(1)],
                    vec![Value::Integer(1)],
                    vec![Value::Null],
                ],
            }),
            op: SetOp::Intersect,
            right: Box::new(PlanNode::Values {
                rows: vec![
                    vec![Value::Integer(1)],
                    vec![Value::Null],
                    vec![Value::Null],
                ],
            }),
            all: false,
            columns: vec![ColumnMeta::new("col", SqlType::Integer)],
        };

        let result = execute_plan(&plan, &db).expect("execute");
        assert_eq!(
            result.rows,
            vec![vec![Value::Integer(1)], vec![Value::Null]]
        );
    }

    #[test]
    fn except_distinct_excludes_with_nulls() {
        let db = Database::create(temp_db_path("except_distinct")).expect("create");
        let plan = PlanNode::SetOp {
            left: Box::new(PlanNode::Values {
                rows: vec![
                    vec![Value::Integer(1)],
                    vec![Value::Integer(1)],
                    vec![Value::Null],
                    vec![Value::Integer(2)],
                ],
            }),
            op: SetOp::Except,
            right: Box::new(PlanNode::Values {
                rows: vec![vec![Value::Null], vec![Value::Integer(3)]],
            }),
            all: false,
            columns: vec![ColumnMeta::new("col", SqlType::Integer)],
        };

        let result = execute_plan(&plan, &db).expect("execute");
        assert_eq!(
            result.rows,
            vec![vec![Value::Integer(1)], vec![Value::Integer(2)]]
        );
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
            modifier: FunctionModifier::All,
            args: vec![FunctionArgPlan::Expr(ExprPlan::Literal(Value::Integer(-5)))],
        };
        let value = eval_expr(&expr, &row).expect("eval");
        assert_eq!(value, Value::Integer(5));

        let expr = ExprPlan::Function {
            name: "ABS".to_string(),
            modifier: FunctionModifier::All,
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
            modifier: FunctionModifier::All,
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
            modifier: FunctionModifier::All,
            args: vec![
                FunctionArgPlan::Expr(ExprPlan::Literal(Value::Null)),
                FunctionArgPlan::Expr(ExprPlan::Literal(Value::Null)),
            ],
        };
        let value = eval_expr(&expr, &row).expect("eval");
        assert_eq!(value, Value::Null);
    }

    #[test]
    fn nullif_returns_null_when_equal() {
        let row = vec![];
        let expr = ExprPlan::Function {
            name: "NULLIF".to_string(),
            modifier: FunctionModifier::All,
            args: vec![
                FunctionArgPlan::Expr(ExprPlan::Literal(Value::Integer(7))),
                FunctionArgPlan::Expr(ExprPlan::Literal(Value::Integer(7))),
            ],
        };
        let value = eval_expr(&expr, &row).expect("eval");
        assert_eq!(value, Value::Null);

        let expr = ExprPlan::Function {
            name: "nullif".to_string(),
            modifier: FunctionModifier::All,
            args: vec![
                FunctionArgPlan::Expr(ExprPlan::Literal(Value::Integer(7))),
                FunctionArgPlan::Expr(ExprPlan::Literal(Value::Integer(9))),
            ],
        };
        let value = eval_expr(&expr, &row).expect("eval");
        assert_eq!(value, Value::Integer(7));

        let expr = ExprPlan::Function {
            name: "nullif".to_string(),
            modifier: FunctionModifier::All,
            args: vec![
                FunctionArgPlan::Expr(ExprPlan::Literal(Value::Null)),
                FunctionArgPlan::Expr(ExprPlan::Literal(Value::Integer(7))),
            ],
        };
        let value = eval_expr(&expr, &row).expect("eval");
        assert_eq!(value, Value::Null);

        let expr = ExprPlan::Function {
            name: "nullif".to_string(),
            modifier: FunctionModifier::All,
            args: vec![
                FunctionArgPlan::Expr(ExprPlan::Literal(Value::Integer(7))),
                FunctionArgPlan::Expr(ExprPlan::Literal(Value::Null)),
            ],
        };
        let value = eval_expr(&expr, &row).expect("eval");
        assert_eq!(value, Value::Integer(7));
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
                modifier: FunctionModifier::All,
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
                modifier: FunctionModifier::All,
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
                modifier: FunctionModifier::All,
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
