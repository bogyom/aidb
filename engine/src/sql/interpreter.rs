use super::lexer::Literal;
use super::parser::{
    BinaryOp, Expr, FunctionArg, OrderBy, OrderDirection, Select, SelectItem, UnaryOp,
};
use crate::{ColumnMeta, Database, EngineError, QueryResult, SqlType, TableSchema, Value};
use std::cell::RefCell;
use std::cmp::Ordering;
use std::collections::HashMap;
use std::rc::Rc;

#[derive(Clone, Copy)]
struct RowContext<'a> {
    table: &'a TableSchema,
    alias: Option<&'a str>,
    row: &'a [Value],
}

#[derive(Clone)]
struct EvalContext<'a> {
    db: &'a Database,
    current: Option<RowContext<'a>>,
    outer: Option<RowContext<'a>>,
    subquery_cache: Rc<RefCell<HashMap<usize, Value>>>,
    correlation_probe: Option<Rc<RefCell<bool>>>,
}

pub fn execute_select(select: &Select, db: &Database) -> Result<QueryResult, EngineError> {
    let cache = Rc::new(RefCell::new(HashMap::new()));
    execute_select_inner(select, db, None, cache, None)
}

fn execute_select_inner(
    select: &Select,
    db: &Database,
    outer: Option<RowContext<'_>>,
    subquery_cache: Rc<RefCell<HashMap<usize, Value>>>,
    correlation_probe: Option<Rc<RefCell<bool>>>,
) -> Result<QueryResult, EngineError> {
    let (table, alias) = match &select.from {
        Some(from) => (
            Some(
                db.catalog()
                    .table(&from.table)
                    .ok_or(EngineError::TableNotFound)?,
            ),
            from.alias.as_deref(),
        ),
        None => (None, None),
    };

    let columns = build_columns(select, table)?;
    let has_aggregate = select
        .items
        .iter()
        .any(|item| matches!(item, SelectItem::Expr(expr) if expr_is_aggregate(expr)));

    if has_aggregate {
        let rows = if let Some(table) = table {
            db.scan_table_rows_unlocked(&table.name)?
        } else {
            Vec::new()
        };
        let mut context = EvalContext {
            db,
            current: None,
            outer,
            subquery_cache: subquery_cache.clone(),
            correlation_probe: correlation_probe.clone(),
        };
        let aggregated = evaluate_aggregates(select, table, alias, &rows, &mut context)?;
        return Ok(QueryResult {
            columns,
            rows: vec![aggregated],
        });
    }

    let input_rows = if let Some(table) = table {
        db.scan_table_rows_unlocked(&table.name)?
            .into_iter()
            .map(|row| row.values)
            .collect::<Vec<_>>()
    } else {
        vec![Vec::new()]
    };

    let mut collected: Vec<(Vec<Value>, Vec<Value>)> = Vec::new();

    for row in &input_rows {
        let current = table.map(|table| RowContext {
            table,
            alias,
            row,
        });
        let context = EvalContext {
            db,
            current,
            outer,
            subquery_cache: subquery_cache.clone(),
            correlation_probe: correlation_probe.clone(),
        };

        if let Some(filter) = &select.filter {
            if !eval_predicate(filter, &context)? {
                continue;
            }
        }

        let mut output_row = Vec::with_capacity(select.items.len());
        for item in &select.items {
            match item {
                SelectItem::Wildcard(_) => {
                    if table.is_some() {
                        output_row.extend_from_slice(row);
                    } else {
                        return Err(EngineError::InvalidSql);
                    }
                }
                SelectItem::Expr(expr) => output_row.push(eval_expr(expr, &context)?),
            }
        }

        let mut order_keys = Vec::with_capacity(select.order_by.len());
        for order in &select.order_by {
            let key = order_key(order, &output_row, &context)?;
            order_keys.push(key);
        }

        collected.push((order_keys, output_row));
    }

    if !select.order_by.is_empty() {
        collected.sort_by(|(left_keys, _), (right_keys, _)| {
            for (idx, order) in select.order_by.iter().enumerate() {
                let ordering = value_cmp(
                    left_keys.get(idx).unwrap_or(&Value::Null),
                    right_keys.get(idx).unwrap_or(&Value::Null),
                );
                if ordering != Ordering::Equal {
                    return if matches!(order.direction, OrderDirection::Desc) {
                        ordering.reverse()
                    } else {
                        ordering
                    };
                }
            }
            Ordering::Equal
        });
    }

    let mut rows: Vec<Vec<Value>> = collected.into_iter().map(|(_, row)| row).collect();
    if let Some(limit) = select.limit {
        rows.truncate(limit as usize);
    }

    Ok(QueryResult { columns, rows })
}

fn build_columns(
    select: &Select,
    table: Option<&TableSchema>,
) -> Result<Vec<ColumnMeta>, EngineError> {
    let mut columns = Vec::new();
    for item in &select.items {
        match item {
            SelectItem::Wildcard(_) => {
                let table = table.ok_or(EngineError::InvalidSql)?;
                for column in &table.columns {
                    columns.push(ColumnMeta::new(column.name.clone(), column.sql_type));
                }
            }
            SelectItem::Expr(expr) => {
                let label = label_for_expr(expr);
                let sql_type = expr_result_type(expr, table)?;
                columns.push(ColumnMeta::new(label, sql_type));
            }
        }
    }
    Ok(columns)
}

fn evaluate_aggregates(
    select: &Select,
    table: Option<&TableSchema>,
    alias: Option<&str>,
    rows: &[crate::RowRecord],
    context: &mut EvalContext<'_>,
) -> Result<Vec<Value>, EngineError> {
    let mut aggregates = Vec::new();
    for item in &select.items {
        match item {
            SelectItem::Expr(expr) if expr_is_aggregate(expr) => {
                aggregates.push(AggregateState::new(expr)?);
            }
            _ => return Err(EngineError::InvalidSql),
        }
    }

    for row in rows {
        let current = table.map(|table| RowContext {
            table,
            alias,
            row: &row.values,
        });
        let row_context = EvalContext {
            db: context.db,
            current,
            outer: context.outer,
            subquery_cache: context.subquery_cache.clone(),
            correlation_probe: context.correlation_probe.clone(),
        };

        if let Some(filter) = &select.filter {
            if !eval_predicate(filter, &row_context)? {
                continue;
            }
        }

        for state in &mut aggregates {
            state.consume(expr_arg_for_aggregate(state.expr(), &row_context)?);
        }
    }

    Ok(aggregates.into_iter().map(|state| state.finish()).collect())
}

struct AggregateState {
    expr: Expr,
    kind: AggregateKind,
}

enum AggregateKind {
    Count { count: i64 },
    Avg { sum: f64, count: i64 },
}

impl AggregateState {
    fn new(expr: &Expr) -> Result<Self, EngineError> {
        match expr {
            Expr::Function { name, args: _ } => {
                let name = name.to_ascii_lowercase();
                if name == "count" {
                    Ok(Self {
                        expr: expr.clone(),
                        kind: AggregateKind::Count { count: 0 },
                    })
                } else if name == "avg" {
                    Ok(Self {
                        expr: expr.clone(),
                        kind: AggregateKind::Avg { sum: 0.0, count: 0 },
                    })
                } else {
                    Err(EngineError::InvalidSql)
                }
            }
            _ => Err(EngineError::InvalidSql),
        }
    }

    fn expr(&self) -> &Expr {
        &self.expr
    }

    fn consume(&mut self, value: Option<Value>) {
        match &mut self.kind {
            AggregateKind::Count { count } => {
                if value.is_some() {
                    *count += 1;
                }
            }
            AggregateKind::Avg { sum, count } => {
                if let Some(value) = value.and_then(value_to_f64) {
                    *sum += value;
                    *count += 1;
                }
            }
        }
    }

    fn finish(self) -> Value {
        match self.kind {
            AggregateKind::Count { count } => Value::Integer(count),
            AggregateKind::Avg { sum, count } => {
                if count == 0 {
                    Value::Null
                } else {
                    Value::Real(sum / count as f64)
                }
            }
        }
    }
}

fn expr_arg_for_aggregate(
    expr: &Expr,
    context: &EvalContext<'_>,
) -> Result<Option<Value>, EngineError> {
    match expr {
        Expr::Function { name, args } => {
            let name = name.to_ascii_lowercase();
            if name == "count" {
                if args.is_empty() {
                    return Ok(None);
                }
                match &args[0] {
                    FunctionArg::Star => Ok(Some(Value::Integer(1))),
                    FunctionArg::Expr(expr) => {
                        let value = eval_expr(expr, context)?;
                        Ok(if matches!(value, Value::Null) {
                            None
                        } else {
                            Some(value)
                        })
                    }
                }
            } else if name == "avg" {
                if args.len() != 1 {
                    return Err(EngineError::InvalidSql);
                }
                match &args[0] {
                    FunctionArg::Expr(expr) => {
                        let value = eval_expr(expr, context)?;
                        Ok(if matches!(value, Value::Null) {
                            None
                        } else {
                            Some(value)
                        })
                    }
                    FunctionArg::Star => Err(EngineError::InvalidSql),
                }
            } else {
                Err(EngineError::InvalidSql)
            }
        }
        _ => Err(EngineError::InvalidSql),
    }
}

fn expr_is_aggregate(expr: &Expr) -> bool {
    match expr {
        Expr::Function { name, .. } => {
            let name = name.to_ascii_lowercase();
            name == "count" || name == "avg"
        }
        _ => false,
    }
}

fn eval_predicate(expr: &Expr, context: &EvalContext<'_>) -> Result<bool, EngineError> {
    match eval_expr(expr, context)? {
        Value::Boolean(value) => Ok(value),
        _ => Err(EngineError::InvalidSql),
    }
}

fn eval_expr(expr: &Expr, context: &EvalContext<'_>) -> Result<Value, EngineError> {
    match expr {
        Expr::Identifier(parts) => resolve_identifier(parts, context),
        Expr::Literal(literal) => literal_to_value(literal),
        Expr::Unary { op, expr } => {
            let value = eval_expr(expr, context)?;
            match op {
                UnaryOp::Not => match value {
                    Value::Boolean(value) => Ok(Value::Boolean(!value)),
                    _ => Err(EngineError::InvalidSql),
                },
                UnaryOp::Neg => numeric_negate(value),
            }
        }
        Expr::Binary { left, op, right } => {
            let left = eval_expr(left, context)?;
            let right = eval_expr(right, context)?;
            eval_binary(op, left, right)
        }
        Expr::IsNull { expr, negated } => {
            let value = eval_expr(expr, context)?;
            let is_null = matches!(value, Value::Null);
            Ok(Value::Boolean(if *negated { !is_null } else { is_null }))
        }
        Expr::Between {
            expr,
            low,
            high,
            negated,
        } => {
            let value = eval_expr(expr, context)?;
            let low = eval_expr(low, context)?;
            let high = eval_expr(high, context)?;
            let between = value_cmp(&value, &low) != Ordering::Less
                && value_cmp(&value, &high) != Ordering::Greater;
            Ok(Value::Boolean(if *negated { !between } else { between }))
        }
        Expr::Case {
            operand,
            when_thens,
            else_expr,
        } => {
            if let Some(operand) = operand {
                let base = eval_expr(operand, context)?;
                for (when_expr, then_expr) in when_thens {
                    let when_value = eval_expr(when_expr, context)?;
                    if values_equal(&base, &when_value) {
                        return eval_expr(then_expr, context);
                    }
                }
            } else {
                for (when_expr, then_expr) in when_thens {
                    let condition = eval_expr(when_expr, context)?;
                    let is_true = match condition {
                        Value::Boolean(value) => value,
                        _ => return Err(EngineError::InvalidSql),
                    };
                    if is_true {
                        return eval_expr(then_expr, context);
                    }
                }
            }
            if let Some(else_expr) = else_expr {
                eval_expr(else_expr, context)
            } else {
                Ok(Value::Null)
            }
        }
        Expr::Function { name, args } => {
            let name = name.to_ascii_lowercase();
            if name == "abs" {
                if args.len() != 1 {
                    return Err(EngineError::InvalidSql);
                }
                match &args[0] {
                    FunctionArg::Expr(expr) => {
                        let value = eval_expr(expr, context)?;
                        numeric_abs(value)
                    }
                    FunctionArg::Star => Err(EngineError::InvalidSql),
                }
            } else if expr_is_aggregate(expr) {
                Err(EngineError::InvalidSql)
            } else {
                Err(EngineError::InvalidSql)
            }
        }
        Expr::Subquery(select) => {
            let key = select.as_ref() as *const Select as usize;
            if let Some(value) = context.subquery_cache.borrow().get(&key) {
                return Ok(value.clone());
            }

            let probe = Rc::new(RefCell::new(false));
            let result = execute_select_inner(
                select,
                context.db,
                context.current,
                context.subquery_cache.clone(),
                Some(probe.clone()),
            )?;
            let value = scalar_from_query(result)?;
            if !*probe.borrow() {
                context.subquery_cache.borrow_mut().insert(key, value.clone());
            }
            Ok(value)
        }
        Expr::Exists(select) => {
            let result = execute_select_inner(
                select,
                context.db,
                context.current,
                context.subquery_cache.clone(),
                None,
            )?;
            Ok(Value::Boolean(!result.rows.is_empty()))
        }
    }
}

fn scalar_from_query(result: QueryResult) -> Result<Value, EngineError> {
    if result.rows.len() != 1 || result.rows[0].len() != 1 {
        return Err(EngineError::InvalidSql);
    }
    Ok(result.rows[0][0].clone())
}

fn resolve_identifier(
    parts: &[String],
    context: &EvalContext<'_>,
) -> Result<Value, EngineError> {
    match parts {
        [column] => {
            if let Some(current) = context.current {
                if let Some(value) = column_value(current, column) {
                    return Ok(value);
                }
            }
            if let Some(outer) = context.outer {
                if let Some(value) = column_value(outer, column) {
                    mark_correlated(context);
                    return Ok(value);
                }
            }
            Err(EngineError::InvalidSql)
        }
        [table_name, column] => {
            if let Some(current) = context.current {
                if matches_table(table_name, current) {
                    if let Some(value) = column_value(current, column) {
                        return Ok(value);
                    }
                }
            }
            if let Some(outer) = context.outer {
                if matches_table(table_name, outer) {
                    if let Some(value) = column_value(outer, column) {
                        mark_correlated(context);
                        return Ok(value);
                    }
                }
            }
            Err(EngineError::InvalidSql)
        }
        _ => Err(EngineError::InvalidSql),
    }
}

fn mark_correlated(context: &EvalContext<'_>) {
    if let Some(probe) = &context.correlation_probe {
        *probe.borrow_mut() = true;
    }
}

fn matches_table(name: &str, row: RowContext<'_>) -> bool {
    if let Some(alias) = row.alias {
        name == alias
    } else {
        name == row.table.name
    }
}

fn column_value(row: RowContext<'_>, column: &str) -> Option<Value> {
    row.table
        .columns
        .iter()
        .position(|col| col.name == column)
        .and_then(|index| row.row.get(index).cloned())
}

fn eval_binary(op: &BinaryOp, left: Value, right: Value) -> Result<Value, EngineError> {
    match op {
        BinaryOp::Eq => Ok(Value::Boolean(values_equal(&left, &right))),
        BinaryOp::NotEq => Ok(Value::Boolean(!values_equal(&left, &right))),
        BinaryOp::Lt => Ok(Value::Boolean(value_cmp(&left, &right) == Ordering::Less)),
        BinaryOp::Lte => Ok(Value::Boolean(
            value_cmp(&left, &right) != Ordering::Greater,
        )),
        BinaryOp::Gt => Ok(Value::Boolean(value_cmp(&left, &right) == Ordering::Greater)),
        BinaryOp::Gte => Ok(Value::Boolean(
            value_cmp(&left, &right) != Ordering::Less,
        )),
        BinaryOp::And => Ok(Value::Boolean(
            value_to_bool(&left)? && value_to_bool(&right)?,
        )),
        BinaryOp::Or => Ok(Value::Boolean(
            value_to_bool(&left)? || value_to_bool(&right)?,
        )),
        BinaryOp::Add => numeric_add(left, right),
        BinaryOp::Sub => numeric_sub(left, right),
        BinaryOp::Mul => numeric_mul(left, right),
        BinaryOp::Div => numeric_div(left, right),
    }
}

fn value_to_bool(value: &Value) -> Result<bool, EngineError> {
    match value {
        Value::Boolean(value) => Ok(*value),
        _ => Err(EngineError::InvalidSql),
    }
}

fn literal_to_value(literal: &Literal) -> Result<Value, EngineError> {
    match literal {
        Literal::Number(raw) => {
            if raw.contains('.') {
                let value = raw
                    .parse::<f64>()
                    .map_err(|_| EngineError::InvalidSql)?;
                Ok(Value::Real(value))
            } else {
                let value = raw
                    .parse::<i64>()
                    .map_err(|_| EngineError::InvalidSql)?;
                Ok(Value::Integer(value))
            }
        }
        Literal::String(value) => Ok(Value::Text(value.clone())),
        Literal::Boolean(value) => Ok(Value::Boolean(*value)),
        Literal::Null => Ok(Value::Null),
    }
}

fn value_to_f64(value: Value) -> Option<f64> {
    match value {
        Value::Integer(value) => Some(value as f64),
        Value::Real(value) => Some(value),
        _ => None,
    }
}

fn numeric_negate(value: Value) -> Result<Value, EngineError> {
    match value {
        Value::Integer(value) => Ok(Value::Integer(-value)),
        Value::Real(value) => Ok(Value::Real(-value)),
        _ => Err(EngineError::InvalidSql),
    }
}

fn numeric_abs(value: Value) -> Result<Value, EngineError> {
    match value {
        Value::Integer(value) => Ok(Value::Integer(value.abs())),
        Value::Real(value) => Ok(Value::Real(value.abs())),
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
        (Value::Integer(a), Value::Real(b)) => Ok(Value::Real(a as f64 / b)),
        (Value::Real(a), Value::Integer(b)) => Ok(Value::Real(a / b as f64)),
        (Value::Real(a), Value::Real(b)) => Ok(Value::Real(a / b)),
        _ => Err(EngineError::InvalidSql),
    }
}

fn numeric_op<F>(left: Value, right: Value, op: F) -> Result<Value, EngineError>
where
    F: FnOnce(f64, f64) -> f64,
{
    match (left, right) {
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

fn expr_result_type(expr: &Expr, table: Option<&TableSchema>) -> Result<SqlType, EngineError> {
    match expr {
        Expr::Identifier(parts) => {
            let table = table.ok_or(EngineError::InvalidSql)?;
            let column = match parts.as_slice() {
                [name] => table.column(name),
                [_, name] => table.column(name),
                _ => None,
            };
            column
                .map(|col| col.sql_type)
                .ok_or(EngineError::InvalidSql)
        }
        Expr::Literal(literal) => match literal {
            Literal::Number(raw) => Ok(if raw.contains('.') {
                SqlType::Real
            } else {
                SqlType::Integer
            }),
            Literal::String(_) => Ok(SqlType::Text),
            Literal::Boolean(_) => Ok(SqlType::Boolean),
            Literal::Null => Ok(SqlType::Text),
        },
        Expr::Unary { op, expr } => match op {
            UnaryOp::Not => Ok(SqlType::Boolean),
            UnaryOp::Neg => expr_result_type(expr, table),
        },
        Expr::Binary { op, left, right } => match op {
            BinaryOp::And
            | BinaryOp::Or
            | BinaryOp::Eq
            | BinaryOp::NotEq
            | BinaryOp::Lt
            | BinaryOp::Lte
            | BinaryOp::Gt
            | BinaryOp::Gte => Ok(SqlType::Boolean),
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul => {
                let left = expr_result_type(left, table)?;
                let right = expr_result_type(right, table)?;
                if left == SqlType::Real || right == SqlType::Real {
                    Ok(SqlType::Real)
                } else {
                    Ok(SqlType::Integer)
                }
            }
            BinaryOp::Div => Ok(SqlType::Real),
        },
        Expr::Between { .. } => Ok(SqlType::Boolean),
        Expr::IsNull { .. } => Ok(SqlType::Boolean),
        Expr::Case { when_thens, else_expr, .. } => {
            if let Some((_, then_expr)) = when_thens.first() {
                expr_result_type(then_expr, table)
            } else if let Some(else_expr) = else_expr {
                expr_result_type(else_expr, table)
            } else {
                Ok(SqlType::Text)
            }
        }
        Expr::Function { name, .. } => {
            let name = name.to_ascii_lowercase();
            if name == "count" {
                Ok(SqlType::Integer)
            } else if name == "avg" {
                Ok(SqlType::Real)
            } else if name == "abs" {
                Ok(SqlType::Integer)
            } else {
                Ok(SqlType::Text)
            }
        }
        Expr::Subquery(select) => {
            if let Some(item) = select.items.first() {
                match item {
                    SelectItem::Expr(expr) => expr_result_type(expr, table),
                    SelectItem::Wildcard(_) => Ok(SqlType::Text),
                }
            } else {
                Ok(SqlType::Text)
            }
        }
        Expr::Exists(_) => Ok(SqlType::Boolean),
    }
}

fn label_for_expr(expr: &Expr) -> String {
    match expr {
        Expr::Identifier(parts) => parts.last().cloned().unwrap_or_else(|| "expr".to_string()),
        Expr::Literal(literal) => match literal {
            Literal::Number(raw) => raw.clone(),
            Literal::String(value) => value.clone(),
            Literal::Boolean(value) => value.to_string(),
            Literal::Null => "NULL".to_string(),
        },
        _ => "expr".to_string(),
    }
}

fn order_key(
    order: &OrderBy,
    output_row: &[Value],
    context: &EvalContext<'_>,
) -> Result<Value, EngineError> {
    if let Expr::Literal(Literal::Number(value)) = &order.expr {
        if let Ok(index) = value.parse::<usize>() {
            if index >= 1 && index <= output_row.len() {
                return Ok(output_row[index - 1].clone());
            }
        }
    }
    eval_expr(&order.expr, context)
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

fn value_cmp(left: &Value, right: &Value) -> Ordering {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sql::parser::{parse, Statement};
    use std::cell::RefCell;
    use std::collections::HashMap;
    use std::path::PathBuf;
    use std::rc::Rc;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_db_path(label: &str) -> PathBuf {
        let mut path = std::env::temp_dir();
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        path.push(format!("aidb_{label}_{nanos}.db"));
        path
    }

    #[test]
    fn uncorrelated_scalar_subquery_uses_cached_value() {
        let path = temp_db_path("uncorrelated_scalar_subquery_cache");
        let mut db =
            Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        db.execute("CREATE TABLE users (id INTEGER)")
            .expect("create table");
        db.execute("INSERT INTO users (id) VALUES (1)")
            .expect("insert row");
        db.execute("INSERT INTO users (id) VALUES (2)")
            .expect("insert row");

        let statements =
            parse("SELECT (SELECT COUNT(*) FROM users) FROM users").expect("parse");
        let select = match &statements[0] {
            Statement::Select(select) => select,
            _ => panic!("expected select statement"),
        };
        let subquery = match &select.items[0] {
            SelectItem::Expr(Expr::Subquery(subquery)) => subquery,
            _ => panic!("expected scalar subquery"),
        };

        let cache = Rc::new(RefCell::new(HashMap::new()));
        let key = subquery.as_ref() as *const Select as usize;
        cache
            .borrow_mut()
            .insert(key, Value::Integer(42));

        let result = execute_select_inner(select, &db, None, cache, None).expect("select");
        assert_eq!(
            result.rows,
            vec![
                vec![Value::Integer(42)],
                vec![Value::Integer(42)],
            ]
        );
    }
}
