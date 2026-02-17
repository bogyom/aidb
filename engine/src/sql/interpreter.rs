use super::lexer::Literal;
use super::parser::{
    BinaryOp, Expr, FunctionArg, OrderBy, OrderDirection, Select, SelectItem, SetExpr,
    SetOp, UnaryOp,
};
use super::predicate::{collect_conjuncts, predicate_table_indices, TableRef, TableUsageError};
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
    current: Vec<RowContext<'a>>,
    outer: Vec<RowContext<'a>>,
    subquery_cache: Rc<RefCell<HashMap<usize, Value>>>,
    correlation_probe: Option<Rc<RefCell<bool>>>,
}

struct TableContext<'a> {
    table: &'a TableSchema,
    alias: Option<&'a str>,
    rows: Vec<Vec<Value>>,
}

#[derive(Clone)]
struct JoinPredicate {
    indices: Vec<usize>,
    expr: Expr,
}

pub fn execute_select(select: &Select, db: &Database) -> Result<QueryResult, EngineError> {
    let cache = Rc::new(RefCell::new(HashMap::new()));
    execute_select_inner(select, db, Vec::new(), cache, None)
}

pub fn execute_set_expr(expr: &SetExpr, db: &Database) -> Result<QueryResult, EngineError> {
    let cache = Rc::new(RefCell::new(HashMap::new()));
    execute_set_expr_inner(expr, db, cache)
}

fn execute_select_inner(
    select: &Select,
    db: &Database,
    outer: Vec<RowContext<'_>>,
    subquery_cache: Rc<RefCell<HashMap<usize, Value>>>,
    correlation_probe: Option<Rc<RefCell<bool>>>,
) -> Result<QueryResult, EngineError> {
    let mut tables = Vec::new();
    for from in &select.from {
        let table = db
            .catalog()
            .table(&from.table)
            .ok_or(EngineError::TableNotFound)?;
        tables.push(TableContext {
            table,
            alias: from.alias.as_deref(),
            rows: db
                .scan_table_rows_unlocked(&table.name)?
                .into_iter()
                .map(|row| row.values)
                .collect(),
        });
    }

    let columns = build_columns(select, &tables)?;
    let has_aggregate = select
        .items
        .iter()
        .any(|item| matches!(item, SelectItem::Expr(expr) if expr_is_aggregate(expr)));

    if tables.is_empty() {
        let context = EvalContext {
            db,
            current: Vec::new(),
            outer,
            subquery_cache: subquery_cache.clone(),
            correlation_probe: correlation_probe.clone(),
        };

        if has_aggregate {
            let mut aggregates = Vec::new();
            for item in &select.items {
                match item {
                    SelectItem::Expr(expr) if expr_is_aggregate(expr) => {
                        aggregates.push(AggregateState::new(expr)?);
                    }
                    _ => return Err(EngineError::InvalidSql),
                }
            }

            let should_consume = match select.filter.as_ref() {
                Some(filter) => eval_predicate(filter, &context)?,
                None => true,
            };
            if should_consume {
                for state in aggregates.iter_mut() {
                    state.consume(expr_arg_for_aggregate(state.expr(), &context)?);
                }
            }

            return Ok(QueryResult {
                columns,
                rows: vec![aggregates
                    .into_iter()
                    .map(|state| state.finish())
                    .collect()],
            });
        }

        if let Some(filter) = select.filter.as_ref() {
            if !eval_predicate(filter, &context)? {
                return Ok(QueryResult {
                    columns,
                    rows: Vec::new(),
                });
            }
        }

        let mut output_row = Vec::with_capacity(select.items.len());
        for item in &select.items {
            match item {
                SelectItem::Wildcard(_) => return Err(EngineError::InvalidSql),
                SelectItem::Expr(expr) => output_row.push(eval_expr(expr, &context)?),
            }
        }

        for order in &select.order_by {
            let _ = order_key(order, &output_row, &context)?;
        }

        let mut rows = vec![output_row];
        if let Some(limit) = select.limit {
            rows.truncate(limit as usize);
        }

        return Ok(QueryResult { columns, rows });
    }

    let remaining_filter = apply_filter_pushdown(
        &mut tables,
        select.filter.as_ref(),
        &outer,
        db,
        &subquery_cache,
        correlation_probe.clone(),
    )?;
    let table_refs: Vec<TableRef<'_>> = tables
        .iter()
        .map(|table| TableRef::new(table.table, table.alias))
        .collect();
    let (join_predicates, remaining_filter) =
        split_join_predicates(remaining_filter.as_ref(), &table_refs)?;

    if has_aggregate {
        let mut context = EvalContext {
            db,
            current: Vec::new(),
            outer,
            subquery_cache: subquery_cache.clone(),
            correlation_probe: correlation_probe.clone(),
        };
        let aggregated = evaluate_aggregates(remaining_filter.as_ref(), select, &tables, &mut context)?;
        return Ok(QueryResult {
            columns,
            rows: vec![aggregated],
        });
    }

    let mut collected: Vec<(Vec<Value>, Vec<Value>)> = Vec::new();
    let join_order = join_order_for_query(&tables, &join_predicates);
    let mut selected: Vec<Option<&[Value]>> = vec![None; tables.len()];
    execute_join_incremental(
        &tables,
        &join_order,
        0,
        &mut selected,
        &outer,
        db,
        &subquery_cache,
        &correlation_probe,
        &join_predicates,
        &remaining_filter,
        select,
        &mut collected,
    )?;

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

fn execute_set_expr_inner(
    expr: &SetExpr,
    db: &Database,
    subquery_cache: Rc<RefCell<HashMap<usize, Value>>>,
) -> Result<QueryResult, EngineError> {
    match expr {
        SetExpr::Select(select) => execute_select_inner(
            select,
            db,
            Vec::new(),
            subquery_cache,
            None,
        ),
        SetExpr::SetOp {
            left,
            op,
            right,
            all,
        } => {
            let left_result = execute_set_expr_inner(left, db, subquery_cache.clone())?;
            let right_result = execute_set_expr_inner(right, db, subquery_cache)?;
            if left_result.columns.len() != right_result.columns.len() {
                return Err(EngineError::InvalidSql);
            }
            let rows = match op {
                SetOp::Union => {
                    if *all {
                        let mut rows = left_result.rows;
                        rows.extend(right_result.rows);
                        rows
                    } else {
                        union_distinct_rows(left_result.rows, right_result.rows)
                    }
                }
                SetOp::Intersect => {
                    if *all {
                        return Err(EngineError::InvalidSql);
                    }
                    intersect_distinct_rows(left_result.rows, right_result.rows)
                }
                SetOp::Except => {
                    if *all {
                        return Err(EngineError::InvalidSql);
                    }
                    except_distinct_rows(left_result.rows, right_result.rows)
                }
            };
            Ok(QueryResult {
                columns: left_result.columns,
                rows,
            })
        }
    }
}

fn apply_filter_pushdown(
    tables: &mut [TableContext<'_>],
    filter: Option<&Expr>,
    outer: &[RowContext<'_>],
    db: &Database,
    subquery_cache: &Rc<RefCell<HashMap<usize, Value>>>,
    correlation_probe: Option<Rc<RefCell<bool>>>,
) -> Result<Option<Expr>, EngineError> {
    let filter = match filter {
        Some(filter) => filter,
        None => return Ok(None),
    };

    let conjuncts = collect_conjuncts(filter);

    let mut per_table = vec![Vec::<Expr>::new(); tables.len()];
    let mut remaining = Vec::new();
    let table_refs: Vec<TableRef<'_>> = tables
        .iter()
        .map(|table| TableRef::new(table.table, table.alias))
        .collect();

    for expr in conjuncts {
        match predicate_table_indices(&expr, &table_refs) {
            Ok(indices) if indices.len() == 1 => {
                let idx = indices[0];
                per_table[idx].push(expr);
            }
            Ok(_) => remaining.push(expr),
            Err(TableUsageError::UnsupportedSubquery)
            | Err(TableUsageError::UnknownColumn(_))
            | Err(TableUsageError::UnknownTable(_)) => remaining.push(expr),
            Err(TableUsageError::AmbiguousColumn(_))
            | Err(TableUsageError::AmbiguousTable(_)) => return Err(EngineError::InvalidSql),
        }
    }

    for (idx, table) in tables.iter_mut().enumerate() {
        if per_table[idx].is_empty() {
            continue;
        }
        let mut filtered = Vec::new();
        for row in &table.rows {
            let row_context = EvalContext {
                db,
                current: vec![RowContext {
                    table: table.table,
                    alias: table.alias,
                    row,
                }],
                outer: outer.to_vec(),
                subquery_cache: subquery_cache.clone(),
                correlation_probe: correlation_probe.clone(),
            };
            let mut keep = true;
            for expr in &per_table[idx] {
                if !eval_predicate(expr, &row_context)? {
                    keep = false;
                    break;
                }
            }
            if keep {
                filtered.push(row.clone());
            }
        }
        table.rows = filtered;
    }

    Ok(combine_conjuncts(remaining))
}

fn execute_exists(
    select: &Select,
    context: &EvalContext<'_>,
) -> Result<bool, EngineError> {
    if matches!(select.limit, Some(0)) {
        return Ok(false);
    }

    let has_aggregate = select
        .items
        .iter()
        .any(|item| matches!(item, SelectItem::Expr(expr) if expr_is_aggregate(expr)));
    if has_aggregate {
        let result = execute_select_inner(
            select,
            context.db,
            context.current.clone(),
            context.subquery_cache.clone(),
            None,
        )?;
        return Ok(!result.rows.is_empty());
    }

    if select.from.is_empty() {
        let row_context = EvalContext {
            db: context.db,
            current: Vec::new(),
            outer: context.current.clone(),
            subquery_cache: context.subquery_cache.clone(),
            correlation_probe: None,
        };
        if let Some(filter) = select.filter.as_ref() {
            if !eval_predicate(filter, &row_context)? {
                return Ok(false);
            }
        }
        return Ok(true);
    }

    let mut tables = Vec::new();
    for from in &select.from {
        let table = context
            .db
            .catalog()
            .table(&from.table)
            .ok_or(EngineError::TableNotFound)?;
        tables.push(TableContext {
            table,
            alias: from.alias.as_deref(),
            rows: context
                .db
                .scan_table_rows_unlocked(&table.name)?
                .into_iter()
                .map(|row| row.values)
                .collect(),
        });
    }

    let table_refs: Vec<TableRef<'_>> = tables
        .iter()
        .map(|table| TableRef::new(table.table, table.alias))
        .collect();
    let (join_predicates, remaining_filter) =
        split_join_predicates(select.filter.as_ref(), &table_refs)?;
    let join_order = join_order_for_query(&tables, &join_predicates);
    let mut selected: Vec<Option<&[Value]>> = vec![None; tables.len()];
    return exists_join_incremental(
        &tables,
        &join_order,
        0,
        &mut selected,
        context.db,
        &context.current,
        &context.subquery_cache,
        &join_predicates,
        remaining_filter.as_ref(),
    );
}

fn build_columns(
    select: &Select,
    tables: &[TableContext<'_>],
) -> Result<Vec<ColumnMeta>, EngineError> {
    let mut columns = Vec::new();
    for item in &select.items {
        match item {
            SelectItem::Wildcard(prefix) => {
                if tables.is_empty() {
                    return Err(EngineError::InvalidSql);
                }
                if let Some(prefix) = prefix {
                    let mut matched = false;
                    for table in tables {
                        if matches_table(prefix, RowContext {
                            table: table.table,
                            alias: table.alias,
                            row: &[],
                        }) {
                            for column in &table.table.columns {
                                columns.push(ColumnMeta::new(column.name.clone(), column.sql_type));
                            }
                            matched = true;
                            break;
                        }
                    }
                    if !matched {
                        return Err(EngineError::InvalidSql);
                    }
                } else {
                    for table in tables {
                        for column in &table.table.columns {
                            columns.push(ColumnMeta::new(column.name.clone(), column.sql_type));
                        }
                    }
                }
            }
            SelectItem::Expr(expr) => {
                let label = label_for_expr(expr);
                let sql_type = expr_result_type(expr, tables)?;
                columns.push(ColumnMeta::new(label, sql_type));
            }
        }
    }
    Ok(columns)
}

fn evaluate_aggregates(
    filter: Option<&Expr>,
    select: &Select,
    tables: &[TableContext<'_>],
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

    let table_refs: Vec<TableRef<'_>> = tables
        .iter()
        .map(|table| TableRef::new(table.table, table.alias))
        .collect();
    let (join_predicates, remaining_filter) =
        split_join_predicates(filter, &table_refs)?;
    let join_order = join_order_for_query(tables, &join_predicates);
    let mut selected: Vec<Option<&[Value]>> = vec![None; tables.len()];
    aggregate_join_incremental(
        tables,
        &join_order,
        0,
        &mut selected,
        context.db,
        &context.outer,
        &context.subquery_cache,
        context.correlation_probe.clone(),
        &join_predicates,
        remaining_filter.as_ref(),
        &mut aggregates,
    )?;

    Ok(aggregates.into_iter().map(|state| state.finish()).collect())
}

fn combine_conjuncts(conjuncts: Vec<Expr>) -> Option<Expr> {
    let mut iter = conjuncts.into_iter();
    let mut expr = iter.next()?;
    for next in iter {
        expr = Expr::Binary {
            left: Box::new(expr),
            op: BinaryOp::And,
            right: Box::new(next),
        };
    }
    Some(expr)
}

fn join_order_by_row_count(tables: &[TableContext<'_>]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..tables.len()).collect();
    order.sort_by_key(|&idx| (tables[idx].rows.len(), idx));
    order
}

fn join_order_for_query(tables: &[TableContext<'_>], join_predicates: &[JoinPredicate]) -> Vec<usize> {
    if tables.is_empty() {
        return Vec::new();
    }

    let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); tables.len()];
    for predicate in join_predicates {
        let indices = &predicate.indices;
        for (pos, &left) in indices.iter().enumerate() {
            for &right in indices.iter().skip(pos + 1) {
                adjacency[left].push(right);
                adjacency[right].push(left);
            }
        }
    }
    for neighbors in &mut adjacency {
        neighbors.sort_unstable();
        neighbors.dedup();
    }

    let mut order = Vec::with_capacity(tables.len());
    let mut remaining: Vec<usize> = (0..tables.len()).collect();
    let mut remaining_mask = vec![true; tables.len()];
    let mut selected_mask = vec![false; tables.len()];

    let mut pick_min = |candidates: &[usize]| -> usize {
        *candidates
            .iter()
            .min_by_key(|&&idx| (tables[idx].rows.len(), idx))
            .expect("candidates non-empty")
    };

    let mut pick_start = |remaining: &[usize], remaining_mask: &[bool]| -> usize {
        let with_edges: Vec<usize> = remaining
            .iter()
            .copied()
            .filter(|&idx| adjacency[idx].iter().any(|&neighbor| remaining_mask[neighbor]))
            .collect();
        if with_edges.is_empty() {
            pick_min(remaining)
        } else {
            pick_min(&with_edges)
        }
    };

    while !remaining.is_empty() {
        let next = if order.is_empty() {
            pick_start(&remaining, &remaining_mask)
        } else {
            let connected: Vec<usize> = remaining
                .iter()
                .copied()
                .filter(|&idx| adjacency[idx].iter().any(|&neighbor| selected_mask[neighbor]))
                .collect();
            if connected.is_empty() {
                pick_start(&remaining, &remaining_mask)
            } else {
                pick_min(&connected)
            }
        };

        order.push(next);
        selected_mask[next] = true;
        remaining_mask[next] = false;
        remaining.retain(|&idx| idx != next);
    }

    order
}

fn split_join_predicates(
    filter: Option<&Expr>,
    table_refs: &[TableRef<'_>],
) -> Result<(Vec<JoinPredicate>, Option<Expr>), EngineError> {
    let Some(filter) = filter else {
        return Ok((Vec::new(), None));
    };
    let mut join_predicates = Vec::new();
    let mut remaining = Vec::new();
    for expr in collect_conjuncts(filter) {
        match predicate_table_indices(&expr, table_refs) {
            Ok(indices) => {
                if indices.len() >= 2 {
                    let mut indices = indices;
                    indices.sort_unstable();
                    indices.dedup();
                    join_predicates.push(JoinPredicate { indices, expr });
                } else {
                    remaining.push(expr);
                }
            }
            Err(TableUsageError::UnsupportedSubquery)
            | Err(TableUsageError::UnknownColumn(_))
            | Err(TableUsageError::UnknownTable(_)) => remaining.push(expr),
            Err(TableUsageError::AmbiguousColumn(_))
            | Err(TableUsageError::AmbiguousTable(_)) => return Err(EngineError::InvalidSql),
        }
    }
    Ok((join_predicates, combine_conjuncts(remaining)))
}

fn build_row_contexts<'a>(
    tables: &'a [TableContext<'a>],
    selected: &[Option<&'a [Value]>],
) -> (Vec<RowContext<'a>>, Vec<Value>) {
    let mut current = Vec::with_capacity(tables.len());
    let mut combined = Vec::new();
    for (idx, table) in tables.iter().enumerate() {
        let row = selected[idx].expect("row selected");
        current.push(RowContext {
            table: table.table,
            alias: table.alias,
            row,
        });
        combined.extend_from_slice(row);
    }
    (current, combined)
}

fn execute_join_incremental<'a>(
    tables: &'a [TableContext<'a>],
    order: &[usize],
    level: usize,
    selected: &mut Vec<Option<&'a [Value]>>,
    outer: &[RowContext<'a>],
    db: &'a Database,
    subquery_cache: &Rc<RefCell<HashMap<usize, Value>>>,
    correlation_probe: &Option<Rc<RefCell<bool>>>,
    join_predicates: &[JoinPredicate],
    remaining_filter: &Option<Expr>,
    select: &Select,
    collected: &mut Vec<(Vec<Value>, Vec<Value>)>,
) -> Result<(), EngineError> {
    if tables.is_empty() {
        return Ok(());
    }
    if level == order.len() {
        let (current, combined) = build_row_contexts(tables, selected);
        let context = EvalContext {
            db,
            current,
            outer: outer.to_vec(),
            subquery_cache: subquery_cache.clone(),
            correlation_probe: correlation_probe.clone(),
        };
        if let Some(filter) = remaining_filter {
            if !eval_predicate(filter, &context)? {
                return Ok(());
            }
        }
        let mut output_row = Vec::with_capacity(select.items.len());
        for item in &select.items {
            match item {
                SelectItem::Wildcard(prefix) => {
                    if tables.is_empty() {
                        return Err(EngineError::InvalidSql);
                    }
                    if let Some(prefix) = prefix {
                        let mut matched = false;
                        for row_ctx in &context.current {
                            if matches_table(prefix, *row_ctx) {
                                output_row.extend_from_slice(row_ctx.row);
                                matched = true;
                                break;
                            }
                        }
                        if !matched {
                            return Err(EngineError::InvalidSql);
                        }
                    } else {
                        output_row.extend_from_slice(&combined);
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
        return Ok(());
    }

    if select.order_by.is_empty() {
        if let Some(limit) = select.limit {
            if collected.len() >= limit as usize {
                return Ok(());
            }
        }
    }

    let table_idx = order[level];
    for row in &tables[table_idx].rows {
        selected[table_idx] = Some(row.as_slice());
        if !eval_join_predicates(
            tables,
            selected,
            outer,
            db,
            subquery_cache,
            correlation_probe,
            join_predicates,
            table_idx,
        )? {
            continue;
        }
        execute_join_incremental(
            tables,
            order,
            level + 1,
            selected,
            outer,
            db,
            subquery_cache,
            correlation_probe,
            join_predicates,
            remaining_filter,
            select,
            collected,
        )?;
        if select.order_by.is_empty() {
            if let Some(limit) = select.limit {
                if collected.len() >= limit as usize {
                    return Ok(());
                }
            }
        }
    }
    selected[table_idx] = None;
    Ok(())
}

fn exists_join_incremental<'a>(
    tables: &'a [TableContext<'a>],
    order: &[usize],
    level: usize,
    selected: &mut Vec<Option<&'a [Value]>>,
    db: &'a Database,
    outer: &[RowContext<'a>],
    subquery_cache: &Rc<RefCell<HashMap<usize, Value>>>,
    join_predicates: &[JoinPredicate],
    filter: Option<&Expr>,
) -> Result<bool, EngineError> {
    if tables.is_empty() {
        return Ok(false);
    }
    if level == order.len() {
        let (current, _combined) = build_row_contexts(tables, selected);
        let row_context = EvalContext {
            db,
            current,
            outer: outer.to_vec(),
            subquery_cache: subquery_cache.clone(),
            correlation_probe: None,
        };
        if let Some(filter) = filter {
            if !eval_predicate(filter, &row_context)? {
                return Ok(false);
            }
        }
        return Ok(true);
    }

    let table_idx = order[level];
    for row in &tables[table_idx].rows {
        selected[table_idx] = Some(row.as_slice());
        if !eval_join_predicates(
            tables,
            selected,
            outer,
            db,
            subquery_cache,
            &None,
            join_predicates,
            table_idx,
        )? {
            continue;
        }
        if exists_join_incremental(
            tables,
            order,
            level + 1,
            selected,
            db,
            outer,
            subquery_cache,
            join_predicates,
            filter,
        )? {
            return Ok(true);
        }
    }
    selected[table_idx] = None;
    Ok(false)
}

fn aggregate_join_incremental<'a>(
    tables: &'a [TableContext<'a>],
    order: &[usize],
    level: usize,
    selected: &mut Vec<Option<&'a [Value]>>,
    db: &'a Database,
    outer: &[RowContext<'a>],
    subquery_cache: &Rc<RefCell<HashMap<usize, Value>>>,
    correlation_probe: Option<Rc<RefCell<bool>>>,
    join_predicates: &[JoinPredicate],
    filter: Option<&Expr>,
    aggregates: &mut Vec<AggregateState>,
) -> Result<(), EngineError> {
    if tables.is_empty() {
        return Ok(());
    }
    if level == order.len() {
        let (current, _combined) = build_row_contexts(tables, selected);
        let row_context = EvalContext {
            db,
            current,
            outer: outer.to_vec(),
            subquery_cache: subquery_cache.clone(),
            correlation_probe: correlation_probe.clone(),
        };

        if let Some(filter) = filter {
            if !eval_predicate(filter, &row_context)? {
                return Ok(());
            }
        }

        for state in aggregates.iter_mut() {
            state.consume(expr_arg_for_aggregate(state.expr(), &row_context)?);
        }
        return Ok(());
    }

    let table_idx = order[level];
    for row in &tables[table_idx].rows {
        selected[table_idx] = Some(row.as_slice());
        if !eval_join_predicates(
            tables,
            selected,
            outer,
            db,
            subquery_cache,
            &correlation_probe,
            join_predicates,
            table_idx,
        )? {
            continue;
        }
        aggregate_join_incremental(
            tables,
            order,
            level + 1,
            selected,
            db,
            outer,
            subquery_cache,
            correlation_probe.clone(),
            join_predicates,
            filter,
            aggregates,
        )?;
    }
    selected[table_idx] = None;
    Ok(())
}

fn eval_join_predicates<'a>(
    tables: &'a [TableContext<'a>],
    selected: &[Option<&'a [Value]>],
    outer: &[RowContext<'a>],
    db: &'a Database,
    subquery_cache: &Rc<RefCell<HashMap<usize, Value>>>,
    correlation_probe: &Option<Rc<RefCell<bool>>>,
    join_predicates: &[JoinPredicate],
    newly_selected: usize,
) -> Result<bool, EngineError> {
    let mut should_check = false;
    for predicate in join_predicates {
        if predicate.indices.contains(&newly_selected)
            && predicate
                .indices
                .iter()
                .all(|idx| selected[*idx].is_some())
        {
            should_check = true;
            break;
        }
    }
    if !should_check {
        return Ok(true);
    }

    let mut current = Vec::new();
    for (idx, table) in tables.iter().enumerate() {
        if let Some(row) = selected[idx] {
            current.push(RowContext {
                table: table.table,
                alias: table.alias,
                row,
            });
        }
    }

    let context = EvalContext {
        db,
        current,
        outer: outer.to_vec(),
        subquery_cache: subquery_cache.clone(),
        correlation_probe: correlation_probe.clone(),
    };

    for predicate in join_predicates {
        if !predicate.indices.contains(&newly_selected) {
            continue;
        }
        if !predicate
            .indices
            .iter()
            .all(|idx| selected[*idx].is_some())
        {
            continue;
        }
        if !eval_predicate(&predicate.expr, &context)? {
            return Ok(false);
        }
    }

    Ok(true)
}

fn resolve_column_type(
    parts: &[String],
    tables: &[TableContext<'_>],
) -> Result<SqlType, EngineError> {
    match parts {
        [name] => {
            let mut found = None;
            for table in tables {
                if let Some(column) = table.table.column(name) {
                    if found.is_some() {
                        return Err(EngineError::InvalidSql);
                    }
                    found = Some(column.sql_type);
                }
            }
            found.ok_or(EngineError::InvalidSql)
        }
        [table_name, column] => {
            let table = tables.iter().find(|table| {
                table.table.name == *table_name
                    || table
                        .alias
                        .map(|alias| alias == table_name)
                        .unwrap_or(false)
            });
            let table = table.ok_or(EngineError::InvalidSql)?;
            table
                .table
                .column(column)
                .map(|col| col.sql_type)
                .ok_or(EngineError::InvalidSql)
        }
        _ => Err(EngineError::InvalidSql),
    }
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
        Value::Null => Ok(false),
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
                    Value::Null => Ok(Value::Null),
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
            if matches!(value, Value::Null) || matches!(low, Value::Null) || matches!(high, Value::Null) {
                return Ok(Value::Null);
            }
            let between = value_cmp(&value, &low) != Ordering::Less
                && value_cmp(&value, &high) != Ordering::Greater;
            Ok(Value::Boolean(if *negated { !between } else { between }))
        }
        Expr::InList {
            expr,
            list,
            negated,
        } => {
            let value = eval_expr(expr, context)?;
            if matches!(value, Value::Null) {
                return Ok(Value::Null);
            }
            let mut has_null = false;
            for item in list {
                let item_value = eval_expr(item, context)?;
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
                        Value::Null => false,
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
            } else if name == "coalesce" {
                for arg in args {
                    if let FunctionArg::Expr(expr) = arg {
                        let value = eval_expr(expr, context)?;
                        if !matches!(value, Value::Null) {
                            return Ok(value);
                        }
                    } else {
                        return Err(EngineError::InvalidSql);
                    }
                }
                Ok(Value::Null)
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
                context.current.clone(),
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
            Ok(Value::Boolean(execute_exists(select, context)?))
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
            if let Some(value) = find_unqualified_value(column, &context.current)? {
                return Ok(value);
            }
            if let Some(value) = find_unqualified_value(column, &context.outer)? {
                mark_correlated(context);
                return Ok(value);
            }
            Err(EngineError::InvalidSql)
        }
        [table_name, column] => {
            if let Some(value) =
                find_qualified_value(table_name, column, &context.current)?
            {
                return Ok(value);
            }
            if let Some(value) =
                find_qualified_value(table_name, column, &context.outer)?
            {
                mark_correlated(context);
                return Ok(value);
            }
            Err(EngineError::InvalidSql)
        }
        _ => Err(EngineError::InvalidSql),
    }
}

fn find_unqualified_value(
    column: &str,
    contexts: &[RowContext<'_>],
) -> Result<Option<Value>, EngineError> {
    let mut found = None;
    for ctx in contexts {
        if let Some(value) = column_value(*ctx, column) {
            if found.is_some() {
                return Err(EngineError::InvalidSql);
            }
            found = Some(value);
        }
    }
    Ok(found)
}

fn find_qualified_value(
    table_name: &str,
    column: &str,
    contexts: &[RowContext<'_>],
) -> Result<Option<Value>, EngineError> {
    let mut found = None;
    for ctx in contexts {
        if matches_table(table_name, *ctx) {
            if let Some(value) = column_value(*ctx, column) {
                if found.is_some() {
                    return Err(EngineError::InvalidSql);
                }
                found = Some(value);
            }
        }
    }
    Ok(found)
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
        BinaryOp::Eq => compare_with_nulls(left, right, |l, r| {
            Value::Boolean(values_equal(&l, &r))
        }),
        BinaryOp::NotEq => compare_with_nulls(left, right, |l, r| {
            Value::Boolean(!values_equal(&l, &r))
        }),
        BinaryOp::Lt => compare_with_nulls(left, right, |l, r| {
            Value::Boolean(value_cmp(&l, &r) == Ordering::Less)
        }),
        BinaryOp::Lte => compare_with_nulls(left, right, |l, r| {
            Value::Boolean(value_cmp(&l, &r) != Ordering::Greater)
        }),
        BinaryOp::Gt => compare_with_nulls(left, right, |l, r| {
            Value::Boolean(value_cmp(&l, &r) == Ordering::Greater)
        }),
        BinaryOp::Gte => compare_with_nulls(left, right, |l, r| {
            Value::Boolean(value_cmp(&l, &r) != Ordering::Less)
        }),
        BinaryOp::And => Ok(three_valued_and(left, right)?),
        BinaryOp::Or => Ok(three_valued_or(left, right)?),
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

fn expr_result_type(expr: &Expr, tables: &[TableContext<'_>]) -> Result<SqlType, EngineError> {
    match expr {
        Expr::Identifier(parts) => resolve_column_type(parts, tables),
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
            UnaryOp::Neg => expr_result_type(expr, tables),
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
                let left = expr_result_type(left, tables)?;
                let right = expr_result_type(right, tables)?;
                if left == SqlType::Real || right == SqlType::Real {
                    Ok(SqlType::Real)
                } else {
                    Ok(SqlType::Integer)
                }
            }
            BinaryOp::Div => Ok(SqlType::Real),
        },
        Expr::Between { .. } => Ok(SqlType::Boolean),
        Expr::InList { .. } => Ok(SqlType::Boolean),
        Expr::IsNull { .. } => Ok(SqlType::Boolean),
        Expr::Case { when_thens, else_expr, .. } => {
            if let Some((_, then_expr)) = when_thens.first() {
                expr_result_type(then_expr, tables)
            } else if let Some(else_expr) = else_expr {
                expr_result_type(else_expr, tables)
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
                    SelectItem::Expr(expr) => expr_result_type(expr, tables),
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

fn value_equal_with_nulls(left: &Value, right: &Value) -> bool {
    match (left, right) {
        (Value::Null, Value::Null) => true,
        _ => values_equal(left, right),
    }
}

fn rows_equal_with_nulls(left: &[Value], right: &[Value]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right.iter())
        .all(|(l, r)| value_equal_with_nulls(l, r))
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

        let result = execute_select_inner(select, &db, Vec::new(), cache, None).expect("select");
        assert_eq!(
            result.rows,
            vec![
                vec![Value::Integer(42)],
                vec![Value::Integer(42)],
            ]
        );
    }

    #[test]
    fn arithmetic_with_null_returns_null() {
        let path = temp_db_path("arithmetic_null");
        let db = Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        let context = EvalContext {
            db: &db,
            current: Vec::new(),
            outer: Vec::new(),
            subquery_cache: Rc::new(RefCell::new(HashMap::new())),
            correlation_probe: None,
        };

        let expr = Expr::Binary {
            left: Box::new(Expr::Literal(Literal::Null)),
            op: BinaryOp::Add,
            right: Box::new(Expr::Literal(Literal::Number("1".to_string()))),
        };
        let value = eval_expr(&expr, &context).expect("eval");
        assert_eq!(value, Value::Null);

        let expr = Expr::Unary {
            op: UnaryOp::Neg,
            expr: Box::new(Expr::Literal(Literal::Null)),
        };
        let value = eval_expr(&expr, &context).expect("eval");
        assert_eq!(value, Value::Null);
    }

    #[test]
    fn division_by_zero_returns_error() {
        let path = temp_db_path("division_by_zero");
        let db = Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        let context = EvalContext {
            db: &db,
            current: Vec::new(),
            outer: Vec::new(),
            subquery_cache: Rc::new(RefCell::new(HashMap::new())),
            correlation_probe: None,
        };

        let expr = Expr::Binary {
            left: Box::new(Expr::Literal(Literal::Number("10".to_string()))),
            op: BinaryOp::Div,
            right: Box::new(Expr::Literal(Literal::Number("0".to_string()))),
        };
        assert!(eval_expr(&expr, &context).is_err());

        let expr = Expr::Binary {
            left: Box::new(Expr::Literal(Literal::Number("10.0".to_string()))),
            op: BinaryOp::Div,
            right: Box::new(Expr::Literal(Literal::Number("0.0".to_string()))),
        };
        assert!(eval_expr(&expr, &context).is_err());
    }

    #[test]
    fn comparison_with_null_returns_null() {
        let path = temp_db_path("comparison_null");
        let db = Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        let context = EvalContext {
            db: &db,
            current: Vec::new(),
            outer: Vec::new(),
            subquery_cache: Rc::new(RefCell::new(HashMap::new())),
            correlation_probe: None,
        };

        let expr = Expr::Binary {
            left: Box::new(Expr::Literal(Literal::Null)),
            op: BinaryOp::Eq,
            right: Box::new(Expr::Literal(Literal::Number("1".to_string()))),
        };
        let value = eval_expr(&expr, &context).expect("eval");
        assert_eq!(value, Value::Null);
    }

    #[test]
    fn predicate_treats_null_as_false() {
        let path = temp_db_path("predicate_null");
        let db = Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        let context = EvalContext {
            db: &db,
            current: Vec::new(),
            outer: Vec::new(),
            subquery_cache: Rc::new(RefCell::new(HashMap::new())),
            correlation_probe: None,
        };

        let expr = Expr::Literal(Literal::Null);
        let value = eval_predicate(&expr, &context).expect("eval predicate");
        assert!(!value);
    }

    #[test]
    fn pushdown_filters_single_table_predicates_before_join() {
        let path = temp_db_path("filter_pushdown");
        let mut db = Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        db.execute("CREATE TABLE users (id INTEGER, age INTEGER)")
            .expect("create users");
        db.execute("CREATE TABLE orders (id INTEGER, user_id INTEGER)")
            .expect("create orders");
        db.execute("INSERT INTO users (id, age) VALUES (1, 10)")
            .expect("insert user");
        db.execute("INSERT INTO users (id, age) VALUES (2, 25)")
            .expect("insert user");
        db.execute("INSERT INTO users (id, age) VALUES (3, 30)")
            .expect("insert user");
        db.execute("INSERT INTO orders (id, user_id) VALUES (1, 1)")
            .expect("insert order");
        db.execute("INSERT INTO orders (id, user_id) VALUES (2, 2)")
            .expect("insert order");
        db.execute("INSERT INTO orders (id, user_id) VALUES (3, 2)")
            .expect("insert order");

        let statements = parse(
            "SELECT users.id, orders.id \
             FROM users, orders \
             WHERE users.age > 20 AND orders.id = 2 AND users.id = orders.user_id",
        )
        .expect("parse");
        let select = match &statements[0] {
            Statement::Select(select) => select,
            _ => panic!("expected select statement"),
        };

        let mut tables = Vec::new();
        for from in &select.from {
            let table = db
                .catalog()
                .table(&from.table)
                .expect("table exists");
            tables.push(TableContext {
                table,
                alias: from.alias.as_deref(),
                rows: db
                    .scan_table_rows_unlocked(&table.name)
                    .expect("scan table")
                    .into_iter()
                    .map(|row| row.values)
                    .collect(),
            });
        }

        let cache = Rc::new(RefCell::new(HashMap::new()));
        let remaining = apply_filter_pushdown(
            &mut tables,
            select.filter.as_ref(),
            &[],
            &db,
            &cache,
            None,
        )
        .expect("apply pushdown")
        .expect("remaining filter");

        let user_ids: Vec<i64> = tables[0]
            .rows
            .iter()
            .map(|row| match row[0] {
                Value::Integer(value) => value,
                _ => panic!("expected integer"),
            })
            .collect();
        let order_ids: Vec<i64> = tables[1]
            .rows
            .iter()
            .map(|row| match row[0] {
                Value::Integer(value) => value,
                _ => panic!("expected integer"),
            })
            .collect();
        assert_eq!(user_ids, vec![2, 3]);
        assert_eq!(order_ids, vec![2]);

        match remaining {
            Expr::Binary { op: BinaryOp::Eq, .. } => {}
            _ => panic!("expected remaining join predicate"),
        }

        let result = execute_select(select, &db).expect("execute select");
        assert_eq!(
            result.rows,
            vec![vec![Value::Integer(2), Value::Integer(2)]]
        );
    }

    #[test]
    fn join_order_uses_filtered_row_counts() {
        let path = temp_db_path("join_order_filtered");
        let mut db = Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        db.execute("CREATE TABLE users (id INTEGER)")
            .expect("create users");
        db.execute("CREATE TABLE orders (id INTEGER)")
            .expect("create orders");
        db.execute("INSERT INTO users (id) VALUES (1)")
            .expect("insert users");
        db.execute("INSERT INTO users (id) VALUES (2)")
            .expect("insert users");
        db.execute("INSERT INTO users (id) VALUES (3)")
            .expect("insert users");
        db.execute("INSERT INTO users (id) VALUES (4)")
            .expect("insert users");
        db.execute("INSERT INTO users (id) VALUES (5)")
            .expect("insert users");
        db.execute("INSERT INTO orders (id) VALUES (1)")
            .expect("insert orders");
        db.execute("INSERT INTO orders (id) VALUES (2)")
            .expect("insert orders");

        let statements =
            parse("SELECT * FROM users, orders WHERE users.id = 1").expect("parse");
        let select = match &statements[0] {
            Statement::Select(select) => select,
            _ => panic!("expected select statement"),
        };

        let mut tables = Vec::new();
        for from in &select.from {
            let table = db
                .catalog()
                .table(&from.table)
                .expect("table exists");
            tables.push(TableContext {
                table,
                alias: from.alias.as_deref(),
                rows: db
                    .scan_table_rows_unlocked(&table.name)
                    .expect("scan table")
                    .into_iter()
                    .map(|row| row.values)
                    .collect(),
            });
        }

        let cache = Rc::new(RefCell::new(HashMap::new()));
        apply_filter_pushdown(
            &mut tables,
            select.filter.as_ref(),
            &[],
            &db,
            &cache,
            None,
        )
        .expect("pushdown");

        let order = join_order_by_row_count(&tables);
        assert_eq!(order, vec![0, 1]);
    }

    #[test]
    fn join_order_is_deterministic_with_equal_counts() {
        let path = temp_db_path("join_order_deterministic");
        let mut db = Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        db.execute("CREATE TABLE t1 (id INTEGER)")
            .expect("create t1");
        db.execute("CREATE TABLE t2 (id INTEGER)")
            .expect("create t2");
        db.execute("INSERT INTO t1 (id) VALUES (1)")
            .expect("insert t1");
        db.execute("INSERT INTO t1 (id) VALUES (2)")
            .expect("insert t1");
        db.execute("INSERT INTO t2 (id) VALUES (3)")
            .expect("insert t2");
        db.execute("INSERT INTO t2 (id) VALUES (4)")
            .expect("insert t2");

        let statements = parse("SELECT * FROM t1, t2").expect("parse");
        let select = match &statements[0] {
            Statement::Select(select) => select,
            _ => panic!("expected select statement"),
        };

        let mut tables = Vec::new();
        for from in &select.from {
            let table = db
                .catalog()
                .table(&from.table)
                .expect("table exists");
            tables.push(TableContext {
                table,
                alias: from.alias.as_deref(),
                rows: db
                    .scan_table_rows_unlocked(&table.name)
                    .expect("scan table")
                    .into_iter()
                    .map(|row| row.values)
                    .collect(),
            });
        }

        let order1 = join_order_by_row_count(&tables);
        let order2 = join_order_by_row_count(&tables);
        assert_eq!(order1, order2);
        assert_eq!(order1, vec![0, 1]);
    }

    #[test]
    fn incremental_join_matches_cross_product_results() {
        let path = temp_db_path("incremental_join_compare");
        let mut db = Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        db.execute("CREATE TABLE a (id INTEGER)")
            .expect("create a");
        db.execute("CREATE TABLE b (id INTEGER)")
            .expect("create b");
        db.execute("INSERT INTO a (id) VALUES (1)")
            .expect("insert a");
        db.execute("INSERT INTO a (id) VALUES (2)")
            .expect("insert a");
        db.execute("INSERT INTO b (id) VALUES (10)")
            .expect("insert b");
        db.execute("INSERT INTO b (id) VALUES (20)")
            .expect("insert b");

        let statements = parse("SELECT * FROM a, b").expect("parse");
        let select = match &statements[0] {
            Statement::Select(select) => select,
            _ => panic!("expected select statement"),
        };

        let mut tables = Vec::new();
        for from in &select.from {
            let table = db
                .catalog()
                .table(&from.table)
                .expect("table exists");
            tables.push(TableContext {
                table,
                alias: from.alias.as_deref(),
                rows: db
                    .scan_table_rows_unlocked(&table.name)
                    .expect("scan table")
                    .into_iter()
                    .map(|row| row.values)
                    .collect(),
            });
        }

        let expected = build_combinations_ordered(&tables, &[0, 1]);
        let join_order = join_order_by_row_count(&tables);
        let actual = build_combinations_ordered(&tables, &join_order);
        assert_eq!(actual, expected);
    }

    #[test]
    fn join_predicates_prune_intermediate_rows() {
        let path = temp_db_path("join_predicate_prune");
        let mut db = Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        db.execute("CREATE TABLE a (id INTEGER)")
            .expect("create a");
        db.execute("CREATE TABLE b (id INTEGER)")
            .expect("create b");
        db.execute("CREATE TABLE c (id INTEGER)")
            .expect("create c");
        db.execute("INSERT INTO a (id) VALUES (1)")
            .expect("insert a");
        db.execute("INSERT INTO a (id) VALUES (2)")
            .expect("insert a");
        db.execute("INSERT INTO b (id) VALUES (1)")
            .expect("insert b");
        db.execute("INSERT INTO b (id) VALUES (3)")
            .expect("insert b");
        db.execute("INSERT INTO c (id) VALUES (10)")
            .expect("insert c");
        db.execute("INSERT INTO c (id) VALUES (20)")
            .expect("insert c");
        db.execute("INSERT INTO c (id) VALUES (30)")
            .expect("insert c");
        db.execute("INSERT INTO c (id) VALUES (40)")
            .expect("insert c");
        db.execute("INSERT INTO c (id) VALUES (50)")
            .expect("insert c");

        let statements =
            parse("SELECT * FROM a, b, c WHERE a.id = b.id").expect("parse");
        let select = match &statements[0] {
            Statement::Select(select) => select,
            _ => panic!("expected select statement"),
        };

        let mut tables = Vec::new();
        for from in &select.from {
            let table = db
                .catalog()
                .table(&from.table)
                .expect("table exists");
            tables.push(TableContext {
                table,
                alias: from.alias.as_deref(),
                rows: db
                    .scan_table_rows_unlocked(&table.name)
                    .expect("scan table")
                    .into_iter()
                    .map(|row| row.values)
                    .collect(),
            });
        }

        let table_refs: Vec<TableRef<'_>> = tables
            .iter()
            .map(|table| TableRef::new(table.table, table.alias))
            .collect();
        let (join_predicates, remaining) =
            split_join_predicates(select.filter.as_ref(), &table_refs).expect("split");
        assert!(remaining.is_none());

        let join_order = join_order_by_row_count(&tables);
        let leaf_count = count_join_leaves_with_predicates(
            &tables,
            &join_order,
            &join_predicates,
            &db,
        )
        .expect("count");

        let cross_product_count =
            tables.iter().map(|table| table.rows.len()).product::<usize>();
        assert_eq!(leaf_count, 5);
        assert!(leaf_count < cross_product_count);
    }

    #[test]
    fn exists_exits_on_first_match() {
        let path = temp_db_path("exists_early_exit");
        let mut db = Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        db.execute("CREATE TABLE a (id INTEGER)")
            .expect("create a");
        db.execute("CREATE TABLE b (id INTEGER)")
            .expect("create b");
        db.execute("CREATE TABLE outer_t (id INTEGER)")
            .expect("create outer_t");
        db.execute("INSERT INTO a (id) VALUES (1)")
            .expect("insert a");
        db.execute("INSERT INTO a (id) VALUES (0)")
            .expect("insert a");
        db.execute("INSERT INTO b (id) VALUES (1)")
            .expect("insert b");
        db.execute("INSERT INTO outer_t (id) VALUES (1)")
            .expect("insert outer_t");

        let result = db
            .execute(
                "SELECT EXISTS(SELECT 1 FROM a, b WHERE b.id = 1 AND 1 / a.id = 1) FROM outer_t",
            )
            .expect("execute exists");

        assert_eq!(result.rows, vec![vec![Value::Boolean(true)]]);
    }

    #[test]
    fn limit_stops_join_execution_after_enough_rows() {
        let path = temp_db_path("limit_join_early_exit");
        let mut db = Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        db.execute("CREATE TABLE a (id INTEGER)")
            .expect("create a");
        db.execute("CREATE TABLE b (id INTEGER)")
            .expect("create b");
        db.execute("INSERT INTO a (id) VALUES (1)")
            .expect("insert a");
        db.execute("INSERT INTO a (id) VALUES (2)")
            .expect("insert a");
        db.execute("INSERT INTO b (id) VALUES (10)")
            .expect("insert b");
        db.execute("INSERT INTO b (id) VALUES (20)")
            .expect("insert b");

        let result = db
            .execute("SELECT a.id, b.id FROM a, b LIMIT 1")
            .expect("execute limit");

        assert_eq!(result.rows.len(), 1);
        assert_eq!(
            result.rows[0],
            vec![Value::Integer(1), Value::Integer(10)]
        );
    }

    #[test]
    fn join_two_tables_with_reordered_from() {
        let path = temp_db_path("join_reorder_two");
        let mut db = Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        db.execute("CREATE TABLE users (id INTEGER)")
            .expect("create users");
        db.execute("CREATE TABLE orders (user_id INTEGER, amount INTEGER)")
            .expect("create orders");
        db.execute("INSERT INTO users (id) VALUES (1)")
            .expect("insert user");
        db.execute("INSERT INTO users (id) VALUES (2)")
            .expect("insert user");
        db.execute("INSERT INTO orders (user_id, amount) VALUES (1, 10)")
            .expect("insert order");
        db.execute("INSERT INTO orders (user_id, amount) VALUES (1, 20)")
            .expect("insert order");
        db.execute("INSERT INTO orders (user_id, amount) VALUES (2, 30)")
            .expect("insert order");

        let result = db
            .execute(
                "SELECT users.id, orders.amount \
                 FROM orders, users \
                 WHERE orders.user_id = users.id \
                 ORDER BY orders.amount",
            )
            .expect("execute join");

        assert_eq!(
            result.rows,
            vec![
                vec![Value::Integer(1), Value::Integer(10)],
                vec![Value::Integer(1), Value::Integer(20)],
                vec![Value::Integer(2), Value::Integer(30)],
            ]
        );
    }

    #[test]
    fn join_three_tables_with_reordered_from() {
        let path = temp_db_path("join_reorder_three");
        let mut db = Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        db.execute("CREATE TABLE a (id INTEGER)")
            .expect("create a");
        db.execute("CREATE TABLE b (id INTEGER, a_id INTEGER)")
            .expect("create b");
        db.execute("CREATE TABLE c (id INTEGER, b_id INTEGER)")
            .expect("create c");
        db.execute("INSERT INTO a (id) VALUES (1)")
            .expect("insert a");
        db.execute("INSERT INTO a (id) VALUES (2)")
            .expect("insert a");
        db.execute("INSERT INTO b (id, a_id) VALUES (10, 1)")
            .expect("insert b");
        db.execute("INSERT INTO b (id, a_id) VALUES (20, 2)")
            .expect("insert b");
        db.execute("INSERT INTO c (id, b_id) VALUES (100, 10)")
            .expect("insert c");
        db.execute("INSERT INTO c (id, b_id) VALUES (200, 20)")
            .expect("insert c");

        let result = db
            .execute(
                "SELECT a.id, b.id, c.id \
                 FROM c, a, b \
                 WHERE a.id = b.a_id AND b.id = c.b_id \
                 ORDER BY a.id",
            )
            .expect("execute join");

        assert_eq!(
            result.rows,
            vec![
                vec![Value::Integer(1), Value::Integer(10), Value::Integer(100)],
                vec![Value::Integer(2), Value::Integer(20), Value::Integer(200)],
            ]
        );
    }

    #[test]
    fn correlated_subquery_in_join_select_list() {
        let path = temp_db_path("correlated_join_select");
        let mut db = Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        db.execute("CREATE TABLE users (id INTEGER, region_id INTEGER)")
            .expect("create users");
        db.execute("CREATE TABLE regions (id INTEGER, name TEXT)")
            .expect("create regions");
        db.execute("CREATE TABLE orders (user_id INTEGER)")
            .expect("create orders");
        db.execute("INSERT INTO users (id, region_id) VALUES (1, 10)")
            .expect("insert user");
        db.execute("INSERT INTO users (id, region_id) VALUES (2, 20)")
            .expect("insert user");
        db.execute("INSERT INTO regions (id, name) VALUES (10, 'NA')")
            .expect("insert region");
        db.execute("INSERT INTO regions (id, name) VALUES (20, 'EU')")
            .expect("insert region");
        db.execute("INSERT INTO orders (user_id) VALUES (1)")
            .expect("insert order");
        db.execute("INSERT INTO orders (user_id) VALUES (1)")
            .expect("insert order");
        db.execute("INSERT INTO orders (user_id) VALUES (2)")
            .expect("insert order");

        let result = db
            .execute(
                "SELECT users.id, regions.name, \
                 (SELECT COUNT(*) FROM orders WHERE orders.user_id = users.id) \
                 FROM users, regions \
                 WHERE users.region_id = regions.id \
                 ORDER BY users.id",
            )
            .expect("execute join with correlated subquery");

        assert_eq!(
            result.rows,
            vec![
                vec![
                    Value::Integer(1),
                    Value::Text("NA".to_string()),
                    Value::Integer(2),
                ],
                vec![
                    Value::Integer(2),
                    Value::Text("EU".to_string()),
                    Value::Integer(1),
                ],
            ]
        );
    }

    #[test]
    fn correlated_exists_with_join_filtering() {
        let path = temp_db_path("correlated_join_exists");
        let mut db = Database::create(path.to_string_lossy().as_ref()).expect("create should succeed");
        db.execute("CREATE TABLE users (id INTEGER, region_id INTEGER)")
            .expect("create users");
        db.execute("CREATE TABLE regions (id INTEGER, name TEXT)")
            .expect("create regions");
        db.execute("CREATE TABLE orders (user_id INTEGER, amount INTEGER)")
            .expect("create orders");
        db.execute("INSERT INTO users (id, region_id) VALUES (1, 10)")
            .expect("insert user");
        db.execute("INSERT INTO users (id, region_id) VALUES (2, 10)")
            .expect("insert user");
        db.execute("INSERT INTO users (id, region_id) VALUES (3, 20)")
            .expect("insert user");
        db.execute("INSERT INTO regions (id, name) VALUES (10, 'NA')")
            .expect("insert region");
        db.execute("INSERT INTO regions (id, name) VALUES (20, 'EU')")
            .expect("insert region");
        db.execute("INSERT INTO orders (user_id, amount) VALUES (1, 5)")
            .expect("insert order");
        db.execute("INSERT INTO orders (user_id, amount) VALUES (2, 50)")
            .expect("insert order");
        db.execute("INSERT INTO orders (user_id, amount) VALUES (3, 7)")
            .expect("insert order");

        let result = db
            .execute(
                "SELECT users.id, regions.name \
                 FROM regions, users \
                 WHERE users.region_id = regions.id \
                   AND EXISTS(SELECT 1 FROM orders \
                              WHERE orders.user_id = users.id \
                                AND orders.amount >= 10) \
                 ORDER BY users.id",
            )
            .expect("execute join with correlated exists");

        assert_eq!(
            result.rows,
            vec![vec![Value::Integer(2), Value::Text("NA".to_string())]]
        );
    }
}

#[cfg(test)]
fn build_combinations_ordered<'a>(
    tables: &'a [TableContext<'a>],
    order: &[usize],
) -> Vec<Vec<Value>> {
    if tables.is_empty() {
        return vec![Vec::new()];
    }
    let mut results = Vec::new();
    let mut selected: Vec<Option<&'a [Value]>> = vec![None; tables.len()];
    build_combinations_recursive_test(tables, order, 0, &mut selected, &mut results);
    results
}

#[cfg(test)]
fn build_combinations_recursive_test<'a>(
    tables: &'a [TableContext<'a>],
    order: &[usize],
    level: usize,
    selected: &mut Vec<Option<&'a [Value]>>,
    results: &mut Vec<Vec<Value>>,
) {
    if level == order.len() {
        let (current, combined) = build_row_contexts(tables, selected);
        let _ = current;
        results.push(combined);
        return;
    }
    let table_idx = order[level];
    for row in &tables[table_idx].rows {
        selected[table_idx] = Some(row.as_slice());
        build_combinations_recursive_test(tables, order, level + 1, selected, results);
    }
    selected[table_idx] = None;
}

#[cfg(test)]
fn count_join_leaves_with_predicates<'a>(
    tables: &'a [TableContext<'a>],
    order: &[usize],
    join_predicates: &[JoinPredicate],
    db: &'a Database,
) -> Result<usize, EngineError> {
    let mut selected: Vec<Option<&'a [Value]>> = vec![None; tables.len()];
    let cache = Rc::new(RefCell::new(HashMap::new()));
    count_join_leaves_recursive(
        tables,
        order,
        0,
        &mut selected,
        db,
        &cache,
        join_predicates,
    )
}

#[cfg(test)]
fn count_join_leaves_recursive<'a>(
    tables: &'a [TableContext<'a>],
    order: &[usize],
    level: usize,
    selected: &mut Vec<Option<&'a [Value]>>,
    db: &'a Database,
    subquery_cache: &Rc<RefCell<HashMap<usize, Value>>>,
    join_predicates: &[JoinPredicate],
) -> Result<usize, EngineError> {
    if tables.is_empty() {
        return Ok(0);
    }
    if level == order.len() {
        return Ok(1);
    }
    let table_idx = order[level];
    let mut count = 0;
    for row in &tables[table_idx].rows {
        selected[table_idx] = Some(row.as_slice());
        if eval_join_predicates(
            tables,
            selected,
            &[],
            db,
            subquery_cache,
            &None,
            join_predicates,
            table_idx,
        )? {
            count += count_join_leaves_recursive(
                tables,
                order,
                level + 1,
                selected,
                db,
                subquery_cache,
                join_predicates,
            )?;
        }
    }
    selected[table_idx] = None;
    Ok(count)
}
