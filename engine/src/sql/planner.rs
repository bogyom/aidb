use super::lexer::Literal as AstLiteral;
use super::parser::{
    CastType, Expr as AstExpr, FunctionArg as AstFunctionArg, FunctionModifier, OrderDirection,
    Select, SelectItem, SetExpr, SetOp, Statement, UnaryOp as AstUnaryOp,
};
use super::parser::BinaryOp as AstBinaryOp;
use crate::{Catalog, ColumnMeta, SqlType, TableSchema, Value};

#[derive(Debug, Clone, PartialEq)]
pub struct Plan {
    pub root: PlanNode,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PlanNode {
    Values { rows: Vec<Vec<Value>> },
    Scan(TableScan),
    Filter { predicate: ExprPlan, input: Box<PlanNode> },
    GroupBy {
        keys: Vec<ExprPlan>,
        aggregates: Vec<ProjectionItem>,
        input: Box<PlanNode>,
    },
    Having { predicate: ExprPlan, input: Box<PlanNode> },
    Order { by: Vec<OrderByPlan>, input: Box<PlanNode> },
    Limit { limit: usize, input: Box<PlanNode> },
    Projection { items: Vec<ProjectionItem>, input: Box<PlanNode> },
    SetOp {
        left: Box<PlanNode>,
        op: SetOp,
        right: Box<PlanNode>,
        all: bool,
        columns: Vec<ColumnMeta>,
    },
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
    pub is_aggregate: bool,
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
    Cast { expr: Box<ExprPlan>, ty: CastType },
    Between {
        expr: Box<ExprPlan>,
        low: Box<ExprPlan>,
        high: Box<ExprPlan>,
        negated: bool,
    },
    InList {
        expr: Box<ExprPlan>,
        list: Vec<ExprPlan>,
        negated: bool,
    },
    Case {
        operand: Option<Box<ExprPlan>>,
        when_thens: Vec<(ExprPlan, ExprPlan)>,
        else_expr: Option<Box<ExprPlan>>,
    },
    Function {
        name: String,
        modifier: FunctionModifier,
        args: Vec<FunctionArgPlan>,
    },
    Subquery(Box<Select>),
    Exists(Box<Select>),
    IsNull { expr: Box<ExprPlan>, negated: bool },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Not,
    Neg,
    Pos,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Eq,
    NotEq,
    Lt,
    Lte,
    Gt,
    Gte,
    Add,
    Sub,
    Mul,
    Div,
    DivInt,
    And,
    Or,
}

#[derive(Debug, Clone, PartialEq)]
pub enum FunctionArgPlan {
    Expr(ExprPlan),
    Star,
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
        Statement::SetOperation(expr) => {
            let root = plan_set_expr(expr, catalog)?;
            Ok(Some(Plan { root }))
        }
        _ => Ok(None),
    }
}

pub fn plan_select(select: &Select, catalog: &Catalog) -> Result<Plan, PlanError> {
    let table = match select.from.as_slice() {
        [] => None,
        [from] => catalog.table(&from.table),
        _ => {
            return Err(PlanError::new(
                "select with multiple tables is not supported",
            ))
        }
    };

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

    let planned_items = plan_projection_items(&select.items, table)?;
    let has_aggregate = planned_items.iter().any(|item| item.is_aggregate);
    let needs_grouping =
        !select.group_by.is_empty() || select.having.is_some() || has_aggregate;

    if needs_grouping {
        let mut keys = Vec::with_capacity(select.group_by.len());
        for expr in &select.group_by {
            keys.push(plan_expr(expr, table)?);
        }

        let mut aggregates: Vec<ProjectionItem> = planned_items
            .iter()
            .filter(|item| item.is_aggregate)
            .cloned()
            .collect();

        let planned_having = if let Some(having) = &select.having {
            let planned_having = plan_expr(having, table)?;
            collect_aggregate_exprs(&planned_having, &mut aggregates);
            Some(planned_having)
        } else {
            None
        };

        node = PlanNode::GroupBy {
            keys,
            aggregates,
            input: Box::new(node),
        };

        if let Some(predicate) = planned_having {
            node = PlanNode::Having {
                predicate,
                input: Box::new(node),
            };
        }
    }

    if !select.order_by.is_empty() {
        let mut by = Vec::with_capacity(select.order_by.len());
        for order_by in &select.order_by {
            let expr = plan_order_expr(&order_by.expr, &select.items, table)?;
            by.push(OrderByPlan {
                expr,
                direction: order_by.direction.clone(),
            });
        }
        node = PlanNode::Order {
            by,
            input: Box::new(node),
        };
    }

    if let Some(limit) = select.limit {
        node = PlanNode::Limit {
            limit: limit as usize,
            input: Box::new(node),
        };
    }

    let items = planned_items;
    node = PlanNode::Projection {
        items,
        input: Box::new(node),
    };

    Ok(Plan { root: node })
}

fn plan_set_expr(expr: &SetExpr, catalog: &Catalog) -> Result<PlanNode, PlanError> {
    match expr {
        SetExpr::Select(select) => Ok(plan_select(select, catalog)?.root),
        SetExpr::SetOp { left, op, right, all } => {
            let left_plan = plan_set_expr(left, catalog)?;
            let right_plan = plan_set_expr(right, catalog)?;
            let left_columns = output_columns(&left_plan)?;
            let right_columns = output_columns(&right_plan)?;
            if left_columns.len() != right_columns.len() {
                return Err(PlanError::new(
                    "set operation requires operands with same column count",
                ));
            }
            Ok(PlanNode::SetOp {
                left: Box::new(left_plan),
                op: *op,
                right: Box::new(right_plan),
                all: *all,
                columns: left_columns,
            })
        }
    }
}

fn output_columns(plan: &PlanNode) -> Result<Vec<ColumnMeta>, PlanError> {
    match plan {
        PlanNode::Values { rows } => {
            let width = rows.first().map(|row| row.len()).unwrap_or(0);
            let mut columns = Vec::with_capacity(width);
            for _ in 0..width {
                columns.push(ColumnMeta::new("expr", SqlType::Text));
            }
            Ok(columns)
        }
        PlanNode::Scan(scan) => Ok(scan
            .columns
            .iter()
            .map(|col| ColumnMeta::new(col.name.clone(), col.sql_type))
            .collect()),
        PlanNode::Filter { input, .. }
        | PlanNode::Having { input, .. }
        | PlanNode::Order { input, .. }
        | PlanNode::Limit { input, .. } => output_columns(input),
        PlanNode::GroupBy {
            keys, aggregates, ..
        } => {
            let mut columns = Vec::with_capacity(keys.len() + aggregates.len());
            for key in keys {
                columns.push(ColumnMeta::new(label_for_expr(key), expr_plan_type(key)));
            }
            for aggregate in aggregates {
                columns.push(ColumnMeta::new(
                    aggregate.label.clone(),
                    aggregate.sql_type,
                ));
            }
            Ok(columns)
        }
        PlanNode::Projection { items, .. } => Ok(items
            .iter()
            .map(|item| ColumnMeta::new(item.label.clone(), item.sql_type))
            .collect()),
        PlanNode::SetOp { columns, .. } => Ok(columns.clone()),
    }
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
                        is_aggregate: false,
                    });
                }
            }
            SelectItem::Expr(expr) => {
                let planned_expr = plan_expr(expr, table)?;
                let sql_type = expr_plan_type(&planned_expr);
                let label = label_for_expr(&planned_expr);
                let is_aggregate = expr_contains_aggregate(&planned_expr);
                planned.push(ProjectionItem {
                    label,
                    sql_type,
                    expr: planned_expr,
                    is_aggregate,
                });
            }
        }
    }
    Ok(planned)
}

fn plan_order_expr(
    expr: &AstExpr,
    items: &[SelectItem],
    table: Option<&TableSchema>,
) -> Result<ExprPlan, PlanError> {
    if let AstExpr::Literal(AstLiteral::Number(raw)) = expr {
        if !raw.contains('.') {
            if let Ok(index) = raw.parse::<usize>() {
                if index == 0 || index > items.len() {
                    return Err(PlanError::new("ORDER BY position out of range"));
                }
                return match &items[index - 1] {
                    SelectItem::Expr(expr) => plan_expr(expr, table),
                    SelectItem::Wildcard(_) => Err(PlanError::new(
                        "ORDER BY position refers to wildcard select item",
                    )),
                };
            }
        }
    }
    plan_expr(expr, table)
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

fn expr_contains_aggregate(expr: &ExprPlan) -> bool {
    match expr {
        ExprPlan::Function { name, .. } => {
            let name = name.to_ascii_lowercase();
            name == "count" || name == "avg" || name == "sum" || name == "min" || name == "max"
        }
        ExprPlan::Unary { expr, .. } => expr_contains_aggregate(expr),
        ExprPlan::Cast { expr, .. } => expr_contains_aggregate(expr),
        ExprPlan::Binary { left, right, .. } => {
            expr_contains_aggregate(left) || expr_contains_aggregate(right)
        }
        ExprPlan::Between {
            expr,
            low,
            high,
            ..
        } => {
            expr_contains_aggregate(expr)
                || expr_contains_aggregate(low)
                || expr_contains_aggregate(high)
        }
        ExprPlan::InList { expr, list, .. } => {
            expr_contains_aggregate(expr) || list.iter().any(expr_contains_aggregate)
        }
        ExprPlan::Case {
            operand,
            when_thens,
            else_expr,
        } => {
            operand
                .as_ref()
                .map_or(false, |expr| expr_contains_aggregate(expr))
                || when_thens.iter().any(|(when_expr, then_expr)| {
                    expr_contains_aggregate(when_expr) || expr_contains_aggregate(then_expr)
                })
                || else_expr
                    .as_ref()
                    .map_or(false, |expr| expr_contains_aggregate(expr))
        }
        ExprPlan::IsNull { expr, .. } => expr_contains_aggregate(expr),
        ExprPlan::Subquery(_) | ExprPlan::Exists(_) => false,
        ExprPlan::Column(_) | ExprPlan::Literal(_) => false,
    }
}

fn collect_aggregate_exprs(expr: &ExprPlan, aggregates: &mut Vec<ProjectionItem>) {
    match expr {
        ExprPlan::Function { name, .. } => {
            let name = name.to_ascii_lowercase();
            if name == "count" || name == "avg" {
                if !aggregates.iter().any(|item| item.expr == *expr) {
                    let sql_type = expr_plan_type(expr);
                    let label = label_for_expr(expr);
                    aggregates.push(ProjectionItem {
                        expr: expr.clone(),
                        label,
                        sql_type,
                        is_aggregate: true,
                    });
                }
            }
        }
        ExprPlan::Unary { expr, .. } => collect_aggregate_exprs(expr, aggregates),
        ExprPlan::Cast { expr, .. } => collect_aggregate_exprs(expr, aggregates),
        ExprPlan::Binary { left, right, .. } => {
            collect_aggregate_exprs(left, aggregates);
            collect_aggregate_exprs(right, aggregates);
        }
        ExprPlan::Between { expr, low, high, .. } => {
            collect_aggregate_exprs(expr, aggregates);
            collect_aggregate_exprs(low, aggregates);
            collect_aggregate_exprs(high, aggregates);
        }
        ExprPlan::InList { expr, list, .. } => {
            collect_aggregate_exprs(expr, aggregates);
            for item in list {
                collect_aggregate_exprs(item, aggregates);
            }
        }
        ExprPlan::Case {
            operand,
            when_thens,
            else_expr,
        } => {
            if let Some(operand) = operand {
                collect_aggregate_exprs(operand, aggregates);
            }
            for (when_expr, then_expr) in when_thens {
                collect_aggregate_exprs(when_expr, aggregates);
                collect_aggregate_exprs(then_expr, aggregates);
            }
            if let Some(else_expr) = else_expr {
                collect_aggregate_exprs(else_expr, aggregates);
            }
        }
        ExprPlan::IsNull { expr, .. } => collect_aggregate_exprs(expr, aggregates),
        ExprPlan::Column(_)
        | ExprPlan::Literal(_)
        | ExprPlan::Subquery(_)
        | ExprPlan::Exists(_) => {}
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
                AstUnaryOp::Neg => UnaryOp::Neg,
                AstUnaryOp::Pos => UnaryOp::Pos,
            };
            Ok(ExprPlan::Unary {
                op,
                expr: Box::new(expr),
            })
        }
        AstExpr::Cast { expr, ty } => {
            let expr = plan_expr(expr, table)?;
            Ok(ExprPlan::Cast {
                expr: Box::new(expr),
                ty: *ty,
            })
        }
        AstExpr::Binary { left, op, right } => {
            let left = plan_expr(left, table)?;
            let right = plan_expr(right, table)?;
            let op = match op {
                AstBinaryOp::Eq => BinaryOp::Eq,
                AstBinaryOp::NotEq => BinaryOp::NotEq,
                AstBinaryOp::Lt => BinaryOp::Lt,
                AstBinaryOp::Lte => BinaryOp::Lte,
                AstBinaryOp::Gt => BinaryOp::Gt,
                AstBinaryOp::Gte => BinaryOp::Gte,
                AstBinaryOp::Add => BinaryOp::Add,
                AstBinaryOp::Sub => BinaryOp::Sub,
                AstBinaryOp::Mul => BinaryOp::Mul,
                AstBinaryOp::Div => BinaryOp::Div,
                AstBinaryOp::DivInt => BinaryOp::DivInt,
                AstBinaryOp::And => BinaryOp::And,
                AstBinaryOp::Or => BinaryOp::Or,
            };
            ensure_comparable(&op, &left, &right)?;
            Ok(ExprPlan::Binary {
                left: Box::new(left),
                op,
                right: Box::new(right),
            })
        }
        AstExpr::Between {
            expr,
            low,
            high,
            negated,
        } => Ok(ExprPlan::Between {
            expr: Box::new(plan_expr(expr, table)?),
            low: Box::new(plan_expr(low, table)?),
            high: Box::new(plan_expr(high, table)?),
            negated: *negated,
        }),
        AstExpr::InList {
            expr,
            list,
            negated,
        } => {
            let planned_expr = plan_expr(expr, table)?;
            let mut planned_list = Vec::with_capacity(list.len());
            for item in list {
                let planned_item = plan_expr(item, table)?;
                ensure_comparable(&BinaryOp::Eq, &planned_expr, &planned_item)?;
                planned_list.push(planned_item);
            }
            Ok(ExprPlan::InList {
                expr: Box::new(planned_expr),
                list: planned_list,
                negated: *negated,
            })
        }
        AstExpr::Case {
            operand,
            when_thens,
            else_expr,
        } => {
            let operand = operand
                .as_ref()
                .map(|expr| plan_expr(expr, table).map(Box::new))
                .transpose()?;
            let mut planned_when_thens = Vec::with_capacity(when_thens.len());
            for (when_expr, then_expr) in when_thens {
                planned_when_thens.push((
                    plan_expr(when_expr, table)?,
                    plan_expr(then_expr, table)?,
                ));
            }
            let else_expr = else_expr
                .as_ref()
                .map(|expr| plan_expr(expr, table).map(Box::new))
                .transpose()?;
            Ok(ExprPlan::Case {
                operand,
                when_thens: planned_when_thens,
                else_expr,
            })
        }
        AstExpr::Function { name, modifier, args } => {
            let mut planned_args = Vec::with_capacity(args.len());
            for arg in args {
                match arg {
                    AstFunctionArg::Expr(expr) => {
                        planned_args.push(FunctionArgPlan::Expr(plan_expr(expr, table)?));
                    }
                    AstFunctionArg::Star => planned_args.push(FunctionArgPlan::Star),
                }
            }
            validate_function_arity(name, *modifier, &planned_args)?;
            Ok(ExprPlan::Function {
                name: name.clone(),
                modifier: *modifier,
                args: planned_args,
            })
        }
        AstExpr::Subquery(select) => Ok(ExprPlan::Subquery(select.clone())),
        AstExpr::Exists(select) => Ok(ExprPlan::Exists(select.clone())),
        AstExpr::IsNull { expr, negated } => Ok(ExprPlan::IsNull {
            expr: Box::new(plan_expr(expr, table)?),
            negated: *negated,
        }),
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

fn validate_function_arity(
    name: &str,
    modifier: FunctionModifier,
    args: &[FunctionArgPlan],
) -> Result<(), PlanError> {
    let lower = name.to_ascii_lowercase();
    let is_aggregate = matches!(lower.as_str(), "count" | "avg" | "sum" | "min" | "max");
    if matches!(modifier, FunctionModifier::Distinct) && !is_aggregate {
        return Err(PlanError::new(
            "DISTINCT is only supported for aggregate functions",
        ));
    }
    if lower == "abs" {
        if args.len() != 1 {
            return Err(PlanError::new("ABS expects 1 argument"));
        }
        if matches!(args[0], FunctionArgPlan::Star) {
            return Err(PlanError::new("ABS does not accept '*'"));
        }
    } else if lower == "count" {
        if args.len() != 1 {
            return Err(PlanError::new("COUNT expects 1 argument"));
        }
        if matches!(modifier, FunctionModifier::Distinct)
            && matches!(args[0], FunctionArgPlan::Star)
        {
            return Err(PlanError::new("COUNT DISTINCT does not accept '*'"));
        }
    } else if lower == "avg" {
        if args.len() != 1 {
            return Err(PlanError::new("AVG expects 1 argument"));
        }
        if matches!(args[0], FunctionArgPlan::Star) {
            return Err(PlanError::new("AVG does not accept '*'"));
        }
    } else if lower == "sum" {
        if args.len() != 1 {
            return Err(PlanError::new("SUM expects 1 argument"));
        }
        if matches!(args[0], FunctionArgPlan::Star) {
            return Err(PlanError::new("SUM does not accept '*'"));
        }
    } else if lower == "min" {
        if args.len() != 1 {
            return Err(PlanError::new("MIN expects 1 argument"));
        }
        if matches!(args[0], FunctionArgPlan::Star) {
            return Err(PlanError::new("MIN does not accept '*'"));
        }
    } else if lower == "max" {
        if args.len() != 1 {
            return Err(PlanError::new("MAX expects 1 argument"));
        }
        if matches!(args[0], FunctionArgPlan::Star) {
            return Err(PlanError::new("MAX does not accept '*'"));
        }
    } else if lower == "coalesce" {
        if args.is_empty() {
            return Err(PlanError::new("COALESCE expects at least 1 argument"));
        }
        if args.iter().any(|arg| matches!(arg, FunctionArgPlan::Star)) {
            return Err(PlanError::new("COALESCE does not accept '*'"));
        }
    } else if lower == "nullif" {
        if args.len() != 2 {
            return Err(PlanError::new("NULLIF expects 2 arguments"));
        }
        if args.iter().any(|arg| matches!(arg, FunctionArgPlan::Star)) {
            return Err(PlanError::new("NULLIF does not accept '*'"));
        }
    }
    Ok(())
}

fn ensure_comparable(
    op: &BinaryOp,
    left: &ExprPlan,
    right: &ExprPlan,
) -> Result<(), PlanError> {
    let is_comparison = matches!(
        op,
        BinaryOp::Eq
            | BinaryOp::NotEq
            | BinaryOp::Lt
            | BinaryOp::Lte
            | BinaryOp::Gt
            | BinaryOp::Gte
    );
    if !is_comparison {
        return Ok(());
    }

    let left_type = comparison_operand_type(left);
    let right_type = comparison_operand_type(right);

    if left_type.is_none() || right_type.is_none() {
        return Ok(());
    }

    let left_type = left_type.unwrap();
    let right_type = right_type.unwrap();
    let comparable = match op {
        BinaryOp::Eq | BinaryOp::NotEq => types_compatible_eq(left_type, right_type),
        BinaryOp::Lt | BinaryOp::Lte | BinaryOp::Gt | BinaryOp::Gte => {
            types_compatible_order(left_type, right_type)
        }
        _ => true,
    };

    if comparable {
        Ok(())
    } else {
        Err(PlanError::new(format!(
            "unsupported comparison between {:?} and {:?}",
            left_type, right_type
        )))
    }
}

fn comparison_operand_type(expr: &ExprPlan) -> Option<SqlType> {
    match expr {
        ExprPlan::Literal(Value::Null) => None,
        ExprPlan::Subquery(_) => None,
        _ => Some(expr_plan_type(expr)),
    }
}

fn types_compatible_eq(left: SqlType, right: SqlType) -> bool {
    left == right || matches!((left, right), (SqlType::Integer, SqlType::Real) | (SqlType::Real, SqlType::Integer))
}

fn types_compatible_order(left: SqlType, right: SqlType) -> bool {
    left == right || matches!((left, right), (SqlType::Integer, SqlType::Real) | (SqlType::Real, SqlType::Integer))
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
        ExprPlan::Cast { ty, .. } => match ty {
            CastType::Signed | CastType::Integer => SqlType::Integer,
            CastType::Decimal | CastType::Real => SqlType::Real,
        },
        ExprPlan::Unary { op, expr } => match op {
            UnaryOp::Not => SqlType::Boolean,
            UnaryOp::Neg => expr_plan_type(expr),
            UnaryOp::Pos => expr_plan_type(expr),
        },
        ExprPlan::Binary { op, left, right } => match op {
            BinaryOp::And
            | BinaryOp::Or
            | BinaryOp::Eq
            | BinaryOp::NotEq
            | BinaryOp::Lt
            | BinaryOp::Lte
            | BinaryOp::Gt
            | BinaryOp::Gte => SqlType::Boolean,
            BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul => {
                let left = expr_plan_type(left);
                let right = expr_plan_type(right);
                if left == SqlType::Real || right == SqlType::Real {
                    SqlType::Real
                } else if left == SqlType::Integer && right == SqlType::Integer {
                    SqlType::Integer
                } else {
                    SqlType::Text
                }
            }
            BinaryOp::Div => SqlType::Real,
            BinaryOp::DivInt => SqlType::Integer,
        },
        ExprPlan::Between { .. } => SqlType::Boolean,
        ExprPlan::InList { .. } => SqlType::Boolean,
        ExprPlan::IsNull { .. } => SqlType::Boolean,
        ExprPlan::Exists(_) => SqlType::Boolean,
        ExprPlan::Case {
            when_thens,
            else_expr,
            ..
        } => {
            if let Some((_, then_expr)) = when_thens.first() {
                expr_plan_type(then_expr)
            } else if let Some(else_expr) = else_expr {
                expr_plan_type(else_expr)
            } else {
                SqlType::Text
            }
        }
        ExprPlan::Function {
            name,
            modifier: _,
            args,
        } => {
            let name = name.to_ascii_lowercase();
            if name == "count" {
                SqlType::Integer
            } else if name == "avg" {
                SqlType::Real
            } else if name == "sum" {
                let arg_type = args.iter().find_map(|arg| match arg {
                    FunctionArgPlan::Expr(expr) => Some(expr_plan_type(expr)),
                    FunctionArgPlan::Star => None,
                });
                if matches!(arg_type, Some(SqlType::Real)) {
                    SqlType::Real
                } else {
                    SqlType::Integer
                }
            } else if name == "min" || name == "max" {
                args.iter()
                    .find_map(|arg| match arg {
                        FunctionArgPlan::Expr(expr) => Some(expr_plan_type(expr)),
                        FunctionArgPlan::Star => None,
                    })
                    .unwrap_or(SqlType::Text)
            } else if name == "abs" {
                SqlType::Integer
            } else if name == "coalesce" {
                args.iter().find_map(|arg| match arg {
                    FunctionArgPlan::Expr(expr) => Some(expr_plan_type(expr)),
                    FunctionArgPlan::Star => None,
                }).unwrap_or(SqlType::Text)
            } else {
                SqlType::Text
            }
        }
        ExprPlan::Subquery(_) => SqlType::Text,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Catalog, ColumnMeta, ColumnSchema, SqlType, TableSchema};
    use super::super::parser;
    use std::path::Path;

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

    fn catalog_with_t1() -> Catalog {
        let mut catalog = Catalog::new();
        catalog.add_table(TableSchema::new(
            "t1",
            vec![
                ColumnSchema::new("a", SqlType::Integer, false),
                ColumnSchema::new("b", SqlType::Integer, false),
                ColumnSchema::new("c", SqlType::Integer, false),
                ColumnSchema::new("d", SqlType::Integer, false),
                ColumnSchema::new("e", SqlType::Integer, false),
            ],
        ));
        catalog
    }

    fn load_queries(path: &Path) -> Vec<String> {
        let contents = std::fs::read_to_string(path).expect("read sqllogictest file");
        let mut queries = Vec::new();
        let mut current = Vec::new();
        let mut in_query = false;

        for line in contents.lines() {
            let trimmed = line.trim_end();
            if trimmed.starts_with('#') {
                continue;
            }

            if in_query {
                if trimmed == "----" {
                    let sql = current.join("\n");
                    if !sql.trim().is_empty() {
                        queries.push(sql.trim().to_string());
                    }
                    current.clear();
                    in_query = false;
                } else if !trimmed.is_empty() {
                    current.push(trimmed.to_string());
                }
                continue;
            }

            if trimmed.starts_with("query ") {
                in_query = true;
                current.clear();
            }
        }

        queries
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
    fn planner_plans_multiple_order_by_keys() {
        let catalog = catalog_with_users();
        let statements = parser::parse(
            "SELECT id, name FROM users ORDER BY id, name DESC",
        )
        .expect("parse");
        let select = match &statements[0] {
            Statement::Select(select) => select,
            _ => panic!("expected select"),
        };
        let plan = plan_select(select, &catalog).expect("plan");
        let order_by = match plan.root {
            PlanNode::Projection { input, .. } => match *input {
                PlanNode::Order { by, .. } => by,
                _ => panic!("expected order"),
            },
            _ => panic!("expected projection"),
        };

        assert_eq!(order_by.len(), 2);
        match &order_by[0].expr {
            ExprPlan::Column(column) => assert_eq!(column.name, "id"),
            _ => panic!("expected column expression"),
        }
        assert!(matches!(order_by[0].direction, OrderDirection::Asc));

        match &order_by[1].expr {
            ExprPlan::Column(column) => assert_eq!(column.name, "name"),
            _ => panic!("expected column expression"),
        }
        assert!(matches!(order_by[1].direction, OrderDirection::Desc));
    }

    #[test]
    fn planner_resolves_order_by_ordinal() {
        let catalog = catalog_with_users();
        let statements =
            parser::parse("SELECT id, name FROM users ORDER BY 1").expect("parse");
        let select = match &statements[0] {
            Statement::Select(select) => select,
            _ => panic!("expected select"),
        };
        let plan = plan_select(select, &catalog).expect("plan");
        let order_by = match plan.root {
            PlanNode::Projection { input, .. } => match *input {
                PlanNode::Order { by, .. } => by,
                _ => panic!("expected order"),
            },
            _ => panic!("expected projection"),
        };
        assert_eq!(order_by.len(), 1);
        match &order_by[0].expr {
            ExprPlan::Column(column) => assert_eq!(column.name, "id"),
            _ => panic!("expected column expression"),
        }
    }

    #[test]
    fn planner_rejects_out_of_range_order_by_ordinal() {
        let catalog = catalog_with_users();
        let statements =
            parser::parse("SELECT id FROM users ORDER BY 2").expect("parse");
        let select = match &statements[0] {
            Statement::Select(select) => select,
            _ => panic!("expected select"),
        };
        let err = plan_select(select, &catalog).expect_err("expected error");
        assert!(err.message.contains("ORDER BY position out of range"));
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

    fn planned_filter_op(sql: &str, catalog: &Catalog) -> BinaryOp {
        let statements = parser::parse(sql).expect("parse");
        let select = match &statements[0] {
            Statement::Select(select) => select,
            _ => panic!("expected select"),
        };
        let plan = plan_select(select, catalog).expect("plan");
        let predicate = match plan.root {
            PlanNode::Projection { input, .. } => match *input {
                PlanNode::Filter { predicate, .. } => predicate,
                _ => panic!("expected filter"),
            },
            _ => panic!("expected projection"),
        };
        match predicate {
            ExprPlan::Binary { op, .. } => op,
            _ => panic!("expected binary predicate"),
        }
    }

    #[test]
    fn planner_plans_comparison_ops() {
        let catalog = catalog_with_users();
        let cases = vec![
            ("SELECT name FROM users WHERE id = 1", BinaryOp::Eq),
            ("SELECT name FROM users WHERE id <> 1", BinaryOp::NotEq),
            ("SELECT name FROM users WHERE id < 1", BinaryOp::Lt),
            ("SELECT name FROM users WHERE id <= 1", BinaryOp::Lte),
            ("SELECT name FROM users WHERE id > 1", BinaryOp::Gt),
            ("SELECT name FROM users WHERE id >= 1", BinaryOp::Gte),
        ];

        for (sql, expected) in cases {
            let op = planned_filter_op(sql, &catalog);
            assert_eq!(op, expected);
        }
    }

    #[test]
    fn planner_rejects_incompatible_comparison() {
        let catalog = catalog_with_users();
        let statements = parser::parse("SELECT name FROM users WHERE name < id").expect("parse");
        let select = match &statements[0] {
            Statement::Select(select) => select,
            _ => panic!("expected select"),
        };
        let err = plan_select(select, &catalog).expect_err("expected error");
        assert!(err.message.contains("unsupported comparison"));
    }

    #[test]
    fn planner_respects_arithmetic_precedence() {
        let catalog = catalog_with_t1();
        let statements = parser::parse("SELECT a + b * c FROM t1").expect("parse");
        let select = match &statements[0] {
            Statement::Select(select) => select,
            _ => panic!("expected select"),
        };
        let plan = plan_select(select, &catalog).expect("plan");
        let projection = match plan.root {
            PlanNode::Projection { items, .. } => items,
            _ => panic!("expected projection"),
        };
        let expr = &projection[0].expr;
        match expr {
            ExprPlan::Binary { op, left, right } => {
                assert_eq!(*op, BinaryOp::Add);
                assert!(matches!(**left, ExprPlan::Column(_)));
                match &**right {
                    ExprPlan::Binary {
                        op,
                        left: mul_left,
                        right: mul_right,
                    } => {
                        assert_eq!(*op, BinaryOp::Mul);
                        assert!(matches!(**mul_left, ExprPlan::Column(_)));
                        assert!(matches!(**mul_right, ExprPlan::Column(_)));
                    }
                    _ => panic!("expected multiplication on right"),
                }
            }
            _ => panic!("expected binary expression"),
        }
    }

    #[test]
    fn planner_supports_unary_negation() {
        let catalog = catalog_with_t1();
        let statements = parser::parse("SELECT -a FROM t1").expect("parse");
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
            ExprPlan::Unary { op, expr } => {
                assert_eq!(*op, UnaryOp::Neg);
                assert!(matches!(**expr, ExprPlan::Column(_)));
            }
            _ => panic!("expected unary negation"),
        }
    }

    fn planned_filter_expr(sql: &str, catalog: &Catalog) -> ExprPlan {
        let statements = parser::parse(sql).expect("parse");
        let select = match &statements[0] {
            Statement::Select(select) => select,
            _ => panic!("expected select"),
        };
        let plan = plan_select(select, catalog).expect("plan");
        match plan.root {
            PlanNode::Projection { input, .. } => match *input {
                PlanNode::Filter { predicate, .. } => predicate,
                _ => panic!("expected filter"),
            },
            _ => panic!("expected projection"),
        }
    }

    fn planned_projection_expr(sql: &str, catalog: &Catalog) -> ExprPlan {
        let statements = parser::parse(sql).expect("parse");
        let select = match &statements[0] {
            Statement::Select(select) => select,
            _ => panic!("expected select"),
        };
        let plan = plan_select(select, catalog).expect("plan");
        match plan.root {
            PlanNode::Projection { items, .. } => items
                .into_iter()
                .next()
                .expect("expected projection item")
                .expr,
            _ => panic!("expected projection"),
        }
    }

    fn planned_projection_item(sql: &str, catalog: &Catalog) -> ProjectionItem {
        let statements = parser::parse(sql).expect("parse");
        let select = match &statements[0] {
            Statement::Select(select) => select,
            _ => panic!("expected select"),
        };
        let plan = plan_select(select, catalog).expect("plan");
        match plan.root {
            PlanNode::Projection { items, .. } => items
                .into_iter()
                .next()
                .expect("expected projection item"),
            _ => panic!("expected projection"),
        }
    }

    #[test]
    fn planner_plans_between_expressions() {
        let catalog = catalog_with_users();
        let expr = planned_filter_expr("SELECT name FROM users WHERE id BETWEEN 1 AND 3", &catalog);
        match expr {
            ExprPlan::Between { expr, low, high, negated } => {
                assert!(!negated);
                assert!(matches!(*expr, ExprPlan::Column(_)));
                assert!(matches!(*low, ExprPlan::Literal(Value::Integer(1))));
                assert!(matches!(*high, ExprPlan::Literal(Value::Integer(3))));
            }
            _ => panic!("expected BETWEEN expression"),
        }
    }

    #[test]
    fn planner_plans_not_between_expressions() {
        let catalog = catalog_with_users();
        let expr = planned_filter_expr("SELECT name FROM users WHERE id NOT BETWEEN 1 AND 3", &catalog);
        match expr {
            ExprPlan::Between { negated, .. } => assert!(negated),
            _ => panic!("expected NOT BETWEEN expression"),
        }
    }

    #[test]
    fn planner_rejects_mismatched_set_op_columns() {
        let catalog = catalog_with_users();
        let statements = parser::parse(
            "SELECT id FROM users UNION SELECT id, name FROM users",
        )
        .expect("parse");
        let err = plan_statement(&statements[0], &catalog).expect_err("expected error");
        assert!(err.message.contains("column count"));
    }

    #[test]
    fn planner_uses_left_columns_for_set_op_output() {
        let catalog = catalog_with_users();
        let statements = parser::parse(
            "SELECT id, name FROM users UNION SELECT name, id FROM users",
        )
        .expect("parse");
        let plan = plan_statement(&statements[0], &catalog)
            .expect("plan")
            .expect("plan");
        match plan.root {
            PlanNode::SetOp { columns, .. } => {
                assert_eq!(
                    columns,
                    vec![
                        ColumnMeta::new("id", SqlType::Integer),
                        ColumnMeta::new("name", SqlType::Text)
                    ]
                );
            }
            _ => panic!("expected set operation plan"),
        }
    }

    #[test]
    fn planner_plans_in_list_expressions() {
        let catalog = catalog_with_users();
        let expr = planned_filter_expr("SELECT name FROM users WHERE id IN (1, 2, NULL)", &catalog);
        match expr {
            ExprPlan::InList { expr, list, negated } => {
                assert!(!negated);
                assert!(matches!(*expr, ExprPlan::Column(_)));
                assert_eq!(list.len(), 3);
                assert!(matches!(list[0], ExprPlan::Literal(Value::Integer(1))));
                assert!(matches!(list[2], ExprPlan::Literal(Value::Null)));
            }
            _ => panic!("expected IN list expression"),
        }
    }

    #[test]
    fn planner_plans_not_in_list_expressions() {
        let catalog = catalog_with_users();
        let expr = planned_filter_expr("SELECT name FROM users WHERE id NOT IN (1, 2)", &catalog);
        match expr {
            ExprPlan::InList { negated, list, .. } => {
                assert!(negated);
                assert_eq!(list.len(), 2);
            }
            _ => panic!("expected NOT IN list expression"),
        }
    }

    #[test]
    fn planner_plans_abs_function() {
        let catalog = catalog_with_users();
        let expr = planned_projection_expr("SELECT ABS(id) FROM users", &catalog);
        match expr {
            ExprPlan::Function { name, args, .. } => {
                assert_eq!(name.to_ascii_lowercase(), "abs");
                assert_eq!(args.len(), 1);
            }
            _ => panic!("expected function expression"),
        }
    }

    #[test]
    fn planner_plans_coalesce_function() {
        let catalog = catalog_with_users();
        let expr =
            planned_projection_expr("SELECT COALESCE(name, 'n/a') FROM users", &catalog);
        match expr {
            ExprPlan::Function { name, args, .. } => {
                assert_eq!(name.to_ascii_lowercase(), "coalesce");
                assert_eq!(args.len(), 2);
            }
            _ => panic!("expected function expression"),
        }
    }

    #[test]
    fn planner_marks_aggregate_projections() {
        let catalog = catalog_with_users();
        let aggregate = planned_projection_item("SELECT COUNT(*) FROM users", &catalog);
        assert!(aggregate.is_aggregate);

        let scalar = planned_projection_item("SELECT id FROM users", &catalog);
        assert!(!scalar.is_aggregate);
    }

    #[test]
    fn planner_plans_group_by_and_having() {
        let catalog = catalog_with_users();
        let statements = parser::parse(
            "SELECT id, COUNT(*) FROM users GROUP BY id HAVING COUNT(*) > 1",
        )
        .expect("parse");
        let select = match &statements[0] {
            Statement::Select(select) => select,
            _ => panic!("expected select"),
        };
        let plan = plan_select(select, &catalog).expect("plan");
        let (having, group_by) = match plan.root {
            PlanNode::Projection { input, .. } => match *input {
                PlanNode::Having { predicate, input } => (predicate, input),
                _ => panic!("expected having"),
            },
            _ => panic!("expected projection"),
        };

        match &having {
            ExprPlan::Binary { left, .. } => match &**left {
                ExprPlan::Function { name, .. } => {
                    assert_eq!(name.to_ascii_lowercase(), "count");
                }
                _ => panic!("expected count in having"),
            },
            _ => panic!("expected binary having predicate"),
        }

        match *group_by {
            PlanNode::GroupBy { keys, aggregates, input } => {
                assert_eq!(keys.len(), 1);
                match &keys[0] {
                    ExprPlan::Column(column) => assert_eq!(column.name, "id"),
                    _ => panic!("expected column key"),
                }
                assert_eq!(aggregates.len(), 1);
                match &aggregates[0].expr {
                    ExprPlan::Function { name, .. } => {
                        assert_eq!(name.to_ascii_lowercase(), "count");
                    }
                    _ => panic!("expected count aggregate"),
                }
                match *input {
                    PlanNode::Scan(_) => {}
                    _ => panic!("expected scan"),
                }
            }
            _ => panic!("expected group by"),
        }
    }

    #[test]
    fn planner_rejects_invalid_count_arity() {
        let catalog = catalog_with_users();
        let statements = parser::parse("SELECT COUNT() FROM users").expect("parse");
        let select = match &statements[0] {
            Statement::Select(select) => select,
            _ => panic!("expected select"),
        };
        let err = plan_select(select, &catalog).expect_err("expected error");
        assert!(err.message.contains("COUNT expects 1 argument"));
    }

    #[test]
    fn planner_rejects_invalid_avg_arity() {
        let catalog = catalog_with_users();
        let statements = parser::parse("SELECT AVG(*) FROM users").expect("parse");
        let select = match &statements[0] {
            Statement::Select(select) => select,
            _ => panic!("expected select"),
        };
        let err = plan_select(select, &catalog).expect_err("expected error");
        assert!(err.message.contains("AVG does not accept '*'"));
    }

    #[test]
    fn planner_plans_scalar_subquery_expression() {
        let catalog = catalog_with_users();
        let statements = parser::parse("SELECT (SELECT 1) FROM users").expect("parse");
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
            ExprPlan::Subquery(_) => {}
            _ => panic!("expected subquery expression"),
        }
    }

    #[test]
    fn planner_plans_exists_subquery_expression() {
        let catalog = catalog_with_users();
        let statements =
            parser::parse("SELECT name FROM users WHERE EXISTS (SELECT 1)").expect("parse");
        let select = match &statements[0] {
            Statement::Select(select) => select,
            _ => panic!("expected select"),
        };
        let plan = plan_select(select, &catalog).expect("plan");
        let predicate = match plan.root {
            PlanNode::Projection { input, .. } => match *input {
                PlanNode::Filter { predicate, .. } => predicate,
                _ => panic!("expected filter"),
            },
            _ => panic!("expected projection"),
        };
        match predicate {
            ExprPlan::Exists(_) => {}
            _ => panic!("expected exists expression"),
        }
    }

    #[test]
    fn planner_rejects_invalid_abs_arity() {
        let catalog = catalog_with_users();
        let statements = parser::parse("SELECT ABS(id, name) FROM users").expect("parse");
        let select = match &statements[0] {
            Statement::Select(select) => select,
            _ => panic!("expected select"),
        };
        let err = plan_select(select, &catalog).expect_err("expected error");
        assert!(err.message.contains("ABS expects 1 argument"));
    }

    #[test]
    fn planner_rejects_invalid_coalesce_arity() {
        let catalog = catalog_with_users();
        let statements = parser::parse("SELECT COALESCE() FROM users").expect("parse");
        let select = match &statements[0] {
            Statement::Select(select) => select,
            _ => panic!("expected select"),
        };
        let err = plan_select(select, &catalog).expect_err("expected error");
        assert!(err.message.contains("COALESCE expects at least 1 argument"));
    }

    #[test]
    fn planner_plans_searched_case_expression() {
        let catalog = catalog_with_users();
        let statements = parser::parse(
            "SELECT CASE WHEN id > 1 THEN name ELSE 'n/a' END FROM users",
        )
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
            ExprPlan::Case {
                operand,
                when_thens,
                else_expr,
            } => {
                assert!(operand.is_none());
                assert_eq!(when_thens.len(), 1);
                assert!(else_expr.is_some());
            }
            _ => panic!("expected case expression"),
        }
    }

    #[test]
    fn planner_plans_simple_case_expression() {
        let catalog = catalog_with_users();
        let statements = parser::parse(
            "SELECT CASE id WHEN 1 THEN name ELSE 'n/a' END FROM users",
        )
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
            ExprPlan::Case {
                operand,
                when_thens,
                else_expr,
            } => {
                assert!(operand.is_some());
                assert_eq!(when_thens.len(), 1);
                assert!(else_expr.is_some());
            }
            _ => panic!("expected case expression"),
        }
    }

    #[test]
    fn planner_compiles_select2_and_select3_queries() {
        let catalog = catalog_with_t1();
        let manifest_dir = Path::new(env!("CARGO_MANIFEST_DIR"));
        let base = manifest_dir.join("../tests/sqllogictest/test");
        let files = ["select2.test", "select3.test"];

        for file in files {
            let path = base.join(file);
            let queries = load_queries(&path);
            assert!(!queries.is_empty(), "no queries loaded from {}", path.display());
            for sql in queries {
                let statements = parser::parse(&sql).unwrap_or_else(|err| {
                    panic!("failed to parse {}: {} ({:?})", file, sql, err)
                });
                for stmt in statements {
                    if let Statement::Select(select) = stmt {
                        plan_select(&select, &catalog).unwrap_or_else(|err| {
                            panic!("failed to plan {}: {} ({})", file, sql, err.message)
                        });
                    }
                }
            }
        }
    }
}
