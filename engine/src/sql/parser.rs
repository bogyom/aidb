use super::lexer::{lex, Keyword, LexError, Literal, Symbol, Token, TokenKind};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Statement {
    CreateTable(CreateTable),
    DropTable(DropTable),
    Insert(Insert),
    Select(Select),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CreateTable {
    pub name: String,
    pub columns: Vec<ColumnDef>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ColumnDef {
    pub name: String,
    pub data_type: TypeName,
    pub primary_key: bool,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TypeName {
    Integer,
    Real,
    Text,
    Boolean,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DropTable {
    pub name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Insert {
    pub table: String,
    pub columns: Option<Vec<String>>,
    pub values: Vec<Expr>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Select {
    pub items: Vec<SelectItem>,
    pub from: Option<String>,
    pub filter: Option<Expr>,
    pub order_by: Option<OrderBy>,
    pub limit: Option<u64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectItem {
    Wildcard(Option<String>),
    Expr(Expr),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderBy {
    pub expr: Expr,
    pub direction: OrderDirection,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OrderDirection {
    Asc,
    Desc,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Expr {
    Identifier(Vec<String>),
    Literal(Literal),
    Unary { op: UnaryOp, expr: Box<Expr> },
    Binary { left: Box<Expr>, op: BinaryOp, right: Box<Expr> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnaryOp {
    Not,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BinaryOp {
    Eq,
    And,
    Or,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParseError {
    pub message: String,
    pub index: Option<usize>,
}

impl ParseError {
    fn new(message: impl Into<String>, index: Option<usize>) -> Self {
        Self {
            message: message.into(),
            index,
        }
    }
}

impl From<LexError> for ParseError {
    fn from(value: LexError) -> Self {
        match value {
            LexError::UnexpectedChar { ch, index } => {
                ParseError::new(format!("unexpected character '{ch}'"), Some(index))
            }
            LexError::UnterminatedString { start } => {
                ParseError::new("unterminated string literal", Some(start))
            }
        }
    }
}

pub fn parse(sql: &str) -> Result<Vec<Statement>, ParseError> {
    let tokens = lex(sql).map_err(ParseError::from)?;
    let mut stream = TokenStream::new(tokens);
    let mut statements = Vec::new();

    while !stream.is_eof() {
        if stream.consume_symbol(Symbol::Semicolon) {
            continue;
        }
        let stmt = parse_statement(&mut stream)?;
        statements.push(stmt);
        stream.consume_symbol(Symbol::Semicolon);
    }

    Ok(statements)
}

fn parse_statement(stream: &mut TokenStream) -> Result<Statement, ParseError> {
    if stream.consume_keyword(Keyword::Create) {
        parse_create_table(stream).map(Statement::CreateTable)
    } else if stream.consume_keyword(Keyword::Drop) {
        parse_drop_table(stream).map(Statement::DropTable)
    } else if stream.consume_keyword(Keyword::Insert) {
        parse_insert(stream).map(Statement::Insert)
    } else if stream.consume_keyword(Keyword::Select) {
        parse_select(stream).map(Statement::Select)
    } else {
        Err(stream.error("expected statement", None))
    }
}

fn parse_create_table(stream: &mut TokenStream) -> Result<CreateTable, ParseError> {
    stream.expect_keyword(Keyword::Table)?;
    let name = stream.expect_identifier()?;
    stream.expect_symbol(Symbol::LParen)?;

    let mut columns = Vec::new();
    loop {
        let column_name = stream.expect_identifier()?;
        let data_type = parse_type_name(stream)?;
        let mut primary_key = false;
        if stream.consume_keyword(Keyword::Primary) {
            stream.expect_keyword(Keyword::Key)?;
            primary_key = true;
        }
        columns.push(ColumnDef {
            name: column_name,
            data_type,
            primary_key,
        });

        if stream.consume_symbol(Symbol::Comma) {
            continue;
        }
        break;
    }

    stream.expect_symbol(Symbol::RParen)?;

    Ok(CreateTable { name, columns })
}

fn parse_drop_table(stream: &mut TokenStream) -> Result<DropTable, ParseError> {
    stream.expect_keyword(Keyword::Table)?;
    let name = stream.expect_identifier()?;
    Ok(DropTable { name })
}

fn parse_insert(stream: &mut TokenStream) -> Result<Insert, ParseError> {
    stream.expect_keyword(Keyword::Into)?;
    let table = stream.expect_identifier()?;

    let columns = if stream.consume_symbol(Symbol::LParen) {
        let mut cols = Vec::new();
        loop {
            cols.push(stream.expect_identifier()?);
            if stream.consume_symbol(Symbol::Comma) {
                continue;
            }
            break;
        }
        stream.expect_symbol(Symbol::RParen)?;
        Some(cols)
    } else {
        None
    };

    stream.expect_keyword(Keyword::Values)?;
    stream.expect_symbol(Symbol::LParen)?;
    let mut values = Vec::new();
    loop {
        values.push(parse_expr(stream)?);
        if stream.consume_symbol(Symbol::Comma) {
            continue;
        }
        break;
    }
    stream.expect_symbol(Symbol::RParen)?;

    Ok(Insert {
        table,
        columns,
        values,
    })
}

fn parse_select(stream: &mut TokenStream) -> Result<Select, ParseError> {
    let mut items = Vec::new();

    if stream.consume_symbol(Symbol::Star) {
        items.push(SelectItem::Wildcard(None));
    } else {
        loop {
            if let Some(wildcard) = parse_qualified_wildcard(stream)? {
                items.push(SelectItem::Wildcard(Some(wildcard)));
            } else {
                items.push(SelectItem::Expr(parse_expr(stream)?));
            }

            if stream.consume_symbol(Symbol::Comma) {
                continue;
            }
            break;
        }
    }

    let from = if stream.consume_keyword(Keyword::From) {
        Some(stream.expect_identifier()?)
    } else {
        None
    };

    let filter = if stream.consume_keyword(Keyword::Where) {
        Some(parse_expr(stream)?)
    } else {
        None
    };

    let order_by = if stream.consume_keyword(Keyword::Order) {
        stream.expect_keyword(Keyword::By)?;
        let expr = parse_expr(stream)?;
        let direction = if stream.consume_keyword(Keyword::Asc) {
            OrderDirection::Asc
        } else if stream.consume_keyword(Keyword::Desc) {
            OrderDirection::Desc
        } else {
            OrderDirection::Asc
        };
        Some(OrderBy { expr, direction })
    } else {
        None
    };

    let limit = if stream.consume_keyword(Keyword::Limit) {
        let literal = stream.expect_literal()?;
        match literal {
            Literal::Number(value) => value.parse::<u64>().ok(),
            _ => None,
        }
        .ok_or_else(|| stream.error("expected numeric limit", stream.last_span_start()))
        .map(Some)?
    } else {
        None
    };

    Ok(Select {
        items,
        from,
        filter,
        order_by,
        limit,
    })
}

fn parse_qualified_wildcard(stream: &mut TokenStream) -> Result<Option<String>, ParseError> {
    let checkpoint = stream.position();
    if let Some(name) = stream.consume_identifier() {
        if stream.consume_symbol(Symbol::Dot) {
            if stream.consume_symbol(Symbol::Star) {
                return Ok(Some(name));
            }
        }
    }
    stream.reset(checkpoint);
    Ok(None)
}

fn parse_type_name(stream: &mut TokenStream) -> Result<TypeName, ParseError> {
    if stream.consume_keyword(Keyword::Integer) {
        Ok(TypeName::Integer)
    } else if stream.consume_keyword(Keyword::Real) {
        Ok(TypeName::Real)
    } else if stream.consume_keyword(Keyword::Text) {
        Ok(TypeName::Text)
    } else if stream.consume_keyword(Keyword::Boolean) {
        Ok(TypeName::Boolean)
    } else {
        Err(stream.error("expected type name", None))
    }
}

fn parse_expr(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    parse_or(stream)
}

fn parse_or(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    let mut expr = parse_and(stream)?;
    while stream.consume_keyword(Keyword::Or) {
        let right = parse_and(stream)?;
        expr = Expr::Binary {
            left: Box::new(expr),
            op: BinaryOp::Or,
            right: Box::new(right),
        };
    }
    Ok(expr)
}

fn parse_and(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    let mut expr = parse_not(stream)?;
    while stream.consume_keyword(Keyword::And) {
        let right = parse_not(stream)?;
        expr = Expr::Binary {
            left: Box::new(expr),
            op: BinaryOp::And,
            right: Box::new(right),
        };
    }
    Ok(expr)
}

fn parse_not(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    if stream.consume_keyword(Keyword::Not) {
        let expr = parse_not(stream)?;
        Ok(Expr::Unary {
            op: UnaryOp::Not,
            expr: Box::new(expr),
        })
    } else {
        parse_comparison(stream)
    }
}

fn parse_comparison(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    let mut expr = parse_primary(stream)?;
    if stream.consume_symbol(Symbol::Equal) {
        let right = parse_primary(stream)?;
        expr = Expr::Binary {
            left: Box::new(expr),
            op: BinaryOp::Eq,
            right: Box::new(right),
        };
    }
    Ok(expr)
}

fn parse_primary(stream: &mut TokenStream) -> Result<Expr, ParseError> {
    if stream.consume_symbol(Symbol::LParen) {
        let expr = parse_expr(stream)?;
        stream.expect_symbol(Symbol::RParen)?;
        return Ok(expr);
    }

    if let Some(literal) = stream.consume_literal() {
        return Ok(Expr::Literal(literal));
    }

    if let Some(identifier) = stream.consume_identifier() {
        let mut parts = vec![identifier];
        while stream.consume_symbol(Symbol::Dot) {
            parts.push(stream.expect_identifier()?);
        }
        return Ok(Expr::Identifier(parts));
    }

    Err(stream.error("expected expression", None))
}

struct TokenStream {
    tokens: Vec<Token>,
    index: usize,
    last_span_start: Option<usize>,
}

impl TokenStream {
    fn new(tokens: Vec<Token>) -> Self {
        Self {
            tokens,
            index: 0,
            last_span_start: None,
        }
    }

    fn position(&self) -> usize {
        self.index
    }

    fn reset(&mut self, index: usize) {
        self.index = index;
    }

    fn is_eof(&self) -> bool {
        self.index >= self.tokens.len()
    }

    fn peek(&self) -> Option<&Token> {
        self.tokens.get(self.index)
    }

    fn consume(&mut self) -> Option<Token> {
        let token = self.tokens.get(self.index).cloned();
        if let Some(token) = &token {
            self.last_span_start = Some(token.span.start);
            self.index += 1;
        }
        token
    }

    fn error(&self, message: &str, index: Option<usize>) -> ParseError {
        let index = index.or_else(|| self.peek().map(|token| token.span.start));
        ParseError::new(message, index)
    }

    fn last_span_start(&self) -> Option<usize> {
        self.last_span_start
    }

    fn consume_keyword(&mut self, keyword: Keyword) -> bool {
        matches!(self.peek(), Some(Token { kind: TokenKind::Keyword(k), .. }) if *k == keyword)
            && self.consume().is_some()
    }

    fn expect_keyword(&mut self, keyword: Keyword) -> Result<(), ParseError> {
        if self.consume_keyword(keyword) {
            Ok(())
        } else {
            Err(self.error("expected keyword", None))
        }
    }

    fn consume_symbol(&mut self, symbol: Symbol) -> bool {
        matches!(self.peek(), Some(Token { kind: TokenKind::Symbol(s), .. }) if *s == symbol)
            && self.consume().is_some()
    }

    fn expect_symbol(&mut self, symbol: Symbol) -> Result<(), ParseError> {
        if self.consume_symbol(symbol) {
            Ok(())
        } else {
            Err(self.error("expected symbol", None))
        }
    }

    fn consume_identifier(&mut self) -> Option<String> {
        match self.peek() {
            Some(Token {
                kind: TokenKind::Identifier(name),
                ..
            }) => {
                let name = name.clone();
                self.consume();
                Some(name)
            }
            _ => None,
        }
    }

    fn expect_identifier(&mut self) -> Result<String, ParseError> {
        self.consume_identifier()
            .ok_or_else(|| self.error("expected identifier", None))
    }

    fn consume_literal(&mut self) -> Option<Literal> {
        match self.peek() {
            Some(Token {
                kind: TokenKind::Literal(literal),
                ..
            }) => {
                let literal = literal.clone();
                self.consume();
                Some(literal)
            }
            _ => None,
        }
    }

    fn expect_literal(&mut self) -> Result<Literal, ParseError> {
        self.consume_literal()
            .ok_or_else(|| self.error("expected literal", None))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_create_table_statement() {
        let statements = parse("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT);")
            .expect("parse");
        assert_eq!(statements.len(), 1);
        match &statements[0] {
            Statement::CreateTable(stmt) => {
                assert_eq!(stmt.name, "users");
                assert_eq!(stmt.columns.len(), 2);
                assert!(stmt.columns[0].primary_key);
            }
            _ => panic!("expected create table"),
        }
    }

    #[test]
    fn parse_insert_statement() {
        let statements = parse("INSERT INTO users (id, name) VALUES (1, 'Ada')")
            .expect("parse");
        match &statements[0] {
            Statement::Insert(stmt) => {
                assert_eq!(stmt.table, "users");
                assert_eq!(stmt.columns.as_ref().unwrap().len(), 2);
                assert_eq!(stmt.values.len(), 2);
            }
            _ => panic!("expected insert"),
        }
    }

    #[test]
    fn parse_select_with_where_order_limit() {
        let sql = "SELECT id, name FROM users WHERE id = 1 ORDER BY name DESC LIMIT 10";
        let statements = parse(sql).expect("parse");
        match &statements[0] {
            Statement::Select(stmt) => {
                assert_eq!(stmt.items.len(), 2);
                assert_eq!(stmt.from.as_deref(), Some("users"));
                assert!(stmt.filter.is_some());
                assert!(matches!(stmt.order_by.as_ref().unwrap().direction, OrderDirection::Desc));
                assert_eq!(stmt.limit, Some(10));
            }
            _ => panic!("expected select"),
        }
    }

    #[test]
    fn parse_drop_table_statement() {
        let statements = parse("DROP TABLE users").expect("parse");
        match &statements[0] {
            Statement::DropTable(stmt) => assert_eq!(stmt.name, "users"),
            _ => panic!("expected drop table"),
        }
    }

    #[test]
    fn parse_invalid_statement_reports_error() {
        let err = parse("BOGUS").expect_err("error");
        assert!(err.message.contains("expected statement"));
    }
}
