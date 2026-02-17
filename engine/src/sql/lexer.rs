#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub kind: TokenKind,
    pub span: Span,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TokenKind {
    Identifier(String),
    Keyword(Keyword),
    Literal(Literal),
    Symbol(Symbol),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Keyword {
    Select,
    From,
    Where,
    Insert,
    Into,
    Values,
    Create,
    Drop,
    Table,
    Primary,
    Key,
    Integer,
    Real,
    Text,
    Boolean,
    And,
    Or,
    Not,
    Is,
    Between,
    Case,
    When,
    Then,
    Else,
    End,
    Exists,
    As,
    Order,
    By,
    Limit,
    Asc,
    Desc,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Literal {
    Number(String),
    String(String),
    Boolean(bool),
    Null,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Symbol {
    LParen,
    RParen,
    Comma,
    Semicolon,
    Star,
    Equal,
    NotEqual,
    Less,
    LessEqual,
    Greater,
    GreaterEqual,
    Plus,
    Minus,
    Slash,
    Dot,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LexError {
    UnexpectedChar { ch: char, index: usize },
    UnterminatedString { start: usize },
}

pub fn lex(input: &str) -> Result<Vec<Token>, LexError> {
    let mut tokens = Vec::new();
    let mut chars = input.char_indices().peekable();

    while let Some((index, ch)) = chars.next() {
        if ch.is_whitespace() {
            continue;
        }

        let token = match ch {
            '(' => Token {
                kind: TokenKind::Symbol(Symbol::LParen),
                span: Span {
                    start: index,
                    end: index + ch.len_utf8(),
                },
            },
            ')' => Token {
                kind: TokenKind::Symbol(Symbol::RParen),
                span: Span {
                    start: index,
                    end: index + ch.len_utf8(),
                },
            },
            ',' => Token {
                kind: TokenKind::Symbol(Symbol::Comma),
                span: Span {
                    start: index,
                    end: index + ch.len_utf8(),
                },
            },
            ';' => Token {
                kind: TokenKind::Symbol(Symbol::Semicolon),
                span: Span {
                    start: index,
                    end: index + ch.len_utf8(),
                },
            },
            '*' => Token {
                kind: TokenKind::Symbol(Symbol::Star),
                span: Span {
                    start: index,
                    end: index + ch.len_utf8(),
                },
            },
            '=' => Token {
                kind: TokenKind::Symbol(Symbol::Equal),
                span: Span {
                    start: index,
                    end: index + ch.len_utf8(),
                },
            },
            '!' => {
                if let Some((i, next)) = chars.peek().copied() {
                    if next == '=' {
                        chars.next();
                        Token {
                            kind: TokenKind::Symbol(Symbol::NotEqual),
                            span: Span {
                                start: index,
                                end: i + next.len_utf8(),
                            },
                        }
                    } else {
                        return Err(LexError::UnexpectedChar { ch, index });
                    }
                } else {
                    return Err(LexError::UnexpectedChar { ch, index });
                }
            }
            '<' => {
                if let Some((i, next)) = chars.peek().copied() {
                    if next == '=' {
                        chars.next();
                        Token {
                            kind: TokenKind::Symbol(Symbol::LessEqual),
                            span: Span {
                                start: index,
                                end: i + next.len_utf8(),
                            },
                        }
                    } else if next == '>' {
                        chars.next();
                        Token {
                            kind: TokenKind::Symbol(Symbol::NotEqual),
                            span: Span {
                                start: index,
                                end: i + next.len_utf8(),
                            },
                        }
                    } else {
                        Token {
                            kind: TokenKind::Symbol(Symbol::Less),
                            span: Span {
                                start: index,
                                end: index + ch.len_utf8(),
                            },
                        }
                    }
                } else {
                    Token {
                        kind: TokenKind::Symbol(Symbol::Less),
                        span: Span {
                            start: index,
                            end: index + ch.len_utf8(),
                        },
                    }
                }
            }
            '>' => {
                if let Some((i, next)) = chars.peek().copied() {
                    if next == '=' {
                        chars.next();
                        Token {
                            kind: TokenKind::Symbol(Symbol::GreaterEqual),
                            span: Span {
                                start: index,
                                end: i + next.len_utf8(),
                            },
                        }
                    } else {
                        Token {
                            kind: TokenKind::Symbol(Symbol::Greater),
                            span: Span {
                                start: index,
                                end: index + ch.len_utf8(),
                            },
                        }
                    }
                } else {
                    Token {
                        kind: TokenKind::Symbol(Symbol::Greater),
                        span: Span {
                            start: index,
                            end: index + ch.len_utf8(),
                        },
                    }
                }
            }
            '+' => Token {
                kind: TokenKind::Symbol(Symbol::Plus),
                span: Span {
                    start: index,
                    end: index + ch.len_utf8(),
                },
            },
            '-' => Token {
                kind: TokenKind::Symbol(Symbol::Minus),
                span: Span {
                    start: index,
                    end: index + ch.len_utf8(),
                },
            },
            '/' => Token {
                kind: TokenKind::Symbol(Symbol::Slash),
                span: Span {
                    start: index,
                    end: index + ch.len_utf8(),
                },
            },
            '.' => Token {
                kind: TokenKind::Symbol(Symbol::Dot),
                span: Span {
                    start: index,
                    end: index + ch.len_utf8(),
                },
            },
            '\'' => {
                let start = index;
                let mut value = String::new();
                let mut end = None;

                while let Some((i, c)) = chars.next() {
                    if c == '\'' {
                        if let Some((_, next)) = chars.peek() {
                            if *next == '\'' {
                                value.push('\'');
                                chars.next();
                                continue;
                            }
                        }
                        end = Some(i + c.len_utf8());
                        break;
                    }
                    value.push(c);
                }

                let end = end.ok_or(LexError::UnterminatedString { start })?;

                Token {
                    kind: TokenKind::Literal(Literal::String(value)),
                    span: Span { start, end },
                }
            }
            '0'..='9' => {
                let start = index;
                let mut end = index + ch.len_utf8();
                let mut seen_dot = false;

                while let Some((i, next)) = chars.peek().copied() {
                    if next.is_ascii_digit() {
                        chars.next();
                        end = i + next.len_utf8();
                        continue;
                    }

                    if next == '.' && !seen_dot {
                        let mut lookahead = chars.clone();
                        lookahead.next();
                        if let Some((_, after_dot)) = lookahead.peek().copied() {
                            if after_dot.is_ascii_digit() {
                                chars.next();
                                seen_dot = true;
                                end = i + next.len_utf8();
                                continue;
                            }
                        }
                    }

                    break;
                }

                let literal = &input[start..end];
                Token {
                    kind: TokenKind::Literal(Literal::Number(literal.to_string())),
                    span: Span { start, end },
                }
            }
            _ if is_identifier_start(ch) => {
                let start = index;
                let mut end = index + ch.len_utf8();

                while let Some((i, next)) = chars.peek().copied() {
                    if is_identifier_continue(next) {
                        chars.next();
                        end = i + next.len_utf8();
                    } else {
                        break;
                    }
                }

                let ident = &input[start..end];
                match keyword_or_literal(ident) {
                    KeywordOrLiteral::Keyword(keyword) => Token {
                        kind: TokenKind::Keyword(keyword),
                        span: Span { start, end },
                    },
                    KeywordOrLiteral::Literal(literal) => Token {
                        kind: TokenKind::Literal(literal),
                        span: Span { start, end },
                    },
                    KeywordOrLiteral::Identifier => Token {
                        kind: TokenKind::Identifier(ident.to_string()),
                        span: Span { start, end },
                    },
                }
            }
            other => return Err(LexError::UnexpectedChar { ch: other, index }),
        };

        tokens.push(token);
    }

    Ok(tokens)
}

fn is_identifier_start(ch: char) -> bool {
    ch.is_ascii_alphabetic() || ch == '_'
}

fn is_identifier_continue(ch: char) -> bool {
    ch.is_ascii_alphanumeric() || ch == '_'
}

enum KeywordOrLiteral {
    Keyword(Keyword),
    Literal(Literal),
    Identifier,
}

fn keyword_or_literal(ident: &str) -> KeywordOrLiteral {
    match ident.to_ascii_uppercase().as_str() {
        "SELECT" => KeywordOrLiteral::Keyword(Keyword::Select),
        "FROM" => KeywordOrLiteral::Keyword(Keyword::From),
        "WHERE" => KeywordOrLiteral::Keyword(Keyword::Where),
        "INSERT" => KeywordOrLiteral::Keyword(Keyword::Insert),
        "INTO" => KeywordOrLiteral::Keyword(Keyword::Into),
        "VALUES" => KeywordOrLiteral::Keyword(Keyword::Values),
        "CREATE" => KeywordOrLiteral::Keyword(Keyword::Create),
        "DROP" => KeywordOrLiteral::Keyword(Keyword::Drop),
        "TABLE" => KeywordOrLiteral::Keyword(Keyword::Table),
        "PRIMARY" => KeywordOrLiteral::Keyword(Keyword::Primary),
        "KEY" => KeywordOrLiteral::Keyword(Keyword::Key),
        "INTEGER" => KeywordOrLiteral::Keyword(Keyword::Integer),
        "REAL" => KeywordOrLiteral::Keyword(Keyword::Real),
        "TEXT" => KeywordOrLiteral::Keyword(Keyword::Text),
        "BOOLEAN" => KeywordOrLiteral::Keyword(Keyword::Boolean),
        "AND" => KeywordOrLiteral::Keyword(Keyword::And),
        "OR" => KeywordOrLiteral::Keyword(Keyword::Or),
        "NOT" => KeywordOrLiteral::Keyword(Keyword::Not),
        "IS" => KeywordOrLiteral::Keyword(Keyword::Is),
        "BETWEEN" => KeywordOrLiteral::Keyword(Keyword::Between),
        "CASE" => KeywordOrLiteral::Keyword(Keyword::Case),
        "WHEN" => KeywordOrLiteral::Keyword(Keyword::When),
        "THEN" => KeywordOrLiteral::Keyword(Keyword::Then),
        "ELSE" => KeywordOrLiteral::Keyword(Keyword::Else),
        "END" => KeywordOrLiteral::Keyword(Keyword::End),
        "EXISTS" => KeywordOrLiteral::Keyword(Keyword::Exists),
        "AS" => KeywordOrLiteral::Keyword(Keyword::As),
        "ORDER" => KeywordOrLiteral::Keyword(Keyword::Order),
        "BY" => KeywordOrLiteral::Keyword(Keyword::By),
        "LIMIT" => KeywordOrLiteral::Keyword(Keyword::Limit),
        "ASC" => KeywordOrLiteral::Keyword(Keyword::Asc),
        "DESC" => KeywordOrLiteral::Keyword(Keyword::Desc),
        "TRUE" => KeywordOrLiteral::Literal(Literal::Boolean(true)),
        "FALSE" => KeywordOrLiteral::Literal(Literal::Boolean(false)),
        "NULL" => KeywordOrLiteral::Literal(Literal::Null),
        _ => KeywordOrLiteral::Identifier,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(tokens: &[Token]) -> Vec<TokenKind> {
        tokens.iter().map(|token| token.kind.clone()).collect()
    }

    #[test]
    fn lexes_keywords_identifiers_and_symbols() {
        let tokens = lex("SELECT col, t1.* FROM my_table;").expect("lex");
        let expected = vec![
            TokenKind::Keyword(Keyword::Select),
            TokenKind::Identifier("col".to_string()),
            TokenKind::Symbol(Symbol::Comma),
            TokenKind::Identifier("t1".to_string()),
            TokenKind::Symbol(Symbol::Dot),
            TokenKind::Symbol(Symbol::Star),
            TokenKind::Keyword(Keyword::From),
            TokenKind::Identifier("my_table".to_string()),
            TokenKind::Symbol(Symbol::Semicolon),
        ];

        assert_eq!(kinds(&tokens), expected);
    }

    #[test]
    fn lexes_literals() {
        let tokens = lex("VALUES (1, 3.14, 'hi', TRUE, NULL)").expect("lex");
        let expected = vec![
            TokenKind::Keyword(Keyword::Values),
            TokenKind::Symbol(Symbol::LParen),
            TokenKind::Literal(Literal::Number("1".to_string())),
            TokenKind::Symbol(Symbol::Comma),
            TokenKind::Literal(Literal::Number("3.14".to_string())),
            TokenKind::Symbol(Symbol::Comma),
            TokenKind::Literal(Literal::String("hi".to_string())),
            TokenKind::Symbol(Symbol::Comma),
            TokenKind::Literal(Literal::Boolean(true)),
            TokenKind::Symbol(Symbol::Comma),
            TokenKind::Literal(Literal::Null),
            TokenKind::Symbol(Symbol::RParen),
        ];

        assert_eq!(kinds(&tokens), expected);
    }

    #[test]
    fn lexes_is_keyword() {
        let tokens = lex("col IS NULL").expect("lex");
        let expected = vec![
            TokenKind::Identifier("col".to_string()),
            TokenKind::Keyword(Keyword::Is),
            TokenKind::Literal(Literal::Null),
        ];

        assert_eq!(kinds(&tokens), expected);
    }

    #[test]
    fn lexes_escaped_quotes() {
        let tokens = lex("'O''Reilly'").expect("lex");
        assert_eq!(kinds(&tokens), vec![TokenKind::Literal(Literal::String("O'Reilly".to_string()))]);
    }

    #[test]
    fn rejects_unterminated_string() {
        let err = lex("'unterminated").expect_err("error");
        assert_eq!(err, LexError::UnterminatedString { start: 0 });
    }
}
