use std::io::{Read, Write};
use std::io::{BufRead, BufReader};

const DEFAULT_SERVER_URL: &str = "http://127.0.0.1:8080";
const ENV_SERVER_URL: &str = "AIDB_SERVER_URL";

#[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq)]
struct QueryResult {
    columns: Vec<Column>,
    rows: Vec<Vec<serde_json::Value>>,
}

#[derive(Debug, serde::Serialize, serde::Deserialize, PartialEq)]
struct Column {
    name: String,
    #[serde(rename = "type")]
    data_type: String,
}

#[derive(Debug, serde::Deserialize)]
struct ErrorResponse {
    error: ErrorBody,
}

#[derive(Debug, serde::Deserialize)]
struct ErrorBody {
    code: String,
    message: String,
}

#[derive(Debug, PartialEq, Eq)]
enum CliError {
    Request(String),
    Response { status: u16, code: String, message: String },
    ResponseRaw { status: u16, body: String },
    Parse(String),
    Url(String),
}

#[derive(Debug, PartialEq, Eq)]
struct Config {
    server_url: String,
    show_help: bool,
    sql: Option<String>,
}

fn usage_hint() -> &'static str {
    "Usage: aidb [--server <url>] [--execute <sql>] [--help]\n\
\n\
Options:\n\
  --server <url>   Server URL (env: AIDB_SERVER_URL)\n\
  -e, --execute    Execute a single SQL statement\n\
  -h, --help       Show this help message\n\
\n\
If --execute is omitted, the CLI starts an interactive REPL.\n\
\n\
Default server: http://127.0.0.1:8080"
}

fn parse_config<I, S>(args: I) -> Result<Config, String>
where
    I: IntoIterator<Item = S>,
    S: AsRef<str>,
{
    let mut server_url =
        std::env::var(ENV_SERVER_URL).unwrap_or_else(|_| DEFAULT_SERVER_URL.to_string());
    let mut show_help = false;
    let mut sql: Option<String> = None;
    let mut iter = args.into_iter();
    let _program = iter.next();

    while let Some(arg) = iter.next() {
        let arg = arg.as_ref();
        if arg == "-h" || arg == "--help" {
            show_help = true;
            continue;
        }
        if let Some(value) = arg.strip_prefix("--server=") {
            if value.is_empty() {
                return Err("missing value for --server".to_string());
            }
            server_url = value.to_string();
            continue;
        }
        if let Some(value) = arg.strip_prefix("--execute=") {
            if value.is_empty() {
                return Err("missing value for --execute".to_string());
            }
            sql = Some(value.to_string());
            continue;
        }
        if arg == "--server" {
            let value = iter
                .next()
                .ok_or_else(|| "missing value for --server".to_string())?;
            server_url = value.as_ref().to_string();
            continue;
        }
        if arg == "--execute" || arg == "-e" {
            let value = iter
                .next()
                .ok_or_else(|| "missing value for --execute".to_string())?;
            sql = Some(value.as_ref().to_string());
            continue;
        }
        return Err(format!("unknown argument: {}", arg));
    }

    Ok(Config {
        server_url,
        show_help,
        sql,
    })
}

fn execute_sql(server_url: &str, sql: &str) -> Result<QueryResult, CliError> {
    let address = parse_server_url(server_url)?;
    let payload = serde_json::json!({ "sql": sql }).to_string();
    let request = format!(
        "POST /execute HTTP/1.1\r\nHost: {}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
        address.host,
        payload.as_bytes().len(),
        payload
    );

    let mut stream = std::net::TcpStream::connect((address.host.as_str(), address.port))
        .map_err(|err| CliError::Request(format!("connect failed: {}", err)))?;
    stream
        .write_all(request.as_bytes())
        .map_err(|err| CliError::Request(format!("write failed: {}", err)))?;
    let mut response_bytes = Vec::new();
    stream
        .read_to_end(&mut response_bytes)
        .map_err(|err| CliError::Request(format!("read failed: {}", err)))?;

    let (status, body) = parse_http_response(&response_bytes)?;
    handle_execute_response(status, &body)
}

#[derive(Debug, PartialEq, Eq)]
struct ServerAddress {
    host: String,
    port: u16,
}

fn parse_server_url(server_url: &str) -> Result<ServerAddress, CliError> {
    let trimmed = server_url.trim_end_matches('/');
    let rest = trimmed
        .strip_prefix("http://")
        .ok_or_else(|| CliError::Url("server url must start with http://".to_string()))?;
    let mut parts = rest.splitn(2, '/');
    let host_port = parts.next().unwrap_or("");
    if let Some(path) = parts.next() {
        if !path.is_empty() {
            return Err(CliError::Url(
                "server url must not include a path".to_string(),
            ));
        }
    }
    let (host, port) = match host_port.rsplit_once(':') {
        Some((host, port)) if !port.is_empty() => {
            let port = port
                .parse::<u16>()
                .map_err(|_| CliError::Url("invalid port in server url".to_string()))?;
            (host.to_string(), port)
        }
        _ => (host_port.to_string(), 80),
    };
    if host.is_empty() {
        return Err(CliError::Url("server url missing host".to_string()));
    }
    Ok(ServerAddress { host, port })
}

fn parse_http_response(bytes: &[u8]) -> Result<(u16, String), CliError> {
    let separator = b"\r\n\r\n";
    let header_end = bytes
        .windows(separator.len())
        .position(|window| window == separator)
        .ok_or_else(|| CliError::Parse("invalid http response".to_string()))?;
    let (header_bytes, body_bytes) = bytes.split_at(header_end);
    let header_str = std::str::from_utf8(header_bytes)
        .map_err(|_| CliError::Parse("invalid http headers".to_string()))?;
    let mut lines = header_str.lines();
    let status_line = lines
        .next()
        .ok_or_else(|| CliError::Parse("missing status line".to_string()))?;
    let status = status_line
        .split_whitespace()
        .nth(1)
        .ok_or_else(|| CliError::Parse("invalid status line".to_string()))?
        .parse::<u16>()
        .map_err(|_| CliError::Parse("invalid status code".to_string()))?;

    let mut content_length: Option<usize> = None;
    let mut chunked = false;
    for line in lines {
        if let Some((name, value)) = line.split_once(':') {
            let name = name.trim().to_ascii_lowercase();
            let value = value.trim();
            if name == "content-length" {
                if let Ok(len) = value.parse::<usize>() {
                    content_length = Some(len);
                }
            } else if name == "transfer-encoding" && value.to_ascii_lowercase().contains("chunked")
            {
                chunked = true;
            }
        }
    }

    let body = &body_bytes[separator.len()..];
    let decoded = if chunked {
        decode_chunked(body)?
    } else if let Some(len) = content_length {
        if body.len() < len {
            return Err(CliError::Parse("truncated response body".to_string()));
        }
        body[..len].to_vec()
    } else {
        body.to_vec()
    };

    let body_str = String::from_utf8(decoded)
        .map_err(|_| CliError::Parse("response body not utf-8".to_string()))?;
    Ok((status, body_str))
}

fn decode_chunked(mut body: &[u8]) -> Result<Vec<u8>, CliError> {
    let mut decoded = Vec::new();
    loop {
        let line_end = body
            .windows(2)
            .position(|window| window == b"\r\n")
            .ok_or_else(|| CliError::Parse("invalid chunked response".to_string()))?;
        let line = &body[..line_end];
        body = &body[line_end + 2..];
        let size_str = std::str::from_utf8(line)
            .map_err(|_| CliError::Parse("invalid chunk size".to_string()))?;
        let size_str = size_str
            .split(';')
            .next()
            .unwrap_or("")
            .trim();
        let size = usize::from_str_radix(size_str, 16)
            .map_err(|_| CliError::Parse("invalid chunk size".to_string()))?;
        if size == 0 {
            break;
        }
        if body.len() < size + 2 {
            return Err(CliError::Parse("truncated chunked body".to_string()));
        }
        decoded.extend_from_slice(&body[..size]);
        body = &body[size + 2..];
    }
    Ok(decoded)
}

fn handle_execute_response(status: u16, body: &str) -> Result<QueryResult, CliError> {
    if status == 200 {
        serde_json::from_str(body)
            .map_err(|err| CliError::Parse(format!("invalid json response: {}", err)))
    } else {
        match serde_json::from_str::<ErrorResponse>(body) {
            Ok(error) => Err(CliError::Response {
                status,
                code: error.error.code,
                message: error.error.message,
            }),
            Err(_) => Err(CliError::ResponseRaw {
                status,
                body: body.to_string(),
            }),
        }
    }
}

fn format_result(result: &QueryResult) -> String {
    if result.columns.is_empty() && result.rows.is_empty() {
        return "(no results)".to_string();
    }

    let headers: Vec<String> = result.columns.iter().map(|c| c.name.clone()).collect();
    let mut rows: Vec<Vec<String>> = result
        .rows
        .iter()
        .map(|row| row.iter().map(value_to_string).collect::<Vec<_>>())
        .collect();

    let mut widths: Vec<usize> = headers.iter().map(|h| h.len()).collect();
    for row in &rows {
        for (idx, value) in row.iter().enumerate() {
            if let Some(width) = widths.get_mut(idx) {
                *width = (*width).max(value.len());
            }
        }
    }

    let header_line = format_row(&headers, &widths);
    let separator = widths
        .iter()
        .map(|w| "-".repeat(*w))
        .collect::<Vec<_>>()
        .join("-+-");

    let mut output = String::new();
    output.push_str(&header_line);
    output.push('\n');
    output.push_str(&separator);

    if rows.is_empty() {
        output.push('\n');
        output.push_str("(0 rows)");
        return output;
    }

    for row in rows.drain(..) {
        output.push('\n');
        output.push_str(&format_row(&row, &widths));
    }

    output
}

fn format_row(values: &[String], widths: &[usize]) -> String {
    let mut pieces = Vec::with_capacity(values.len());
    for (idx, value) in values.iter().enumerate() {
        let width = widths.get(idx).copied().unwrap_or(0);
        pieces.push(format!("{value:<width$}"));
    }
    pieces.join(" | ")
}

fn value_to_string(value: &serde_json::Value) -> String {
    match value {
        serde_json::Value::Null => "NULL".to_string(),
        serde_json::Value::Bool(value) => value.to_string(),
        serde_json::Value::Number(value) => value.to_string(),
        serde_json::Value::String(value) => value.clone(),
        other => other.to_string(),
    }
}

fn run(config: Config) -> i32 {
    if config.show_help {
        println!("{}", usage_hint());
        return 0;
    }

    if let Some(sql) = config.sql.as_deref() {
        return execute_one_shot(&config.server_url, sql);
    }

    run_repl(&config.server_url)
}

fn execute_one_shot(server_url: &str, sql: &str) -> i32 {
    match execute_sql(server_url, sql) {
        Ok(result) => {
            println!("{}", format_result(&result));
            0
        }
        Err(error) => {
            report_error(error);
            1
        }
    }
}

fn run_repl(server_url: &str) -> i32 {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock()).lines();
    let mut buffer = String::new();
    let mut had_error = false;

    loop {
        if buffer.trim().is_empty() {
            print!("aidb> ");
        } else {
            print!(" ...> ");
        }
        let _ = std::io::stdout().flush();

        match reader.next() {
            Some(Ok(line)) => {
                if let Some(statement) = consume_statement(&mut buffer, &line) {
                    match execute_sql(server_url, &statement) {
                        Ok(result) => {
                            println!("{}", format_result(&result));
                        }
                        Err(error) => {
                            report_error(error);
                            had_error = true;
                        }
                    }
                }
            }
            Some(Err(err)) => {
                eprintln!("input error: {}", err);
                had_error = true;
                break;
            }
            None => break,
        }
    }

    if !buffer.trim().is_empty() {
        eprintln!("incomplete statement ignored");
        had_error = true;
    }

    if had_error { 1 } else { 0 }
}

fn consume_statement(buffer: &mut String, line: &str) -> Option<String> {
    if !buffer.is_empty() {
        buffer.push('\n');
    }
    buffer.push_str(line);
    let trimmed = buffer.trim_end();
    if trimmed.ends_with(';') {
        let statement = trimmed
            .trim_end_matches(';')
            .trim_end()
            .to_string();
        buffer.clear();
        if statement.is_empty() {
            None
        } else {
            Some(statement)
        }
    } else {
        None
    }
}

fn report_error(error: CliError) {
    match error {
        CliError::Request(message) => {
            eprintln!("request error: {}", message);
        }
        CliError::Response {
            status,
            code,
            message,
        } => {
            eprintln!("server error ({}): {} - {}", status, code, message);
        }
        CliError::ResponseRaw { status, body } => {
            eprintln!("server error ({}): {}", status, body);
        }
        CliError::Parse(message) => {
            eprintln!("response error: {}", message);
        }
        CliError::Url(message) => {
            eprintln!("invalid server url: {}", message);
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let exit_code = match parse_config(args) {
        Ok(config) => run(config),
        Err(error) => {
            eprintln!("{}", error);
            println!("{}", usage_hint());
            2
        }
    };
    std::process::exit(exit_code);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn usage_hint_is_not_empty() {
        assert!(!usage_hint().is_empty());
    }

    #[test]
    fn usage_hint_mentions_server_flag_and_env() {
        let usage = usage_hint();
        assert!(usage.contains("--server"));
        assert!(usage.contains(ENV_SERVER_URL));
        assert!(usage.contains("--execute"));
    }

    #[test]
    fn default_server_is_used_when_unset() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::remove_var(ENV_SERVER_URL);
        let config = parse_config(vec!["aidb"]).expect("config");
        assert_eq!(config.server_url, DEFAULT_SERVER_URL);
        assert!(!config.show_help);
    }

    #[test]
    fn env_server_is_used_when_set() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var(ENV_SERVER_URL, "http://env:9999");
        let config = parse_config(vec!["aidb"]).expect("config");
        assert_eq!(config.server_url, "http://env:9999");
        std::env::remove_var(ENV_SERVER_URL);
    }

    #[test]
    fn flag_overrides_env_server() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::set_var(ENV_SERVER_URL, "http://env:9999");
        let config =
            parse_config(vec!["aidb", "--server", "http://flag:8888"]).expect("config");
        assert_eq!(config.server_url, "http://flag:8888");
        std::env::remove_var(ENV_SERVER_URL);
    }

    #[test]
    fn help_flag_sets_show_help() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::remove_var(ENV_SERVER_URL);
        let config = parse_config(vec!["aidb", "--help"]).expect("config");
        assert!(config.show_help);
    }

    #[test]
    fn execute_flag_sets_sql() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::remove_var(ENV_SERVER_URL);
        let config = parse_config(vec!["aidb", "--execute", "SELECT 1"]).expect("config");
        assert_eq!(config.sql, Some("SELECT 1".to_string()));
    }

    #[test]
    fn short_execute_flag_sets_sql() {
        let _guard = ENV_LOCK.lock().unwrap();
        std::env::remove_var(ENV_SERVER_URL);
        let config = parse_config(vec!["aidb", "-e", "SELECT 2"]).expect("config");
        assert_eq!(config.sql, Some("SELECT 2".to_string()));
    }

    #[test]
    fn handle_execute_response_success() {
        let body = r#"{"columns":[{"name":"id","type":"Integer"}],"rows":[[1]]}"#;
        let result = handle_execute_response(200, body).expect("result");
        assert_eq!(
            result,
            QueryResult {
                columns: vec![Column {
                    name: "id".to_string(),
                    data_type: "Integer".to_string(),
                }],
                rows: vec![vec![serde_json::json!(1)]],
            }
        );
    }

    #[test]
    fn handle_execute_response_error() {
        let body = r#"{"error":{"code":"table_not_found","message":"table not found"}}"#;
        let err = handle_execute_response(404, body).unwrap_err();
        assert_eq!(
            err,
            CliError::Response {
                status: 404,
                code: "table_not_found".to_string(),
                message: "table not found".to_string(),
            }
        );
    }

    #[test]
    fn consume_statement_returns_single_line() {
        let mut buffer = String::new();
        let statement = consume_statement(&mut buffer, "SELECT 1;").unwrap();
        assert_eq!(statement, "SELECT 1");
        assert!(buffer.is_empty());
    }

    #[test]
    fn consume_statement_supports_multiline() {
        let mut buffer = String::new();
        assert!(consume_statement(&mut buffer, "SELECT *").is_none());
        let statement = consume_statement(&mut buffer, "FROM t;").unwrap();
        assert_eq!(statement, "SELECT *\nFROM t");
        assert!(buffer.is_empty());
    }

    #[test]
    fn consume_statement_ignores_empty_statement() {
        let mut buffer = String::new();
        let statement = consume_statement(&mut buffer, ";");
        assert!(statement.is_none());
        assert!(buffer.is_empty());
    }

    #[test]
    fn format_result_renders_table_with_headers() {
        let result = QueryResult {
            columns: vec![
                Column {
                    name: "id".to_string(),
                    data_type: "Integer".to_string(),
                },
                Column {
                    name: "name".to_string(),
                    data_type: "Text".to_string(),
                },
            ],
            rows: vec![
                vec![serde_json::json!(1), serde_json::json!("Ada")],
                vec![serde_json::json!(20), serde_json::json!("Bob")],
            ],
        };
        let output = format_result(&result);
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines[0], "id | name");
        assert_eq!(lines[1], "---+-----");
        assert_eq!(lines[2], "1  | Ada ");
        assert_eq!(lines[3], "20 | Bob ");
    }

    #[test]
    fn format_result_renders_empty_rows_sensibly() {
        let result = QueryResult {
            columns: vec![Column {
                name: "id".to_string(),
                data_type: "Integer".to_string(),
            }],
            rows: vec![],
        };
        let output = format_result(&result);
        let lines: Vec<&str> = output.lines().collect();
        assert_eq!(lines[0], "id");
        assert_eq!(lines[1], "--");
        assert_eq!(lines[2], "(0 rows)");
    }

    #[test]
    fn format_result_renders_no_results() {
        let result = QueryResult {
            columns: vec![],
            rows: vec![],
        };
        assert_eq!(format_result(&result), "(no results)");
    }
}
