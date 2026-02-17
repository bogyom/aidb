use engine::Database;
use engine::EngineError;
use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Method, Request, Response, Server, StatusCode};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone)]
struct AppState {
    db: Arc<Mutex<Database>>,
}

#[derive(Debug, Deserialize)]
struct SqlRequest {
    sql: String,
}

#[derive(Debug, Serialize)]
struct ErrorResponse {
    error: ErrorBody,
}

#[derive(Debug, Serialize)]
struct ErrorBody {
    code: String,
    message: String,
}

#[tokio::main]
async fn main() {
    let addr = SocketAddr::from(([127, 0, 0, 1], 8080));
    let db_path = temp_db_path("server");
    let state = Arc::new(Mutex::new(
        Database::create(db_path.to_string_lossy().as_ref()).expect("create db"),
    ));

    let make_svc = make_service_fn(move |_conn| {
        let state = AppState { db: Arc::clone(&state) };
        async move {
            Ok::<_, Infallible>(service_fn(move |req| {
                let state = state.clone();
                async move { handle_request(req, state).await }
            }))
        }
    });

    let server = Server::bind(&addr).serve(make_svc);
    println!("listening on http://{}", addr);
    if let Err(err) = server.await {
        eprintln!("server error: {}", err);
    }
}

fn temp_db_path(label: &str) -> std::path::PathBuf {
    let mut path = std::env::temp_dir();
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("clock")
        .as_nanos();
    path.push(format!("aidb_{label}_{nanos}.db"));
    path
}

async fn handle_request(
    req: Request<Body>,
    state: AppState,
) -> Result<Response<Body>, Infallible> {
    match (req.method(), req.uri().path()) {
        (&Method::GET, "/health") => Ok(Response::new(Body::from("ok"))),
        (&Method::POST, "/execute") => handle_execute(req, state).await,
        _ => {
            let mut response = Response::new(Body::from("not found"));
            *response.status_mut() = StatusCode::NOT_FOUND;
            Ok(response)
        }
    }
}

async fn handle_execute(
    req: Request<Body>,
    state: AppState,
) -> Result<Response<Body>, Infallible> {
    let bytes = match hyper::body::to_bytes(req.into_body()).await {
        Ok(bytes) => bytes,
        Err(_) => return Ok(json_error(StatusCode::BAD_REQUEST, "invalid_body", "invalid body")),
    };

    let payload: SqlRequest = match serde_json::from_slice(&bytes) {
        Ok(payload) => payload,
        Err(_) => return Ok(json_error(StatusCode::BAD_REQUEST, "invalid_json", "invalid json")),
    };

    let mut db = match state.db.lock() {
        Ok(db) => db,
        Err(_) => {
            return Ok(json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "db_lock",
                "db lock poisoned",
            ))
        }
    };

    match db.execute(&payload.sql) {
        Ok(result) => json_ok(result.to_json()),
        Err(err) => Ok(engine_error_to_response(err)),
    }
}

fn json_ok(value: serde_json::Value) -> Result<Response<Body>, Infallible> {
    let body = match serde_json::to_vec(&value) {
        Ok(body) => body,
        Err(_) => {
            return Ok(json_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "encode_error",
                "encode error",
            ))
        }
    };
    let response = Response::builder()
        .status(StatusCode::OK)
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap();
    Ok(response)
}

fn json_error(status: StatusCode, code: &str, message: &str) -> Response<Body> {
    let body = serde_json::to_vec(&ErrorResponse {
        error: ErrorBody {
            code: code.to_string(),
            message: message.to_string(),
        },
    })
    .unwrap_or_else(|_| b"{\"error\":{\"code\":\"internal\",\"message\":\"internal\"}}".to_vec());
    Response::builder()
        .status(status)
        .header("content-type", "application/json")
        .body(Body::from(body))
        .unwrap()
}

fn engine_error_to_response(error: EngineError) -> Response<Body> {
    match error {
        EngineError::InvalidSql => json_error(StatusCode::BAD_REQUEST, "invalid_sql", "invalid sql"),
        EngineError::TableNotFound => {
            json_error(StatusCode::NOT_FOUND, "table_not_found", "table not found")
        }
        EngineError::TableAlreadyExists => json_error(
            StatusCode::CONFLICT,
            "table_exists",
            "table already exists",
        ),
        EngineError::InvalidPath
        | EngineError::InvalidHeader
        | EngineError::InvalidPageHeader
        | EngineError::InvalidPageType(_)
        | EngineError::PageOutOfBounds(_)
        | EngineError::CatalogCorrupt
        | EngineError::CatalogTooLarge
        | EngineError::RowTooLarge
        | EngineError::RowCorrupt
        | EngineError::UnsupportedVersion(_)
        | EngineError::Closed
        | EngineError::Io(_) => json_error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "engine_error",
            "engine error",
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hyper::body::to_bytes;

    #[tokio::test]
    async fn health_endpoint_returns_ok() {
        let state = AppState {
            db: Arc::new(Mutex::new(
                Database::create(temp_db_path("health").to_string_lossy().as_ref()).unwrap(),
            )),
        };
        let req = Request::builder()
            .method(Method::GET)
            .uri("/health")
            .body(Body::empty())
            .unwrap();
        let response = handle_request(req, state).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body()).await.unwrap();
        assert_eq!(&body[..], b"ok");
    }

    #[tokio::test]
    async fn sql_endpoint_rejects_invalid_json() {
        let state = AppState {
            db: Arc::new(Mutex::new(
                Database::create(temp_db_path("bad_json").to_string_lossy().as_ref()).unwrap(),
            )),
        };
        let req = Request::builder()
            .method(Method::POST)
            .uri("/execute")
            .header("content-type", "application/json")
            .body(Body::from("not json"))
            .unwrap();
        let response = handle_request(req, state).await.unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn execute_endpoint_returns_rows() {
        let state = AppState {
            db: Arc::new(Mutex::new(
                Database::create(temp_db_path("execute_rows").to_string_lossy().as_ref()).unwrap(),
            )),
        };
        {
            let mut db = state.db.lock().unwrap();
            db.execute("CREATE TABLE t (id INTEGER)")
                .expect("create table");
            db.execute("INSERT INTO t (id) VALUES (1)")
                .expect("insert row");
        }

        let payload = serde_json::json!({ "sql": "SELECT id FROM t" }).to_string();
        let req = Request::builder()
            .method(Method::POST)
            .uri("/execute")
            .header("content-type", "application/json")
            .body(Body::from(payload))
            .unwrap();
        let response = handle_request(req, state).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = to_bytes(response.into_body()).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "columns": [{ "name": "id", "type": "Integer" }],
                "rows": [[1]],
            })
        );
    }

    #[tokio::test]
    async fn execute_endpoint_reports_error() {
        let state = AppState {
            db: Arc::new(Mutex::new(
                Database::create(temp_db_path("execute_error").to_string_lossy().as_ref()).unwrap(),
            )),
        };
        let payload = serde_json::json!({ "sql": "SELECT * FROM missing" }).to_string();
        let req = Request::builder()
            .method(Method::POST)
            .uri("/execute")
            .header("content-type", "application/json")
            .body(Body::from(payload))
            .unwrap();
        let response = handle_request(req, state).await.unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
        let body = to_bytes(response.into_body()).await.unwrap();
        let json: serde_json::Value = serde_json::from_slice(&body).unwrap();
        assert_eq!(
            json,
            serde_json::json!({
                "error": {
                    "code": "table_not_found",
                    "message": "table not found"
                }
            })
        );
    }

    #[test]
    fn engine_error_maps_to_status() {
        let response = engine_error_to_response(EngineError::TableAlreadyExists);
        assert_eq!(response.status(), StatusCode::CONFLICT);
        let response = engine_error_to_response(EngineError::InvalidSql);
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }
}
