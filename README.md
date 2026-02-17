# AIDB

## Goal
Demonstrate that an AI system can generate and evolve a very large, complex database codebase **without human-in-the-loop edits**, while still meeting correctness, durability, and API requirements. The target is a SQLite-level core engine implemented in Rust, with a REST service and a CLI client that communicates with the service.

## Repository Layout
This is a Rust workspace with three crates:

- `engine`: core database engine library.
- `server`: REST service entry point (binary crate).
- `cli`: command-line client for interacting with the service.

The workspace root `Cargo.toml` aggregates these crates for builds and tests.

## Build
```bash
cargo build
```

## Run the Server
```bash
cargo run -p server
```

## Use the CLI
Single statement:
```bash
cargo run -p cli -- --sql "SELECT 1"
```

Interactive REPL:
```bash
cargo run -p cli
```

You can point the CLI at a custom server address via flags; run:
```bash
cargo run -p cli -- --help
```

## REST API
- `GET /health` returns a basic health check.
- `POST /execute` executes SQL with a JSON body `{ "sql": "..." }`.
- `GET /openapi.json` serves the OpenAPI specification for the service.
- `GET /docs` serves Swagger UI for the OpenAPI spec.

## Testing
Unit tests:
```bash
timeout 10m cargo test
```

Run a specific sqllogictest file:
```bash
timeout 10m cargo test -p engine sqllogictest_runs_select1_file
```

Run the full vendored sqllogictest suite via the harness:
```bash
timeout 60m cargo run -p engine --bin sqllogictest_harness -- tests/sqllogictest/test target/aidb_sqllogictest.db
```

## Notes
The SQL compatibility target is defined by the vendored SQLite `sqllogictest` suite. Features outside that suite are out of scope.
