# AIDB Workspace

This repository is organized as a Rust workspace with three crates:

- `engine`: core database engine library.
- `server`: REST service entry point (binary crate).
- `cli`: command-line client for interacting with the service (binary crate).

The workspace root `Cargo.toml` aggregates these crates for builds and tests.

## REST API
- `GET /health` returns a basic health check.
- `POST /execute` executes SQL with a JSON body `{ "sql": "..." }`.
- `GET /openapi.json` serves the OpenAPI specification for the service.
- `GET /docs` serves Swagger UI for the OpenAPI spec.
