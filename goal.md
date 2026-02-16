# Goal
Build a fully functional database engine in Rust with SQLite-level core capabilities, plus a REST server and a CLI that is based on (and talks to) the service. One service instance should be able to serve multiple CLIs and other clients.

## Constraints
- Implement the core database engine in Rust.
- General-purpose libraries are allowed.
- Database-specific libraries are NOT allowed (except `sqllogictest-rs` for testing).
- Do not wrap or embed SQLite.

## Official SQL Logic Test Suite (SQLite)
Use the official SQLite `sqllogictest` repository test set. The suite is defined as all files under `/test`:

### Required test files and directories
- `select1.test`
- `select2.test`
- `select3.test`
- `select4.test`
- `select5.test`
- `random/`
- `index/`
- `evidence/`

### Vendoring and provenance
- Vendor the suite into this repo at `tests/sqllogictest/test/` (preserve the same layout).
- Record the upstream source URL and exact check-in ID in `tests/sqllogictest/UPSTREAM.txt`.

## Acceptance criteria (done = all true)
- `cargo build` succeeds on a clean checkout.
- `sqllogictest-rs` runs against *all* `.test` files under `tests/sqllogictest/test/` and passes 100%.
- REST server exposes an API to execute SQL and returns results/errors in JSON. 
- The service must support multiple concurrent clients (e.g., multiple CLIs and other clients).
- REST API publishes Swagger UI and the OpenAPI spec at documented endpoints.
- CLI can execute SQL statements (interactive and single-command modes) by calling the REST service API; 

## Minimal file-based backend requirements (SQLite-aligned, rollback-journal)
- Storage: data persists to a single on-disk database file; create/open semantics are supported.
- Catalog: schema metadata is stored in the database file and survives restart.
- Recovery: statement-level atomicity via rollback-journal (journal file created during writes, removed on commit).
- Transactions: autocommit by default; support `BEGIN`, `COMMIT`, `ROLLBACK` (single connection required).
- Concurrency: single-writer only; readers may be blocked during write transactions.
- Scope: SQL compatibility is defined by the `sqllogictest` suite; features not exercised there are out of scope.
