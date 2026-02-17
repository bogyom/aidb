# AIDB Agent Guide

## Project Summary
AIDB is a Rust workspace that implements a SQLite-level core database engine, plus a REST server and a CLI client that talks to the service. The engine must persist to a single on-disk file, support rollback-journal recovery, and pass the vendored SQLite `sqllogictest` suite.

## Repo Layout
- `engine/`: core database engine library (Rust).
- `server/`: REST service binary crate.
- `cli/`: CLI client binary crate.
- `tests/sqllogictest/`: vendored SQLite SQL logic tests.
- `vendor/`: vendored dependencies (includes `sqllogictest`).

## Non-Negotiable Constraints
- The core database engine must be implemented in Rust.
- Do not wrap or embed SQLite.
- Database-specific libraries are not allowed, except `sqllogictest-rs` for testing.
- Persist data to a single on-disk database file with a page-based layout and header.
- Rollback-journal recovery is required; autocommit by default with `BEGIN/COMMIT/ROLLBACK` support.
- Enforce single-writer semantics; concurrent access must be safe.

## Engineering Standards (Production Grade)
- Prioritize correctness, data integrity, and deterministic behavior.
- Public APIs must have Rust doc comments (`///`) that explain behavior, invariants, error cases, and examples where useful.
- Keep modules cohesive with clear responsibilities; document non-obvious algorithms and storage formats.
- Validate inputs and surface errors with actionable messages.
- Add or update unit/integration tests for behavior changes. Aim for high coverage on critical paths.

## Testing (Always Use `timeout`)
Test runs can hang or take a long time. Every test command must be wrapped with `timeout`.

Examples (adjust durations as needed):
- Build: `cargo build` (no timeout required)
- Unit tests: `timeout 10m cargo test`
- Engine tests only: `timeout 10m cargo test -p engine`
- Targeted test: `timeout 5m cargo test -p engine sqllogictest_runs_select1_file`
- Full SQL logic suite via harness: `timeout 60m cargo run -p engine --bin sqllogictest_harness -- tests/sqllogictest/test target/aidb_sqllogictest.db`

## When Modifying SQL Logic Tests
- Keep the vendored suite under `tests/sqllogictest/test/` intact.
- Record upstream URL and check-in ID in `tests/sqllogictest/UPSTREAM.txt` if updated.

## Suggested Tooling
- Format: `cargo fmt`
- Lint: `cargo clippy`

## Communication
- Be explicit about assumptions, error handling, and performance tradeoffs.
- If a change impacts storage format or recovery, document the migration/compatibility impact.
