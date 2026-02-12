# AGENTS.md

Purpose: local instructions for Codex working in this repo.

## Project summary
- Build a production-grade Rust database engine with REST server and CLI.
- Follow the constraints in `goal.md` (no DB-specific libs, no SQLite embedding).

## Rust project best practices
- Keep public APIs small and documented; use `#[deny(missing_docs)]` on library crates.
- Use a clear error strategy; avoid leaking implementation details across crate boundaries.
- Use structured logging for observability.
- Enforce `cargo fmt` and `cargo clippy -- -D warnings` on touched crates.
- Favor explicit types and small modules; avoid megafiles.
- Add unit tests next to modules and integration tests under `tests/` when stable.

## Testing policy (all tests must pass)
We have a large upstream SQL logic test suite and must keep a strict all-tests-pass policy by enabling tests gradually.

Rules:
- Vendor the upstream suite under `tests/sqllogictest/test/` and track provenance in `tests/sqllogictest/UPSTREAM.txt`.
- Only run a curated subset by default. Use an allowlist manifest so new tests are enabled only when they pass.
- Do not add failing tests to the allowlist. Grow coverage incrementally as features land.
- Keep the allowlist sorted and comment entries with the feature that unblocked them.
- Add additional project-specific tests (unit, integration, and end-to-end) beyond SQL compatibility to cover engine behavior and APIs.
- Each code-bearing commit must include relevant tests unless the change is already covered; explain omissions explicitly in the commit message.

Recommended files for the incremental policy (create when needed):
- `tests/sqllogictest/ALLOWLIST.txt` (one `.test` path per line)
- `tests/sqllogictest/README.md` (how to run the allowlisted suite)

## Workflow expectations for Codex
- Read `goal.md` before major changes.
- Keep changes minimal and focused; avoid refactors unrelated to the task.
- If you must introduce a dependency, justify it and keep it general-purpose.
- If a decision is ambiguous, ask a short question rather than guessing.

## Commands (when implemented)
- Build: `cargo build`
- Lint: `cargo clippy -- -D warnings`
- Format: `cargo fmt`
- Unit tests: `cargo test`
- SQL logic tests: a runner that uses the allowlist above.
