# AIDB Workspace

This repository is organized as a Rust workspace with three crates:

- `engine`: core database engine library.
- `server`: REST service entry point (binary crate).
- `cli`: command-line client for interacting with the service (binary crate).

The workspace root `Cargo.toml` aggregates these crates for builds and tests.
