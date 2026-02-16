# Requirements

## 1. Scope and Constraints
### 1.1 Core Implementation
#### 1.1.1 The core database engine MUST be implemented in Rust.
#### 1.1.2 General-purpose libraries MAY be used.
#### 1.1.3 Database-specific libraries MUST NOT be used, except `sqllogictest-rs` for testing.
#### 1.1.4 SQLite MUST NOT be wrapped or embedded.

### 1.2 Primary Goal
#### 1.2.1 Deliver a fully functional database engine with SQLite-level core capabilities, plus a REST service and a CLI that talks to the service.
#### 1.2.2 A single service instance MUST be able to serve multiple concurrent clients (multiple CLIs and other clients).

## 2. Storage and Persistence
### 2.1 Single-File Backend
#### 2.1.1 Data MUST persist to a single on-disk database file.
#### 2.1.2 The engine MUST support create and open semantics for the database file.

### 2.2 Catalog and Schema Persistence
#### 2.2.1 Schema metadata (catalog) MUST be stored in the database file.
#### 2.2.2 Schema metadata MUST survive process restart and be recoverable on startup.

### 2.3 File Format and Page Management
#### 2.3.1 The on-disk format MUST be a single-file binary format with a page-based layout.
#### 2.3.2 The file MUST include a header sufficient to identify and load the database file across restarts.

## 3. Transactions, Recovery, and Concurrency
### 3.1 Transaction Model
#### 3.1.1 Autocommit MUST be the default behavior.
#### 3.1.2 Explicit transaction control MUST support `BEGIN`, `COMMIT`, and `ROLLBACK` (single connection required).

### 3.2 Rollback Journal
#### 3.2.1 Statement-level atomicity MUST be implemented via a rollback-journal file.
#### 3.2.2 The journal file MUST be created during writes and removed on commit.

### 3.3 Recovery
#### 3.3.1 On restart after a crash, the engine MUST recover to a consistent state using the rollback journal.

### 3.4 Concurrency Policy
#### 3.4.1 The engine MUST enforce single-writer semantics.
#### 3.4.2 Readers MAY be blocked during write transactions.
#### 3.4.3 The core engine MUST be safe for concurrent access (e.g., via synchronization primitives).

## 4. SQL Support and Execution
### 4.1 Compatibility Scope
#### 4.1.1 SQL compatibility is defined strictly by the SQLite `sqllogictest` suite.
#### 4.1.2 Features not exercised by the required test suite are out of scope.

### 4.2 DDL and DML
#### 4.2.1 The engine MUST support DDL and DML statements required by the test suite, including `CREATE`/`DROP` and `INSERT`/`UPDATE`/`DELETE` as exercised by tests.

### 4.3 Query Execution
#### 4.3.1 The engine MUST correctly execute `SELECT` statements required by the test suite.
#### 4.3.2 Result sets MUST match expected values and ordering as defined by the suite.

## 5. Testing and Verification
### 5.1 Vendored SQLite SQL Logic Test Suite
#### 5.1.1 The official SQLite `sqllogictest` repository test set MUST be vendored under `tests/sqllogictest/test/` with the same layout.
#### 5.1.2 The upstream source URL and exact check-in ID MUST be recorded in `tests/sqllogictest/UPSTREAM.txt`.

### 5.2 Required Test Coverage
#### 5.2.1 All `.test` files under `tests/sqllogictest/test/` MUST be executed by `sqllogictest-rs`.
#### 5.2.2 The required coverage MUST include `select1.test`.
#### 5.2.3 The required coverage MUST include `select2.test`.
#### 5.2.4 The required coverage MUST include `select3.test`.
#### 5.2.5 The required coverage MUST include `select4.test`.
#### 5.2.6 The required coverage MUST include `select5.test`.
#### 5.2.7 The required coverage MUST include `random/`.
#### 5.2.8 The required coverage MUST include `index/`.
#### 5.2.9 The required coverage MUST include `evidence/`.

### 5.3 Pass Criteria
#### 5.3.1 `cargo build` MUST succeed on a clean checkout.
#### 5.3.2 `sqllogictest-rs` MUST pass 100% against all required tests.

## 6. REST Service
### 6.1 Core Service Behavior
#### 6.1.1 The service MUST expose an HTTP API to execute SQL statements.
#### 6.1.2 The service MUST return results and errors in JSON.
#### 6.1.3 The service MUST handle multiple concurrent client requests.

### 6.2 API Documentation
#### 6.2.1 The service MUST publish an OpenAPI specification at a documented endpoint.
#### 6.2.2 The service MUST provide a Swagger UI at a documented endpoint.

## 7. Command-Line Interface (CLI)
### 7.1 Service Integration
#### 7.1.1 The CLI MUST execute SQL statements by calling the REST service API.
#### 7.1.2 The CLI MUST allow configuration of the server address.

### 7.2 Interactive Mode
#### 7.2.1 The CLI MUST provide an interactive REPL for executing SQL.

### 7.3 Non-Interactive Mode
#### 7.3.1 The CLI MUST support single-command execution via arguments.
#### 7.3.2 The CLI MUST support scripting via STDIN.

## 8. Non-Functional Requirements
### 8.1 Reliability
#### 8.1.1 The engine MUST preserve database consistency across crashes via rollback journal recovery.

### 8.2 Portability
#### 8.2.1 The system MUST build and run via Rust tooling with no dependency on external database engines.
