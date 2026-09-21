# Phase 10 — PostgreSQL Profile Storage

## 1. Objective

Persist enrollment profiles in **PostgreSQL** instead of process-local memory,
with **zero change to Phase 3–9B ML behaviour**. The routes, ML service and
`StoredProfile` value object are untouched in behaviour; only **where** a
profile lives changes:

```
Phase 9B:  FastAPI routes → InMemoryProfileStore      (lost on restart)
Phase 10:  FastAPI routes → PostgresProfileRepository (survives restarts)
```

Storage is strictly scoped to the six **aggregate** columns already flowing
through Phase 9B — no raw behaviour, no new ML, no auth, no WebSockets, no
Redis, no retraining, no recalibration.

## 2. What is stored (and what is never stored)

```sql
CREATE TABLE behavioral_profiles (
    user_ref       TEXT             PRIMARY KEY,
    centroid       DOUBLE PRECISION[] NOT NULL,
    embedding_dim  INTEGER          NOT NULL CHECK (embedding_dim > 0),
    session_count  INTEGER          NOT NULL CHECK (session_count >= 0),
    created_at     TIMESTAMPTZ      NOT NULL,
    updated_at     TIMESTAMPTZ      NOT NULL
);
```

Only the Phase 8 enrollment aggregate is written: `user_ref`,
`centroid` (as a `DOUBLE PRECISION[]`), `embedding_dim`, `session_count`,
`created_at`, `updated_at`. **Never** stored: keyboard/mouse events, key
identity, characters, passwords, tokens, session traces, or any secret. The
migration is idempotent and non-destructive. The repository never logs or
echoes SQL, connection strings, hostnames, or credentials; the API only ever
sees the structured `code` + `message` envelope.

## 3. Files

| File | Role |
| --- | --- |
| `db/migrations/001_create_behavioral_profiles.sql` | schema DDL (idempotent) |
| `backend/app/repositories/profile_repository.py` | `ProfileRepository` abstract contract (Phase 10) |
| `backend/app/repositories/postgres_profile_repository.py` | psycopg3 + `psycopg_pool` implementation |
| `backend/app/repositories/errors.py` | `DatabaseUnavailableError` (503) / `DatabaseOperationError` (500) |
| `backend/app/repositories/__init__.py` | public re-exports |
| `backend/app/dependencies.py` | `get_profile_store` chooses repository from `database_url` |
| `backend/app/routes/verification.py` | depends on `ProfileRepository` contract now |
| `tests/test_postgres_profile_repository.py` | 37 unit/API tests (no DB required) |
| `tests/test_postgres_integration.py` | 16 real-PostgreSQL tests (gated by `TEST_DATABASE_URL`) |

## 4. Repository design

`PostgresProfileRepository` implements the same `ProfileRepository` contract as
the Phase 9B in-memory store, so the ML service and routes are unchanged in
behaviour. Key properties:

* **Lazy pool** — a small `psycopg_pool.ConnectionPool` is built only on the
  first database operation (`min_size=0`, `open=False` then explicit
  `open()`). `create_app()`, `/health` and dependency resolution never touch
  the database stack; `psycopg` / `psycopg_pool` are imported inside method
  bodies.
* **Connection hygiene** — every operation runs inside
  `with pool.connection():` so the lease is always returned; writes are wrapped
  in `conn.transaction()` on an `autocommit` connection. After a failed
  (unique-violation) insert the pooled connection is reusable for fresh rows.
* **Race-safe duplicates** — `save()` simply executes the `INSERT` with the
  profile's values; the `user_ref` PRIMARY KEY is the duplicate guard. There is
  **no pre-check** (pre-checks race); a concurrent second insert for the same
  user raises `psycopg.errors.UniqueViolation`, which `_run()` translates into
  `ProfileExistsError` → HTTP 409.
* **Structured failures** (no SQL/URL/credential leakage):
  * missing URL / connect failure / pool timeout → `DatabaseUnavailableError`
    (503, `database_unavailable`)
  * unique violation → `ProfileExistsError` (409, `profile_exists`)
  * anything else DB-related → `DatabaseOperationError` (500, `database_error`)

### The save path (bug fix)

`save()` must pass the computed values into the INSERT helper. The helper
signature is `_insert_values(conn, values)`, so the connection and the values
tuple are delivered together while `_run` still only supplies the connection:

```python
def save(self, profile: StoredProfile) -> StoredProfile:
    values = profile_to_values(profile)
    self._run(
        lambda conn: _insert_values(conn, values),   # conn ← pool, values ← profile
        unique_user=profile.user_ref,
    )
    return profile
```

The regression test `test_insert_runs_with_exact_profile_values` asserts the
INSERT executes with `(INSERT_PROFILE_SQL, profile_to_values(profile))`; the
previous code dropped `values` and would have raised for every save.

## 5. Wiring

`get_profile_store` (unchanged dependency name, so every Phase 9B dependency
override keeps working) now returns:

* `PostgresProfileRepository` when `Settings.database_url` is configured
  (`BBA_DATABASE_URL`); and
* `InMemoryProfileStore` when it is not (the Phase 9B fallback).

`routes/verification.py` now type-annotates the dependency as
`ProfileRepository` (the contract both stores implement). Resolution is lazy, so
a configured URL never costs a connection at import/app-boot time.

## 6. Tests

### 6.1 Unit / API tests — `tests/test_postgres_profile_repository.py` (37 tests, no DB)

* Value/row mapping: exact ORDER matches the INSERT columns and the
  placeholder count; `row_to_profile` round-trips driver timestamps; metadata
  never contains the centroid.
* Error translation: `UniqueViolation` → 409 `ProfileExistsError`;
  `OperationalError`/`InterfaceError`/`PoolTimeout`/`PoolClosed` → 503;
  other DB / unknown errors → 500; all messages verified free of SQL, hosts and
  credentials.
* Repository over fakes: `save` executes the INSERT with the real value tuple
  and commits inside a transaction; duplicate insert raises 409 and never
  overwrites (primary-key guard, no pre-check); missing URL / connect failure /
  DB error map to the right domain errors; `get`/`contains`/`count`/
  `list_profiles` run the expected statements; every connection lease is
  returned; `close()` is idempotent; pool is created lazily and never at
  import/dependency-resolution time.
* Dependency wiring: in-memory fallback without a URL, persistent repository
  with one, caching on `app.state`, and no connection on resolution.
* API over the persistent store (TestClient + dependency override with a
  stateful fake PostgreSQL): enroll 201 and stored through the repository;
  duplicate 409 via the repository's unique guard; verify against a persisted
  profile 200; unknown user 404.

### 6.2 Real-PostgreSQL integration tests — `tests/test_postgres_integration.py` (16 tests)

Every test is **skipped unless `TEST_DATABASE_URL` is set** — the normal test
run never touches a database:

```
TEST_DATABASE_URL=postgresql://user:pass@host:5432/testdb \
    python -m pytest tests/test_postgres_integration.py -q
```

Covered: schema creation (columns, types, PRIMARY KEY); insert/retrieve;
centroid round-trip (128-dim float precision, timestamps); duplicate → 409
`ProfileExistsError` with the original row untouched; missing profile → `None`;
`contains`/`count`/`list_profiles`; a **repository A → save → close A →
repository B → read** restart-persistence test; durability across independent
connections; pool reuse after an aborted transaction; table consistency after a
failed insert; every connection returned to the pool; `close()` idempotence;
503 routing for a missing/unreachable URL; and a guard enforcing the six-column
(aggregates-only) schema.

The module operates on an isolated `behavioral_profiles` table which it DROPs
and recreates around the run — it must only ever be pointed at a dedicated
**test** database.

### 6.3 Verification status for this run

* Unit + API (no-DB) suite: **37/37 passed**.
* Real-PostgreSQL integration tests: **written but NOT executed**.
  *Real PostgreSQL persistence smoke test was NOT executed because no
  PostgreSQL instance / `TEST_DATABASE_URL` was available.*
  They collect and skip (16 skipped) under `pytest -q`.
* Full suite: **609 passed, 16 skipped**; frontend **19/19 passed**;
  `compileall` and `create_app(Settings(environment='test'))` pass.

## 7. Limitations

* Real-PostgreSQL behaviour was verified through fakes that mirror psycopg
  semantics (unique-violation propagation, row/columns, transaction commit) —
  **not** against a live server in this environment. The gated integration
  module is ready and must be run against a real PostgreSQL before production
  claims are made.
* Profiles are insert-only: `save` never updates an existing row (matches the
  Phase 9B “do not silently overwrite” rule).
* The pool reopens lazily if a closed repository is used again (safe, by
  design).
* Still an unauthenticated dev/demo API with synthetic artifacts; auth, real
  enrollment UX, and deployment remain future phases.