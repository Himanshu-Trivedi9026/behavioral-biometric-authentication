# Phase 13 — Live PostgreSQL Persistence & Production-Style Data Integration

## Objective

Validate the PostgreSQL-backed repository layer (Phase 10/11) against a real
PostgreSQL server and prove that accounts and behavioural profiles survive
backend process restarts — the minimum bar for a production-style persistence
claim.

## PostgreSQL Environment

A user-local PostgreSQL **16.15** cluster was installed (no sudo / Docker
required) and is **not** committed to the repository:

* Binaries: `/tmp/opencode/pg/bin` (from `embedded-postgres-binaries-linux-amd64-16.15.0.jar`)
* Data: `/tmp/opencode/pgdata`
* Endpoint: `127.0.0.1:5432`
* Superuser: `himanshu` (trust auth)
* App DB: `behavioral` — both migration files applied; verified schema identical
  to repository expectations
* Test DB: `bba_test` — dedicated integration-test database; per-module
  drop+recreate fixtures

## Acceptance Results

| Criterion | Result |
|---|---|
| Real users table schema correct | **VERIFIED** |
| Real behavioural_profiles schema correct | **VERIFIED** |
| Register + login against real PostgreSQL | **201 / 200** |
| Enroll against real PostgreSQL | **201** |
| Cross-process login + verify (restart survival) | **200 / 200 VERIFIED** |
| Duplicate username → 409 + single row in DB | **VERIFIED** |
| Duplicate enrollment → 409 + single row in DB | **VERIFIED** |
| Health independent of DB | **200 / 200** |
| Unreachable DB → 503 `database_unavailable`, no leakage | **VERIFIED** |
| Connection safety (no leaks after process exit) | **0 stray connections** |
| Full pytest suite (with `TEST_DATABASE_URL`) | **758 passed, 0 failed, 0 skipped** |
| Full pytest suite (without `TEST_DATABASE_URL`) | **36 DB tests skipped cleanly** |
| `npm test` (frontend) | **41/41 passed** |
| `npm run build` | **PASS** |
| `python3 -m compileall backend ml scripts` | **PASS** |
| Phase boundary audit | **clean** (Phase 14 not touched) |

**PHASE 13 — PASSED**

## Live Restart Smoke Test

Two real uvicorn processes (`backend.app.main:app`) ran sequentially on port
8010 against the `behavioral` database with real ML artifacts and a real JWT
secret.

### Instance A

```
health  → 200
register → 201  (user_id: 4edf0625...)
login    → 200  (bearer token issued)
enroll   → 201  (embedding_dim: 128, session_count: 1)
```

Process A terminated (SIGTERM). PostgreSQL remained running independently.

### PostgreSQL (rows persisted across the restart)

```
users:              id=4edf0625...  username=smoke_1789830951  created_at=2026-09-19T15:15:52.288341+00:00
behavioral_profiles: user_ref=smoke_1789830951  embedding_dim=128  session_count=1  created_at=2026-09-19T15:15:54.755421+00:00
```

### Instance B (fresh process, same DB)

```
health  → 200
login    → 200
verify   → 200  decision: VERIFIED  checkpoint: real  embedding_dim: 128
```

## Real-Database Integration Tests

Three dedicated modules gate every real-DB test behind `TEST_DATABASE_URL`.

### tests/test_postgres_api_integration.py (5 tests — NEW)

App-level persistence test: two independent `create_app` instances share the
same real PostgreSQL database via a fake ML verifier (the persistence contract
is the database, not the model). Includes duplicate guards, health independence,
and structured 503 without credential leakage.

### tests/test_postgres_integration.py (17 tests)

Profile-repository behaviour over real PostgreSQL (centroid insert, dimension
checks, count/list, duplicates, transaction rollback, missing user, close).

### tests/test_postgres_user_integration.py (14 tests — 1 fixed)

User-repository behaviour over real PostgreSQL (register, get-by-id/username,
duplicate username 409, transaction rollback, missing user, list, timestamps).
The module-scoped `test_contains_count_list_profiles` count assertion was
fixed: it previously assumed a clean table but ran after other tests inserted
rows; it is now order-independent (delta-based, set membership, no-centroid
check).

### Full Suite Result

```
758 passed in 33.29s  (with TEST_DATABASE_URL set)
```

When `TEST_DATABASE_URL` is unset, all 36 real-DB tests skip cleanly:

```
36 skipped in 2.07s
```

## Privacy Audit

Two tables in the `behavioral` database:

**users**
| Column | Type |
|---|---|
| id | uuid (PK) |
| username | text (UNIQUE, NOT NULL) |
| password_hash | text (NOT NULL, Argon2id) |
| created_at | timestamptz |
| updated_at | timestamptz |

**behavioral_profiles**
| Column | Type |
|---|---|
| user_ref | text (PK) |
| centroid | double precision[] (summary embedding) |
| embedding_dim | integer |
| session_count | integer |
| created_at | timestamptz |
| updated_at | timestamptz |

* No raw keyboard/mouse events stored anywhere in PostgreSQL.
* Centroid is a numeric array (mean of session embeddings), not raw event data.
* `password_hash` is Argon2id; no plaintext password is ever stored.
* `user_ref` is never a model feature — it is metadata only.
* No JWT tokens, secrets, or connection URLs are persisted or logged.

## Connection Safety

Zero leaked PostgreSQL connections were observed after every smoke test,
integration-test module, and process termination. Both repositories use
`psycopg_pool.ConnectionPool` with `autocommit=True`; all `close()` calls
were exercised and no dangling connections remain in `pg_stat_activity`.

## Files Changed

* `tests/test_postgres_api_integration.py` — **NEW** (app-level persistence test)
* `tests/test_postgres_integration.py` — fixed order-dependent count assertion
* `docs/PHASE_13_POSTGRESQL_PERSISTENCE.md` — this document
* `docs/PROJECT_STATUS.md` — updated to Phase 13 Completed

No backend source code was modified. No ML code, checkpoint, threshold, or
preprocessing configuration was touched. No frontend code was touched.

## Honest Guard-Rails

* No headless browser was available; the full real-typing browser E2E was not
  automated (covered by backend live smoke + 41 frontend unit/integration
  tests + build pass + preview-route verification from Phase 12; manual browser
  pass recommended).
* Synthetic session events (3 keyboard, 3 mouse events) were used for the live
  smoke; distances and decisions are a functional correctness proof, not a
  real-biometric performance claim.
* A pre-existing uvicorn process (PID 249198, port 8000) from an earlier
  Phase 12 diagnostic session was observed running and left untouched (no
  Phase-13 conflict; it was noted and not killed unilaterally).
