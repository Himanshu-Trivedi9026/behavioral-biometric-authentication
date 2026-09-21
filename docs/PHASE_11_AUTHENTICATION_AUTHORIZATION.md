# Phase 11 — Authentication & Authorization

## 1. Objective

Add account registration and login to the API and make the ML endpoints
(enrollment / verification) operate **only on the authenticated user's own
profile**. Passwords are hashed with Argon2id and callers present an HS256 JWT
bearer token. The authenticated identity — never a request-body field — is the
single source of authorization.

Phase 11 changes **no** ML behaviour: preprocessing, encoder, Siamese model,
checkpoint, calibrated threshold and the ML service are untouched. Identity is
control-plane metadata and never becomes a model feature.

Not in scope (and deliberately not implemented): React login UI, refresh
tokens, OAuth/Google/GitHub login, password reset, email verification, MFA,
Redis, WebSockets, continuous/behavioral re-authentication, Docker/deployment,
new ML models, retraining, recalibration, rate limiting.

## 2. Authentication flow

```
POST /api/v1/auth/register   {username, password}
    -> 201 {id, username, created_at}          (never the password/hash)

POST /api/v1/auth/login      {username, password}
    -> 200 {access_token, token_type:"bearer"} (HS256 JWT, sub = user id)

POST /api/v1/enrollment      Authorization: Bearer <JWT>
POST /api/v1/verification    Authorization: Bearer <JWT>
```

* Registration enforces a username charset (`^[A-Za-z0-9._-]+$`, 3–64 chars)
  and a password length policy (8–128 chars); violations are `422`.
* Duplicate usernames are rejected by the database unique constraint → `409`
  `username_exists` (no pre-check, so concurrent registration cannot race).
* Login answers the **identical** generic `401 invalid_credentials` for an
  unknown username and a wrong password, so account existence is never
  disclosed.
* `/health` and `/api/v1/health` remain public and never require a token.
  Register/login are public by necessity; enrollment/verification are protected.

## 3. Password hashing (Argon2id)

`backend/app/auth/security.py`:

```python
hash_password(password) -> str          # argon2-cffi PasswordHasher (Argon2id)
verify_password(password, password_hash) -> bool
```

* `verify_password` calls `PasswordHasher.verify(password_hash, password)`
  (argon2-cffi argument order is `hash` first).
* It returns `False` — never raises — for a wrong password, a malformed hash,
  a non-string input, or a hash that is not Argon2id (`$argon2id$…`). Only
  Argon2id hashes are ever produced or accepted.
* Defaults are the argon2-cffi secure defaults (`t=3`, `m=65536`, `p=4`,
  32-byte hash, 16-byte salt). The salt is generated per hash, so hashing the
  same password twice yields different values.
* Passwords and hashes are never logged and never returned by the API.

## 4. JWT

* **Library/algorithm:** PyJWT, **HS256 only** (`Settings.jwt_algorithm`).
* **Secret:** read from `BBA_JWT_SECRET_KEY`. There is **no usable hard-coded
  default** (`jwt_secret_key = ""`). A configured secret shorter than 32
  characters or matching a known placeholder list is rejected at
  `Settings` construction; a `production` environment without a secret is
  rejected. When the secret is empty, token creation fails closed with `503
  auth_unavailable` and decoding fails as `401`.
* **Claims:** `sub` (user id), `iat`, `exp` only. No password/hash/embedding/
  raw-event/secret is ever placed in the token.
* **Validation:** `decode_access_token` verifies the signature, requires
  `exp` and `sub`, enforces expiration, and rejects a non-string/blank `sub`.
  Malformed tokens, wrong signatures and expired tokens all collapse to the
  same `401 invalid_credentials`.

## 5. Authorization / profile ownership

`GET`/`POST` enrollment and verification depend on `get_current_user`
(`backend/app/auth/dependencies.py`), which reads the bearer token, validates
it, loads the user by `sub`, and returns a hash-free `CurrentUser`. The route
then resolves the profile key with `_authorized_user_ref`:

| Request-body `user_ref` | Result |
| --- | --- |
| omitted | authenticated username is used |
| equals authenticated username | allowed |
| different from authenticated username | `403 forbidden` |

A request-body `user_ref` can therefore **never** select another user's
profile, and an authenticated user A can never enroll into or verify against
user B's profile. Identity comes exclusively from the JWT.

## 6. Users schema

`db/migrations/002_create_users.sql` (idempotent, non-destructive; does not
touch `behavioral_profiles`):

```sql
CREATE TABLE IF NOT EXISTS users (
    id            UUID        PRIMARY KEY,
    username      TEXT        NOT NULL UNIQUE,
    password_hash TEXT        NOT NULL,
    created_at    TIMESTAMPTZ NOT NULL,
    updated_at    TIMESTAMPTZ NOT NULL,
    CONSTRAINT users_username_non_blank CHECK (length(btrim(username)) > 0)
);
```

Only identity records are stored. No behavioural events, embeddings, JWT
secrets or tokens. The `UNIQUE` constraint on `username` is the race-safe
duplicate guard.

## 7. User repository & wiring

* `backend/app/repositories/user_repository.py` — abstract `UserRepository`
  contract + immutable `UserRecord` (`id`, `username`, `password_hash`,
  `created_at`, `updated_at`).
* `backend/app/repositories/memory_user_repository.py` — thread-safe
  non-persistent fallback.
* `backend/app/repositories/postgres_user_repository.py` — psycopg3 +
  `psycopg_pool` implementation. Lazy pool (never opened during
  `create_app()` or dependency resolution), connections always returned,
  `create()` passes the real `user_to_values(user)` tuple to
  `_insert_values(conn, values)` (the Phase 10 `save()` bug is **not**
  repeated), unique violation → `409 username_exists`, no SQL/URL/credential
  leakage.
* `get_user_repository()` returns `PostgresUserRepository` when
  `BBA_DATABASE_URL` is configured and `InMemoryUserRepository` otherwise.
  Resolution never opens a database connection.
* The auth dependencies (`get_current_user`) use the same `get_user_repository`
  dependency, so tests override one dependency cleanly.

## 8. ML / privacy isolation

Verified by `tests/test_auth_privacy.py` and `tests/test_api_privacy.py`:

* The authenticated identity (`username` / `user_ref` / JWT claims) is used
  **only** as the profile key and response label. A recording verifier proves
  processed session dicts passed to the encoder contain none of
  `user_ref`, `user_id`, `password`, `token`, `jwt`, key identity or characters.
* Passwords, Argon2id hashes, bearer tokens and the JWT secret never appear in
  any response (including OpenAPI).
* `Settings.__repr__`/`__str__` redact `database_url` and `jwt_secret_key`.
* The `users` record and the `users` table persist only identity fields.

## 9. Files

| File | Role |
| --- | --- |
| `backend/app/auth/security.py` | Argon2id hashing + PyJWT create/decode |
| `backend/app/auth/schemas.py` | register/login/token/user Pydantic schemas |
| `backend/app/auth/errors.py` | 401/403/503 structured auth errors |
| `backend/app/auth/dependencies.py` | `get_current_user`, `CurrentUser` |
| `backend/app/routes/auth.py` | `POST /auth/register`, `POST /auth/login` |
| `backend/app/repositories/user_repository.py` | `UserRepository` contract, `UserRecord`, `UsernameExistsError` |
| `backend/app/repositories/memory_user_repository.py` | in-memory implementation |
| `backend/app/repositories/postgres_user_repository.py` | PostgreSQL implementation |
| `backend/app/config.py` | JWT settings, weak-secret/production validation, secret redaction |
| `backend/app/dependencies.py` | `get_user_repository` (lazy, config-driven) |
| `backend/app/routes/verification.py` | protected + owner-scoped ML routes |
| `backend/app/main.py` | registers the auth router |
| `db/migrations/002_create_users.sql` | idempotent users table |
| `.env.example` | `BBA_*` example config (empty JWT secret) |
| `requirements.txt` | `PyJWT>=2.8.0`, `argon2-cffi>=23.1.0` |
| `tests/auth_testing.py` | shared test-only auth helpers |

## 10. Tests

### 10.1 Dedicated Phase 11 tests

* `tests/test_auth_security.py` (11) — Argon2id format/salt, correct/wrong
  password, malformed/non-Argon2id/non-string safety, defaults.
* `tests/test_auth_jwt.py` (19) — round-trip, minimal claims, expiration,
  malformed/wrong-signature/missing-sub/non-string-sub/missing-exp, fail-closed
  when unconfigured, settings validation + redaction.
* `tests/test_auth_api.py` (25) — register 201/409/422, no password/hash in
  responses, login token shape, identical generic 401, 503 without a secret.
* `tests/test_auth_authorization.py` (30) — 401 for missing/malformed/invalid/
  expired tokens, own-profile access, omitted/matching/mismatching `user_ref`,
  cross-user `403`, body cannot override identity, full register→login→ML flow.
* `tests/test_auth_privacy.py` (8) — secrets never in responses/OpenAPI,
  identity never in model sessions, store holds identity only.
* `tests/test_user_repository.py` (20) — mapping helpers, PG repository over
  fakes, duplicate guard, error translation, in-memory implementation,
  dependency wiring.
* `tests/test_postgres_user_integration.py` (15) — real-PostgreSQL tests gated
  by `TEST_DATABASE_URL`.

Dedicated result: **113 passed** (integration tests collect/skip without a
database).

### 10.2 Affected Phase 9B/10 tests updated for auth

`tests/test_enrollment_api.py`, `tests/test_verification_api.py`,
`tests/test_api_privacy.py`, `tests/test_api_health_isolation.py`,
`tests/test_api_ml_integration.py`, `tests/test_api_errors.py` and
`tests/test_postgres_profile_repository.py` now run the ML endpoints as an
authenticated identity (dependency override or real JWT). The Phase 9A
"empty-body POST → 422" routing assertion became "unauthenticated POST → 401"
(auth is evaluated before body validation), which still proves the routes exist.

### 10.3 Verification status for this run

* Dedicated Phase 11 suite: **113/113 passed**.
* Full suite: **722 passed, 31 skipped**.
* The 31 skips are the Phase 10 + Phase 11 real-PostgreSQL integration tests;
  `TEST_DATABASE_URL` was unavailable, so **no live PostgreSQL test was
  executed**.
* Frontend `npm test`: **19/19 passed** (untouched).
* `compileall backend ml scripts`: **passed**;
  `create_app(Settings(environment='test', …))`: **passed**.
* Live uvicorn smoke test: health 200, register 201, login 200, authenticated
  enrollment 201, authentication verification 200, missing/invalid JWT 401,
  cross-user `user_ref` 403 — all passed. Temporary server stopped afterwards.

## 11. Security review

* `verify_password` calls `PasswordHasher.verify(hash, password)` and fails
  safe for wrong/malformed/invalid/non-string input; Argon2id-only; no
  password/hash logging.
* JWT: PyJWT, HS256 only, secret from `BBA_JWT_SECRET_KEY`, no usable default,
  ≥32-character enforcement, production fails closed, minimal claims, `exp`
  enforced, `sub` required, wrong signature/malformed rejected.
* Authorization: authenticated identity is authoritative; omitted → own
  identity, matching → allowed, different → 403; body `user_ref` never
  overrides the JWT.
* ML privacy: username / user id / user_ref / JWT claims / password / hash
  never enter feature tensors or processed behavioural sessions.
* `Settings` repr/str redact `database_url` and `jwt_secret_key`; no
  hard-coded secrets or credentials anywhere in the source.

## 12. Limitations

* **Live PostgreSQL was not tested in this environment** (`TEST_DATABASE_URL`
  unavailable). `PostgresUserRepository` is verified against fakes plus the
  gated integration module, which must be run against a real PostgreSQL before
  any production claim.
* Profiles remain in memory unless `BBA_DATABASE_URL` is configured.
* Access tokens are short-lived bearer tokens with no refresh/revocation list.
* This is a development service with synthetic ML artifacts; it makes **no
  claim of production readiness** and no claim of real-world biometric
  accuracy.
* No later phase (refresh/OAuth/MFA/Redis/WebSockets/continuous auth/
  deployment/retraining/rate limiting) was implemented.
