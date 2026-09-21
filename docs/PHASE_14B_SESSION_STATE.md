# Phase 14B — Server-Authoritative Session Verification State & Re-Verification Gating

**Status: Implemented — automated checks PASSED. Manual browser E2E pending**
(no headless browser is available in this environment, matching Phases 12/13/14A).

## 1. What this phase adds

Phase 14A returns a one-shot decision per continuous window but forgets nothing
that can be *acted on*: the browser could be refreshed mid-lifecycle and the
identity verdict was never authoritative anywhere. Phase 14B makes the
**server** the single source of truth for a session's behavioral verification
state and closes protected actions until a session re-verifies:

- A new persisted table **`behavioral_session_states`** holding *only
  aggregates* per session: `state`, `consecutive_suspicious`,
  `last_verified_at` — never raw events/embeddings/tokens.
- A deterministic per-session key derived server-side as a **SHA-256 digest of
  the validated bearer token itself** (the payload stays exactly
  `{sub, iat, exp}`). A fresh login always issues a distinct token and starts a
  clean, verified session; the gate is per-session, never account-wide.
- **Fail-closed gating**: `POST /api/v1/enrollment` and
  `POST /api/v1/verification` return **403 `reverification_required`** while a
  session is not verified. `GET /api/v1/continuous-verification/state` and
  `POST /api/v1/continuous-verification` stay available so a user can always
  recover by recording a verified continuous window.
- The continuous response is additively extended with
  `session_state`, `consecutive_suspicious`, `last_verified_at`, and a new
  read endpoint returns the authoritative state on page load/refresh.

No phase interacts with the Phase 3 preprocessing pipeline, the Phase 7 CNN+GRU
checkpoint, the Phase 8 verification/evaluation logic, or the calibrated
threshold **0.463513**.

## 2. Hard boundaries (explicitly NOT done)

- **No JWT payload change**: the payload is still exactly `{sub, iat, exp}`. A
  per-issuance `nonce` is signed into the JOSE *header* (not a claim) purely so
  every issued token is a distinct byte string — the Phase 11 token-shape test
  stays green.
- No per-window raw data storage, no vector/embedding persistence, no new PII.
- No account-level locks, no forced logout on a suspicious window (the
  frontend logs out only on 401), no rate limiting, no revocation.
- No Phase 15 scope (refresh tokens, OAuth, MFA, WebSockets, deployment).
- Threshold and the Phase 14A window bounds (300 kbd / 1200 mouse / 1500 total /
  30 s) are unchanged.

## 3. Data model

`db/migrations/003_create_session_states.sql`:

| Column | Type / constraints |
|---|---|
| `user_ref` | TEXT NOT NULL |
| `session_id` | TEXT NOT NULL |
| `state` | TEXT CHECK IN (`'verified'`, `'reverification_required'`) |
| `consecutive_suspicious` | INT NOT NULL DEFAULT 0 CHECK (`>= 0`) |
| `last_verified_at` | TIMESTAMPTZ NULL |
| `created_at` / `updated_at` | TIMESTAMPTZ NOT NULL |
| PRIMARY KEY | (`user_ref`, `session_id`) |

Only these aggregates are persisted. `mark_verified` (on a verified continuous
window) clears a `reverification_required` state and resets the counter;
`mark_suspicious` increments the per-session counter, records `last_verified_at`
preservation semantics, and flips the state to `reverification_required`.

## 4. Backend

### 4.1 Session identity (`backend/app/auth/dependencies.py`)

`CurrentUser` gained `session_id: str = ""` populated by
`_session_id_from_token` — a **SHA-256 digest of the validated bearer token
itself**, not of its claims. The token's header carries a per-issuance
`nonce` (never a payload claim), so every fresh login produces a distinct
token string and therefore a distinct session id **even when two logins share
`sub`, the same `iat` second and the same `exp`**. Replaying an already-issued
token re-derives the same session id and state.

### 4.2 Repository (`backend/app/repositories/`)

- `session_state_repository.py` — frozen `SessionVerificationState`, module
  constants `SESSION_STATE_VERIFIED` / `SESSION_STATE_REVERIFICATION_REQUIRED`,
  the ABC, and `default_session_state`.
- `memory_session_state_store.py` — RLock-synchronized dict store (default when
  no DB URL is configured).
- `postgres_session_state_repository.py` — atomic `INSERT ... ON CONFLICT (…)
  DO UPDATE/NOTHING … RETURNING`, lazy pool, `row_to_session_state`, `get`,
  `get_or_create`, `mark_verified`, `mark_suspicious`, `count`, `close`.
- Wiring in `backend/app/dependencies.py` (`get_session_state_store`) mirrors
  the Phase 13 profile-store selection logic.

### 4.3 Gating (`backend/app/routes/verification.py`)

- `_require_reverified_session(current_user)` is invoked by **POST
  `/enrollment`** and **POST `/verification`**. When the DB reports the session
  as `reverification_required` (or is unreachable → `database_unavailable`),
  the fork fails **closed** with a 403.
- `ContinuousVerificationResponse` adds `session_state`,
  `consecutive_suspicious`, `last_verified_at`; the continuous handler calls
  `mark_verified`/`mark_suspicious` so its decision is immediately durable.
- New **`GET /api/v1/continuous-verification/state`** returns
  `{session_state, consecutive_suspicious, last_verified_at}` for the current
  session.

### 4.4 HTTP contract

| Outcome | Status / code |
|---|---|
| Unauthenticated / bad token | 401 |
| Enroll/verify while session unverified | **403 `reverification_required`** |
| DB unreachable | 503 `database_unavailable` (fail closed) |
| State read on a fresh session | 200 `{verified, 0, null}` |

## 5. Frontend

- `src/api.js` — `getContinuousSessionState` (GET state endpoint).
- `src/lib/continuous.js` — `SESSION_STATES`, `sessionDecisionToState`,
  `restoreServerState`; the `DECISION` reducer action is **server-authoritative**
  (`session_state` wins, decision mapping is only the defensive fallback); new
  `RESTORE` action hydrates the machine from the server on page load and can
  never fabricate a state.
- `src/hooks/useContinuousVerification.js` — fetches the state endpoint once
  per token on mount and dispatches `RESTORE` (401 → logout).
- Windows remain **transient React state** — no localStorage/sessionStorage/
  IndexedDB/cookies for the verdict; a refresh re-fetches from the server.

## 6. Test coverage

Backend: `tests/test_session_state.py` (token-derived session identity),
`tests/test_session_state_gating.py` (message + code cohesion; real-token
replay/fresh-login isolation; no `session_id` in API responses or ML
features), `tests/test_auth_jwt.py` (per-issuance nonce + payload integrity),
`tests/test_postgres_session_state_integration.py`,
`tests/test_postgres_gating_integration.py` (incl. persistence across a fresh
repository pool), and updates to `tests/test_postgres_api_integration.py` /
`tests/test_continuous_verification_api.py`.
Frontend: `tests/continuous.test.js` (RESTORE, server-authoritative DECISION,
fallback, endpoint client).

## 7. Verification summary

- Backend **no DB**: 778 passed, 60 skipped.
- Backend **with real PostgreSQL**: 838 passed, 0 failed.
- Frontend: `npm test` 69/69; `npm run build` clean; `compileall` clean;
  `git diff --check` clean.
- DB restart durability: a `reverification_required` row written, a full
  PostgreSQL server restart performed, and the row read back intact from a
  brand-new pool.

## 8. Known caveats

- A reissued access token cannot recover an older session's behavioral state:
  only the EXACT token that created a `behavioral_session_states` row re-derives
  that session id (the same user can log in again freely and starts clean).
  This is the intended per-login-session isolation — no cross-token state
  migration exists.