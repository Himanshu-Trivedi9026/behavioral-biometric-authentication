# Phase 14A — Periodic Continuous Behavioral Verification

**Status: Implemented — automated checks PASSED. Manual browser E2E pending**
(no headless browser is available in this environment, matching Phases 12/13).

## 1. What this phase adds

Continuous (periodic) re-verification on top of the existing authentication
and one-shot behavioral verification:

- One new protected endpoint **`POST /api/v1/continuous-verification`**.
- A frontend **Continuous Verification** page (`/continuous`) that captures a
  *bounded* behavioral window, submits it, and reflects the machine state
  `authenticated → collecting → verifying → verified | re-verification-required`.
- Pure, dependency-free window logic + a reducer state machine in the frontend
  (`src/lib/continuous.js`) and a small hook (`src/hooks/useContinuousVerification.js`).

No phase interacts with the Phase 3 preprocessing pipeline, the Phase 7 CNN+GRU
checkpoint, the Phase 8 verification/evaluation logic, or the calibrated
threshold **0.463513**. Continuous verification reuses the exact same
fidelity path a one-shot probe uses.

## 2. Hard boundaries (explicitly NOT done)

This phase deliberately does **not** implement any Phase 14B+ behavior:

- No refresh / sliding tokens, no OAuth, no MFA, no password reset.
- No automatic logout / account locking / forced re-login on a suspicious window.
- No rate limiting, no Redis, no WebSockets, no Docker/deployment.
- No re-training, no re-calibration, no threshold change, no ML-architecture change.
- No storing of the behavioural windows themselves (raw events are never
  persisted anywhere in this phase).
- No capture of key identity / typed text (the Phase 2 collector contract is
  unchanged and privacy-clean).
- No changes to enrollment/verification behaviour or to existing responses.

## 3. Backend

### 3.1 Schema (`backend/app/schemas.py`)

- `ContinuousVerificationRequest(RawSession)` — a single bounded behavioural
  window in the exact Phase 2 raw session shape, with
  `model_config = ConfigDict(extra="forbid")`. Identity fields (`user_ref`,
  `username`, …) and any extra fields are rejected with **422**, so identity
  can never be chosen by the client.
- `ContinuousVerificationResponse` — intentionally minimal:
  `{decision, distance, threshold}`. No `user_ref`, no centroid, no embedding,
  no raw events, no credentials.

### 3.2 Endpoint (`backend/app/routes/verification.py`)

`POST /api/v1/continuous-verification`, protected by the Phase 11
`get_current_user` (HTTP-bearer JWT). Behaviour:

1. The authenticated JWT identity is the **sole** profile selector:
   `user_ref = current_user.username`.
2. `store.get(user_ref)` → **404 `profile_not_found`** when the user has no
   enrolled profile (never a 403, matching one-shot verification).
3. The window is run through the existing service
   `verify_stored(user_ref, stored_profile, window)` — same Phase 3
   preprocessing, same Phase 7 encoder, same Phase 8 distance/threshold rule.
   An empty/invalid window is rejected by the ML service with
   **422 `invalid_session`** (established Phase 9B behaviour).
4. Only the decision/distance/threshold are returned; **the window and its raw
   events are never stored** (no `store.save`, no logging of events).

The route is registered inside the existing `build_versioned_router`, so the
`/api/v1` prefix, auth, error envelope, 405 behaviour, and OpenAPI generation
are shared with enrollment/verification.

### 3.3 Security properties (test-proven)

| Property | Mechanism | Test |
|---|---|---|
| Unauthenticated → 401 | `get_current_user` (bearer required) | `test_unauthenticated_request_is_401` |
| Invalid token → 401 | JWT signature check | `test_invalid_bearer_token_is_401` |
| Client can't select a profile | body extra fields forbidden | `test_client_cannot_select_another_users_profile` |
| JWT identity selects profile | `user_ref = current_user.username` | `test_jwt_identity_determines_profile` |
| Server-side threshold only | client `threshold` in body → 422 | `test_client_supplied_threshold_is_rejected` |
| Minimal response | exactly `{decision,distance,threshold}` | `test_response_shape_is_minimal` |
| No leakage | response has no centroid/embedding/raw/identity | `test_response_leaks_no_...` |
| Profile untouched | verify never mutates the stored profile | `test_stored_profile_is_not_modified_by_verification` |

## 4. Bounded windows (frontend)

Each window is one `BehavioralSession` (Phase 2 collector, unchanged). Limits
were derived from the collector's own sampling so a window can never approach
the backend `MAX_SESSION_EVENTS = 20000` ceiling:

| Bound | Value | Derivation |
|---|---|---|
| `MAX_KEYBOARD_EVENTS` | 300 | typing ≈ ≤10 Hz × 30 s |
| `MAX_MOUSE_EVENTS` | 1200 | mousemove throttled to ≤ 40 Hz × 30 s |
| `MAX_TOTAL_EVENTS` | 1500 | combined upper bound, checked first |
| `MAX_DURATION_MS` | 30 000 | window never runs past 30 s |

Checks are order: total → keyboard → mouse → duration. The hook auto-stops the
collector at the first cap (`WINDOW_CAP_LABELS` shown in the UI) and only a
window with both keyboard and mouse activity (≥ 2 events each, the existing
data-sufficiency gate) and no cap breach is submittable
(`canSubmitWindow` in `src/lib/continuous.js`).

## 5. Frontend

- **`src/lib/continuous.js`** — bounds constants, `windowEventCounts`,
  `windowDurationMs`, `windowFullReason`, `shouldStopCollecting`,
  `windowHasUsableContent`, `canSubmitWindow`, `buildContinuousWindow`
  (bare, identity-free deep copy), `initialState`, and the pure
  `continuousReducer` state machine
  (`authenticated → collecting → verifying → verified | reverification_required`,
  invalid transitions are identity-safe no-ops).
- **`src/hooks/useContinuousVerification.js`** — bridges the reducer, the
  (unchanged) `useCollector`, auto-stop polling (500 ms) for caps that occur
  without fresh events, and the `continuousVerify` API call. On 401 it logs
  out; on `profile_not_found` it surfaces the enrollment prompt; the captured
  window exists only in React state and is replaced by fresh state on every
  window (`window: null` on START/RESET).
- **`src/api.js`** — `continuousVerify({session, token})` posts the *bare*
  window (not wrapped, no `user_ref`) with the bearer token.
- **`src/continuous/`** — `ContinuousVerificationPage.jsx` + `components/`
  (`ContinuousStatus` chips, `WindowCapture`, `ContinuousResult`,
  `ContinuousSpecification`) + `styles/continuous.css` mirroring the existing
  dark design system.
- Routing: `router.jsx` `/continuous` route; `App.jsx` adds `continuous` to
  `PROTECTED_ROUTES` (unauthenticated → `/login`); `AuthenticatedHeader` adds
  a `CONTINUOUS` nav link.

### 5.1 Privacy on the frontend

A captured window lives only in the hook's React state. It is sent exactly
once and is cleared on submit/reset. No window is written to
`localStorage`/`sessionStorage` (the only persisted item in the app remains the
auth session). The payload never contains a `user_ref` or any key identity.

## 6. Tests

### 6.1 Backend (`tests/test_continuous_verification_api.py` — 21 tests)

Authentication (missing/invalid token, unknown profile), validation (malformed
window, empty window → 422 `invalid_session`, missing `session_id`, body
identity → 422), decisions (same-session VERIFIED, zero-threshold SUSPICIOUS,
decision ↔ threshold consistency), identity isolation (JWT-selects-profile via
a recording wrapper; cross-user body → 422), response privacy (exact shape, no
leakage), profile integrity (profile unmodified, no new profiles), threshold
integrity (server threshold only), and existing endpoints unchanged
(enrollment/verification still return the established shapes/decisions).

### 6.2 Frontend (`tests/continuous.test.js` — 20 tests)

Window bounds and caps, duration semantics, `canSubmitWindow` gates, the bare
identity-free payload builder, the full reducer state machine (valid/invalid
transitions, decision mapping, RESET semantics), and the `continuousVerify`
API client (missing token, bare-window body + bearer header).

## 7. Regression results

- Full backend suite with `TEST_DATABASE_URL`: **779 passed** (758 baseline
  + 21 new) — 0 failed, 0 skipped.
- Backend suite without `TEST_DATABASE_URL`: 743 passed, 36 skipped (real-DB
  tests skip cleanly as before).
- `compileall backend ml scripts`: **PASS**.
- Frontend `npm test`: **61 passed** (41 baseline + 20 new).
- `npm run build`: **PASS** (93 modules — CSS/JS bundles grew by the new page).
- `git diff --check`: clean.
- Pre-existing uvicorn on port 8000 (PID 249198) left untouched; Phase 13
  PostgreSQL 16.15 still running at `127.0.0.1:5432`.

## 8. Honest guard-rails

- **No browser is available** in this environment, so the real-typing browser
  E2E for `/continuous` cannot be automated/manually run here. The page is
  covered by tests + production build; a manual browser pass is recommended
  before claiming PASSED.
- **Known Phase 12 limitation kept on purpose:** real-browser probes on this
  development checkpoint verify as **SUSPICIOUS** (distances ≈1.30 and ≈1.52
  vs threshold 0.463513). The model/threshold are intentionally **not**
  changed to fix this; a continuous window taken by the same real-browser
  flows is expected to request re-verification. This is documented, not
  "fixed".
- Like one-shot verification, this is a **development system**: synthetic-dev
  artifacts, no production claims.

## 9. Phase 14B and later

Not started: refresh tokens, auto-logout/locking on repeated suspicion, rate
limiting, Redis/WebSockets, multi-device sessions, deployment, retraining.