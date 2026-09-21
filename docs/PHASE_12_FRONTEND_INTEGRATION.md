# Phase 12 — Frontend Full Integration

## 1. Objective

Connect the React frontend to the Phase 11 backend for a **real, end-to-end
behavioral authentication flow**:

```
register  →  login  →  authenticated session  →  enrollment (real collector)
           →  POST /api/v1/enrollment         →  verification (real probe)
           →  POST /api/v1/verification       →  VERIFIED / SUSPICIOUS
```

Everything is driven by the existing Phase 2 browser collector
(`frontend/src/lib/collector.js` + `hooks/useCollector.js`) and the Phase 11
protected API (`POST /api/v1/enrollment`, `POST /api/v1/verification`). **No
backend, schema, ML model, checkpoint or threshold was modified** — Phase 12 is
frontend-only.

Not in scope (deliberately not implemented): refresh tokens, OAuth, MFA,
password reset, email verification, rate limiting, Redis/WebSockets,
continuous/behavioral re-authentication, a new dashboard page, deployment,
retraining, recalibration, and any new backend endpoint.

## 2. Resulting flow

* **Public routes** — `/`, `/register`, `/login`, `/collector`.
* **Protected routes** — `/enrollment`, `/verification`. An unauthenticated
  visitor is redirected to `/login`; an authenticated visitor who lands on
  `/login` is redirected to `/enrollment`.
* **Registration** (unchanged Page 3) calls `POST /auth/register`; on success
  the card's button leads to `/login`.
* **Login** calls `POST /auth/login`; on success the token is stored and the
  flow proceeds to `/enrollment` (the success card's CTA is now
  "Continue to Enrollment →").
* **Enrollment** captures 1+ real sessions with the collector, then submits
  them in one authenticated `POST /enrollment` and renders the returned
  `status` / `embedding_dimension` / `session_count` (and a verified-good
  `user_ref`).
* **Verification** captures exactly one probe session and submits it as one
  authenticated `POST /verification`, then renders the backend
  decision (`VERIFIED`/`SUSPICIOUS`) with `distance` and `threshold`.
* **Authenticated header** shows the current username and a working "Log Out"
  action plus an active-item nav to Enrollment / Verification.

## 3. Authentication state layer (`frontend/src/auth.js`)

`auth.js` is now a small reactive store built on `useSyncExternalStore`:

* `logIn({ token, username })`, `logOut()`, `getAccessToken()`,
  `getUsername()`, `isAuthenticated()`, and the `useAuth()` hook.
* A `createSessionStore(storage)` factory is exported so tests and non-browser
  environments get a controllable/in-memory store.

### Chosen client-side token-storage approach (required documentation)

The Phase 11 JWT access token is stored in **`sessionStorage`** under the key
`bba.session.v1` (value: `{ accessToken, username }`), with an in-memory mirror
so the app works even if storage is unavailable (quota/security).

* **Why:** the task requires authentication state to survive a page refresh,
  and the smallest approach that does so without long-lived persistence is
  `sessionStorage` — it is scoped to the tab, automatically cleared when the
  tab/window closes, and never sent to the server on its own.
* **Returned from the server:** only `access_token` (JWT) + `username` (the
  identity the user just authenticated with). **A password is never stored** —
  in `sessionStorage`, in `localStorage`, in memory, in logs, or anywhere else.
* **Delivery detail:** the JWT is only ever attached as the
  `Authorization: Bearer` header inside `frontend/src/api.js`; it is never
  rendered, never logged, and never placed in a URL.
* **Trade-offs accepted:** an XSS attacker who can read page context could read
  the token from `sessionStorage` — the same exposure as any tab-scoped
  credential. Because the token is session-scoped and cannot be revoked without
  a refresh-token mechanism (out of scope), expiry is enforced by the JWT `exp`
  claim: an expired/missing token becomes the clear "session expired — log in
  again" state and the route guard redirects to `/login`.

## 4. API client (`frontend/src/api.js`)

Refactored around a single `request(path, { method, token, body, timeoutMs })`
helper that:

* attaches `Content-Type` and `Authorization: Bearer` only when needed;
* aborts after a 30 s timeout (`AbortController`) and maps an abort to a
  `timeout` error (status `0`);
* parses the backend error envelope `{"error": {code, message}}` into
  `{ ok, status, code, message }`, keeping the previous `registerAccount` /
  `loginAccount` contracts identical for the existing pages.

New endpoints added (frontend-only):

* `enrollSessions({ sessions, token })` → `POST /enrollment` with the raw
  sessions; returns `{ ok, result }` with `user_ref`, `status`,
  `embedding_dimension`, `session_count` on `201`.
* `verifySession({ session, token })` → `POST /verification` with the raw
  probe; returns `{ ok, result }` with `user_ref`, `decision`, `distance`,
  `threshold` on `200`.

Missing/absent tokens fail closed client-side (`401` `missing_token`) before any
request is made.

## 5. Collector integration

`hooks/useCollector.js` gained one additive field: it now also exposes the
current session object (`session`, from `BehavioralSession.getSession()`) so
pages can submit the actual captured raw session to the API. Existing behaviour
(listeners only while running, event sampling, counts, download/clear) is
unchanged, and the 19 collector unit tests still pass untouched.

* Keyboard events store only `event_type`/`event`/`timestamp`; mouse events
  store `event_type`/`event`/`x`/`y`/`timestamp`. No key identities, text or
  values anywhere.
* The TypeZone-style capture areas are **uncontrolled** textareas — their typed
  content is never read, rendered, or sent.

## 6. Enrollment page (`frontend/src/enrollment/`)

Real flow, two-column layout matching Register/Login (7fr/5fr grid, dark navy
visual system, detailed "REGISTRATION STATUS"-style strip):

1. **LIVE CAPTURE** area + counters; "Start Session" / "Finish Session".
2. On finish, the collected session is validated with a **frontend
   data-sufficiency gate** (`lib/enrollment.js: hasUsableContent`):
   * ≥ 2 keyboard events **and** ≥ 2 mouse events. This is a UX quality gate to
     avoid submitting effort-free/empty sessions; it is *not* presented as a
     backend rule. The backend remains authoritative: any session that does not
     satisfy the Phase 9B schema or has no usable behaviour is answered by the
     API with a structured `422 invalid_session`.
3. Completed sessions accumulate in a "SESSIONS READY TO SUBMIT" list (1–16,
   mirroring `MAX_ENROLLMENT_SESSIONS`). The user may add more.
4. "Send N Sessions to Server" → one authenticated `POST /enrollment`.
5. `201` → real result card (`user_ref`, `status`, `embedding_dimension`,
   `session_count`) + "Continue to Verification →".
6. `409 profile_exists` → clear message + "Continue to Verification →".
7. `401` → store cleared and the protected-route guard sends the user to
   `/login`. `422/503/500/0` → specific, honest error text; "Record Sessions
   Again" restarts cleanly.

The button stays disabled until at least one session is captured; the identical
gate message guides the user when a session carries too little activity.

## 7. Verification page (`frontend/src/verification/`)

1. **LIVE PROBE** capture of exactly one session (same gate as enrollment).
2. "Run Verification" → one authenticated `POST /verification`.
3. `200` → result card with the backend decision prominently shown
   (`VERIFIED` in green / `SUSPICIOUS` in amber), plus `DISTANCE`,
   `CALIBRATED THRESHOLD`, `USER_REF`, and the honest rule "distance ≤
   threshold verifies; distance > threshold is suspicious" with a note that the
   threshold is server-calibrated and never client-supplied.
4. `404 profile_not_found` → "No behavioral profile exists yet. Enroll first."
   with a working "Continue to Enrollment →" action.
5. `401` → same session-expired handling as enrollment.

## 8. Routing & guards (`frontend/src/router.jsx`, `frontend/src/App.jsx`)

* Routes added: `/enrollment` → `enrollment`, `/verification` → `verification`.
* `App.jsx` subscribes to the auth store (`useAuth`). Protected routes render
  only when a session exists; otherwise a redirect effect pushes `/login`.
* A logged-in user cannot view `/login` (redirected to `/enrollment`).
* Any unknown path still collapses back to `/` as before.

## 9. Tests

`npm test` now runs `../tests/collector.test.js` **and**
`../tests/frontend.test.js` (`node --test`, no new npm dependencies):

* **19** collector tests — unchanged, still passing.
* **18** new frontend tests:
  * auth store: login/logout, empty-token rejection, **restoration from
    sessionStorage (page refresh)**, never-persist-password, subscriber
    notification;
  * API client: register payload/content-type, structured error envelope,
    login token parsing, missing token, network failure (status 0), abort →
    `timeout`, bearer auth on enrollment and verification;
  * payloads/privacy: payload builders deep-copy, collected sessions carry
    **no key identity fields**, `hasUsableContent` gate.

Found & fixed during this phase: the auth store accidentally omitted the
`createSessionStore` export (test caught it before shipping).

## 10. Verification (all performed in this environment)

* `npm test` → **37/37 pass** (19 collector + 18 Phase 12).
* `npm run build` → **PASS**: `index-DANil5Pi.js` 226.37 kB, `index-D87O2CEh.css`
  63.64 kB.
* `vite preview` (port 4318): `/`, `/register`, `/login`, `/collector`,
  `/enrollment`, `/verification` all **200**.
* Production-bundle greps confirm the new pages' key strings are shipped.
* **Live API smoke test** (real uvicorn on port 8232, `BBA_JWT_SECRET_KEY` set,
  in-memory store, real Phase 7 checkpoint + Phase 9B artifacts); sessions were
  produced with the **actual Phase 2 collector module** (synthetic-but-valid
  event streams — a headless browser was not available to type for real):

  | Check | Result |
  |---|---|
  | health | 200 ok |
  | register unknown user | 201, `{id, username, created_at}` |
  | duplicate register | 409 `username_exists` |
  | wrong password | 401 `invalid_credentials` |
  | unknown username | 401 `invalid_credentials` |
  | login | 200 bearer JWT (188 chars), keys `{access_token, token_type}` |
  | enroll unauthenticated | 401 `authentication_required` |
  | enroll 2 sessions | 201, dim **128**, `session_count` **2**, `status` enrolled, keys `{user_ref, status, embedding_dimension, session_count}` (no centroids/embeddings) |
  | duplicate enroll | 409 `profile_exists` |
  | verify with cross-user `user_ref` | 403 `forbidden` |
  | self verify (enrolled session) | 200 `VERIFIED`, distance 0.0243 ≤ threshold 0.46351 |
  | second probe | 200, distance 0.1164 ≤ threshold → `VERIFIED` |
  | verify keys | `{user_ref, decision, distance, threshold}` |

* Source audit (`Math.random`, `console.*`, `localStorage`, fake/mock markers,
  `TODO`/`FIXME`): **clean** — the only storage used is `sessionStorage` for the
  auth session (documented above), and the only password references are the
  form payloads the API requires.

## 11. Honest guard-rails

* **No headless browser** was available, so the full click-through browser E2E
  (real typing into the capture area) was **not** automated here. It is
  verified by: 37 automated tests, the production build, all six preview routes,
  and the live API smoke above. A manual browser pass is recommended before any
  claim of "works in a real browser with real typing".
* Behavioral similarity on this **development (synthetic checkpoint)** model is
  generous: a distinct synthetic probe (distance 0.1164) also fell within the
  calibrated threshold. This is the model's real behaviour on these artifacts
  and is **not** a claim of production biometric performance.
* Profiles/accounts remain in memory unless `BBA_DATABASE_URL` is configured;
  the frontend does not implement refresh tokens or token revocation.
* The data-sufficiency gate (≥2 keyboard + ≥2 mouse events) is a frontend UX
  quality gate; the API's own validation is the authority.
* Phase 13 items (refresh tokens, rate limiting, continuous re-auth, OAuth,
  deployment, retraining, recalibration) are **not** implemented.