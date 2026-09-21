# Project Status — Behavioral Biometric Authentication

## Current Phase

**Phase 15A — Containerization Foundation (Implemented; full-stack Compose deployment exercised)**

Status: automated implementation checks all pass; manual browser E2E pending
(no browser available in this environment — see Phase 14B section).

## Status

Complete

## Phase 2 — Completed

- Browser collection interface (React + Vite)
- Keyboard event collector
- Mouse event collector
- Session data structure
- JSON export
- Basic validation
- Tests/checks

## Phase 3 — Completed

- Raw session validation
- Keyboard preprocessing and feature extraction (hold / flight time)
- Mouse preprocessing and feature extraction (dx, dy, dt, distance, speed)
- Numerical normalization (`FeatureScaler`: fit / transform / fit_transform)
- Processed-session output format
- Synthetic test fixtures and unit tests (cases A–R)
- Manual verification: PASSED (incl. overlapping-keyboard flight-time fix)
- See [`PHASE_3_PREPROCESSING.md`](PHASE_3_PREPROCESSING.md) for full details

## Phase 4 — Completed

- Dataset schema + validation + privacy checks (`dataset/schema.py`)
- Deterministic synthetic generator (5 users x 10 sessions, seeded)
- Real-session loading (raw Phase 2 -> Phase 3 -> Phase 4) (`dataset/loader.py`)
- Per-user deterministic train/test split, no leakage (`dataset/split.py`)
- Statistical enrollment profiles (`baseline/profile.py`)
- Keyboard / mouse / combined distance scoring (`baseline/scoring.py`)
- Threshold calibration (training data only) + TAR/FAR/FRR/ROC evaluation
  (`baseline/evaluator.py`)
- Dataset + baseline CLI scripts and unit tests (81/81 Phase 4 tests;
  156/156 full suite)
- Byte-identical determinism verified (synthetic dataset + baseline)
- Persisted dataset artifact `data/datasets/dataset_synthetic.json`
  (5 users, 50 sessions, 1,228 keyboard / 2,639 mouse samples) — git-ignored
  by policy
- See [`PHASE_4_DATASET_BASELINE.md`](PHASE_4_DATASET_BASELINE.md) for full details

## Phase 5 — Completed

- PyTorch CPU 2.14.0 + numpy installed for Python 3.14 (`torch==2.14.0+cpu`)
- Sequence preparation (`ml/encoder/input.py`): validation, tensor conversion,
  variable-length collation (pad + lengths), per-modality `ModalityBatch`,
  pluggable `FeatureScaler` reuse
- CNN + GRU encoder (`ml/encoder/cnn_gru.py`): `ConvGruEncoder` per modality,
  `BehavioralEncoder` fusion (default 64+64 -> 128-dim), packed variable-length
  support, documented zero-embedding for absent modalities, rejection of
  behaviour-less sessions
- Tests: 55/55 new (input + encoder) — shapes, padding non-influence, batch
  forward, determinism (byte-identical), gradient flow, identity independence,
  empty-modality semantics, end-to-end dataset & real-session encoding
- Real-session smoke test: PASSED (55 keyboard / 571 mouse samples -> 128-dim,
  deterministic)
- See [`PHASE_5_CNN_GRU_ENCODER.md`](PHASE_5_CNN_GRU_ENCODER.md) for full details

## Phase 6 — Completed

- Deterministic genuine/impostor pair construction (`ml/verification/pairs.py`),
  per-partition generation prevents train/test leakage; labels 1 = genuine,
  0 = impostor
- Shared-weight Siamese verifier (`ml/verification/siamese.py`) around the
  Phase 5 encoder; L2 distance with `eps=1e-8`; identity metadata never a model
  feature
- Contrastive loss (`ml/verification/loss.py`):
  `L = mean(y·D² + (1−y)·max(0, margin−D)²)`
- Distance-score decision helpers (`ml/verification/scoring.py`) — documented
  placeholder, not production-ready thresholds
- Tests: 119/119 new (pairs, siamese, contrastive loss, scoring, integration)
- Full suite: 334/334 passed; frontend 19/19 passed
- Real-session smoke test + Phase 6 gradient smoke test: PASSED
- See [`PHASE_6_SIAMESE_VERIFICATION.md`](PHASE_6_SIAMESE_VERIFICATION.md)

## Phase 7 — Completed

- Training data pipeline (`ml/training/dataset.py`): loads the persisted
  dataset, user-aware 25/10/15 train/val/test split, per-partition pair
  generation, `FeatureScaler` fitted on training rows only, `SiamesePairDataset`
  + `siamese_collate` + DataLoaders (`num_workers=0`)
- Trainer (`ml/training/trainer.py`): per-epoch train + validation loop,
  Adam optimizer, optional gradient clipping, finite-loss recording,
  seed-stable per-epoch shuffling, `TrainingHistory`
- Config (`ml/training/config.py`): immutable `TrainingConfig` with validated
  defaults (seed 42, epochs 10, batch 8, lr 1e-3, margin 1.0)
- Checkpointing (`ml/training/checkpoint.py`): atomic save, `weights_only`
  load, fresh-model reconstruction, optimizer-state restore, reload parity
  verification (`models/siamese_behavioral_encoder.pt`, git-ignored)
- CLI `scripts/train_siamese.py`: loads the persisted dataset (never
  regenerates it), refuses to overwrite an existing checkpoint without
  `--overwrite`
- Tests: 49/49 new (training dataset, trainer, checkpoint, integration)
- Full suite: 383/383 passed; frontend 19/19 passed; compileall + imports PASS
- Training smoke test (10 epochs, batch 8, lr 1e-3, margin 1.0, seed 42):
  train 0.124812 -> 0.011864, val ~0.024, all finite; synthetic test loss
  0.168076; checkpoint reload PASS (0.0 deviation); same-seed determinism PASS
- **Interpretation guard-rail:** results are on SYNTHETIC generated identities
  and are NOT real-world biometric performance.
- See [`PHASE_7_SIAMESE_TRAINING.md`](PHASE_7_SIAMESE_TRAINING.md) for full details

## Phase 8 — Completed

- Inference-only checkpoint wrapper (`ml/evaluation/model.py`): loads the Phase
  7 checkpoint, `eval()` + `torch.no_grad()` inference, embed / pair-distance
  API, parameter-invariance guarantees, loud failures on missing/corrupt files
- Enrollment + verification (`ml/evaluation/enrollment.py`): centroid
  profiles, `verify(profile, probe)` -> `VERIFIED`/`SUSPICIOUS` via Phase 6 L2
- Threshold calibration (`ml/evaluation/calibration.py`): largest threshold
  meeting the target-FAR budget, closest-feasible fallback, development
  partition only
- Metrics (`ml/evaluation/metrics.py`): FAR/TAR/FRR, confusion-matrix counts,
  ROC points + AUC (trapezoid = U-statistic), EER
- Evaluator (`ml/evaluation/evaluator.py`): deterministic 25/10/15 split +
  train-only scalers -> dev/test pairs -> cached embeddings -> calibration ->
  JSON-serialisable test report (ROC/AUC/EER, leakage guard)
- CLI `scripts/evaluate_verification.py`: full report + enroll/verify smoke,
  optional `--report` JSON output, exit codes 0/1/2
- Phase 8 results (seed 42, target FAR 0.05): dev pairs 5+40, test pairs
  15+90; threshold 0.463513 (dev FAR exactly 0.05); test TAR 0.6000, FAR
  0.0000, FRR 0.4000, accuracy 0.9429, AUC 0.9474, EER 0.1833
- Tests: 70/70 new (metrics, model, enrollment, verification-evaluation)
- Full suite: 453/453 passed; frontend 19/19 passed; compileall + imports PASS
- Determinism: byte-identical reports across repeated runs
- **Interpretation guard-rail:** results are on SYNTHETIC generated identities
  and are NOT real-world biometric performance.
- See [`PHASE_8_ENROLLMENT_VERIFICATION_EVALUATION.md`](PHASE_8_ENROLLMENT_VERIFICATION_EVALUATION.md) for full details

## Phase 9A — Completed

- FastAPI 0.141.1 + uvicorn 0.53.0 + httpx 0.28.1 installed in the existing
  .venv (Python 3.14); `requirements.txt` gained a Phase 9A section
- Backend package (`backend/app/`): application factory `create_app()`,
  typed `Settings` config (frozen dataclass, `BBA_*` env overrides),
  centralized error handling, CORS (explicit origins, `*` rejected),
  DI foundation (`get_settings`), minimal lifespan, health/metadata routes
- Routes: `GET /` + `GET /api/v1/` (service info), `GET /health` +
  `GET /api/v1/health` (canonical `{"status": "ok"}`, no model/db I/O)
- Error envelope `{"error": {"code", "message"}}` for 404/405/422/403/500;
  validation details included; stack traces/paths never leaked
- OpenAPI: `/docs`, `/redoc`, `/openapi.json` verified via TestClient and a
  live uvicorn smoke test; `/api/v1/enrollment` + `/api/v1/verification`
  confirmed absent (structured 404)
- Tests: 47/47 new (foundation, errors, config)
- Full suite: 500/500 passed; frontend 19/19 passed (unchanged); compileall
  + import PASS
- ML code (ml/encoder, ml/verification, ml/training, ml/evaluation), dataset
  and baseline untouched; no database, no JWT, no auth endpoints
- See [`PHASE_9A_FASTAPI_FOUNDATION.md`](PHASE_9A_FASTAPI_FOUNDATION.md) for full details

## Phase 9B — Completed

- ML inference service (`backend/app/services/ml_service.py`): lazy
  `ml.*`/`torch` imports, read-only checkpoint wrapper, injectable seams
  (verifier factory, preprocessing loader, verification config loader,
  session processor), structured ML error codes
- Profile store (`backend/app/services/profile_store.py`): thread-safe
  in-memory store holding ONLY aggregates (centroid tuple, embedding_dim,
  session_count, timestamps); never raw events/keys/secrets; non-persistent
- Pydantic request/response schemas (`backend/app/schemas.py`): Phase 2 raw
  session mirror, finite-float rejection (`NaN`/`±inf` → 422), payload limits
  (≤16 enrollment sessions, ≤20k events/session), non-blank `user_ref`
- Routes (`backend/app/routes/verification.py`): `POST /api/v1/enrollment`
  (201) + `POST /api/v1/verification`; decision rule = Phase 6 L2 distance vs
  server-side calibrated threshold (never client-supplied)
- Inference artifacts (`scripts/export_inference_artifacts.py` →
  git-ignored `models/behavioral_preprocessing.json` + `models/verification_config.json`):
  train-only FeatureScaler state dicts (seed 42) + calibrated threshold
  0.463513 (Phase 8 calibration, target FAR 0.05)
- Health isolation: `/health` never imports/loads ML; ML endpoints fail with
  structured 503 envelopes when artifacts are missing, health stays 200
- Privacy: `user_ref` is metadata only (never a model feature); raw events,
  centroids, embeddings, secrets and paths never returned
- Tests: 72/72 new (service, store, enrollment API, verification API,
  privacy, health isolation, real-pipeline integration A–G)
- Full suite: 572/572 passed; frontend 19/19 passed (untouched); compileall
  + import PASS
- Live uvicorn smoke test PASSED (enroll 201 dim 128, duplicate 409,
  genuine VERIFIED 0.0001 ≤ 0.4635, impostor SUSPICIOUS 0.886 > 0.4635,
  unknown 404, empty sessions 422, POST 405)
- Stale 9A tests updated (endpoints now exist: GET → 405, empty-body POST → 422)
- **Honest guard-rails:** profiles are in-memory (lost on restart); artifacts
  are synthetic-development artifacts that must be regenerated for deployment;
  unauthenticated dev/demo API (no JWT/login/register)
- See [`PHASE_9B_ML_SERVICE_API.md`](PHASE_9B_ML_SERVICE_API.md) for full details

## Phase 10 — Completed

- PostgreSQL profile storage (`backend/app/repositories/`): abstract
  `ProfileRepository` contract + psycopg3/`psycopg_pool` implementation
  (`PostgresProfileRepository`), structured storage errors
  (503 `database_unavailable` / 500 `database_error`), lazy pool (never
  created at import/create_app/dependency-resolution time), race-safe
  insert-only `save()` guarded by the `user_ref` PRIMARY KEY
- **Bug fixed:** `save()` previously called the INSERT helper without the
  computed profile values, so a PostgreSQL save could never work; it now closes
  over `profile_to_values(profile)` and executes the INSERT with the real
  values (regression-tested)
- Migration `db/migrations/001_create_behavioral_profiles.sql`: idempotent
  DDL creating only the six aggregate columns (`user_ref`, `centroid`
  `DOUBLE PRECISION[]`, `embedding_dim`, `session_count`, `created_at`,
  `updated_at`); no raw behavioural data, no secrets
- Wiring: `get_profile_store` (unchanged name, so all 9B dependency overrides
  keep working) returns the persistent repository when `database_url`
  (`BBA_DATABASE_URL`) is configured and the in-memory store otherwise;
  routes now type-hint the `ProfileRepository` contract
- Error envelope unchanged: duplicate insert → 409 `profile_exists`, unknown
  user → 404, missing/unreachable DB → 503; no SQL/URL/hostname/credential
  leakage in messages
- Tests: 37/37 new (value/row mapping, error translation, repository behaviour
  over fakes, dependency wiring, API over a PostgreSQL-backed store) +
  16 new real-PostgreSQL integration tests in
  `tests/test_postgres_integration.py` gated by `TEST_DATABASE_URL`
- Full suite: 609 passed, 16 skipped (integration tests remain skipped here);
  frontend 19/19 passed (untouched); compileall + `create_app` import PASS
- **Honest guard-rail:** *Real PostgreSQL persistence smoke test was NOT
  executed because no PostgreSQL instance / `TEST_DATABASE_URL` was
  available.* Real-DB behaviour (schema, insert/retrieve, duplicate,
  restart-persistence across repository instances, transactions, connection
  cleanup) is implemented in the gated integration tests and verified against
  fakes only; it must be run against a live PostgreSQL before production claims
- **Resolved in Phase 13:** the gated real-PostgreSQL integration tests were
  executed against live PostgreSQL 16.15 and passed (see the Phase 13 section)
- Phase 9B ML/service/routes behaviour unchanged; no JWT/auth, no frontend
  integration, no WebSockets, no Redis, no retraining/recalibration
- See [`PHASE_10_POSTGRES_PROFILE_STORAGE.md`](PHASE_10_POSTGRES_PROFILE_STORAGE.md) for full details

## Phase 11 — Completed

- Authentication (`backend/app/auth/`): Argon2id password hashing
  (argon2-cffi defaults; `verify(hash, password)` order; fails safe for
  wrong/malformed/non-string input) and PyJWT HS256 access tokens with a
  deliberately minimal payload (`sub`/`iat`/`exp`); no password/hash/emails/
  token/secrets ever in responses or logs
- No usable hard-coded secret: `BBA_JWT_SECRET_KEY` defaults to empty; weak or
  short (< 32 chars) secrets are rejected at `Settings` construction and a
  production environment without a secret fails closed; unconfigured auth
  endpoints fail with 503 `auth_unavailable`
- Routes: `POST /api/v1/auth/register` (201, safe user info; duplicates → 409
  `username_exists`) and `POST /api/v1/auth/login` (200 bearer token; identical
  generic 401 for unknown username and wrong password); register/login stay
  public while enrollment/verification are now protected
- Authorization: `get_current_user` derives identity exclusively from the JWT
  `sub`; an authenticated user can only enroll/verify their OWN profile — a
  mismatched request-body `user_ref` is 403 `forbidden`, so body `user_ref`
  can never select another user's profile (ML behaviour unchanged)
- Users storage: abstract `UserRepository` contract + `UserRecord`; in-memory
  fallback and psycopg3 `PostgresUserRepository` (lazy pool, race-safe
  unique-constraint duplicate guard, no SQL/URL/credential leakage);
  wiring via `get_user_repository` (PostgreSQL when `BBA_DATABASE_URL` is set,
  in-memory otherwise, resolution never connects)
- Migration `db/migrations/002_create_users.sql`: idempotent `users` table
  (UUID PK, `username` NOT NULL UNIQUE + non-blank CHECK, `password_hash`
  NOT NULL, timestamps NOT NULL); never touches `behavioral_profiles` and
  contains no behavioural events, embeddings, JWT tokens or secrets
- ML privacy: authenticated identity / `user_ref` / JWT claims never enter
  model features; recorded-verifier tests prove processed sessions contain no
  identity/credential keys
- Tests: 113/113 dedicated Phase 11 (security, JWT, API, authorization,
  privacy, user repository) + 15 real-PostgreSQL integration tests gated by
  `TEST_DATABASE_URL`; affected Phase 9B/10 API tests updated to run behind
  auth/authenticated identity
- Full suite: 722 passed, 31 skipped; frontend 19/19 passed (untouched);
  `compileall backend ml scripts` PASS; `create_app` PASS; live uvicorn smoke
  PASS (health 200, register 201, login 200, authenticated enrollment 201
  dim 128, self-verification VERIFIED 0.0001 ≤ 0.4635, unprotected 401,
  cross-user `user_ref` 403)
- **Honest guard-rail:** *live PostgreSQL integration was NOT executed in this
  environment because `TEST_DATABASE_URL` was unavailable*; the real-DB tests
  were skipped. `PostgresUserRepository` is verified against fakes plus the
  gated integration module and must be run against a live PostgreSQL before
  any production claim. Profiles and accounts remain in memory unless
  `BBA_DATABASE_URL` is configured
  (Resolved in Phase 13: executed against a live PostgreSQL 16.15 and passed —
  see the Phase 13 section)
- At Phase 11 there was still no frontend auth UI, no refresh tokens, no rate
  limiting, and no Redis/WebSockets; continuous behavioral re-verification
  arrives later (Phase 14A). See the Phase 12 section for the completed
  frontend, and the Phase 14A section for periodic continuous verification
- See [`PHASE_11_AUTHENTICATION_AUTHORIZATION.md`](PHASE_11_AUTHENTICATION_AUTHORIZATION.md) for full details

## Phase 12 — Completed

- Full frontend integration (`frontend/src/`): reactive auth store
  (`auth.js`) with `useSyncExternalStore`, focused API client (`api.js`)
  with 30 s timeout/abort, and protected routes in `App.jsx`/`router.jsx`
  (`/enrollment`, `/verification`; unauthenticated → `/login`)
- **Token-storage decision (documented):** the Phase 11 JWT is kept in
  `sessionStorage` (`bba.session.v1`) so auth survives a page refresh while
  staying tab-scoped; a password is never stored anywhere; the token is only
  sent as the `Authorization: Bearer` header
- **Enrollment page** (`frontend/src/enrollment/`): real Phase 2 collector
  sessions → frontend data-sufficiency gate (≥2 keyboard + ≥2 mouse events;
  UX quality gate only) → single authenticated `POST /enrollment` → real
  `user_ref`/`status`/`embedding_dimension`/`session_count` result card; 409
  `profile_exists` handled with a working verification action; 401 clears the
  session and redirects to login
- **Verification page** (`frontend/src/verification/`): one real probe session
  → authenticated `POST /verification` → backend `VERIFIED`/`SUSPICIOUS` with
  `distance` + calibrated `threshold`; `404 profile_not_found` → onboarding
  link to enrollment
- Shared `AuthenticatedHeader` (username chip + working Log Out + nav);
  login success now proceeds to `/enrollment`; landing/register/login
  unchanged in design
- Tests: `npm test` now runs collector (19) + new frontend tests (18)
  → **37/37 passed**; exercises auth store (incl. storage restoration + no
  password persistence), API client (envelope parsing, bearer auth, timeout,
  network failure), payload privacy (no key-identity fields), and the data
  gate. No new npm dependencies
- `npm run build` PASS (226.37 kB JS / 63.64 kB CSS); `vite preview`
  `/`, `/register`, `/login`, `/collector`, `/enrollment`, `/verification`
  all 200; production-bundle string greps PASS
- Live uvicorn smoke PASS (port 8232, `BBA_JWT_SECRET_KEY` set, in-memory
  store, real checkpoint/artifacts): register 201, duplicate 409, login 200
  bearer, wrong/unknown → identical 401, unauthenticated enroll 401, enroll 2
  sessions → 201 dim 128, duplicate 409, cross-user `user_ref` 403,
  self-verify VERIFIED (distance 0.0243 ≤ 0.46351), second probe VERIFIED
  (distance 0.1164 ≤ threshold), verify keys `{user_ref, decision, distance,
  threshold}` with no raw/embedding leakage
- Source audit clean (`Math.random`, `console.*`, `localStorage`, fake/mock
  markers, TODOs): only storage used is `sessionStorage` for the auth session
- **Honest guard-rails:** no headless browser available → the full real-typing
  browser E2E was not automated (covered by tests + build + preview routes +
  live API smoke; manual browser pass recommended); synthetic-probe distances
  both verified on this dev checkpoint — not a real-biometric claim;
  profiles/accounts stay in-memory unless `BBA_DATABASE_URL` is set
- Backend/ML/schema/checkpoint/threshold untouched (Phase 12 is frontend-only)
- See [`PHASE_12_FRONTEND_INTEGRATION.md`](PHASE_12_FRONTEND_INTEGRATION.md) for full details

## Phase 13 — Completed (PASSED)

- **Real PostgreSQL environment:** user-local PostgreSQL **16.15** cluster
  (`127.0.0.1:5432`, superuser `himanshu`, trust auth) installed without
  sudo/Docker; app DB `behavioral` (both migrations applied, schema verified
  column-for-column against repository expectations) and dedicated test DB
  `bba_test` (per-module drop+recreate fixtures). Cluster lives under
  `/tmp/opencode` and is intentionally not committed
- **App-level persistence test (NEW `tests/test_postgres_api_integration.py`,
  5 tests):** two independent `create_app` instances over the same real
  PostgreSQL via a fake-ML-verifier seam; register 201 → login 200 → enroll
  201 on instance A → close repos → login 200 → verify 200 (VERIFIED) on
  instance B; duplicate username/enrollment → structured 409 + exactly one row
  in PostgreSQL; `/health` stays 200 without DB I/O; unreachable DB → 503
  `database_unavailable` with zero connection details leaked; repository
  isolation proven (`PostgresUserRepository`/`PostgresProfileRepository`,
  distinct per-app instances)
- Real-PostgreSQL integration tests enabled via `TEST_DATABASE_URL`:
  profile-repository suite (17), user-repository suite (14), app-level suite
  (5). One latent fixed: in `tests/test_postgres_integration.py` the
  module-scoped `test_contains_count_list_profiles` assumed a clean table but
  ran after earlier tests inserted rows — now order-independent (count-delta +
  set membership)
- **Full suite: 758 passed, 0 failed, 0 skipped** with `TEST_DATABASE_URL`
  set; the 36 real-DB tests skip cleanly when the variable is unset
  (`36 skipped`). Frontend `npm test` **41/41 passed** (unchanged); `npm run
  build` PASS (same hashes as Phase 12); `compileall backend ml scripts` PASS
- **Live restart smoke (real uvicorn A → B, port 8010, real artifacts + JWT):**
  A health 200/register 201/login 200/enroll 201 (dim 128, 1 session) →
  SIGTERM → rows verified present in PostgreSQL (`users` + `behavioral_profiles`
  for `smoke_1789830951`) → B health 200/login 200/verify 200 **VERIFIED**
- Connection safety: **0** leaked PostgreSQL connections after all smokes,
  test modules, and process exits (`pg_stat_activity` clean); pool `close()`
  exercised
- Privacy audit of real DB: only aggregate columns; no raw keyboard/mouse
  events, no embeddings-as-features, Argon2id hash only, no tokens/secrets/URLs
  stored or logged
- **Honest guard-rails:** no headless browser → real-typing browser E2E still
  not automated (backend live smoke + 41 frontend tests cover it; manual
  browser pass recommended); smoke used synthetic session events — functional
  correctness proof, not a real-biometric performance claim; a pre-existing
  Phase 12-diagnostic uvicorn on port 8000 (PID 249198) was observed, noted,
  and left untouched
- Backend/ML/frontend source untouched in Phase 13 (only tests + docs changed)
- **PHASE 13 — PASSED**
- See [`PHASE_13_POSTGRESQL_PERSISTENCE.md`](PHASE_13_POSTGRESQL_PERSISTENCE.md) for full details

## Phase 14A — Implemented (automated checks PASSED; manual browser E2E pending)

- **New endpoint `POST /api/v1/continuous-verification`** (Phase 11 protected,
  same `build_versioned_router`): body is a **bare bounded behavioural window**
  (`ContinuousVerificationRequest(RawSession)` with `extra="forbid"` — any
  `user_ref`/identity field → 422), the authenticated JWT is the **sole** profile
  selector, unknown profile → 404 `profile_not_found`, invalid/empty window →
  422 `invalid_session` (ML-service path), response is **minimal**
  `{decision, distance, threshold}` — no user_ref/centroid/embedding/raw events;
  the window is verified and **never stored**; threshold is always the
  server-side calibrated 0.463513 (client-supplied threshold → 422)
- **Frontend Continuous Verification page** (`/continuous`, protected): bounded
  window capture reusing the unchanged Phase 2 collector; limits derived from
  collector sampling (kbd 300 / mouse 1200 / total 1500 / 30 s) so a window can
  never approach the API 20k-event ceiling; auto-stop at any cap; state machine
  `authenticated → collecting → verifying → verified | reverification_required`
  (pure `lib/continuous.js` reducer + `useContinuousVerification` hook);
  `api.js` `continuousVerify` posts the bare window with the bearer token;
  `AuthenticatedHeader` + nav updated
- **Privacy:** a window lives only in React state, is sent once, and is
  discarded on submit/reset; no localStorage/sessionStorage window writes; no
  key identity / typed-text capture; JWT/identity never a model feature
- **Backend tests:** `tests/test_continuous_verification_api.py` **21/21**
  (auth, validation, decisions, identity isolation, response privacy, profile
  integrity, threshold integrity, existing endpoints unchanged)
- **Frontend tests:** `tests/continuous.test.js` **20/20** (bounds/caps,
  duration, `canSubmitWindow` gates, bare identity-free payload, full reducer
  state machine, API client) — `npm test` total **61/61**
- **Full backend suite: 779 passed** with `TEST_DATABASE_URL` (758 baseline +
  21 new; 0 failed, 0 skipped); 743 passed/36 skipped when unset; compileall
  PASS; `npm run build` PASS; `git diff --check` clean
- **Honest guard-rails:** no headless browser → the `/continuous` real-typing
  E2E is not automated/manual here (like Phase 12/13, a manual browser pass is
  recommended before any "PASSED" claim). The **known Phase 12 limitation**
  (real-browser probes verify SUSPICIOUS, distances ≈1.30/≈1.52 vs 0.463513)
  is deliberately **not** "fixed" — the model/threshold are untouched; such
  windows will request re-verification by design. Development system —
  synthetic-dev artifacts, no production claims
- **Phase 14B and later not started:** refresh tokens, auto-logout/locking on
  repeated suspicion, rate limiting, Redis/WebSockets, multi-device sessions,
  deployment, retraining
- See [`PHASE_14A_CONTINUOUS_VERIFICATION.md`](PHASE_14A_CONTINUOUS_VERIFICATION.md) for full details

## Phase 14B — Implemented (automated checks PASSED; manual browser E2E pending)

- **Server-authoritative session state** — new table
  `behavioral_session_states` (`db/migrations/003_create_session_states.sql`)
  keyed by `(user_ref, session_id)` holding ONLY aggregates: `state` (CHECK
  `'verified' | 'reverification_required'`), `consecutive_suspicious`,
  `last_verified_at`, timestamps. Never raw events/embeddings/tokens
- **Per-session identity derived server-side from the validated bearer token
  itself** (`_session_id_from_token` in `backend/app/auth/dependencies.py` = a
  SHA-256 digest of the raw token; not `sub@iat`); a per-issuance `nonce` in
  the JOSE *header* (never a payload claim) makes every fresh token a distinct
  string, so two logins sharing `sub`/`iat`/`exp` still get two sessions;
  payload stays exactly `{sub, iat, exp}`, a fresh login starts clean
- **Fail-closed gating:** `POST /api/v1/enrollment` and
  `POST /api/v1/verification` → **403 `reverification_required`** while a
  session is unverified; DB unreachable → structured 503 (fail closed);
  `POST /api/v1/continuous-verification` stays open as the recovery path, and a
  verified window (`mark_verified`) clears the gate / resets the counter
  (`mark_suspicious` increments + flips state)
- **New endpoint `GET /api/v1/continuous-verification/state`** →
  `{session_state, consecutive_suspicious, last_verified_at}` for the current
  session; `ContinuousVerificationResponse` additively extended with the same
  three fields (threshold/decision semantics unchanged, still 0.463513)
- **Repository layer** (`backend/app/repositories/`): abstract
  `SessionStateRepository` + `memory_session_state_store` (RLock) +
  `postgres_session_state_repository` (atomic `INSERT … ON CONFLICT …
  RETURNING`, lazy pool); wired via `get_session_state_store` (PG when
  `BBA_DATABASE_URL` is set, in-memory otherwise)
- **Frontend feeds the server, never the browser:** new `RESTORE` reducer action
  + `getContinuousSessionState` in `api.js` hydrate the machine from the server
  once per token on mount (401 → logout only); `DECISION` is
  **server-authoritative** (`session_state` wins; `decision` mapping is only the
  defensive fallback); verdicts are transient React state — a refresh re-fetches
  from the server (no localStorage/sessionStorage/IndexedDB)
- **Backend tests:** `test_session_state.py` (token-derived session identity),
  `test_session_state_gating.py`
  (real-token replay/fresh-login isolation; no `session_id` in API responses or
  ML features), `test_auth_jwt.py` (per-issuance token uniqueness + payload
  integrity), real-PG `test_postgres_session_state_integration.py` +
  `test_postgres_gating_integration.py` (incl. block surviving a brand-new
  repository pool via the SAME token), and updates to
  `test_postgres_api_integration.py` + `test_continuous_verification_api.py`
- **Frontend tests:** `tests/continuous.test.js` **28/28** (69 total) — RESTORE,
  server-authoritative DECISION, decision fallback, state helpers, state-endpoint
  client
- **Full backend suite: 843 passed, 0 failed, 0 skipped** with `TEST_DATABASE_URL`
  (real PostgreSQL 16.15); 783 passed / 60 skipped when unset. `npm run build`
  PASS; `compileall` PASS; `git diff --check` clean
- **DB-restart durability (live):** wrote a `reverification_required` row,
  restarted the PostgreSQL server (`pg_ctl -m fast`), and read the row back
  intact from a brand-new pool — committed session state survives a server
  restart
- **Honest guard-rails:** no headless browser → manual real-typing E2E pending
  (matching Phases 12/13/14A); a reissued token cannot recover an older
  session's state (intended per-login isolation — the same user can log in
  again freely and starts clean); development system — synthetic-dev
  artifacts, no production claims
- **Phase 15 not started (as of Phase 14B):** rate limiting, auto-logout/locking
  on repeated suspicion, refresh/revocation, Redis/WebSockets, multi-device
  sessions, deployment, retraining, or any JWT/model change
- See [`PHASE_14B_SESSION_STATE.md`](PHASE_14B_SESSION_STATE.md) for full details

## Phase 15A — Implemented (full-stack Compose deployment PASSED)

- Containerized the Phase 1–14B application **without behavior changes**:
  PostgreSQL `postgres:16-alpine`, FastAPI backend (CPU PyTorch), React/Vite
  frontend served by nginx with SPA deep-link fallback
- **Dockerfile (backend):** `python:3.14-slim`, CPU-only PyTorch pinned
  (`torch==2.14.0+cpu` from the PyTorch CPU index, libgomp1), requirements
  installed reproducibly, non-root user, `uvicorn backend.app.main:app` on
  0.0.0.0:8000; `.env` and model binaries never baked in
- **Dockerfile (frontend):** multi-stage `node:22-alpine` build -> `nginx`
  serve; `VITE_API_BASE_URL` injected as a build arg from
  `compose.yaml`/`.env`; deep links `/login`, `/enrollment`, `/verification`,
  `/continuous`, `/register`, `/collector` all return the SPA (HTTP 200)
- **`compose.yaml`:** services `db -> migrate -> backend -> frontend`, named
  volume `postgres_data`, ports 8000 (backend) / 3000 (frontend), host-mapped
  via `.env`; PostgreSQL never published to the host; healthchecks wired into
  `depends_on` (`service_healthy`, `service_completed_successfully`); no host
  networking
- **Deterministic migrations:** one-shot `migrate` service runs
  `db/migrations/*.sql` in lexical order (001 → 002 → 003) via
  `psql -v ON_ERROR_STOP=1` after `pg_isready`; backend only starts after
  migrate completes successfully; migration SQL untouched; no tracking table
- **Model artifacts:** `models/*.pt` + JSON artifacts remain gitignored, never
  committed, mounted read-only at `/app/models` with explicit `BBA_*` paths;
  missing artifacts fail closed (503) — no generation/retraining/recalibration
- **Secrets/env:** `BBA_ENVIRONMENT` pinned to `production`; `BBA_JWT_SECRET_KEY`
  and `POSTGRES_PASSWORD` required by Compose (fail closed) and by the backend
  config validation; `.env` gitignored; `.env.example` placeholders only
- **Full-stack smoke (real containers):** register → login → enroll → verify →
  continuous-verify → session-state all PASSED with the existing threshold
  (0.4635127782821655); Phase 14B gating exercised (SUSPICIOUS → 403
  `reverification_required` → VERIFIED window clears); rows persisted in
  PostgreSQL (`users`, `behavioral_profiles`, `behavioral_session_states`)
- **Persistence/restart:** backend restart while DB persisted → data intact,
  app works; `docker compose down` + `up` (no `-v`) → data survives; `down -v`
  → volume reset and fresh migration verified
- **Host test batteries unchanged:** backend pytest 784 passed / 60 PG-gated
  skipped; frontend 75/75; `npm run build` PASS; `compileall` PASS;
  `git diff --check` clean; no secrets or model binaries in Git
- See [`PHASE_15A_CONTAINERIZATION.md`](PHASE_15A_CONTAINERIZATION.md) for
  architecture, services, ports, migration and reset instructions

## Roadmap Status

Completed phases:

- Phase 1–8: Completed
- Phase 9A: Completed
- Phase 9B: Completed
- Phase 10: Completed
- Phase 11: Completed
- Phase 12: Completed
- Phase 13: Completed
- Phase 14A: Implemented (automated checks PASSED; manual browser E2E pending)
- Phase 14B: Implemented (automated checks PASSED; manual browser E2E pending)
- Phase 15A: Implemented (full-stack Compose deployment PASSED)

Phase 15B and later have **not** started: TLS/HTTPS + public reverse proxy,
Kubernetes / cloud / CI-CD, load balancing, autoscaling, refresh tokens,
auto-logout/locking on repeated suspicion, rate limiting, Redis/WebSockets,
multi-device sessions, deployment hardening, retraining, or new ML. The full
real-typing browser E2E for the
Phase 12 flow and the Phase 14A `/continuous` / Phase 14B `/continuous` +
gated-enrollment flows remains unverified without a manual browser pass (no
automated browser was available). Real-PostgreSQL verification of the Phase
10/11/14B repositories is complete (users + behavioural profiles + per-session
behavioural verification state survive backend restarts; the 14B state also
survived a live server restart).
Additionally, Phase 15A verified state persistence and Phase 14B gating against
the real PostgreSQL container inside the Compose stack.

## Development Rule

Each phase must be implemented, tested, reviewed, and verified before proceeding to the
next phase. This file is updated at the end of every phase.