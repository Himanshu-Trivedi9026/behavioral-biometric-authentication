# Phase 9B — ML Service + Enrollment / Verification API

## 1. Objective

Expose the **trained Phase 8 verification pipeline** through the FastAPI
backend built in Phase 9A as two endpoints:

```
POST /api/v1/enrollment    enroll a user_ref from raw behavioral sessions
POST /api/v1/verification  verify a probe session against a stored profile
```

Phase 9B wires the existing Phase 3 (preprocessing), Phase 5/7 (inference
encoder) and Phase 8 (enrollment / verification) code into the API **read-only**:
no training, no re-fitting of scalers, no client-supplied thresholds, no
per-request calibration. It is deliberately not yet a production service
(in-memory profiles, dev/synthetic artifacts, no auth) — those are honest,
documented limits, not silently-missing features.

## 2. Architecture

```
React frontend           (Phase 2 — unchanged)
        ↓   raw Phase 2 JSON sessions
  FastAPI routes         (Phase 9B)
        ↓
  BehavioralMLService    (backend/app/services/ml_service.py — lazy ML init)
        │   process_session          Phase 3 validation + features
        │   FeatureScaler transform  Phase 3 train-only scalers (artifact)
        │   embed_session            Phase 5/7 checkpoint (read-only, eval mode)
        │   enroll / verify          Phase 8 centroid + L2 + calibrated threshold
        ↓
  InMemoryProfileStore   (backend/app/services/profile_store.py)
        │   centroid (tuple of floats), embedding_dim, session_count, timestamps
        ↓
  VerificationResponse   {"user_ref","decision","distance","threshold"}
```

Health stays independent: `/health` and `/api/v1/health` never import `ml` or
`torch`, never load the checkpoint, and return 200 even when every artifact is
missing.

## 3. Decision Rule

L2 distance between the probe embedding and the enrolled centroid (Phase 6,
`eps=1e-8`), compared against a **calibrated, server-side threshold**:

```
VERIFIED   ⟺  distance <= calibrated_threshold
SUSPICIOUS ⟺  distance >  calibrated_threshold
```

The threshold comes from `models/verification_config.json`
(`calibration.threshold = 0.463513`, derived by Phase 8 calibration at target
FAR 0.05 on the development partition). The API never accepts a threshold from
the client — a client-supplied `threshold` field is ignored.

## 4. Inference Artifacts

| Artifact | Path | Producer | Contents |
|---|---|---|---|
| Preprocessing (train-only) | `models/behavioral_preprocessing.json` | `scripts/export_inference_artifacts.py` from `prepare_training_data(TrainingConfig())` | keyboard/mouse `FeatureScaler` state dicts (fitted on the 25 training sessions, seed 42), split metadata |
| Verification config | `models/verification_config.json` | same exporter from `evaluate_verification` | calibrated deployment threshold 0.463513, calibration details, artifact markers |
| Model checkpoint | `models/siamese_behavioral_encoder.pt` | Phase 7 training (git-ignored) | Phase 7 Siamese weights |

Rules tied to the artifacts:

- **Train-only scalers.** Incoming sessions are only *transformed* with scalers
  frozen at training time; they are never used to fit or update scalers.
- **Calibrated threshold.** The endpoint threshold is read from the config
  artifact, never derived from request data, never per-request.
- **No calibration on request data.** No threshold re-calibration happens from
  test or enrollment data.

Both JSON artifacts are git-ignored (generated, not source). Instructions to
regenerate: `.venv/bin/python scripts/export_inference_artifacts.py`.

## 5. Package Structure

```
backend/app/
├── schemas.py                request/response Pydantic models + limits
├── config.py                 + checkpoint_path / preprocessing_artifact_path /
│                             verification_config_path (BBA_* env overrides)
├── errors.py                 + AppError handler (registered before generics)
├── dependencies.py           + get_ml_service, get_profile_store
├── services/
│   ├── __init__.py
│   ├── ml_service.py         BehavioralMLService + error classes + seam loaders
│   └── profile_store.py      InMemoryProfileStore + StoredProfile + errors
└── routes/
    └── verification.py       build_versioned_router(prefix) — enrollment + verification
scripts/
└── export_inference_artifacts.py   generates the two JSON artifacts
models/
├── behavioral_preprocessing.json    generated (git-ignored)
└── verification_config.json         generated (git-ignored)
```

All `ml.*` / `torch` imports in the service layer are **lazy** (inside method
bodies), so importing the app or serving `/health` never touches the ML stack.

## 6. Request Schemas

Mirror the Phase 2 collector export. All numbers are finite floats
(`NaN`/`±inf` rejected at the boundary → 422); `user_ref` is a non-blank,
bounded string.

| Model | Fields | Limits |
|---|---|---|
| `KeyboardEvent` | `event_type:"keyboard"`, `event:"keydown"\|"keyup"`, `timestamp` | — |
| `MouseEvent` | `event_type:"mouse"`, `event:"mousemove"\|"mousedown"\|"mouseup"`, `x`, `y`, `timestamp` | — |
| `RawSession` | `session_id`, `started_at`, `ended_at`, `timestamp_source?`, `keyboard_events[]`, `mouse_events[]` | ≤ 20,000 events/session total |
| `EnrollmentRequest` | `user_ref`, `sessions[]` | 1..16 sessions |
| `VerificationRequest` | `user_ref`, `session` | — |

`user_ref` is **identity metadata only** — it is stripped from sessions before
the model sees them and is never a feature. The API is unauthenticated by
design (dev/demo), so nothing depends on auth.

## 7. Responses

Enrollment success (201):

```json
{"user_ref": "alice", "status": "enrolled",
 "embedding_dimension": 128, "session_count": 1}
```

Verification result (200):

```json
{"user_ref": "alice", "decision": "VERIFIED",
 "distance": 0.0001, "threshold": 0.463513}
```

Responses never expose raw events, embeddings, centroids, scaler internals,
filesystem paths or secrets.

## 8. Error Codes

One envelope: `{"error": {"code", "message"}}`, validation adds `details`.

| Condition | Status | Code |
|---|---|---|
| malformed request body / inf / NaN / too many sessions | 422 | `validation_error` |
| session invalid or has no usable behaviour | 422 | `invalid_session` |
| duplicate enrollment | 409 | `profile_exists` |
| no profile for user_ref | 404 | `profile_not_found` |
| checkpoint missing/corrupt | 503 | `model_load_error` |
| preprocessing artifact missing/malformed | 503 | `preprocessing_artifact_error` |
| verification config missing/malformed | 503 | `verification_config_error` |
| ML service cannot initialise | 503 | `ml_service_unavailable` |
| unexpected inference failure | 500 | `inference_error` |

## 9. Profile Store

`InMemoryProfileStore` holds **only** per-user summary state:

```
user_ref, centroid (tuple of floats), embedding_dim, session_count,
created_at, updated_at
```

- Thread-safe (`RLock`), process-local, **non-persistent** — profiles are lost
  on restart (documented; PostgreSQL is explicitly out of scope).
- Never stores raw events, embeddings, keyboard characters, passwords or
  secrets. `stored_fields()` pins the schema.
- `save()` refuses to overwrite (`ProfileExistsError`); `get()` raises
  `ProfileNotFoundError`; `list_profiles()` returns metadata only.

## 10. ML Service

`BehavioralMLService` composes the Phase 3/5/7/8 pipeline without
re-implementing any of it. Injection seams (`verifier_factory`,
`preprocessing_loader`, `verification_config_loader`, `session_processor`)
make the wiring fully testable offline; the defaults are the real loads.

- `load()` — lazy, idempotent; load failures map to structured error codes.
- `enroll_sessions(user_ref, raw_sessions)` → Phase 8 `EnrollmentProfile`
- `enroll_to_store(user_ref, raw_sessions)` → `StoredProfile` (centroid tuple)
- `verify_stored(user_ref, stored, raw_session)` → Phase 8 `VerificationResult`
  (distance vs server-side threshold)

## 11. Privacy / Security Guarantees (test-enforced)

- `user_ref` never reaches the model (identity stripped before processing).
- Centroid / embeddings / raw events are never returned by the API.
- The store keeps only aggregates; no raw events, keys or secrets persist.
- Responses contain no checkpoint paths, filesystem paths, or environment
  secrets; `Settings.__repr__` shows no secrets.
- No auth is implemented nor required — explicitly a dev/demo stage.

## 12. Testing Strategy

Offline pytest. Files (Phase 9B, 72 tests):

- `tests/test_ml_service.py` (A) — service wiring with a deterministic fake
  verifier: load idempotency, genuine/impostor geometry, threshold control,
  determinism, invalid sessions, structured load/inference failures, scaler
  threading.
- `tests/test_profile_store.py` (B) — save/get/duplicate/instances,
  aggregate-only schema, thread safety.
- `tests/test_enrollment_api.py` (C) — 201/409/422 cases, field shape, no
  overwrite on conflict.
- `tests/test_verification_api.py` (D) — 200 decisions, 404 unknown, threshold
  integrity (client-supplied threshold ignored), 422/routing.
- `tests/test_api_privacy.py` (E) — identity never a model feature, sanitized
  responses, store aggregates only, no secrets, no auth dependency.
- `tests/test_api_health_isolation.py` (F) — health green with missing
  artifacts, ML endpoints fail loudly (503 envelope), no `ml`/`torch` imports
  while serving health.
- `tests/test_api_ml_integration.py` (G) — end-to-end through the **real**
  checkpoint + artifacts: self-verify VERIFIED (distance ~1e-4 floor),
  impostor SUSPICIOUS, duplicate 409, unknown 404, malformed 422, and a
  dataset-derived journey (enroll/verify from `dataset_synthetic.json`).
  Skipped when artifacts/checkpoint are absent.

Stale Phase 9A asserts were updated where the world changed:
- `tests/test_api_foundation.py::test_ml_endpoints_registered` — endpoints
  now exist (GET returns 405 instead of 404).
- `tests/test_api_errors.py::test_future_post_endpoints_are_404` — empty-body
  POST to the real routes is now 422 (request validation), not 404.

## 13. Results

- New dedicated tests: **72/72** (A–G).
- Full Python suite: **572/572** passed.
- Frontend: **19/19** passed (untouched).
- `compileall` + `create_app` import check PASS (no `ml`/`torch` at import).
- Live uvicorn smoke test (port 8111): enroll 201 (dim 128), duplicate 409,
  genuine VERIFIED (`distance` 0.0001 ≤ 0.463513), impostor SUSPICIOUS
  (`distance` 0.886 > 0.463513), unknown 404, empty sessions 422,
  `POST /openapi.json` 405.

## 14. Local Development Command

```
.venv/bin/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Swagger UI at `/docs`, ReDoc at `/redoc`, schema at `/openapi.json`.

## 15. What Is NOT Implemented (Deferred — Honest Limits)

- **Persistence:** profiles live in an in-memory store and are lost on
  restart; PostgreSQL/Redis remain out of scope.
- **Production artifacts:** the threshold/scalers were derived on SYNTHETIC
  data and must be regenerated for a real deployment.
- **Authentication:** no JWT/login/register; the API is unauthenticated by
  design at this stage.
- **Continuous authentication** / WebSocket monitoring.
- **Frontend integration** — the React app is unchanged.
- **API-based training, calibration, ROC/AUC/EER, or repository endpoints.**
- Docker / deployment / cloud configuration.

## 16. Guardrails

- **In-memory store:** profiles disappear on restart by design.
- **Synthetic-development artifacts:** `models/behavioral_preprocessing.json`
  and `models/verification_config.json` are dev artifacts fitted/calibrated on
  synthetic identities; they must be regenerated from production data before
  any real deployment. Neither is a real-world biometric claim.

## 17. Dependencies Added

None — Phase 9B reuses the Phase 9A stack (`fastapi`, `uvicorn`, `httpx`) and
the Phase 5–8 ML stack already installed.