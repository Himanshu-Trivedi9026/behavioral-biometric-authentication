# Phase 9A — FastAPI Foundation

## 1. Objective

Establish the **backend HTTP foundation** for the behavioral biometric
authentication system. FastAPI provides the application skeleton that later
phases will grow into (enrollment, verification, database, authentication,
frontend integration).

Phase 9A deliberately implements **only** the foundation: an application
factory, typed configuration, centralized error handling, CORS, dependency
injection groundwork, health/metadata endpoints, API versioning, and an
offline pytest API suite.

It does **not** yet connect to the Phase 8 ML implementation.

## 2. Backend Architecture

```
React frontend            (Phase 2 — unchanged in Phase 9A)
        ↓
    FastAPI                (this phase: foundation only)
        ↓
    ML service             (ml/evaluation, ml/verification, ml/training —
                            Phase 8, NOT yet wired into the API)
        ↓
Phase 8 evaluation / enrollment / verification
```

Phase 9A stops before the ML-service connection. The ML layer keeps working
independently exactly as Phase 8 left it.

## 3. Package Structure

```
backend/
├── __init__.py            backend package, version 0.1.0
└── app/
    ├── __init__.py
    ├── main.py            create_app() factory + ASGI singleton `app`
    ├── config.py          Settings (frozen dataclass, BBA_* env overrides)
    ├── errors.py          AppError + centralized exception handlers
    ├── dependencies.py    dependency injection foundation (get_settings)
    └── routes/
        ├── __init__.py
        ├── health.py      GET /health + GET /api/v1/health
        └── root.py        GET / + GET /api/v1/
```

Separation of concerns: application creation, configuration, routes, errors
and dependencies each live in their own module (no single giant file).

## 4. Application Factory

```python
from backend.app.main import create_app

app = create_app()                      # reads Settings.from_env()
app = create_app(Settings(...))         # explicit injected settings
```

`create_app()` (backend/app/main.py):

1. Resolves `Settings` (defaults or `BBA_*` environment variables).
2. Builds the `FastAPI` instance (title/version/description, `/docs`,
   `/redoc`, `/openapi.json`).
3. Attaches settings to `app.state` (the dependency-injection source).
4. Adds CORS middleware (explicit origins; never `*` with credentials).
5. Installs the centralized exception handlers.
6. Registers the foundation routers.

No global side effects at import time; `app = create_app()` is provided for
the ASGI entrypoint (`uvicorn backend.app.main:app`). Creating an app does
**not** load the ML checkpoint, open a database, or reach any external
service.

## 5. Configuration

`backend/app/config.py` — a frozen dataclass `Settings`:

| Field            | Default                          | Environment variable   |
|------------------|----------------------------------|------------------------|
| `app_name`       | `Behavioral Biometric Authentication API` | `BBA_APP_NAME` |
| `app_version`    | `0.1.0`                          | `BBA_APP_VERSION`      |
| `environment`    | `development`                    | `BBA_ENVIRONMENT`      |
| `debug`          | `False`                          | `BBA_DEBUG`            |
| `api_prefix`     | `/api/v1`                        | `BBA_API_PREFIX`       |
| `cors_origins`   | `("http://localhost:5173",)`     | `BBA_CORS_ORIGINS`     |

Validation fails loudly with `ValueError` on invalid environments, malformed
prefixes, empty names, and wildcard `*` CORS origins. No database/JWT/secret
configuration exists yet. Settings representations contain no secrets.

## 6. Health Endpoint

```
GET /health        -> 200 {"status": "ok"}
GET /api/v1/health -> 200 {"status": "ok"}   (versioned)
```

The health handler is determinististic and performs no I/O: it never loads
the ML model, queries a database, or contacts external services. `/health` is
the canonical unversioned endpoint for infra/monitoring; `/api/v1/health` is
the versioned API surface. Both share a single handler (no duplication).

## 7. Root / Metadata Endpoints

```
GET /        -> {"service": ..., "version": ..., "status": "ok"}
GET /api/v1/ -> {"api": "v1", "version": ..., "status": "ok"}
```

No internal filesystem paths, checkpoint paths or environment variables are
ever returned.

## 8. API Versioning

Routes are organized under the configurable prefix (`/api/v1` by default).
Future phases can add real endpoints **under the same prefix**:

```
/api/v1/enrollment    (Phase 9B+ — NOT implemented now)
/api/v1/verification  (Phase 9B+ — NOT implemented now)
```

Tests assert these paths currently return structured `404` and do not appear
in `/openapi.json`.

## 9. CORS Strategy

Development-oriented, safe-by-default:

- `allow_origins` comes from `Settings.cors_origins` (explicit list).
- `allow_credentials=True` is allowed **because** origins are explicit.
- A wildcard `*` origin is rejected at configuration time (unsafe with
  credentials).
- The frontend origin `http://localhost:5173` is the documented default but
  remains fully configurable — the frontend itself is untouched in Phase 9A.

## 10. Error Handling

All errors use one JSON envelope:

```json
{"error": {"code": "...", "message": "..."}}
```

Validation errors additionally include a `details` list
(`loc` / `msg` / `type`). Handled cases:

| Case                    | Status | code              |
|-------------------------|--------|-------------------|
| unknown route           | 404    | `not_found`       |
| wrong method            | 405    | `method_not_allowed` |
| `HTTPException`         | as set | e.g. `forbidden`  |
| request validation      | 422    | `validation_error` (with details) |
| unhandled exception     | 500    | `internal_error`  |

Unhandled exceptions are logged server-side and replaced with a generic safe
response — stack traces, paths and internal exception details never reach the
client. Expected domain errors can use `AppError(code, message, status)`.

## 11. Dependency Injection Foundation

`backend/app/dependencies.py` provides `get_settings`, which resolves the
application `Settings` from `app.state` per request. Future providers (model
service, profile repository, authentication service) will extend this module;
**none are implemented in Phase 9A** and no fake services were added.

## 12. Lifespan / Startup

A minimal `lifespan` hook logs start/stop only. It does not load the ML
checkpoint, create a database, or connect to Redis.

## 13. Testing Strategy

Offline pytest suites using `fastapi.testclient.TestClient`:

- `tests/test_api_foundation.py` — factory, health, root, OpenAPI, versioning,
  CORS, no-db / no-model / no-frontend guarantees.
- `tests/test_api_errors.py` — structured 404/405/422/403/500 envelopes and
  the no-leak guarantee.
- `tests/test_api_config.py` — defaults, `BBA_*` overrides, validation
  failures, privacy of settings.

Phase 9A: **47/47 new**; full Python suite **500/500**; frontend **19/19**;
`compileall` + import PASS.

## 14. Local Development Command

```
.venv/bin/python -m uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

Verified smoke test (port 8017):

```
GET /              -> 200
GET /health        -> 200  {"status": "ok"}
GET /api/v1/health -> 200  {"status": "ok"}
GET /api/v1/       -> 200
GET /docs          -> 200  (Swagger UI)
GET /redoc         -> 200  (ReDoc)
GET /openapi.json  -> 200
GET /api/v1/enrollment    -> 404 (not implemented)
GET /api/v1/verification  -> 404 (not implemented)
```

## 15. OpenAPI Documentation

Standard FastAPI docs are live:

- `/docs` — Swagger UI
- `/redoc` — ReDoc
- `/openapi.json` — OpenAPI 3.1 schema (`info.title` / `info.version` match
  `Settings.app_name` / `Settings.app_version`)

## 16. Security Limitations

- The API must never expose model checkpoint contents, user behavioral data,
  keyboard characters, passwords, environment secrets, filesystem internals
  or stack traces. This is enforced by the Phase 9A tests.
- **Authentication is NOT implemented** (Phase 9A) and will arrive in a later
  phase.
- CORS + exception handling alone do **not** make this service
  production-ready; it is a development foundation.

## 17. What Is NOT Implemented (Yet)

- Enrollment endpoints (`/api/v1/enrollment` absent).
- Verification endpoints (`/api/v1/verification` absent).
- Database / PostgreSQL / MongoDB / Redis.
- JWT / login / registration / password / session authentication.
- Continuous authentication / WebSocket monitoring.
- Frontend integration (the React app is unchanged).
- Model serving inside the API, API-based threshold calibration, API-based
  ROC/AUC/EER.
- Docker / deployment / cloud configuration.

No Phase 9B or later functionality was implemented.

## 18. Dependencies Added

- `fastapi>=0.110.0` → installed 0.141.1
- `uvicorn>=0.29.0` → installed 0.53.0
- `httpx>=0.27.0` (TestClient) → installed 0.28.1

`requirements.txt` gained a Phase 9A section; the Phase 5 torch/numpy lines
were made explicit (already installed) and Phase 6 DB/JWT/Redis items remain
documented as future-work comments.