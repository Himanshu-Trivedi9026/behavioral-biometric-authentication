# Phase 15A — Containerization Foundation

**Status: Implemented — full-stack Docker Compose deployment exercised (see PART M).**

Phase 15A containerizes the application built across Phases 1–14B **without
changing application behavior**. It introduces a reproducible local Docker
Compose deployment with three runtime services (PostgreSQL, FastAPI backend,
React/Vite frontend) plus a deterministic one-shot migration service. ML
architecture, model, threshold, preprocessing, collection semantics,
authentication, and database schema are untouched.

## 1. Architecture

```
Browser
   |  http://localhost:3000   (Vite-built SPA, served by nginx)
   v
frontend  (nginx static server, SPA deep-link fallback)
   |  fetch() -> VITE_API_BASE_URL  (http://localhost:8000/api/v1, browser-reachable)
   v
backend   (uvicorn backend.app.main:app, CPU PyTorch inference)
   |  BBA_DATABASE_URL = postgresql://bba:***@db:5432/behavioral
   v
db        (PostgreSQL 16, named volume `postgres_data`)
```

One-shot **migrate** service applies `db/migrations/*.sql` in lexical order
before the backend starts.

## 2. Services

| Service  | Image                                             | Purpose |
|----------|---------------------------------------------------|---------|
| `db`     | `postgres:16-alpine`                              | Persistent PostgreSQL (named volume) |
| `migrate`| `postgres:16-alpine` (one-shot)                   | Applies migrations 001 → 003 to a ready DB |
| `backend`| `behavioral-biometric-auth-backend` (local build) | FastAPI + CPU PyTorch inference, Uvicorn |
| `frontend`| `behavioral-biometric-auth-frontend` (local build)| nginx static server for the Vite SPA |

`migrate` uses `depends_on: db: condition: service_healthy`; `backend` uses
`depends_on: db (service_healthy)` + `migrate (service_completed_successfully)`;
`frontend` uses `depends_on: backend (service_healthy)`.

## 3. Ports

| Service  | Container port | Published host port (default) |
|----------|----------------|-------------------------------|
| `backend`| 8000           | `BACKEND_PORT` (8000)          |
| `frontend`| 80            | `FRONTEND_PORT` (3000)         |
| `db`     | 5432           | **not published** (internal only) |

## 4. Required environment variables

Delivered via the git-ignored `.env` file (see `.env.example` for placeholders
and generation commands). `docker compose` fails closed before starting if a
`${VAR:?}`-marked value is missing.

| Variable            | Required | Notes |
|---------------------|----------|-------|
| `POSTGRES_USER`     | yes      | DB role (default `bba` in examples) |
| `POSTGRES_PASSWORD` | yes      | Random hex/URL-safe; used to build `BBA_DATABASE_URL` |
| `POSTGRES_DB`       | no       | defaults to `behavioral` |
| `BBA_JWT_SECRET_KEY`| yes      | Random secret ≥ 32 chars; empty → compose refuses to start (fail closed) |
| `BBA_CORS_ORIGINS`  | no       | Browser origin of the frontend; default `http://localhost:3000` |
| `VITE_API_BASE_URL` | no       | Browser-facing backend URL baked into the bundle at build time; default `http://localhost:8000/api/v1` |
| `BACKEND_PORT`      | no       | Host port for backend; default `8000` |
| `FRONTEND_PORT`     | no       | Host port for frontend; default `3000` |

`BBA_ENVIRONMENT` is **pinned to `production`** by `compose.yaml`, so the
backend's fail-closed production config validation is always active (a missing
or weak JWT secret is rejected rather than silently weakened).

## 5. Model artifact requirement

These files are intentionally gitignored and **not committed and not baked into
any image**:

- `models/siamese_behavioral_encoder.pt`
- `models/behavioral_preprocessing.json`
- `models/verification_config.json`

The Compose deployment mounts the local `models/` directory **read-only** into
the backend at `/app/models`:

```yaml
volumes:
  - ./models:/app/models:ro
```

with explicit `BBA_CHECKPOINT_PATH` / `BBA_PREPROCESSING_ARTIFACT_PATH` /
`BBA_VERIFICATION_CONFIG_PATH` set to `/app/models/...`. If a required artifact
is absent the ML endpoints fail closed with a structured `503`
(`model_load_error` / `preprocessing_artifact_error` /
`verification_config_error`) exactly as in the non-containerized app. The
system never generates, retrains, or recalibrates artifacts.

## 6. Migration behavior

PostgreSQL's built-in `/docker-entrypoint-initdb.d` mechanism only runs on a
fresh empty volume. To make behavior **explicit regardless of volume state**,
Phase 15A uses a dedicated one-shot **`migrate`** service:

- mounts `./db/migrations` (read-only) at `/migrations`;
- waits for the `db` healthcheck (`pg_isready`);
- runs each `.sql` file in lexical order (001, 002, 003) via
  `psql -v ON_ERROR_STOP=1`;
- any failure aborts the loop with a non-zero exit and the backend (which
  depends on `service_completed_successfully`) never starts.

The migration SQL is unchanged and remains the only source of truth. It is
idempotent (`CREATE TABLE IF NOT EXISTS`), so the migrate service can safely
re-run against an existing database (e.g., after `docker compose down`/`up`). No
migration-tracking table is created.

## 7. How to start the stack

Prerequisites: Docker Desktop + WSL2 running; the three model artifacts present
in `models/`.

```bash
cp .env.example .env          # then fill in real values (see section 4)
docker compose up --build -d   # build images, start db -> migrate -> backend -> frontend
```

Watch the ordering: `migrate` exits after applying migrations; the backend and
frontend start once their dependencies are satisfied.

## 8. How to stop the stack

```bash
docker compose stop    # stop containers, keep the named volume (data persists)
```

## 9. How to inspect logs

```bash
docker compose logs -f backend     # FastAPI / ML inference
docker compose logs -f frontend    # nginx access/error
docker compose logs migrate        # migration output (one-shot)
docker compose logs db             # PostgreSQL
```

## 10. How to verify health

```bash
curl http://localhost:8000/health          # {"status":"ok"}
curl -I http://localhost:3000/             # HTTP 200 (SPA index)
docker compose ps                          # db/backend/frontend "healthy"
docker compose exec db pg_isready -U bba   # inside the db container
```

`depends_on` conditions use these healthchecks so the backend only starts after
PostgreSQL is ready and migrations have completed, and the frontend starts after
the backend is healthy.

## 11. How to run backend tests

On the host (unchanged workflow):

```bash
.venv/bin/python -m pytest -q                                  # 784 passed
TEST_DATABASE_URL=postgresql://... .venv/bin/python -m pytest  # + PG-gated tests
```

The PostgreSQL integration tests can be pointed at the Compose database, e.g.
`TEST_DATABASE_URL=postgresql://bba:<password>@localhost:5432/behavioral`.

## 12. How to run frontend tests

```bash
cd frontend && npm test          # 75/75 tests (Node built-in test runner)
cd frontend && npm run build     # production bundle (dist/)
```

## 13. How to reset the local PostgreSQL volume

```bash
docker compose down -v           # REMOVES the named volume -> full reset
docker compose up -d             # fresh empty DB, migrations re-applied by migrate
```

`docker compose down` (without `-v`) keeps the volume, so data survives.

## 14. Important security notes

- **Secrets are never committed.** `.env` is gitignored; `.env.example` ships
  placeholders only; `compose.yaml` embeds no real secret; the backend image is
  built without `.env` and without model binaries.
- `BBA_JWT_SECRET_KEY` and `POSTGRES_PASSWORD` are required by Compose
  upstream of the backend's own fail-closed validation, so a misconfigured or
  missing secret stops the stack before any credentials are issued.
- Model artifacts stay gitignored; they are mounted read-only, so a container
  compromise cannot mutate them.
- PostgreSQL is not published to the host; only the backend (8000) and the
  frontend (3000) are reachable from the host browser.
- CORS allows only the explicit frontend origin (`BBA_CORS_ORIGINS`, never
  `*`).
- No raw behavioral windows, typed text, key identities, or credentials are
  persisted — the DB tables store only aggregates (enrollment centroids,
  session-state flags, user/Argon2id-hash records).

## 15. What remains outside Phase 15A

Explicitly out of scope (Phase 15B+):

- TLS/HTTPS termination and a public reverse proxy
- Kubernetes, cloud hosting, CI/CD, GitHub Actions
- Load balancing, autoscaling, Redis, WebSockets
- Rate limiting, account lockout, refresh tokens, OAuth, MFA
- ML retraining, threshold recalibration, new model architectures or features
- Raw behavioral window persistence or new telemetry/analytics storage