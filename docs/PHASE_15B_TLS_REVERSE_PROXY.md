# Phase 15B — TLS Reverse Proxy + DB-aware Readiness + Structured Request Logging

Status: **Implemented — automated checks PASSED; manual browser E2E pending user review.**

This phase hardens the Phase 15A container deployment without changing any
feature behaviour: a single TLS-terminating reverse proxy becomes the only
public edge, the backend gains a DB-aware readiness endpoint, and every HTTP
request is logged as structured JSON with a request id. No ML model, threshold,
preprocessing, schema, JWT/session, or frontend feature semantics changed.

---

## 1. Scope and guard-rails

Approved scope (nothing more was implemented):

1. TLS reverse proxy (nginx) in the Compose stack.
2. DB-aware readiness endpoint `GET /health/ready` (+ versioned).
3. Structured JSON request logging with `X-Request-Id`.

Explicitly out of scope (unchanged, not added, not touched):

- ML model / training / threshold (**still `0.4635127782821655`**),
  preprocessing, feature extraction, enrollment/verification/continuous
  decisions, schemas, JWT/session/auth semantics.
- Raw-event, key-identity, or typed-text persistence; new DB tables/migrations.
- Refresh tokens, OAuth/MFA, rate limiting, lockout/auto-logout, Redis,
  WebSockets, multi-device sessions.
- K8s / cloud / CI-CD / autoscaling, Prometheus/Grafana/OTel/log shipping,
  secrets manager, ACME/Let's Encrypt/DNS/CDN.

No commits were created during this phase (working-tree only, per approval).

---

## 2. Architecture

```
Browser --HTTPS 443--> proxy (nginx) --only service with host ports--
                          |  /api/, /health, /health/ready,
                          |  /docs, /openapi.json, /redoc
                          v  --> backend (uvicorn, internal :8000)
                          |  everything else (SPA + deep-link fallback)
                          v  --> frontend (nginx, internal :80)
                              backend --> db (PostgreSQL, private)
```

- HTTP on :80 responds `301 https://$host$request_uri` (plus a tiny
  unauthenticated `GET /nginx_health` used by the proxy healthcheck).
- Unknown `/api/...` paths are proxied to the backend and return its structured
  404 JSON — they never fall through to the SPA.
- Backend and frontend publish **no** host ports; PostgreSQL remains private.
- The browser talks to exactly one origin (HTTPS), so the frontend bundle uses a
  same-origin relative API base (`/api/v1`) and CORS is pinned to that origin.

### Services table

| service  | image             | internal port | host ports | healthcheck                        |
|----------|-------------------|---------------|------------|------------------------------------|
| db       | postgres:16-alpine | 5432          | none       | `pg_isready`                       |
| migrate  | postgres:16-alpine | –             | none       | one-shot, `service_completed_successfully` |
| backend  | FastAPI (CPU PyTorch) | 8000       | none       | `GET /health/ready` (DB-aware)     |
| frontend | nginx SPA         | 80            | none       | `GET /`                            |
| proxy    | nginx:1.27-alpine | 80 + 443      | `${HTTP_PORT:-80}`, `${HTTPS_PORT:-443}` | `GET /nginx_health` |

---

## 3. Files changed / added

| path | change |
|------|--------|
| `proxy/Dockerfile` | new — nginx:1.27-alpine image for the proxy |
| `proxy/nginx.conf` | new — TLS listener, redirect, routing, headers, health probe |
| `scripts/generate_dev_certs.sh` | new — idempotent local dev CA + server cert (openssl only) |
| `proxy/certs/` | generated, gitignored (`.gitkeep` only tracked) |
| `compose.yaml` | proxy service; backend/frontend ports unpublished; same-origin API arg; DB-aware backend healthcheck; TLS cert mounts; `BBA_LOG_LEVEL` passthrough; CORS default `https://localhost` |
| `backend/app/routes/health.py` | `GET /health/ready` + `/api/v1/health/ready`, `default_database_probe`, `DatabaseUnavailableError` |
| `backend/app/observability.py` | new — `JsonFormatter`, `configure_logging`, `RequestIdMiddleware` |
| `backend/app/config.py` | `BBA_LOG_LEVEL` field + validation |
| `backend/app/main.py` | wire `configure_logging` + `RequestIdMiddleware` |
| `.env.example`, local `.env` | same-origin `VITE_API_BASE_URL=/api/v1`, `BBA_CORS_ORIGINS=https://localhost`, proxy ports, TLS paths, log level |
| `.gitignore` | `proxy/certs/*` (+ `!.gitkeep`), `*.crt/key/pem/srl/csr` |
| `tests/test_health_readiness.py` | new — 28 readiness tests (incl. 2 live-PG gated) |
| `tests/test_observability_logging.py` | new — 5 logging/privacy tests |

---

## 4. TLS and certificates

- `proxy/nginx.conf`: `ssl_protocols TLSv1.2 TLSv1.3;` — TLS 1.0/1.1 rejected
  (verified live: TLS 1.1 attempt → rejected, TLS 1.2 + 1.3 accepted), HTTP/2
  on, HSTS `max-age=63072000; includeSubDomains` and `X-Content-Type-Options:
  nosniff` added on HTTPS responses.
- `scripts/generate_dev_certs.sh` (run once before `up`; idempotent; `-f` to
  rotate):
  ```
  scripts/generate_dev_certs.sh
  ```
  Generates `proxy/certs/bba-ca.{crt,key}`, `server.crt`, `server.key` with SAN
  `DNS:localhost` + `IP:127.0.0.1` (+ `BBA_TLS_HOST_NAMES` comma-separated
  extra DNS names). Development-only; gitignored.
- `compose.yaml` mounts the cert/key read-only at `/etc/nginx/certs/`,
  overridable via `BBA_TLS_CERT_PATH` / `BBA_TLS_KEY_PATH` so an operator can
  mount a trusted certificate without rebuilding. **No ACME/Let's Encrypt** —
  that is deliberately deferred.
- Private keys never enter image layers (verified via `docker history`).

---

## 5. DB-aware readiness

`GET /health/ready` (canonical + `/api/v1/health/ready`):

```json
{ "status": "ok", "database": "ok", "ml": "not_loaded" }
```

- `database` = `"ok"` when the configured `BBA_DATABASE_URL` answers `SELECT 1`
  (probe `psycopg.connect(url, connect_timeout=3.0)`, any exception → false);
  `"not_configured"` when no DB is configured (in-memory dev/tests — valid, 200);
  `503` `{"error":{"code":"database_unavailable",...}}` when configured but
  unreachable (no SQL/URL/credentials leaked).
- `ml` reflects `ml_service.is_loaded` without forcing a load.
- `GET /health` remains the unchanged zero-I/O liveness probe (verified: still
  200 while the DB is down).
- The backend container healthcheck now hits `/health/ready`, so a DB outage
  purposely marks the backend unhealthy (fail closed) and readiness recovers
  automatically when the DB returns.

---

## 6. Structured request logging

`backend/app/observability.py` (stdlib only — no new dependencies):

- `JsonFormatter`: whitelisted fields only — `timestamp, level, logger, message,
  request_id, method, path, status, duration_ms, exception`. No body, no query
  string, no `Authorization`, no JWT payload.
- `configure_logging(level, json_format)`: idempotent (handler added once),
  sets the root level, disables uvicorn access logging when JSON is active.
- `RequestIdMiddleware`: accepts an `X-Request-Id` matching
  `^[A-Za-z0-9._:*+-]{1,128}$`, otherwise generates `secrets.token_hex(16)`;
  echoes it as a response header; logs one `backend.app.http` INFO record per
  request with `duration_ms`.

Example backend log line:

```json
{"timestamp": "2026-09-21T20:14:36Z", "level": "INFO", "logger": "backend.app.http",
 "message": "request", "request_id": "72d714d2455129b2a5d8a450d69d25d9",
 "method": "GET", "path": "/health/ready", "status": 200, "duration_ms": 13.213}
```

JSON output is active whenever `BBA_ENVIRONMENT != development`; `BBA_LOG_LEVEL`
(default `INFO`) is validated at `Settings` construction.

---

## 7. Configuration (`.env`)

Relevant compose-related variables (see `.env.example`):

```env
BBA_CORS_ORIGINS=https://localhost
VITE_API_BASE_URL=/api/v1
BBA_LOG_LEVEL=INFO
HTTP_PORT=80
HTTPS_PORT=443
BBA_TLS_CERT_PATH=./proxy/certs/server.crt
BBA_TLS_KEY_PATH=./proxy/certs/server.key
```

The local Vite dev workflow is unchanged: `frontend/src/api.js` still falls back
to `http://localhost:8000/api/v1` when `VITE_API_BASE_URL` is unset. Only the
containerized bundle uses the relative same-origin base, so the production
bundle never contains `localhost:8000`/`localhost:3000` (verified by bundle
scan).

---

## 8. Deployment verification matrix (all PASSED)

| check | result |
|-------|--------|
| `docker compose config` | valid |
| `docker compose up -d --build` | all services healthy; migrate exits 0 |
| HTTP :80 → HTTPS | `301 https://localhost/` |
| HTTPS 200s | `/`, `/login`, `/register`, `/enrollment`, `/verification`, `/continuous`, `/api/v1/health`, `/health`, `/health/ready`, `/docs`, `/openapi.json`, `/redoc` |
| Host ports | only `:80`/`:443` published; backend `8000/tcp`, frontend `80/tcp`, db `5432/tcp` unpublished |
| Readiness DB up→down→up | `200 {"database":"ok"}` → (db stopped) `503 database_unavailable`, `/health` still 200 → (db restarted) `200 {"database":"ok"}` |
| Unknown `/api/v1/does-not-exist` | `404` JSON envelope (not SPA HTML) |
| Headers | HSTS + nosniff present; `X-Request-Id` echoed |
| TLS | 1.2 + 1.3 accepted; 1.0/1.1 rejected; CA-verified against generated CA |
| Bundle scan | no `localhost:8000` / `localhost:3000`; relative `/api/v1` base |
| Containerized tests | backend 815/62 (no PG), 877 passed (with PG via `TEST_DATABASE_URL`); frontend 75/75 |
| Host tests | backend 815 passed / 62 skipped; `compileall` PASS; `git diff --check` clean |
| Secret audit | passwords/JWT secret/DB URLs/raw-event markers: 0 hits across db, migrate, backend, frontend, proxy logs; `.env` untracked; certs gitignored; keys not in image layers; models mounted RO |

---

## 9. Manual Chrome E2E checklist (prepared — NOT yet claimed PASSED)

Flow to perform/observe in Chrome against `https://localhost` (trust the
generated CA / accept the dev-cert warning; password manager off):

1. `https://localhost/` lands on the landing page over HTTPS.
2. Register → login → enrolment captures → `POST /api/v1/enrollment` 201.
3. Verification → `VERIFIED` (threshold unchanged 0.4635127782821655).
4. `/continuous` captures a window → `VERIFIED`; DevTools shows only HTTPS
   requests to `https://localhost`, never `:8000`/`:3000`, no mixed content.
5. Force a `SUSPICIOUS` result (as in Phase 14A/15A) → `reverification_required`
   403 gates, `/continuous` recovery re-verifies and clears the gate.
6. Refresh a page → session restored from `sessionStorage`, server state intact.
7. Log Out → protected routes redirect to `/login`.
8. Note the `X-Request-Id` response header; confirm the backend JSON logs show
   request_id/path/status only (no body/query/secrets/raw events).

This checklist is deliberately **not** executed by automation in this
environment (no headless browser); it remains pending the user's manual pass.

---

## 10. Honest guard-rails

- Development certificates only; a real deployment must mount a trusted
  certificate and re-review HSTS (`includeSubDomains`) and cipher policy.
- No ACME, no secrets manager, no log shipping, no observability pipeline.
- A DB outage now flags the backend container unhealthy via the readiness
  healthcheck (intended fail-closed behaviour); recovery is automatic.
- ReadyState `ml:"not_loaded"` reflects no-forced-load design, not an error.
- No commit was created; `git status` shows the working-tree changes for review.

---
See also: [`PHASE_15A_CONTAINERIZATION.md`](PHASE_15A_CONTAINERIZATION.md)
(base deployment), `compose.yaml`, `proxy/nginx.conf`,
`scripts/generate_dev_certs.sh`.