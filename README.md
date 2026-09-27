# Behavioral Biometric Authentication

**Few-Shot Identity Verification Using Keystroke Dynamics and Mouse-Movement Trajectories**

A college project that verifies an individual's identity by continuously analyzing their
unique typing rhythm and mouse-movement patterns. Instead of a single static credential,
identity is derived from *how* the user behaves at the keyboard and mouse, making it
resilient to credential theft and impersonation.

---

## Main Objective

Build an end-to-end behavioral biometric system that captures keyboard and mouse events in
the browser, converts them into a behavioral embedding using neural sequence models, and
performs **few-shot identity verification** with a Siamese network to decide whether a given
session belongs to the enrolled user (Verified) or a different actor (Suspicious).

---

## Technology Stack

| Layer            | Technology                              |
|------------------|------------------------------------------|
| Frontend         | React + Vite (JavaScript / CSS)                       |
| Backend          | FastAPI                                  |
| Machine Learning | Python + PyTorch                         |
| Data Processing  | NumPy + Pandas                           |
| Sequence Encoder | 1D-CNN (optional) + GRU                  |
| Verification     | Siamese Network + Contrastive Loss       |
| Database         | PostgreSQL                               |
| Session Auth     | JWT (HS256) + Argon2id                   |
| TLS / Proxy      | Nginx reverse proxy (Phase 15B)          |
| Deployment       | Docker + Docker Compose (Phase 15A)      |

---

## High-Level Architecture

The pipeline below is **implemented and running** (Phases 1–15B). Component
documentation lives in the per-phase files linked under
[Development Phases](#development-phases).

```
Keyboard + Mouse Events
        ↓
Preprocessing
        ↓
Optional 1D-CNN
        ↓
GRU
        ↓
Behavioral Embedding
        ↓
Siamese Network
        ↓
Similarity Score
        ↓
Threshold
        ↓
Verified / Suspicious
```

At runtime this pipeline is served by the FastAPI backend in Docker, reached
only through the Nginx HTTPS reverse proxy.

---

## Development Phases

Phases 1–15B are **complete**. Phase 16A is **in progress**.

1. **Phase 1** — Project Setup (`completed`)
2. **Phase 2** — Browser-based Keyboard and Mouse Data Capture (`completed`)
3. **Phase 3** — Data Processing and Feature Extraction (`completed`)
4. **Phase 4** — Dataset Creation + Statistical (Non-Neural) Baseline (`completed`)
5. **Phase 5** — ML Model: Sequence Encoder (1D-CNN + GRU) (`completed`)
6. **Phase 6** — ML Model: Siamese Network + Contrastive Loss (`completed`)
7. **Phase 7** — Siamese Training + Calibrated Threshold (`completed`)
8. **Phase 8** — Enrollment / Verification Evaluation (`completed`)
9. **Phase 9** — FastAPI Foundation + ML Service API (`completed`)
10. **Phase 10** — PostgreSQL Profile Storage (`completed`)
11. **Phase 11** — Authentication & Authorization (`completed`)
12. **Phase 12** — Frontend Integration (`completed`)
13. **Phase 13** — Live PostgreSQL Persistence (`completed`)
14. **Phase 14** — Continuous Verification + Session State (`completed`)
15. **Phase 15** — Containerization (15A) + TLS Reverse Proxy (15B) (`completed`)
16. **Phase 16A** — Real-World Data Collection & Evaluation (`in progress`)

Per-phase documentation: [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md).

---

## Phase 2 — Browser Data Collector (Prototype)

A development/test-only browser page (`frontend/`) captures raw behavioral timing data for
later phases. The frontend is built with **React + Vite** (JavaScript + CSS) — no backend.

### What the collector does

- Starts/stops a uniquely identified **collection session**.
- Records **keyboard** `keydown`/`keyup` events with a high-resolution timestamp.
- Records **mouse** `mousemove`/`mousedown`/`mouseup` events with viewport coordinates
  `x`,`y` and a high-resolution timestamp.
- Lets you export the session as a pretty-printed JSON file and clear in-memory data.

### What is collected (keyboard)

| Field        | Meaning                                        |
|--------------|------------------------------------------------|
| `event_type` | `"keyboard"`                                   |
| `event`      | `"keydown"` or `"keyup"`                       |
| `timestamp`  | Monotonic high-resolution time, milliseconds   |

### What is collected (mouse)

| Field        | Meaning                                        |
|--------------|------------------------------------------------|
| `event_type` | `"mouse"`                                      |
| `event`      | `"mousemove"`, `"mousedown"`, or `"mouseup"`    |
| `x`, `y`     | Viewport coordinates (px)                      |
| `timestamp`  | Monotonic high-resolution time, milliseconds   |

### Timestamp strategy

Event timestamps use `performance.now()` — a **monotonic, high-resolution** timer that is
immune to wall-clock changes, which is the correct basis for behavioral timing. Real-world
reference times are stored only as `started_at`/`ended_at` ISO-8601 strings on the session.
The source is recorded in the session as `timestamp_source`.

### Mouse sampling strategy

`mousemove` fires at the native pointer-event rate (often 1000+ Hz), which would produce an
unbounded dataset. The collector therefore records a move sample when **either** at least
`mouseMoveMinIntervalMs` (default **25 ms**) has elapsed **or** the pointer moved at least
`mouseMoveMinDistancePx` (default **2 px**). This keeps a trajectory-shaping sample set
without exploding in size. Clicks (`mousedown`/`mouseup`) are always recorded.

### What is deliberately NOT collected

- Typed characters / `event.key` — only press/release **timing** is stored
- Passwords, text content, input field values
- Clipboard content
- Cookies, browser history, or other browsing data
- Arbitrary DOM content or personally identifying information

### Privacy note (development collector)

This prototype captures **behavioral metadata only**. No typed text, no key identities, and
no values from the on-page textarea are ever read, persisted, or exported. Never type
passwords or personal information while collecting test data — not because the tool stores
it, but as good practice.

### How a session is exported

1. Press **Start Collection**, type and move/click.
2. Press **Stop Collection**.
3. Press **Download / Save Session (JSON)** — the browser downloads
   `behavioral-session-<session_id>.json`.

### Where raw session files go

Browser JavaScript cannot write directly into the repository. For this prototype workflow,
**manually move/place the downloaded JSON file into `data/raw/`** (for example
`data/raw/session-<session_id>.json`). `data/raw/` is git-ignored, keeping collected data
out of version control.

### Running the frontend locally

The collector is a React + Vite app rooted at `frontend/`:

```bash
cd frontend
npm install        # first time only
npm run dev        # start the Vite dev server
```

Then open the printed URL (default `http://localhost:5173/`). The Vite dev server is a
static development server only — it exposes no API and does not persist any data.

Build the production bundle (static files into `frontend/dist/`, which is git-ignored):

```bash
npm run build
npm run preview    # optionally serve the built bundle locally
```

Run the collector logic unit tests (Node built-in runner, no extra dependencies):

```bash
npm test           # runs node --test ../tests/collector.test.js
```

### Limitations of the browser-only prototype

- No server-side storage; sessions are exported as local files only.
- No data pipeline, feature extraction, or ML (later phases).
- No enrollment/verification semantics yet.
- `performance.now()` values are relative to the page's time origin, not wall-clock — fine
  for behavioral timing, but cross-session alignment needs care in Phase 3.
- The page is for development/testing only; it has no authentication or hardening.

---

## Phase 3 — Data Preprocessing and Feature Extraction

A pure-Python (stdlib-only) package under `ml/preprocessing/` that converts a raw Phase 2
session into machine-learning-ready **behavioral sequences**, keeping keyboard and mouse
separate. Full documentation in [`docs/PHASE_3_PREPROCESSING.md`](docs/PHASE_3_PREPROCESSING.md).

High-level API:

```python
from ml.preprocessing import process_session

raw = { ... }                    # a saved Phase 2 session dict
processed = process_session(raw) # validate → preprocess → extract features
```

What it produces:

- `keyboard_sequence` — `{"hold_time", "flight_time"}` per key press
- `mouse_sequence` — `{"dx", "dy", "dt", "distance", "speed"}` per move pair
- `mouse_action_events` — preserved `mousedown`/`mouseup` (chronological)
- `source` + `preprocessing` metadata for traceability

What it does NOT do: no model, no tensors, no fusion of the two sequences, and it emits only
numerical behavioral features (no key names, characters, passwords, or text content).

Normalization (`ml.preprocessing.normalization.FeatureScaler`) provides
`fit` / `transform` / `fit_transform` (z-score) with constant-feature safety, plus
JSON-serializable `state_dict()` so training statistics can later be applied to
validation/test data without being re-fitted or re-hard-coded.

Run the Phase 3 tests:

```bash
.venv/bin/pip install -r requirements.txt   # first time (installs pytest)
.venv/bin/python -m pytest tests/test_preprocessing.py -v
```

Synthetic test data lives in `tests/fixtures/` (generated, not collected).

---

## Phase 4 — Dataset Creation + Statistical Baseline

A stdlib-only, deterministic dataset layer and a **non-neural** statistical
baseline. Full documentation in [`docs/PHASE_4_DATASET_BASELINE.md`](docs/PHASE_4_DATASET_BASELINE.md).

- `dataset/` — schema + validation + privacy checks, deterministic synthetic
  generator (5 users x 10 sessions), real-session loading, per-user
  leakage-free train/test split.
- `baseline/` — per-user enrollment profiles, keyboard/mouse/combined distance
  scoring, threshold calibration (training data only), TAR/FAR/FRR/ROC metrics.
- CLIs: `scripts/generate_dataset.py`, `scripts/evaluate_baseline.py`.

What it does NOT do: no neural networks, no tensors, no ML frameworks — it is
the reference point the Phase 5 encoder must beat.

---

## Current Development Status

- **Current phase:** Phase 4 — Dataset Creation + Statistical Baseline
- **Status:** In Progress (see [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md))
- **Next phase:** Phase 5 — CNN + GRU Behavioral Encoder

See [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md) for detailed phase tracking.
---

## TEAM MEMBER SETUP

This section describes how to clone the repository and run a locally identical, research-approved instance for Phase 16A behavioral data collection.

### Prerequisites

- **Git** (version control)
- **Docker Desktop** (Windows/macOS/Linux) with the Docker Engine running
- **WSL2 on Windows** if required by Docker Desktop
- A modern **web browser** (Chrome/Edge/Firefox) with permission to trust local certificates

### 1. Clone repository

```bash
git clone <repository-url>
```

### 2. Enter repository

```bash
cd behavioral-biometric-authentication
```

### 3. Create .env from .env.example

Copy the template to your local environment file (git-ignored, never committed):

```bash
cp .env.example .env
```

Edit `.env` and set real values for secrets as instructed in the file comments. The stack uses fail-closed validation: Compose will refuse to start if required secrets (e.g., `BBA_JWT_SECRET_KEY`, `POSTGRES_PASSWORD`) are missing or placeholder in a way that violates production validation.

> Note: `.env.example` ships with placeholders only and contains no real credentials.

### 4. Generate local development certificates

The TLS reverse proxy requires a local development CA and server certificate (Phase 15B). Generate them once (idempotent):

```bash
scripts/generate_dev_certs.sh
```

To rotate/regenerate: `scripts/generate_dev_certs.sh -f`.

The generated files under `proxy/certs/` are **git-ignored** and **must never be committed**. These are development-only.

### 5. Trust the local CA certificate (browser/OS)

Phase 15B uses local HTTPS at `https://localhost`. Your browser will not trust the development CA by default. Trust `proxy/certs/bba-ca.crt` so `https://localhost` is considered secure in your local environment.

- **Chrome/Edge (Linux/macOS/Windows):** Settings → Privacy/Certificates → Import Authorities/Trusted Root → import `proxy/certs/bba-ca.crt`
- **Firefox:** Settings → Privacy & Security → Certificates → View Certificates → Authorities → Import → select `proxy/certs/bba-ca.crt` (check “Trust this CA to identify websites”)
- **macOS Keychain:** open `proxy/certs/bba-ca.crt` and add to “System”/“Login” keychain with “Always Trust”
- **Ubuntu/other Linux:** depending on distro, import into system trust store (e.g., `cp proxy/certs/bba-ca.crt /usr/local/share/ca-certificates/bba-ca.crt && sudo update-ca-certificates`)

After trusting, restart the browser if needed.

### 6. Start the application with Docker Compose

Build images and start all services (db → migrate → backend → frontend → proxy) in the background:

```bash
docker compose up -d --build
```

The proxy is the only service publishing host ports: `80`/`443`. Backend/frontend/db remain internal to the Docker network.

### 7. Verify services are healthy

Check container status:

```bash
docker compose ps
```

All services should reach `healthy` (or `Up` with the expected state). To inspect logs if needed:

```bash
docker compose logs -f
# or docker compose logs -f backend | tail -50
```

Readiness endpoint (canonical): `GET https://localhost/health/ready` (or `http://localhost/health` for liveness). The backend readiness reports database state via structured JSON.

### 8. Open the correct HTTPS URL

Open your browser to **https://localhost**. HTTP requests are automatically redirected to HTTPS (`301`).

All application routes (`/`, `/login`, `/register`, `/enrollment`, `/verification`, `/continuous`) are served via the HTTPS reverse proxy.

### 9. Register / Login

- Register a new account (email/password). Passwords are hashed with **Argon2id** on the server.
- Login with those credentials. JWT access tokens are issued server-side (HS256). No passwords are ever returned by the API.

### 10. Open Enrollment

Navigate to **Enrollment** (`/enrollment`). You will be prompted to capture a small number of behavioral sessions (keyboard + mouse). The enrollment cap is 16 sessions.

### 11. Start behavioral sessions

Click **Start Session** and perform natural typing/mouse activity in the collection area. The collector records:

- **Keyboard**: `keydown`/`keyup` events with high-resolution timestamps (no key characters, no typed text, no field values)
- **Mouse**: `mousemove`/`mousedown`/`mouseup` events with viewport `x,y` and high-resolution timestamps

No passwords, text content, clipboard content, or personally identifying information is read or stored.

### 12. Stop sessions

Click **Stop Session** to end the current collection session. A session appears in the “Sessions Ready to Submit” list with counts: `KBD … · MOUSE …`.

Repeat as instructed to collect the required number of sessions.

### 13. Export Sessions

Click **Export Sessions** to download the collected sessions as a JSON file to your local device. **This is local-only.** The browser writes the file directly to disk; nothing is uploaded, sent over the network, or stored server-side by this action.

The exported file name is e.g. `behavioral-sessions-YYYY-MM-DDTHH-MM-SS.sssZ.json` and its contents contain **only raw keyboard/mouse timing and event-type metadata** (no name, email, password, or account details).

### 14. Understand “Export Sessions is local-only”

- No `fetch`/XHR/`sendBeacon` call is made during export (verified by design).
- The export uses the browser’s download mechanism only.
- The server/database never receives the exported JSON through this button.

### 15. DO NOT click “Send Sessions to Server” for Phase 16A research collection

During Phase 16A research data collection, **do not click “Send … Session(s) to Server”** unless the project owner specifically instructs you to do so. For research collection, the correct workflow is to **export the JSON locally and send that exported file to the project owner**.

The UI label you may see is: “Send N Session(s) to Server” (disabled until sessions exist). This is the normal enrollment submission path for the deployed application; for Phase 16A **research collection**, use **Export Sessions** instead.

### 16. Share the exported JSON with the project owner

After verifying the JSON downloaded successfully, send that exported JSON file to the project owner via the agreed channel. **Do not modify the file.** Keep a local copy only if instructed.

---

## MODEL ARTIFACTS (APPROVED, FROZEN)

This repository includes the **three approved inference artifacts** required at runtime:

| Artifact | Purpose | Size |
|---|---|---|
| `models/siamese_behavioral_encoder.pt` | Siamese encoder checkpoint (epoch 10) | ~583 KB |
| `models/behavioral_preprocessing.json` | Train-only keyboard/mouse feature scalers | ~1.6 KB |
| `models/verification_config.json` | Calibrated threshold `0.4635127782821655` + provenance (`dataset_synthetic.json`) | ~1.2 KB |

These three files are **tracked in version control on purpose** for this research project so every participant runs the **exact same model, preprocessing artifact, and calibrated threshold**. No teammate should retrain, recalibrate, or regenerate them. They are never modified in the course of Phase 16A.

- **Threshold:** `0.4635127782821655` (frozen)
- **Checkpoint epoch:** `10` (frozen)
- **Dataset ID:** `dataset_synthetic.json` (provenance only)
- **Behavior:** preprocessing/scaler behavior is frozen; the system never retrains or recalibrates at runtime

If any of these three files are missing, the backend returns a structured `503` (`model_load_error`, `preprocessing_artifact_error`, or `verification_config_error`) and will not serve verification/enrollment until present.

Other model/checkpoint files (`*.pt`, `*.pth`, etc.) remain **git-ignored**.

---

## HTTPS SETUP (LOCAL DEVELOPMENT)

This project uses a local HTTPS reverse proxy (Phase 15B):

- **URL:** `https://localhost`
- **HTTP → HTTPS:** all HTTP traffic on port 80 is redirected to HTTPS on port 443 (`301`)
- **Certificates:** development-only, generated by `scripts/generate_dev_certs.sh`. The local CA is `proxy/certs/bba-ca.crt`; server cert/key in `proxy/certs/` (git-ignored).
- **Browser trust:** import `proxy/certs/bba-ca.crt` as a trusted root/authority (see step 5). Do not skip this or you will see TLS warnings.
- **Security:** TLS 1.2/1.3 only; HSTS enabled on HTTPS responses; `X-Content-Type-Options: nosniff`. Private keys never enter image layers and are never committed.
- **Do not use production certificates** here — mount operator-provided certs only if deploying outside local research collection.

---

## DOCKER SETUP

Services (from `compose.yaml`): `db` (Postgres 16-alpine, internal only), `migrate` (one-shot SQL migrations), `backend` (FastAPI), `frontend` (nginx serving built bundle), `proxy` (nginx TLS reverse proxy).

Normal workflow:

```bash
docker compose up -d --build
docker compose ps
docker compose down
```

Notes:
- Only `proxy` publishes `:80` and `:443` to the host. `backend` (`8000`), `frontend` (`80`), and `db` (`5432`) are not published.
- Migrations run idempotently via the `migrate` service (`CREATE TABLE IF NOT EXISTS`).
- Model artifacts are mounted read-only into the backend container at `/app/models`.
- The backend healthcheck uses `/health/ready` (DB-aware). Liveness is `/health`.

To reset the local database volume (destructive): `docker compose down -v`.

---

## Oracle Cloud Production Deployment

> **Preparation status.** This section documents the intended production deployment. No Oracle VM, deployment, or certificate has been created by the D2 configuration-preparation phase. Everything below is written for an operator who has been explicitly authorized to provision and deploy.

The same `compose.yaml` used for local testing is used for production. The **only** difference is the values in the VM's git-ignored `.env` — there is no separate production compose file, and none is needed. Local development is unaffected because `compose.yaml` pins `BBA_ENVIRONMENT=production` for the container itself.

### 1. Target architecture

```
Oracle Cloud Ubuntu VM
  └─> Docker Compose
        └─> Nginx (proxy — the ONLY service publishing host ports :80 / :443)
              ├─> Frontend (React/Vite bundle, served by nginx)
              └─> FastAPI (backend, internal :8000)
                    ├─> PostgreSQL 16 (internal :5432, named volume)
                    └─> ML inference (frozen artifacts, read-only mount)
```

The proxy terminates TLS and is the single public edge. The browser and the API share one same-origin path, so `VITE_API_BASE_URL` stays the relative `/api/v1`.

### 2. Recommended VM

- **Oracle Ampere A1, minimum 4 GB RAM.**
- The 1 GB AMD micro is **too small**: the backend image build needs roughly 1.5–2 GB RAM for the CPU PyTorch install. Steady-state usage is far lower (~500–650 MB), but the build peak is what constrains the shape.
- Boot volume: the default 50 GB is ample (~10–15 GB used).

### 3. Network

| Port | Access | Reason |
|---|---|---|
| TCP 443 | Public ingress | HTTPS (the proxy) |
| TCP 80 | Public ingress | HTTP → HTTPS redirect only |
| SSH | **Restrict to the administrator's IP** where possible | Administration |

Only the `proxy` publishes host ports. PostgreSQL, the backend, and the frontend are reachable solely on the internal Compose network and must **not** be given public ingress rules.

### 4. Repository setup

```bash
git clone https://github.com/Himanshu-Trivedi9026/behavioral-biometric-authentication.git
cd behavioral-biometric-authentication
```

### 5. Verify model artifacts

```bash
sha256sum models/siamese_behavioral_encoder.pt \
  models/behavioral_preprocessing.json \
  models/verification_config.json
```

Expected:

```
bbaec09299598fd715f9a66a9fea7ffc4bf416dae5f4db3d76704b919ca156b1  models/siamese_behavioral_encoder.pt
3085a7f0a4c6810c10f90a9c18eab44673132678aa8d71cf95049f18a5f343b1  models/behavioral_preprocessing.json
0221a34d34bfe6057a7bb2978fb09b0ef288aa12533e43179beed0bee16e279e  models/verification_config.json
```

All three values are also recorded in `.env.example`. `git clone` already delivers the files; a mismatch means the clone is not at the expected commit — stop and investigate before deploying.

These three artifacts are **already tracked in Git** and are **frozen**. Do **not** retrain, recalibrate, re-export, regenerate, or manually copy them onto the VM. `compose.yaml` mounts `./models` read-only at `/app/models`.

### 6. Environment

```bash
cp .env.example .env
```

Generate the two secrets **on the VM** (never paste the output into Git):

```bash
python3 -c "import secrets;print(secrets.token_urlsafe(48))"   # -> BBA_JWT_SECRET_KEY
python3 -c "import secrets;print(secrets.token_hex(32))"       # -> POSTGRES_PASSWORD
```

Then set these in `.env`:

```
BBA_CORS_ORIGINS=https://<PUBLIC_ORIGIN>
VITE_API_BASE_URL=/api/v1
```

`BBA_CORS_ORIGINS` must be the exact origin the browser uses, with no trailing slash and no path. Keep `VITE_API_BASE_URL` as the relative `/api/v1`.

```bash
chmod 600 .env
```

### 7. Preflight

```bash
docker compose config --quiet
```

This **fails with a non-zero exit** while any required secret is missing. That is intentional fail-closed behavior — do not work around it. The command does not print secret values.

### 8. Deployment

```bash
docker compose up -d --build
```

### 9. Verify

```bash
docker compose ps
docker compose logs migrate
```

`db`, `backend`, `frontend`, and `proxy` should all report **healthy**; `migrate` is a one-shot service that should exit `0` after applying `001`–`003` idempotently.

### 10. Current TLS limitation

- The certificates in `proxy/certs/` are **development certificates for localhost** (`CN=localhost`, SAN `DNS:localhost, IP:127.0.0.1`) produced by `scripts/generate_dev_certs.sh`.
- They are **not trusted by browsers and must NOT be used as trusted production certificates** for a real public hostname. Browsers will show a TLS warning.
- A trusted public certificate **cannot be issued for a bare VM IP address** — do not plan around that.
- A later domain/TLS phase replaces them with a trusted certificate by overriding `BBA_TLS_CERT_PATH` / `BBA_TLS_KEY_PATH` in `.env`. No compose, Dockerfile, nginx, or code change is required for that swap. Private keys are never committed (`.gitignore` excludes `*.key`, `*.pem`, `*.crt`).

### 11. Backup

PostgreSQL data lives in the named volume `postgres_data`, which is the **only** copy of all accounts and enrollment profiles. Schedule a dump and keep it off the VM:

```bash
docker compose exec -T db pg_dump -U bba -d behavioral | gzip > ~/bba-$(date +%F).sql.gz
```

`docker compose down -v` **destroys** this data. A VM reprovision loses it too.

### 12. Phase 16A protection (unchanged by deployment)

**Deploying this stack does not change the Phase 16A workflow in any way.** Specifically:

- Raw behavioral sessions are still **exported locally** in the browser and shared with the project owner out-of-band. Nothing is uploaded automatically.
- Deployment adds **no** upload endpoint and **no** automatic submission path. There is no `multipart`/`UploadFile` handler anywhere in the backend.
- The collector, the session export, the event schema, and the real-data dataset pipeline are all unchanged.
- The rule in step 15 above — **“DO NOT click ‘Send Sessions to Server’ for Phase 16A research collection”** — applies exactly as it does locally. Read [14. Understand “Export Sessions is local-only”](#14-understand-export-sessions-is-local-only) and [PHASE 16A COLLECTION INSTRUCTIONS](#phase-16a-collection-instructions) before collecting any data.
- The frozen ML artifacts and the calibrated threshold are identical for every participant, whether they run locally or against the deployed instance.

---

## PHASE 16A COLLECTION INSTRUCTIONS

Participant workflow for research collection:

1. **Register/Login** → create/login to your local account
2. **Go to Enrollment** (`/enrollment`)
3. **Start Session** → begin behavioral capture
4. **Behave naturally** for the requested collection period (keyboard + mouse activity)
5. **Stop Session** → end capture
6. **Repeat** as instructed (collect the requested number of sessions)
7. **Export Sessions** → download JSON to this device (**local-only**)
8. **Verify** the JSON file downloaded successfully
9. **Send the exported JSON** to the project owner (do **not** upload via the app)

**Important:** DO NOT click "Send … Session(s) to Server" during Phase 16A research data collection unless the project owner specifically instructs you to do so.

---

## DATA COLLECTION STANDARDIZATION

To ensure comparability across all participants, **all participants must use the same repository version** and must **not modify**:
- Collector logic (`frontend/src/lib/collector.js`)
- Session export (`frontend/src/lib/sessionExport.js`)
- Preprocessing (`ml/preprocessing/`)
- ML model code/weights/artifacts (`ml/`, `models/` — especially the three approved artifacts)
- Calibrated threshold (`models/verification_config.json`)
- Frontend collection/enrollment logic (`frontend/src/enrollment/`, `frontend/src/hooks/useCollector.js`)

### Privacy Guarantees (explicit)

- No passwords are collected
- No typed text is stored
- Actual key identities are not collected
- Only keyboard timing/event-type metadata and mouse movement/click metadata are collected
- Exported JSON is **not automatically uploaded** to any server
- Participants must keep the exported JSON and send it to the project owner
- Participants must **not** modify the dataset or ML model
