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

## Planned Technology Stack

| Layer            | Technology                              |
|------------------|------------------------------------------|
| Frontend         | React + Vite (JavaScript / CSS)                       |
| Backend          | FastAPI                                  |
| Machine Learning | Python + PyTorch                         |
| Data Processing  | NumPy + Pandas                           |
| Sequence Encoder | 1D-CNN (optional) + GRU                  |
| Verification     | Siamese Network + Contrastive Loss       |
| Database         | PostgreSQL                               |
| Session Auth     | JWT                                      |
| Caching          | Redis (later if required)                |
| Deployment       | Docker (later)                           |

---

## High-Level Architecture (Planned Pipeline)

The pipeline below is **documentation only** — its components are implemented in later
phases, not in this setup phase.

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

---

## Development Phases

1. **Phase 1 — Project Setup** (`completed`)
2. **Phase 2 — Browser-based Keyboard and Mouse Data Capture** (`completed`)
3. **Phase 3 — Data Processing and Feature Extraction** (`completed`)
4. **Phase 4 — Dataset Creation + Statistical (Non-Neural) Baseline** (`current`)
5. **Phase 5 — ML Model: Sequence Encoder (1D-CNN + GRU)**
6. **Phase 6 — ML Model: Siamese Network + Contrastive Loss (Few-Shot Verification)**
7. **Phase 7 — Backend API (FastAPI), Database (PostgreSQL), Auth (JWT), Redis**
8. **Phase 8 — Frontend Integration**
9. **Phase 9 — Testing, Deployment (Docker) and Documentation**

---

## Development Rule

**Each phase must be implemented, tested, reviewed, and verified before proceeding to the
next phase.**

- Implement only the scope defined for the current phase.
- Add tests and run automated checks for everything added.
- Review code quality, security, and correctness before moving on.
- Verify the environment and structure before declaring a phase complete.
- No phase is considered "done" until its stated deliverables are checked and confirmed.

Detailed phase-by-phase progress is tracked in [`docs/PROJECT_STATUS.md`](docs/PROJECT_STATUS.md).

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