# Phase 4 — Dataset Creation + Non-Neural Baseline

This phase adds a deterministic **dataset layer** (multi-user, multi-session,
schema-compatible with Phase 3) and a simple **statistical, distance-based
identity-verification baseline** — with **no neural networks, no tensors, and
no machine-learning frameworks** (Python standard library only).

---

## 1. Dataset schema

Every **dataset entry** is one identity-verification sample:

```json
{
  "user_id": "user_001",
  "session_id": "syn-user_001-session-0000",
  "keyboard_sequence": [
    {"hold_time": 60.0, "flight_time": 0.0},
    {"hold_time": 55.0, "flight_time": 150.0}
  ],
  "mouse_sequence": [
    {"dx": 10.0, "dy": 5.0, "dt": 20.0, "distance": 11.18, "speed": 0.56}
  ],
  "mouse_action_events": [
    {"event": "mousedown", "x": 10.0, "y": 10.0, "timestamp": 100.0}
  ],
  "metadata": {
    "generated": true,
    "seed": 0,
    "source": null,
    "preprocessing": {"version": "1.0.0"},
    "keyboard_samples": 2,
    "mouse_samples": 1
  }
}
```

- `user_id` is an **explicit dataset-level identity label** (added by Phase 4;
  it is never present in the Phase 2/3 data because Phase 2 never records who
  is typing).
- Core structure mirrors the Phase 3 processed-session layout so that real
  Phase 3 output can be used unmodified (only the `user_id` envelope is added).
- Keyboard and mouse stay **separate modalities**.

### Feature columns

| Sequence             | Features                                              |
|----------------------|-------------------------------------------------------|
| `keyboard_sequence`  | `hold_time`, `flight_time`                            |
| `mouse_sequence`     | `dx`, `dy`, `dt`, `distance`, `speed`                |
| `mouse_action_events`| preserved `mousedown` / `mouseup` (verbatim)          |

### Privacy contract

The dataset contains **numerical timing/geometry values only**. No typed
characters, key names, key codes, passwords, or raw text input. This is
enforced by `dataset.schema.assert_privacy()` (walks every key, scans the
serialized form) and verified by the automated test suite for both synthetic
and real processed sessions.

---

## 2. Synthetic / demo dataset generation

`dataset/generator.py` — `SyntheticDatasetGenerator`.

Why synthetic? Phase 2 collection needs a live browser, so only a handful of
manually collected sessions exist. The generator produces a deterministic,
behaviorally plausible dataset for development/testing.

### Determinism

Each user and session is generated with a dedicated `random.Random` seeded by
a fixed string (`f"syn:{seed}:user:{i}"`, `f"syn:{seed}:user:{i}:session:{j}"`).
`random.Random(str)` is deterministic across runs and processes, so the same
seed always regenerates the **byte-identical** dataset.

### Behavioural plausibility

- Each user gets a stable per-user "typing profile": their own `hold_time` /
  `flight_time` distributions (some users type faster, some slower).
- Each user gets a mouse profile: step sizes (`dx`/`dy`), speed, and sample
  intervals (`dt` ~ 16–40 ms, like the Phase 2 collector's 25 ms sampling),
  occasional long jumps and pauses.
- Sessions are freshly sampled from the user's distribution (never duplicated
  copies), producing realistic within-user and across-user variation.

### Default scale (rationale)

`seed=0, n_users=5, sessions_per_user=10` → **50 sessions**. With the default
70/30 per-user split that gives 7 enrolment + 3 evaluation sessions per user:
15 genuine + 60 impostor held-out attempts — enough for a meaningful baseline
while staying fully deterministic and stdlib-only. The scale is configurable
(`--n-users`, `--sessions-per-user`).

---

## 3. Real Phase 2 data support

The loader supports the full pipeline **without touching Phase 2/3 code**:

```
raw Phase 2 JSON
    ↓  ml.preprocessing.process_session   (unchanged Phase 3)
processed session
    ↓  dataset.loader.processed_to_entry (adds user_id; sequences passed through)
Phase 4 dataset entry
```

- `load_processed_dir` / `load_processed_files` — load already-processed
  Phase 3 sessions.
- `load_raw_files` / `raw_to_entry` — load raw Phase 2 JSON (runs Phase 3).
- `save_dataset` / `load_dataset` — versioned JSON persistence.

Real sessions need a `user_id_map` (dict `session_id -> user_id`, or a
callable) because the collector never records identity.

---

## 4. Train / test split

`dataset/split.py` — `split_train_test(entries, seed, train_fraction, min_train, min_test)`.

- **Per-user split**: each user's sessions are split independently so every
  user keeps enrolment/training sessions AND separate evaluation sessions.
- **No leakage**: each `session_id` appears in exactly one split; the
  test suite calls `assert_no_leakage` to enforce it.
- **Deterministic**: sessions are stably sorted by `session_id`, then shuffled
  with a per-user RNG derived from the seed (`f"split:{seed}:{user_id}"`).
- Counts: default `train_fraction=0.7`, `min_train=1`, `min_test=1`.

Default split for 10 sessions/user → 7 train + 3 test per user.

---

## 5. Baseline algorithm (non-neural)

### Enrollment profile (`baseline/profile.py`)

For each user, all feature values across that user's **training sessions** are
pooled and summarized per feature as `(mean, std)` — for keyboard
(`hold_time`, `flight_time`) and mouse (`dx, dy, dt, distance, speed`)
separately. This is a pure-statistics fingerprint; no model is trained.

### Scoring (`baseline/scoring.py`)

**Score direction**: scores are *distances* — `0` = identical to the profile,
larger = less similar.

Per feature `f` (e.g. `hold_time`), the session's pooled values are compared
to the profile via mean absolute standardized deviation:

```
d_f = mean( |v - mean_f| / std_f   for v in the session's values of f )
```

- `keyboard_score` = mean of `d_f` over keyboard features.
- `mouse_score` = mean of `d_f` over mouse features.
- `combined_score` = weighted mean (default 0.5 keyboard + 0.5 mouse;
  configurable). If one modality has no data, the combined score falls back to
  the available modality; if both are empty, it is `None`.

Different sequence lengths are handled naturally (each term is an average over
whatever samples exist). Constant features (`std == 0`) contribute `0` — the
code never divides by zero.

### Verification decision

```
accept  if  combined_score <= threshold
```

`threshold` is a **maximum allowable distance**. Larger threshold = more
permissive.

---

## 6. Threshold strategy

- **Configurable** — never hard-coded. Pass `--threshold` explicitly, or let
  the evaluator calibrate it.
- **Calibration uses training/development data ONLY** (`baseline/evaluator.py`):

  * development genuine distances = leave-one-out: each enrolment session scored
    against its user's profile built from the *other* enrolment sessions,
  * development impostor distances = each enrolment session scored against every
    other user's profile.

  Two methods:
  - `target-far` (default): most-permissive threshold whose development impostor
    FAR is still ≤ `far_target` (default 5%). If the budget cannot be met, falls
    back to the strictest observed threshold.
  - `max-tar-far`: threshold maximizing `TAR − FAR` (Youden index) on
    development data, ties resolved toward the stricter threshold.

The **test/held-out set is never used to choose a threshold** — the calibration
function only receives training entries.

---

## 7. Evaluation (`baseline/evaluator.py`)

- **Genuine attempts**: query session scored against its TRUE owner's profile.
- **Impostor attempts**: the same query session scored against every OTHER
  user's profile (i.e. it claims another identity).
- Metrics at a threshold: **TAR** (genuine acceptance), **FAR** (false
  acceptance), **FRR** (= 1 − TAR), **accuracy**, plus confusion-matrix counts
  (TP/FP/FN/TN). Attempts with no usable modality are excluded and reported as
  "unscorable".
- **ROC**: `sweep_thresholds` produces threshold/FAR/TAR points (distance-based
  sweep), serializable for later plotting. The structure is model-agnostic and
  reusable once the neural model arrives in Phase 5.

Representative numbers (default seed, held-out split): calibrated threshold
≈ 0.80 → TAR = 1.00, FAR = 0.05, FRR = 0.00, accuracy = 0.96.

---

## 8. Files

| Path                    | Purpose                                    |
|-------------------------|--------------------------------------------|
| `dataset/__init__.py`   | public API                                 |
| `dataset/schema.py`     | schema validation + privacy checks         |
| `dataset/generator.py`  | deterministic synthetic generator          |
| `dataset/loader.py`     | real-session loading + persistence         |
| `dataset/split.py`      | per-user train/test split                  |
| `baseline/__init__.py`  | public API                                 |
| `baseline/profile.py`   | per-user statistical profiles              |
| `baseline/scoring.py`   | keyboard / mouse / combined distances      |
| `baseline/evaluator.py` | calibration, metrics, ROC, reports         |
| `scripts/generate_dataset.py` | dataset CLI                         |
| `scripts/evaluate_baseline.py`| evaluation CLI                      |
| `tests/test_dataset.py`  | dataset tests                              |
| `tests/test_baseline.py` | baseline tests                             |

---

## 9. Manual verification commands

```bash
# 1. Phase 4 tests
.venv/bin/python -m pytest tests/test_dataset.py tests/test_baseline.py -v

# 2. Full Python suite (Phases 1-4)
.venv/bin/python -m pytest

# 3. Phase 2 (unchanged) Node tests
cd frontend && npm test && cd ..

# 4. compileall + import check
.venv/bin/python -m compileall -q ml dataset baseline scripts
.venv/bin/python -c "import dataset, baseline"

# 5. Dataset generation CLI
.venv/bin/python scripts/generate_dataset.py --out /tmp/ds.json --seed 0

# 6. Deterministic repeatability
.venv/bin/python scripts/generate_dataset.py --out /tmp/ds2.json --seed 0
cmp /tmp/ds.json /tmp/ds2.json        # must be identical

# 7. Baseline evaluation CLI (calibrated threshold)
.venv/bin/python scripts/evaluate_baseline.py --dataset /tmp/ds.json --report-out /tmp/report.json

# 8. Baseline with explicit threshold
.venv/bin/python scripts/evaluate_baseline.py --dataset /tmp/ds.json --threshold 1.0

# 9. Real Phase 2 -> Phase 3 -> Phase 4 loading
.venv/bin/python scripts/evaluate_baseline.py --dataset data/datasets/dataset_synthetic.json
```

---

## 10. Limitations

- Synthetic data is statistically plausible but is **not** real behaviour; real
  collected sessions should replace/augment it before final model work.
- The baseline pools per-feature values, so it ignores *ordering* within a
  session (a deltapoint a sequence model would use). That is exactly why
  Phase 5 (CNN + GRU encoder + Siamese verification) builds on top of it.
- Only **numeric timing/geometry** features carry signal; there is no semantic
  context (deliberately — privacy).
- Performance numbers above are for the synthetic dataset, **not** evidence
  about real-world discrimination power.

## 11. Why a baseline rather than the final model

This baseline offers an end-to-end, deterministic, verifiable reference point
(profiles, threshold calibration, TAR/FAR/FRR, ROC) with **no ML framework
dependency**. It (a) validates the dataset/split/evaluation plumbing, (b) gives
a floor that the Phase 5 neural encoder must beat, and (c) provides a reusable
evaluation harness for the later model — without hiding any threshold or metric
inside opaque code.