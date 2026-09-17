# Phase 3 — Data Preprocessing and Feature Extraction

Converts a raw Phase 2 session (browser-collected keyboard/mouse event JSON) into
machine-learning-ready **behavioral sequences**. Keyboard and mouse are kept as **separate**
sequences; no model, tensors, or fusion are built in this phase.

**Phase 3 status:** In Progress (see [`PROJECT_STATUS.md`](PROJECT_STATUS.md)).

---

## 1. Purpose

Phase 2 captures raw behavioral **metadata**: `keydown`/`keyup` timing and mouse position
samples over time. Phase 3 turns that raw stream into compact, numerical features that
describe **how** a person types and moves — not **what** they type or click on.

The pipeline is deliberately simple and explainable; it produces input shapes suitable for a
later sequence model (e.g. 1D-CNN + GRU / Siamese network) without constructing one here.

## 2. Package Layout

```
ml/
  preprocessing/
    __init__.py       # public API re-exports
    validation.py     # raw-session validation + SessionValidationError
    keyboard.py       # keyboard preprocessing + hold/flight features
    mouse.py          # mouse preprocessing + trajectory/speed features
    normalization.py  # FeatureScaler (fit/transform/fit_transform) + row helpers
    pipeline.py       # process_session() one-call pipeline
```

Implementation is **pure Python (stdlib only)** — no third-party runtime dependencies. Tests
use `pytest` (declared in `requirements.txt`).

## 3. Raw Input Schema (unchanged from Phase 2)

```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "started_at": "2025-09-01T12:00:00.000Z",
  "ended_at":   "2025-09-01T12:00:05.000Z",
  "timestamp_source": "monotonic high-resolution (performance.now, milliseconds)",
  "keyboard_events": [
    { "event_type": "keyboard", "event": "keydown", "timestamp": 100.0 }
  ],
  "mouse_events": [
    { "event_type": "mouse", "event": "mousemove",
      "x": 100.0, "y": 200.0, "timestamp": 110.0 }
  ]
}
```

Validation (`validate_session`) checks, field by field, that:

- `session_id`, `started_at`, `ended_at` exist and are non-empty strings
- `keyboard_events` / `mouse_events` exist and are lists
- every keyboard event has `event_type == "keyboard"`, an `event` of
  `keydown`/`keyup`, and a finite numeric `timestamp`
- every mouse event has `event_type == "mouse"`, an `event` of
  `mousemove`/`mousedown`/`mouseup`, finite numeric `x`, `y`, and `timestamp`
- timestamps/coordinates are finite (no `NaN`/`Inf`); booleans are rejected

`validate_session` returns a list of error strings (`[]` = valid) and never mutates the
input. `process_session` raises `SessionValidationError` (carrying the individual errors) on
invalid input rather than silently accepting it.

## 4. Keyboard Features

Events are processed in chronological order (stable sort by timestamp). Each key press is a
`keydown` → `keyup` pair.

**Pairing rule (event-order based, no key identity used):** `keydown`s are pushed onto a LIFO
stack; each `keyup` pops the most recent unmatched `keydown`. This correctly resolves
interleaved/nested presses even though the collector stores no key names.

| Feature      | Definition                                            | Unit |
|--------------|-------------------------------------------------------|------|
| `hold_time`  | `keyup.timestamp − keydown.timestamp`                  | ms   |
| `flight_time`| `current keydown.timestamp − previous keydown.timestamp` | ms   |

`flight_time` for the **first** press in a session has no predecessor; it is set to `0.0`
(a documented, deterministic placeholder) so the sequence stays fully numeric.

**Malformed-event handling (never aborts the pipeline; each is counted in metadata):**

| Situation                     | Handling                          |
|-------------------------------|-----------------------------------|
| unmatched `keyup` (empty stack) | dropped (`dropped_unmatched_keyup`) |
| unmatched `keydown` left at end | dropped (`dropped_unmatched_keydown`) |
| negative `hold_time`            | press dropped (`dropped_negative_hold`) |
| exact consecutive duplicate     | dropped (`dropped_duplicates`)      |

Each output sample is `{"hold_time": float, "flight_time": float}`.

## 5. Mouse Features

`mousemove` events form the **trajectory sequence** (chronological order). For each pair of
consecutive positions:

| Feature     | Definition                      | Unit   |
|-------------|---------------------------------|--------|
| `dx`        | `x1 − x0`                       | px     |
| `dy`        | `y1 − y0`                       | px     |
| `dt`        | `t1 − t0`                       | ms     |
| `distance`  | `sqrt(dx² + dy²)`               | px     |
| `speed`     | `distance / dt`                 | px/ms  |

**Safe rule for `dt <= 0`:** speed is undefined without a positive time span, so `speed` is
set to `0.0` and the sample is retained (`dx`,`dy`,`dt`,`distance` stay inspectable).
Because events are sorted chronologically, a negative `dt` cannot be produced; `dt == 0`
(two moves at the same timestamp) is the only `dt <= 0` case in practice.

Shape note: the first `mousemove` is a position sample with no predecessor, so the trajectory
sequence has `(move points) − 1` samples.

`mousedown`/`mouseup` are **preserved verbatim** (chronological order) as
`mouse_action_events` for later modeling. No categorical encoding is introduced in this
phase. No screen/browser identity information is used.

Each trajectory sample is
`{"dx": float, "dy": float, "dt": float, "distance": float, "speed": float}`.

## 6. Normalization Strategy

`FeatureScaler` performs per-feature **standardization (z-score)**:

```
standardized = (value − mean_feature) / std_feature
```

- `fit(rows)` computes per-feature mean/std from **training data only**.
- `transform(rows)` standardizes using the stored statistics; `fit_transform(rows)` does both.
- **Constant features** (std == 0) are stored as `1.0`, so every transformed value becomes
  `0` and division by zero is impossible.
- `state_dict()` / `FeatureScaler.load_state_dict()` serialize the fitted statistics as
  plain JSON-compatible lists, so training statistics can later be reused on
  validation/test/inference data. **No statistics are hard-coded.**
- `sequence_to_rows` / `rows_to_sequence` convert a feature sequence (list of dicts) to an
  `(n_samples, n_features)` row matrix (and back), matching `KEYBOARD_FEATURE_COLUMNS`
  (`hold_time`, `flight_time`) and `MOUSE_FEATURE_COLUMNS`
  (`dx`, `dy`, `dt`, `distance`, `speed`).

## 7. Processed Output Schema

`process_session(raw_session)` returns:

```json
{
  "session_id": "...",
  "source": {
    "started_at": "...",
    "ended_at": "...",
    "timestamp_source": "...",
    "raw_keyboard_event_count": 6,
    "raw_mouse_event_count": 5
  },
  "preprocessing": {
    "version": "1.0.0",
    "keyboard": { "input_events": 6, "valid_presses": 3,
                  "dropped_unmatched_keyup": 0, "dropped_unmatched_keydown": 0,
                  "dropped_negative_hold": 0, "dropped_duplicates": 0 },
    "mouse": { "input_events": 5, "move_points": 3, "move_samples": 2,
               "action_events": 2, "dropped_duplicates": 0 }
  },
  "keyboard_sequence": [
    { "hold_time": 60.0, "flight_time": 0.0 },
    { "hold_time": 50.0, "flight_time": 120.0 }
  ],
  "mouse_sequence": [
    { "dx": 10.0, "dy": 20.0, "dt": 25.0, "distance": 22.36, "speed": 0.894 }
  ],
  "mouse_action_events": [
    { "event": "mousedown", "x": 130.0, "y": 225.0, "timestamp": 166.0 }
  ]
}
```

`source` preserves the traceability metadata; the `preprocessing` block records how many raw
events were consumed/dropped. Keyboard and mouse remain **separate sequences**; no tensors or
neural-network inputs are created. The pipeline is deterministic and never mutates raw input.

## 8. Privacy Invariant

Raw keyboard samples contain **timing only**. The pipeline never reads, reconstructs, or
emits key names, characters, passwords, input values, text content, or screen/browser
identity. Test `R` enforces this on the output.

## 9. How to Run the Tests

Prerequisites:

```bash
.venv/bin/pip install -r requirements.txt   # installs pytest
```

Run only the Phase 3 tests:

```bash
.venv/bin/python -m pytest tests/test_preprocessing.py -v
```

Run the full Python suite:

```bash
.venv/bin/python -m pytest -v
```

The Phase 2 collector tests (Node) are unaffected:

```bash
cd frontend && npm test
```

Synthetic fixtures live in `tests/fixtures/` (generated data only — no real collected
sessions are committed).

## 10. Quick API Example

```python
from ml.preprocessing import process_session, FeatureScaler, sequence_to_rows
from ml.preprocessing.keyboard import KEYBOARD_FEATURE_COLUMNS

raw = { ... }                       # a saved Phase 2 session dict
processed = process_session(raw)    # validate + preprocess + features

scaler = FeatureScaler(KEYBOARD_FEATURE_COLUMNS)
scaler.fit(sequence_to_rows(train_sessions_keyboard, KEYBOARD_FEATURE_COLUMNS))
normalized = scaler.transform(sequence_to_rows(processed["keyboard_sequence"],
                                               KEYBOARD_FEATURE_COLUMNS))
```