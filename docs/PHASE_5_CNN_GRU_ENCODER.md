# Phase 5 — CNN + GRU Behavioral Encoder

## Goal

Convert the Phase 3/4 keyboard and mouse feature sequences into a **single
fixed-size behavioral embedding** that a future Phase 6 Siamese
verification network can compare across sessions. This phase delivers the
neural sequence encoder itself — **not** training, loss functions, similarity,
thresholding, or FAR/FRR evaluation (those are Phase 6).

## Input contract (reuse, don't reinvent)

The encoder consumes exactly the Phase 3 feature sequences and Phase 3
normalization statistics:

| modality | feature columns (Phase 3) | tensor shape |
|----------|---------------------------|--------------|
| keyboard | `hold_time`, `flight_time` | `[T_keyboard, 2]` |
| mouse    | `dx, dy, dt, distance, speed` | `[T_mouse, 5]` |

* Feature column definitions come from `ml.preprocessing.KEYBOARD_FEATURE_COLUMNS`
  and `MOUSE_FEATURE_COLUMNS` — one source of truth.
* Normalization reuses the Phase 3 `FeatureScaler` (`fit` on training data
  only, `transform` validation/test/inference) and is pluggable per modality.
* The encoder never sees key names/IDs, characters, passwords, or raw text —
  numeric timing/movement features only (privacy invariant enforced by the
  Phase 4 schema + tests).

## Architecture

```
 keyboard_sequence [T_k, 2]  mouse_sequence [T_m, 5]
        | pad batch             | pad batch
   Conv1d(2->C)            Conv1d(5->C)          C = conv_channels (32)
        | ReLU + Dropout        | ReLU + Dropout
   GRU over time (packed)      GRU over time (packed)   H = gru_hidden (64)
        | final hidden           | final hidden
   Linear(H -> 64)           Linear(H -> 64)
        |                        |
        +------ concat ----------+
                   \|/
        behavioral embedding [B, 128]
```

### Variable length

Batches are zero-padded to the longest sequence in the batch
(`collate_sequences` / `collate_sessions`). Each item's true length is carried
in a `lengths` vector, and `pack_padded_sequence` / `pad_packed_sequence`
(`enforce_sorted=False`, `batch_first=True`) runs the GRU **only over each
item's real timesteps**, so padding never influences the embedding.

### Empty-sequence semantics (documented contract)

Empty sequences are legal in a batch (a session may have no typing, or no mouse
movement):

* An item with an **empty keyboard or mouse sequence** is represented by a
  **zero embedding** for that modality — a well-defined "no evidence"
  representation, identical to training/encoding with a missing modality, never
  influenced by other items' data.
* A session with **no behaviour at all** (both sequences empty) is rejected
  with `ValueError`; use `ml.encoder.input.has_behavior` to filter sessions
  before encoding. This prevents a meaningless zero vector from ever being
  treated as a valid identity signature.

## Package layout

```
ml/encoder/
├── __init__.py      # public API re-exports
├── input.py         # sequence prep: validation, tensors, collation, scalers
└── cnn_gru.py       # EncoderConfig, ConvGruEncoder, BehavioralEncoder
```

### `ml.encoder.input`

* `sequence_tensor(sequence, feature_columns)` — one sequence → `[T, F]` float32
  tensor (empty → `[0, F]`).
* `collate_sequences(sequences, feature_columns)` — list of sequences → `PaddedBatch`.
* `collate_rows(rows, n_features)` — collate pre-scaled rows.
* `collate_sessions(sessions, keyboard_scaler, mouse_scaler)` — Phase 3/4
  sessions/entries → `ModalityBatch` (a modality is `None` only when *every*
  item is empty for it). Applies an optional fitted `FeatureScaler` per modality.
* `PaddedBatch(inputs [B,T,F], lengths [B])` / `ModalityBatch(keyboard, mouse)`.
* `has_behavior(session)` — True if either sequence is non-empty.

### `ml.encoder.cnn_gru`

* `EncoderConfig` — widths (keyboard/mouse features enforce Phase 3 column
  counts to catch pipeline drift), `conv_channels` (32), `conv_kernel` (3),
  `gru_hidden` (64), `embedding_dim` (64/modality → 128 combined), `dropout`
  (0.0).
* `ConvGruEncoder` — one modality: `Conv1d → ReLU → Dropout → GRU(packed) →
  Linear`. Items with length 0 never enter the network and return a zero
  embedding.
* `BehavioralEncoder` — fuses keyboard + mouse encoders by concatenation.
  `forward(keyboard=..., mouse=...)` takes `PaddedBatch | None`; convenience
  wrappers `encode_session` / `encode_entries` work directly on Phase 3/4
  sessions and accept fitted scalers.
* `set_seed(seed)` — `torch.manual_seed`, used for reproducible construction and
  inference. `BehavioralEncoder(seed=...)` also seeds before parameter init.

## Determinism

* Construction is reproducible: `BehavioralEncoder(seed=7)` (or seeding before
  construction) yields identical weights.
* Inference is reproducible: re-seed before encoding and identical inputs give
  byte-identical outputs (verified by tests).

## Verification

* `tests/test_encoder_input.py` — validation, tensor conversion, padding,
  scaler reuse, modality absence flags.
* `tests/test_encoder.py` — dimensions, variable-length/padding non-influence,
  batch forward, empty-modality zero-embedding semantics, determinism, gradient
  flow, identity independence (labels never influence embeddings), feature
  sensitivity, and end-to-end encoding of a scaled Phase 4 synthetic dataset and
  a real Phase 2 → Phase 3 session.
* Real-session smoke test: `behavioral-session-cafdb16d-...json`
  (55 keyboard presses, 571 mouse samples) → 128-dim embedding, finite,
  byte-identical across re-seeded runs.

## Dependencies (Phase 5)

```
torch>=2.14.0
numpy>=2.0.0
```

Install the smaller CPU build (no CUDA toolchain needed):

```
.venv/bin/pip install --index-url https://download.pytorch.org/whl/cpu torch numpy
```

## Out of scope (Phase 6)

Siamese/triplet loss, negative/positive pair construction, training loop,
optimizer, similarity scoring, threshold calibration, and TAR/FAR/FRR evaluation
for the neural model. The encoder API (fixed 128-dim vectors) is shaped so Phase 6
can consume it directly.