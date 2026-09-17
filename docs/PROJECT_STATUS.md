# Project Status — Behavioral Biometric Authentication

## Current Phase

**Phase 5 — CNN + GRU Behavioral Encoder**

## Status

In Progress

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
- See [`PHASE_4_DATASET_BASELINE.md`](PHASE_4_DATASET_BASELINE.md) for full details

## Phase 5 — In Progress

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
- **Awaiting manual verification — do NOT mark complete until verified.**

## Next Phase

**Phase 6 — Siamese Verification Network** (pair construction, contrastive /
triplet loss, training loop, similarity scoring, threshold calibration, neural
FAR/FRR evaluation)

## Development Rule

Each phase must be implemented, tested, reviewed, and verified before proceeding to the
next phase. This file is updated at the end of every phase.