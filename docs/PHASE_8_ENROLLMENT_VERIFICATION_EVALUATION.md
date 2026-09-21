# Phase 8 — Enrollment + Verification Evaluation

## 1. Objective

Turn the Phase 7 trained behavioural encoder into an **enrollment /
verification evaluation layer**:

1. **Enrollment** — average a user's reference sessions into a single
   centroid embedding (`ml/evaluation/enrollment.py`).
2. **Verification** — match a probe session against a profile with the Phase 6
   L2 distance and decide `VERIFIED` / `SUSPICIOUS` when `distance <= threshold`.
3. **Threshold calibration** — pick one decision threshold from the
   **development partition only** against a target FAR budget
   (`ml/evaluation/calibration.py`).
4. **Evaluation** — report TAR / FAR / FRR, confusion-matrix counts, ROC
   points, ROC AUC and EER on the isolated test partition
   (`ml/evaluation/metrics.py`, `ml/evaluation/evaluator.py`).

The model is **never re-trained** and the Phase 7 checkpoint
(`models/siamese_behavioral_encoder.pt`) is **never modified**; all inference
runs under `torch.no_grad()` in `eval()` mode.

## 2. Dataset Used

`data/datasets/dataset_synthetic.json` (Phase 4 artifact) — the same persisted
artifact Phase 7 trained on. Partitions are materialised deterministically by
`ml.training.dataset.prepare_training_data` (same seed → same 25/10/15
train/dev(validation)/test split and the same train-only `FeatureScaler`
statistics as Phase 7).

## 3. Enrollment Model

A profile is the **mean embedding (centroid)** of the user's reference
sessions:

```
profile.centroid = mean_i( embed(session_i) )
```

- `enroll(sessions, verifier, scalers, user_ref)` -> `EnrollmentProfile`
  (requires ≥ 1 behavioural session; embeddings must be finite).
- `verify(profile, probe, verifier, threshold, scalers)` -> `VerificationResult`
  with `decision = "VERIFIED" if distance(profile.centroid, probe) <= threshold
  else "SUSPICIOUS"`.
- `user_id` / `session_id` are **labels/reporting metadata only** — they never
  enter any network input (privacy invariant unchanged).

## 4. Verification Rule

Phase 4/6/7 convention, kept unchanged:

| Quantity | Meaning | At threshold `t` |
|----------|---------|-------------------|
| genuine pair | same user sessions | label `1` |
| impostor pair | different-user sessions | label `0` |
| distance | L2 `sqrt(sum((e_a-e_b)^2) + ε)`, ε = 1e-8 | lower = more similar |
| accept | `distance <= t` | `VERIFIED` |

## 5. Threshold Calibration (Development Only)

`calibrate_threshold(genuine_distances, impostor_distances, target_far=0.05)`
scans the unique candidate distances and keeps the **largest** threshold whose
achieved FAR ≤ target (maximises TAR under the FAR budget; both rates are
monotone in `t`). If no candidate meets the budget, the strictest observed
threshold is returned as the closest feasible fallback (`fallback=True`).

Calibration input comes **only from the development (validation) partition**;
the 15 test sessions never influence `threshold`.

## 6. Evaluation Design

| Step | What | Sessions involved |
|------|------|-------------------|
| partitions | deterministic user-aware split | 25 train / 10 dev / 15 test |
| dev pairs | all genuine + all impostor candidates of the dev partition | 5 + 40 |
| test pairs | all genuine + all impostor candidates of the test partition | 15 + 90 |
| embeddings | read-only checkpoint, scalers from train partition, cached per session | 100% probe-free scalers |
| distances | Phase 6 L2 over the cached embeddings | — |
| calibration | dev distances only | `threshold`, achieved FAR/TAR |
| test report | calibrated threshold + ROC/AUC/EER | test distances only |

Subsampling caps (`--max-dev-impostor`, …) exist for fast runs; defaults use
**every** candidate so the report reflects full capacities.

## 7. Results (seed 42, target FAR 0.05)

```
synthetic users: 5 | sessions: train 25 / dev 10 / test 15
dev pairs:    5 genuine + 40 impostor
test pairs:  15 genuine + 90 impostor

calibration (development only):
  threshold   = 0.463513
  target_far  = 0.0500 | achieved_far = 0.0500 | achieved_tar = 1.0000
  fallback    = False

test @ calibrated threshold:
  TAR = 0.6000 | FAR = 0.0000 | FRR = 0.4000 | accuracy = 0.9429
  confusion: tp=9 tn=90 fp=0 fn=6
ROC AUC (test) = 0.9474
EER  (test)    = 0.1833 at threshold 0.891864

enroll+verify smoke (user_001 enrolled from 2 dev sessions):
  genuine  dist=0.834520 -> SUSPICIOUS   (honest rejection at strict threshold)
  impostor dist=2.112918 -> SUSPICIOUS
```

Two back-to-back CLI runs produced byte-identical JSON reports
(**determinism PASS**).

## 8. Data-Leakage Prevention

- Split **first**, pair **after** — a pair can never straddle two partitions.
- Dev/test pair session ids are asserted disjoint from every other partition.
- The `FeatureScaler` (keyboard + mouse) is fitted **once, on the 25 training
  sessions** (`prepare_training_data`), and reused for dev/test sessions; probe
  sessions never touch scaler statistics.
- The test partition sees **no** training, calibration, or scaler fitting.
- `test_verification_evaluation.py` asserts partition disjointness and reports
  the `leakage_guard` string.

## 9. Privacy

`user_id` and `session_id` appear only as labels / report metadata. The
encoder consumes only the Phase 3 numeric features (`hold_time`, `flight_time`;
`dx, dy, dt, distance, speed`). The evaluation report contains aggregate
distances and counts only — no raw events, keys, characters, or text.

## 10. CLI

```
.venv/bin/python scripts/evaluate_verification.py \
    --dataset data/datasets/dataset_synthetic.json \
    --checkpoint models/siamese_behavioral_encoder.pt \
    --target-far 0.05 --seed 42 [--report models/evaluation_report.json]
```

`--report` writes the JSON-serialisable evaluation report (ROC points
included). Exit codes: `0` success, `1` execution error, `2` usage/file error.

## 11. Tests

- `tests/test_metrics.py` — 18 (confusion counts, ROC endpoints/structure,
  AUC = rank statistic on random data, EER, single-class guards).
- `tests/test_evaluation_model.py` — 13 (embedding shape/determinism,
  parameter invariance under inference, distance symmetry, checkpoint
  error paths).
- `tests/test_enrollment.py` — 14 (centroid correctness, decision rule,
  boundary equality, error paths; one real-model integration test).
- `tests/test_verification_evaluation.py` — 25 (calibration units, pair
  capacities, report structure, self-consistency, JSON round-trip,
  determinism, leakage guard).

Phase 8: **70/70 new**; full Python suite **453/453** passed; frontend
**19/19** passed; `compileall` + import checks PASS.

## 12. Interpretation Guard-rail

All numbers come from **synthetic generated identities** (5 users, 50
sessions). They validate the enrollment/verification/evaluation pipeline and
are **NOT real-world biometric performance** — no accuracy claims are made.

## 13. Scope Boundaries

- No re-training; the checkpoint is read-only.
- Distance threshold is produced by calibration; the verification layer never
  invents thresholds on its own.
- This phase covers enrollment + verification + threshold calibration + ROC /
  AUC / EER evaluation only. Serving, real-time authentication, UI integration
  and model improvement are out of scope and belong to later phases.