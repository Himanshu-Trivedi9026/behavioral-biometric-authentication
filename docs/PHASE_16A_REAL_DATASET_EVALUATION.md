# Phase 16A — Real Behavioral Dataset Evaluation

Status: pipeline built and tested · real-data collection in progress ·
real-world evaluation **not yet complete**
Table of contents
-----------------
1. What this phase is
2. Guard-rails
3. Artifacts produced
4. How to run
5. Interpreting the numbers (honesty statement)
6. Provenance & reproducibility
7. Validation performed

---

## 1. What this phase is

Phase 16A evaluates the **existing, deployed** biometric verification system
against **locally collected real behavioral data**. It is a
*measurement of the deployed system*, not a new model and not a recalibrated
threshold.

* **The checkpoint** (`models/siamese_behavioral_encoder.pt`, Phase 7 / Phase 8)
  is loaded **read-only**. No training, no fine-tuning.
* **The decision threshold**
  (`0.4635127782821655` from `models/verification_config.json`,
  calibrated on synthetic development data) is reused **unchanged**.
  No calibration and no recalibration is ever performed.
* The real development/test subset is measured only to show *how the existing
  system behaves on real data*.

### What it does NOT do
* Does **not** retrain the model.
* Does **not** calibrate or recalibrate the threshold.
* Does **not** tune anything to make real-data metrics look better.
* Does **not** write to `models/` or alter the checkpoint.
* Does **not** replace the synthetic dataset (synthetic data remains the
  canonical training/validation source).

---

## 2. Guard-rails (enforced by construction)

Reused, untouched existing components — exact same conventions, no new logic
invented for scoring:

| Concern | Mechanism |
|---|---|
| No recalibration path | `scripts/evaluate_real_behavior.py` imports only existing metric helpers; the calibration module is never imported. The threshold is a fixed constant. |
| Split/leakage | Reuses `dataset.split` per-user split and Phase 8 pairing conventions *inside each partition*; dev/test never mix subjects' identities across partitions. |
| Feature scaling | `ml.training.prepare_training_data` fits `FeatureScaler` on the real **train** partition only. |
| Read-only model | `load_verifier(checkpoint)` opens the checkpoint read-only; no weights are written. |
| No artifacts written | The script never writes `models/*`. Build script writes only the derived dataset JSON (a Phase 4 schema dataset). |

---

## 3. Artifacts produced

```
data/datasets/dataset_real.json         derived real dataset (Phase 4 schema)
results/real_behavior/
  real_behavior_report.json             machine-readable report
  real_behavior_report.txt              human-readable summary
```

Report structure (JSON) — top-level keys as actually emitted by
`scripts/evaluate_real_behavior.py`:

```
{
  "purpose": "Phase 16A real-behavioral evaluation of the EXISTING deployed system ...",
  "measurement": {
    "statement": "...no recalibration is performed in Phase 16A.",
    "synthetic_vs_real": "...",
    "recalibration_performed": false,
    "synthetic_presented_as_real": false
  },
  "dataset": {"id": "dataset_real.json", "n_sessions": S, "n_users": U,
              "sessions_per_user": {...}, "schema_version": "1.0.0", ...},
  "threshold": {"mode": "fixed deployed threshold (no recalibration)",
                "value": 0.4635127782821655, "provenance": "..."},
  "checkpoint": {"checkpoint_id": "...", "epoch": 10, "seed": 42,
                 "embedding_dim": 128, "encoder_type": "..."},
  "split": {"n_train_sessions": n, "n_dev_sessions": n, "n_test_sessions": n,
            "per_user_partition_counts": {...}, ...},
  "scaler": {"fitted_on": "training partition only", "reused_for": ["dev", "test"]},
  "dev":  {"pairs": {...}, "metrics_at_fixed_threshold": {...},
           "roc": [...], "auc": x, "eer": {...}, "distributions": {...}},
  "test": {"pairs": {...}, "metrics_at_fixed_threshold": {...},
           "roc": [...], "auc": x, "eer": {...}, "distributions": {...}},
  "leakage_checks": {"train_dev_disjoint": true, "train_test_disjoint": true,
                     "dev_test_disjoint": true,
                     "dev_pairs_within_dev_partition": true,
                     "test_pairs_within_test_partition": true}
}
```

Note: `metrics_at_fixed_threshold` and `distributions` live **inside** each of the
`dev` / `test` partition blocks (not at the top level), and ROC/AUC/EER are
reported per partition.

---

## 4. How to run

Participant-facing collection and export instructions (including "do not click
Send … Session(s) to Server during Phase 16A research collection") live in
[`README.md`](../README.md) under **TEAM MEMBER SETUP** and **PHASE 16A
COLLECTION INSTRUCTIONS**. This document covers the owner-side dataset build
and evaluation only.

```bash
# 1) Build the real dataset from raw session JSONs
.venv/bin/python scripts/build_real_dataset.py \
    --raw-dir data/raw                                  \
    --manifest data/raw/participant_manifest.json      \
    --output data/datasets/dataset_real.json

# 2) Evaluate the deployed system against real data (fixed threshold, no recalibration)
.venv/bin/python scripts/evaluate_real_behavior.py \
    --dataset data/datasets/dataset_real.json     \
    --checkpoint models/siamese_behavioral_encoder.pt \
    --output-dir results/real_behavior

# 3) Regression tests
.venv/bin/python -m pytest tests/test_real_dataset_pipeline.py -q
```

Exit codes: `0` success, `1` execution/validation error, `2` usage or
missing-file error.

---

## 5. Interpreting the numbers (honesty statement)

> **Evaluation uses the existing synthetic-calibrated threshold;
> no recalibration is performed in Phase 16A.**

The checkpoint and the deployed threshold were produced from **synthetic
development identities**. The numbers in this report therefore reflect *how
that existing system behaves on real locally-collected data* — this is a
real-world measurement of the **deployed** system. They are **not** a
tuned/rescaled result.

Do not read the real-data numbers as an endorsement that anything "passes":
they answer only "what does the current system do when shown real data?".

---

## 6. Provenance & reproducibility

* `build_real_dataset.py` processes **only** the sessions explicitly listed in
  `data/raw/participant_manifest.json` (it warns on, and skips, any unlisted
  raw file) and emits `data/datasets/dataset_real.json` in the existing Phase 4
  schema. Reruns with identical raw inputs are deterministic. Provenance comes
  from the Phase 4 loader (`metadata.generated=false` plus the existing schema
  validation and `assert_privacy` check), not from a separate filename manifest.
* `evaluate_real_behavior.py` states threshold provenance
  (`models/verification_config.json`) and reuses the deployed threshold in the
  report's `threshold.value` and `threshold.provenance`.
* Raw behavioral data stays under `data/raw/` (gitignored); only the derived
  dataset JSON and the JSON/TXT reports are Phase 16A artifacts.

---

## 7. Validation performed

* Full backend test suite: **passed**.
* Phase 16A pipeline tests: `tests/test_real_dataset_pipeline.py`
  (leakage checks, threshold fixedness, determinism, provenance).
* Byte-compilation of scripts/tests: clean.
* `git diff --check`: clean.
* Checkpoint & `models/verification_config.json` unchanged (read-only).
