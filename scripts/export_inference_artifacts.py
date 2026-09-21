#!/usr/bin/env python3
"""
Phase 9B — export the inference artifacts used by the ML service.

Writes two JSON artifacts under ``models/`` (git-ignored, runtime generated):

1. ``models/behavioral_preprocessing.json`` — the train-only keyboard/mouse
   ``FeatureScaler`` statistics plus the deterministic split provenance. The
   scalers are fitted by the SAME Phase 7 pipeline
   (:func:`ml.training.dataset.prepare_training_data`: seed 42, user-aware
   25/10/15 split) and are reused, never refitted, on incoming sessions.

2. ``models/verification_config.json`` — the calibrated decision threshold and
   provenance, produced by the SAME Phase 8 evaluation
   (:func:`ml.evaluation.evaluator.evaluate_verification`, target FAR 0.05).
   The threshold is calibrated on the DEVELOPMENT (validation) partition only.

Usage:
    .venv/bin/python scripts/export_inference_artifacts.py \
        --dataset data/datasets/dataset_synthetic.json \
        --checkpoint models/siamese_behavioral_encoder.pt \
        --seed 42 --target-far 0.05 \
        --preprocessing-out models/behavioral_preprocessing.json \
        --verification-out models/verification_config.json

Exit code 0 = success; 1 = execution error; 2 = usage/file error.

Guard-rail: both artifacts are derived from the SYNTHETIC development dataset.
They are engineering artifacts that validate the pipeline only; a real-world
deployment must regenerate scalers AND recalibrate the threshold from
production-grade enrolment data.
"""

import argparse
import datetime
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dataset import DATASET_SCHEMA_VERSION  # noqa: E402
from ml.evaluation import TARGET_FAR, evaluate_verification  # noqa: E402
from ml.training import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET,
    TrainingConfig,
    prepare_training_data,
)

PREPROCESSING_ARTIFACT_NAME = "behavioral-preprocessing"
VERIFICATION_CONFIG_ARTIFACT_NAME = "verification-config"
ARTIFACT_VERSION = 1

DEFAULT_PREPROCESSING_OUT = "models/behavioral_preprocessing.json"
DEFAULT_VERIFICATION_OUT = "models/verification_config.json"


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be in [0, 1]")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export the train-only preprocessing artifact and the "
        "calibrated verification config used by the Phase 9B ML service."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="persisted dataset path (default: %(default)s)")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Phase 7 checkpoint path (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42, help="deterministic split seed (default: %(default)s)")
    parser.add_argument("--target-far", type=_probability, default=TARGET_FAR,
                        help="calibration FAR budget (default: %(default)s)")
    parser.add_argument("--preprocessing-out", default=DEFAULT_PREPROCESSING_OUT,
                        help="output path for the preprocessing artifact (default: %(default)s)")
    parser.add_argument("--verification-out", default=DEFAULT_VERIFICATION_OUT,
                        help="output path for the verification config (default: %(default)s)")
    return parser


def _write_json(payload, path: str) -> str:
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return os.path.abspath(path)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if not os.path.exists(args.checkpoint):
        print("error: checkpoint '{}' does not exist".format(args.checkpoint), file=sys.stderr)
        return 2
    if not os.path.exists(args.dataset):
        print("error: dataset '{}' does not exist".format(args.dataset), file=sys.stderr)
        return 2

    config = TrainingConfig(
        dataset_path=args.dataset,
        seed=args.seed,
        checkpoint_path=args.checkpoint,
    )

    generated_at = datetime.datetime.now(datetime.timezone.utc).isoformat()

    print("exporting preprocessing artifact (train-only scalers) ...")
    try:
        data = prepare_training_data(config)
    except Exception as exc:  # noqa: BLE001 - CLI-level report
        print("error: could not materialise training data: {!r}".format(exc), file=sys.stderr)
        return 1

    n_users = len({s["user_id"] for s in data.train_sessions})
    split = data.session_count_summary()
    preprocessing = {
        "artifact": PREPROCESSING_ARTIFACT_NAME,
        "artifact_version": ARTIFACT_VERSION,
        "generated_by": os.path.basename(__file__),
        "generated_at": generated_at,
        "dataset_id": os.path.basename(config.dataset_path),
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "split": {
            "seed": config.seed,
            "train_fraction": config.train_fraction,
            "val_fraction": config.val_fraction,
            "n_users": n_users,
            "train_sessions": split["train_sessions"],
            "dev_sessions": split["val_sessions"],
            "test_sessions": split["test_sessions"],
        },
        "fitted_on": (
            "training partition only (seed 42, user-aware split); the scalers "
            "are reused - never refitted - for enrollment/verification probes"
        ),
        "method": "per-feature z-score standardization (FeatureScaler)",
        "feature_columns_reference": {
            "keyboard": list(data.keyboard_scaler.feature_columns or []),
            "mouse": list(data.mouse_scaler.feature_columns or []),
        },
        "checkpoint_id": os.path.basename(config.checkpoint_path),
        "keyboard_scaler": data.keyboard_scaler.state_dict(),
        "mouse_scaler": data.mouse_scaler.state_dict(),
    }
    try:
        preprocessing_path = _write_json(preprocessing, args.preprocessing_out)
    except Exception as exc:  # noqa: BLE001
        print("error: could not write preprocessing artifact: {!r}".format(exc), file=sys.stderr)
        return 1
    print("  written: {}".format(preprocessing_path))

    print("exporting verification config (development-partition calibration) ...")
    try:
        report = evaluate_verification(config, target_far=args.target_far)
    except Exception as exc:  # noqa: BLE001
        print("error: evaluation failed: {!r}".format(exc), file=sys.stderr)
        return 1

    cal = report["calibration"]
    print("  calibration: threshold={threshold:.6f} target_far={target_far:.4f} "
          "achieved_far={achieved_far:.4f} achieved_tar={achieved_tar:.4f} fallback={fallback}"
          .format(**cal))

    verification = {
        "artifact": VERIFICATION_CONFIG_ARTIFACT_NAME,
        "artifact_version": ARTIFACT_VERSION,
        "generated_by": os.path.basename(__file__),
        "generated_at": generated_at,
        "dataset_id": report["dataset_id"],
        "checkpoint_id": report["checkpoint_id"],
        "checkpoint_epoch": report["checkpoint_epoch"],
        "embedding_dim": report["encoder"]["embedding_dim"],
        "preprocessing_artifact": os.path.basename(args.preprocessing_out),
        "preprocessing_artifact_version": ARTIFACT_VERSION,
        "split": report["split"],
        "calibration": cal,
        "decision_rule": {
            "rule": "VERIFIED iff Phase 6 Euclidean L2 distance <= calibrated threshold",
            "threshold_source": "development (validation) partition only",
            "client_supplied_threshold": "not supported - always ignored",
        },
        "notes": (
            "synthetic-development artifact. Regenerate the preprocessing "
            "artifact and recalibrate the threshold from real enrolment data "
            "before any real-world deployment."
        ),
    }
    try:
        verification_path = _write_json(verification, args.verification_out)
    except Exception as exc:  # noqa: BLE001
        print("error: could not write verification config: {!r}".format(exc), file=sys.stderr)
        return 1
    print("  written: {}".format(verification_path))

    print("SYNTHETIC-data artifacts exported - these are NOT real-world biometric parameters.")
    return 0


if __name__ == "__main__":
    sys.exit(main())