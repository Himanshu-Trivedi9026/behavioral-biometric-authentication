#!/usr/bin/env python3
"""
Phase 8 — Enrollment + Verification evaluation CLI.

Usage:
    .venv/bin/python scripts/evaluate_verification.py \
        --dataset data/datasets/dataset_synthetic.json \
        --checkpoint models/siamese_behavioral_encoder.pt \
        --target-far 0.05 --seed 42

Pipeline (in order):
    persisted dataset -> user-aware 25/10/15 train/dev(validation)/test split
    -> uncontaminated dev & test genuine/impostor pairs -> Phase 7 checkpoint
    (read-only embeddings) -> L2 pair distances -> threshold calibrated on the
    DEVELOPMENT partition only -> test-set TAR/FAR/FRR, ROC, AUC, EER
    -> optional JSON report + an enroll-and-verify smoke demo.

Leakage guard: the test partition is never used for calibration; scale
parameters are re-fitted on the training partition (same as Phase 7), so probe
sessions never touch the scalers.

Interpretation guard-rail: numbers come from a SYNTHETIC dataset of generated
identities. They validate the evaluation pipeline only and are NOT real-world
biometric performance.

Exit code 0 = success; 1 = execution error; 2 = usage/file error.
"""

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ml.evaluation import (  # noqa: E402
    TARGET_FAR,
    enroll,
    evaluate_verification,
    load_verifier,
    save_evaluation_report,
    verify,
)
from ml.training import (
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET,
    TrainingConfig,
    prepare_training_data,
)


def _probability(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be in [0, 1]")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate enrollment + verification on the Phase 7 checkpoint, "
        "calibrating the decision threshold on development data only."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="persisted dataset path (default: %(default)s)")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="Phase 7 checkpoint path (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42, help="deterministic seed (default: %(default)s)")
    parser.add_argument("--val-fraction", type=_probability, default=2 / 7,
                        help="fraction of train sessions used as development (default: %(default)s)")
    parser.add_argument("--target-far", type=_probability, default=TARGET_FAR,
                        help="calibration FAR budget (default: %(default)s)")
    parser.add_argument("--max-dev-genuine", type=_non_negative_int, default=None,
                        help="cap dev genuine pairs (None = all)")
    parser.add_argument("--max-dev-impostor", type=_non_negative_int, default=None,
                        help="cap dev impostor pairs (None = all)")
    parser.add_argument("--max-test-genuine", type=_non_negative_int, default=None,
                        help="cap test genuine pairs (None = all)")
    parser.add_argument("--max-test-impostor", type=_non_negative_int, default=None,
                        help="cap test impostor pairs (None = all)")
    parser.add_argument("--report", default=None,
                        help="write the JSON evaluation report to this path (default: none)")
    parser.add_argument("--enroll-user", default="user_001", help="user id used in the enroll/verify smoke demo")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if args.val_fraction <= 0:
        print("error: --val-fraction must be in (0, 1)", file=sys.stderr)
        return 2
    if not os.path.exists(args.checkpoint):
        print("error: checkpoint '{}' does not exist".format(args.checkpoint), file=sys.stderr)
        return 2

    config = TrainingConfig(
        dataset_path=args.dataset,
        seed=args.seed,
        val_fraction=args.val_fraction,
        checkpoint_path=args.checkpoint,
    )

    print("evaluating checkpoint: {}".format(args.checkpoint))
    print("dataset: {}".format(args.dataset))
    print("seed: {} | target FAR budget: {:.4f}".format(args.seed, args.target_far))

    try:
        report = evaluate_verification(
            config,
            target_far=args.target_far,
            max_dev_genuine=args.max_dev_genuine,
            max_dev_impostor=args.max_dev_impostor,
            max_test_genuine=args.max_test_genuine,
            max_test_impostor=args.max_test_impostor,
        )
    except Exception as exc:  # noqa: BLE001 - report at CLI level
        print("error: evaluation failed: {!r}".format(exc), file=sys.stderr)
        return 1

    split = report["split"]
    print("synthetic users: {n_users} | sessions: train {train_sessions} / dev {dev_sessions} / test {test_sessions}"
          .format(**split))
    dev = report["dev_pairs"]
    test = report["test_pairs"]
    print("dev pairs: {genuine} genuine + {impostor} impostor".format(**dev))
    print("test pairs: {genuine} genuine + {impostor} impostor".format(**test))
    cal = report["calibration"]
    print("calibration (development only): threshold={threshold:.6f} target_far={target_far:.4f} "
          "achieved_far={achieved_far:.4f} achieved_tar={achieved_tar:.4f} fallback={fallback}"
          .format(**cal))
    tm = report["test_metrics_at_threshold"]
    print("test @ calibrated threshold: TAR={tar:.4f} FAR={far:.4f} FRR={frr:.4f} "
          "accuracy={accuracy:.4f} (tp={tp:.0f} tn={tn:.0f} fp={fp:.0f} fn={fn:.0f})".format(**tm))
    print("ROC AUC (test): {auc:.4f} | EER (test): {eer:.4f} at threshold {threshold:.6f}".format(
        auc=report["auc"], eer=report["eer"]["eer"], threshold=report["eer"]["threshold"]))
    print("leakage guard: {}".format(report["leakage_guard"]))

    if args.report:
        try:
            path = save_evaluation_report(report, args.report)
        except Exception as exc:  # noqa: BLE001
            print("error: could not write report: {!r}".format(exc), file=sys.stderr)
            return 1
        print("evaluation report written: {}".format(path))

    # -- enroll-and-verify smoke demo ---------------------------------------
    try:
        data = prepare_training_data(config)
        verifier = load_verifier(config.checkpoint_path)
        threshold = float(report["calibration"]["threshold"])
        enrolled = [s for s in data.val_sessions if s["user_id"] == args.enroll_user][:2]
        if len(enrolled) < 2:
            raise RuntimeError("need >= 2 dev sessions of '{}' for the demo".format(args.enroll_user))
        profile = enroll(
            enrolled, verifier,
            keyboard_scaler=data.keyboard_scaler, mouse_scaler=data.mouse_scaler,
            user_ref=args.enroll_user,
        )
        genuine_probe = next(s for s in data.test_sessions if s["user_id"] == args.enroll_user)
        impostor_probe = next(s for s in data.test_sessions if s["user_id"] != args.enroll_user)
        print("enroll+verify smoke: enrolled '{}' from {} dev session(s), embedding dim {}".format(
            profile.user_id, profile.n_sessions, profile.embedding_dim))
        for label, probe in (("genuine (expected VERIFIED)", genuine_probe),
                             ("impostor (expected SUSPICIOUS)", impostor_probe)):
            result = verify(
                profile, probe, verifier, threshold,
                keyboard_scaler=data.keyboard_scaler, mouse_scaler=data.mouse_scaler,
            )
            flag = "OK" if (label.startswith("genuine") and result.decision == "VERIFIED"
                            or label.startswith("impostor") and result.decision == "SUSPICIOUS") else "UNEXPECTED"
            print("  {:<32} dist={dist:.6f} threshold={threshold:.6f} -> {decision!s:<12} [{flag}]".format(
                label, dist=result.distance, threshold=result.threshold,
                decision=result.decision, flag=flag,
            ))
    except Exception as exc:  # noqa: BLE001
        print("error: enroll/verify smoke failed: {!r}".format(exc), file=sys.stderr)
        return 1

    print("SYNTHETIC-data evaluation complete -- these numbers are NOT real-world biometric performance.")
    return 0


if __name__ == "__main__":
    sys.exit(main())