#!/usr/bin/env python3
"""
Phase 4 — baseline evaluation CLI.

Splits a dataset deterministically (per-user), fits the statistical baseline on
the training split, chooses a threshold (explicit or calibrated on training
data only), and evaluates genuine + impostor attempts on the held-out split.

Usage:
    .venv/bin/python scripts/evaluate_baseline.py [--dataset PATH]
        [--seed N] [--train-fraction F] [--threshold T] [--far-target F]
        [--method target-far|max-tar-far] [--report-out PATH] [--roc-out PATH]

Exit code 0 = success; 1 = evaluation error; 2 = usage/file error.
"""

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from baseline.evaluator import BaselineEvaluator  # noqa: E402
from dataset.loader import load_dataset  # noqa: E402
from dataset.split import (  # noqa: E402
    assert_no_leakage,
    split_report,
    split_train_test,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Evaluate the Phase 4 statistical baseline."
    )
    parser.add_argument(
        "--dataset",
        default=os.path.join("data", "datasets", "dataset_synthetic.json"),
        help="path to a Phase 4 dataset file (default: data/datasets/dataset_synthetic.json)",
    )
    parser.add_argument("--seed", type=int, default=0, help="split seed (default 0)")
    parser.add_argument(
        "--train-fraction", type=float, default=0.7, help="per-user training fraction (default 0.7)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="explicit combined-score distance threshold (default: calibrated from training data)",
    )
    parser.add_argument(
        "--far-target", type=float, default=0.05, help="calibration FAR budget (default 0.05)"
    )
    parser.add_argument(
        "--method",
        choices=("target-far", "max-tar-far"),
        default="target-far",
        help="calibration method (default: target-far)",
    )
    parser.add_argument("--report-out", default=None, help="optional full JSON report path")
    parser.add_argument("--roc-out", default=None, help="optional JSON path for ROC sweep points")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.dataset):
        print("error: no such dataset file: {}".format(args.dataset), file=sys.stderr)
        return 2

    try:
        entries = load_dataset(args.dataset)
        train, test = split_train_test(
            entries, seed=args.seed, train_fraction=args.train_fraction
        )
        assert_no_leakage(train, test)

        evaluator = BaselineEvaluator()
        evaluator.fit(train)
        report = evaluator.evaluate(
            test,
            threshold=args.threshold,
            calibrate=args.threshold is None,
            far_target=args.far_target,
            method=args.method,
            train_entries=train,
        )
    except (ValueError, RuntimeError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1

    metrics = report.metrics
    print("dataset: {}".format(args.dataset))
    print("split seed={} train_fraction={} leakage_check={}".format(
        args.seed, args.train_fraction, "PASS"
    ))
    for user_id, counts in sorted(split_report(entries, train, test).items()):
        print(
            "  user {}: train={} test={} total={}".format(
                user_id, counts["train"], counts["test"], counts["total"]
            )
        )
    print("threshold={:.6f} source={}".format(
        report.threshold, report.threshold_source
    ))
    cal = report.threshold_result
    print(
        "calibration[{}]: dev_genuine_n={} dev_impostor_n={} dev_TAR={:.4f} dev_FAR={:.4f}".format(
            cal.method, len(cal.genuine_dev), len(cal.impostor_dev), cal.dev_tar, cal.dev_far
        )
    )
    print("--- held-out evaluation ---")
    print(
        "genuine attempts={} impostor attempts={}".format(
            metrics["n_genuine"], metrics["n_impostor"]
        )
    )
    print(
        "TAR={:.4f}  FAR={:.4f}  FRR={:.4f}  accuracy={:.4f}".format(
            metrics["tar"], metrics["far"], metrics["frr"], metrics["accuracy"]
        )
    )
    print(
        "confusion: TP={} FP={} FN={} TN={}  unscorable_g={} unscorable_i={}".format(
            metrics["tp"],
            metrics["fp"],
            metrics["fn"],
            metrics["tn"],
            metrics["unscorable_genuine"],
            metrics["unscorable_impostor"],
        )
    )
    print("roc_points={}".format(len(report.roc_points)))

    if args.report_out:
        with open(args.report_out, "w") as f:
            json.dump(report.as_dict(), f, indent=2)
            f.write("\n")
        print("report written to: {}".format(args.report_out))
    if args.roc_out:
        with open(args.roc_out, "w") as f:
            json.dump(report.roc_points, f, indent=2)
            f.write("\n")
        print("roc points written to: {}".format(args.roc_out))

    return 0


if __name__ == "__main__":
    sys.exit(main())