#!/usr/bin/env python3
"""
Phase 16A — evaluate real behavioral data with the EXISTING deployed system.

Purpose
-------
Evaluate locally collected real behavioral sessions against the *already
trained* CNN+GRU + Siamese checkpoint, at the *already deployed* fixed
decision threshold — **without retraining and without recalibration**.

Pipeline (reuses existing, untouched components in order):
    real dataset (Phase 4 schema, `data/datasets/dataset_real.json`)
    -> existing user-aware per-user split (70% train / 20% dev / 10% test,
       `dataset.split.split_train_test` via `ml.training.prepare_training_data`)
    -> FeatureScaler fitted on the TRAINING partition only
    -> dev/test genuine+impostor pairs generated *within* each partition
    -> existing Phase 7 checkpoint loaded read-only (`ml.evaluation.load_verifier`)
    -> L2 pair distances
    -> metrics at the FIXED deployed threshold
        0.4635127782821655
    -> threshold-independent ROC / ROC-AUC / EER
    -> genuine & impostor distance distributions
    -> JSON + text report under `results/real_behavior/`

Guard-rails (enforced by construction)
--------------------------------------
* Calibration is NEVER called — the script imports none of it.
* The checkpoint, models/behavioral_preprocessing.json and
  models/verification_config.json are never written.
* The report stores NO raw events, no key identities, no typed text, and no
  wall-clock activity timestamps.
* The split/leakage/scaler rules are the existing project conventions, reused
  verbatim; no new split algorithm is invented.

Honesty statement
-----------------
"Evaluation uses the existing synthetic-calibrated threshold; no recalibration
is performed in Phase 16A."

The checkpoint and the deployed threshold were produced from SYNTHETIC
development identities. The numbers in this report reflect how that existing
system behaves on real collected data — they are a real-world measurement of
the DEPLOYED system, not a tuned/rescaled result.

Usage:
    .venv/bin/python scripts/evaluate_real_behavior.py \
        --dataset data/datasets/dataset_real.json \
        --checkpoint models/siamese_behavioral_encoder.pt \
        --output-dir results/real_behavior

Exit code 0 = success; 1 = execution error; 2 = usage/file error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dataset import DATASET_SCHEMA_VERSION, load_dataset  # noqa: E402
from ml.evaluation import (  # noqa: E402
    auc,
    build_partition_pairs,
    eer,
    evaluate_at_threshold,
    load_verifier,
    roc_points,
)
from ml.training import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    TrainingConfig,
    prepare_training_data,
)

# Existing deployed decision threshold — models/verification_config.json
# (calibrated on SYNTHETIC development data, Phase 8). Phase 16A reuses this
# exact value and never recalibrates.
DEPLOYED_THRESHOLD = 0.4635127782821655

RECALIBRATION_STATEMENT = (
    "Evaluation uses the existing synthetic-calibrated threshold; "
    "no recalibration is performed in Phase 16A."
)

SYNTHETIC_VS_REAL_STATEMENT = (
    "The existing model and threshold were trained/calibrated using synthetic "
    "development data. Phase 16A evaluates that existing system on real "
    "behavioral data without recalibration."
)

THRESHOLD_PROVENANCE = (
    "models/verification_config.json (synthetic-development calibration; reused unchanged)"
)

DEFAULT_REAL_DATASET = "data/datasets/dataset_real.json"
DEFAULT_OUTPUT_DIR = "results/real_behavior"


# ---------------------------------------------------------------------------
# Statistics / metrics helpers
# ---------------------------------------------------------------------------


def _distribution_stats(distances: Sequence[float]) -> Dict[str, float]:
    """Genuine or impostor distance-distribution summary (JSON-friendly)."""
    arr = np.asarray(distances, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("distribution statistics require a non-empty 1-D array")
    if not bool(np.isfinite(arr).all()):
        raise ValueError("distribution statistics require finite distances")
    percentiles = [float(v) for v in np.percentile(arr, [1, 5, 25, 50, 75, 95, 99])]
    return {
        "count": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "median": float(np.median(arr)),
        "p1": percentiles[0],
        "p5": percentiles[1],
        "p25": percentiles[2],
        "p75": percentiles[4],
        "p95": percentiles[5],
        "p99": percentiles[6],
    }


def _fixed_threshold_metrics(
    distances: Sequence[float], labels: Sequence[float], threshold: float
) -> Dict[str, Any]:
    """Metrics at one fixed threshold, extended with precision/recall/F1 and a
    2x2 confusion matrix (the existing metrics module already supplies the
    counts; these derived values are reported alongside)."""
    base = evaluate_at_threshold(distances, labels, threshold)
    tp, fp, fn, tn = base["tp"], base["fp"], base["fn"], base["tn"]
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {
        **base,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": {
            "tp": float(tp),
            "fp": float(fp),
            "fn": float(fn),
            "tn": float(tn),
            "matrix": [[float(tp), float(fp)], [float(fn), float(tn)]],
            "legend": (
                "rows=[actual genuine, actual impostor]; "
                "columns=[predicted matched (distance<=threshold), predicted not matched]"
            ),
        },
    }


def _pair_scores(
    verifier: Any,
    pairs: Sequence[Any],
    keyboard_scaler: Any,
    mouse_scaler: Any,
    embed_batch_size: int,
) -> tuple:
    """Embed every session once, then L2 distance per pair.

    Returns ``(distances [N], labels [N])`` as numpy arrays, mirroring the
    Phase 8 evaluation embedding/pooling step.
    """
    session_by_id: Dict[str, Mapping[str, Any]] = {}
    order: List[str] = []
    for pair in pairs:
        for session_id, session in (
            (pair.session_id_a, pair.session_a),
            (pair.session_id_b, pair.session_b),
        ):
            if session_id not in session_by_id:
                session_by_id[session_id] = session
                order.append(session_id)

    chunks = verifier.embed_sessions(
        [session_by_id[sid] for sid in order],
        keyboard_scaler,
        mouse_scaler,
        batch_size=embed_batch_size,
    )
    embeddings = {sid: chunks[i] for i, sid in enumerate(order)}

    distances = np.empty(len(pairs), dtype=np.float64)
    labels = np.empty(len(pairs), dtype=np.int64)
    for i, pair in enumerate(pairs):
        a = embeddings[str(pair.session_id_a)].unsqueeze(0)
        b = embeddings[str(pair.session_id_b)].unsqueeze(0)
        distances[i] = float(verifier.pair_distance(a, b)[0])
        labels[i] = int(pair.label)
    return distances, labels


# ---------------------------------------------------------------------------
# Split / leakage checks
# ---------------------------------------------------------------------------


def _partition_user_counts(partition: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for entry in partition:
        counts[entry["user_id"]] = counts.get(entry["user_id"], 0) + 1
    return {user: counts[user] for user in sorted(counts.keys())}


def _leakage_checks(
    train: Sequence[Mapping[str, Any]],
    val: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
    dev_pairs: Sequence[Any],
    test_pairs: Sequence[Any],
) -> Dict[str, Any]:
    train_ids = {e["session_id"] for e in train}
    val_ids = {e["session_id"] for e in val}
    test_ids = {e["session_id"] for e in test}

    def pairs_within(pairs: Sequence[Any], allowed: set) -> bool:
        return all(
            str(p.session_id_a) in allowed and str(p.session_id_b) in allowed
            for p in pairs
        )

    return {
        "train_dev_disjoint": bool(train_ids.isdisjoint(val_ids)),
        "train_test_disjoint": bool(train_ids.isdisjoint(test_ids)),
        "dev_test_disjoint": bool(val_ids.isdisjoint(test_ids)),
        "dev_pairs_within_dev_partition": bool(pairs_within(dev_pairs, val_ids)),
        "test_pairs_within_test_partition": bool(pairs_within(test_pairs, test_ids)),
        "scaler_fitted_on": "training partition only (fulfilled by ml.training.prepare_training_data)",
    }


def _check_dataset_sufficient(entries: Sequence[Mapping[str, Any]]) -> None:
    """Fail loudly when the dataset provably cannot support impersonation pairs.

    The precise partitioning/pair-count limits are still enforced by
    ``prepare_training_data``/``build_partition_pairs``; this check gives a
    clearer message first for the most common failure mode.
    """
    if len(entries) < 2:
        raise RuntimeError(
            "dataset for Phase 16A evaluation must contain at least 2 distinct "
            "users (impostor pairs need two different users)"
        )
    per_user: Dict[str, int] = {}
    for entry in entries:
        per_user[entry["user_id"]] = per_user.get(entry["user_id"], 0) + 1
    lowest = min(per_user.values())
    if lowest < 8:
        print(
            "warning: participant '{}' has only {} session(s); with the 70/20/10 "
            "split this may not produce genuine dev/test pairs (recommended: "
            "10 sessions per participant)".format(
                min(per_user, key=per_user.get), lowest
            ),
            file=sys.stderr,
        )


# ---------------------------------------------------------------------------
# Report building
# ---------------------------------------------------------------------------


def _per_user_partition_counts(
    train: Sequence[Mapping[str, Any]],
    val: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
) -> Dict[str, Dict[str, int]]:
    users = sorted({e["user_id"] for e in train})
    return {
        user: {
            "train": sum(1 for e in train if e["user_id"] == user),
            "dev": sum(1 for e in val if e["user_id"] == user),
            "test": sum(1 for e in test if e["user_id"] == user),
        }
        for user in users
    }


def run_evaluation(
    dataset_path: str,
    *,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    seed: int = 42,
    train_fraction: float = 0.7,
    val_fraction: float = 2 / 7,
    output_dir: Optional[str] = None,
    embed_batch_size: int = 16,
    max_dev_genuine: Optional[int] = None,
    max_dev_impostor: Optional[int] = None,
    max_test_genuine: Optional[int] = None,
    max_test_impostor: Optional[int] = None,
) -> Dict[str, Any]:
    """Run the real-behavioral evaluation; returns a JSON-serialisable report.

    Importable so the CLI and the automated test suite share one implementation.
    Never calibrates a threshold and never writes to ``models/``.
    """
    if not os.path.exists(dataset_path):
        raise RuntimeError(
            "real dataset '{}' does not exist yet -- collect sessions, create "
            "data/raw/participant_manifest.json, then run "
            "scripts/build_real_dataset.py (see docs/PHASE_16A_REAL_DATASET_EVALUATION.md)".format(
                dataset_path
            )
        )
    if not os.path.exists(checkpoint_path):
        raise RuntimeError(
            "checkpoint '{}' does not exist".format(checkpoint_path)
        )

    config = TrainingConfig(
        dataset_path=dataset_path,
        seed=seed,
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        checkpoint_path=checkpoint_path,
        n_genuine_train=1,
        n_impostor_train=1,
        n_genuine_val=1,
        n_impostor_val=1,
        n_genuine_test=1,
        n_impostor_test=1,
    )

    all_entries = load_dataset(dataset_path)
    _check_dataset_sufficient(all_entries)

    data = prepare_training_data(config)
    verifier = load_verifier(checkpoint_path)

    dev_pairs = build_partition_pairs(
        data.val_sessions,
        n_genuine=max_dev_genuine,
        n_impostor=max_dev_impostor,
        seed=seed + 1000,
    )
    test_pairs = build_partition_pairs(
        data.test_sessions,
        n_genuine=max_test_genuine,
        n_impostor=max_test_impostor,
        seed=seed + 2000,
    )

    dev_dist, dev_labels = _pair_scores(
        verifier, dev_pairs, data.keyboard_scaler, data.mouse_scaler, embed_batch_size
    )
    test_dist, test_labels = _pair_scores(
        verifier, test_pairs, data.keyboard_scaler, data.mouse_scaler, embed_batch_size
    )

    dev_gen = dev_dist[dev_labels == 1]
    dev_imp = dev_dist[dev_labels == 0]
    test_gen = test_dist[test_labels == 1]
    test_imp = test_dist[test_labels == 0]

    def partition_block(
        distances: np.ndarray, labels: np.ndarray, gen: np.ndarray, imp: np.ndarray
    ) -> Dict[str, Any]:
        block = {
            "pairs": {
                "genuine": int((labels == 1).sum()),
                "impostor": int((labels == 0).sum()),
                "total": int(labels.size),
            },
            "metrics_at_fixed_threshold": _fixed_threshold_metrics(
                distances, labels, DEPLOYED_THRESHOLD
            ),
            "roc": roc_points(gen, imp),
            "auc": float(auc(roc_points(gen, imp))),
            "eer": {k: float(v) for k, v in eer(gen, imp).items()},
            "distributions": {
                "genuine": _distribution_stats(gen),
                "impostor": _distribution_stats(imp),
            },
        }
        return block

    dev_block = partition_block(dev_dist, dev_labels, dev_gen, dev_imp)
    test_block = partition_block(test_dist, test_labels, test_gen, test_imp)

    report: Dict[str, Any] = {
        "purpose": (
            "Phase 16A real-behavioral evaluation of the EXISTING deployed system "
            "(evaluation and data preparation only; no training, no recalibration)"
        ),
        "dataset": {
            "id": os.path.basename(dataset_path),
            "path": dataset_path,
            "schema_version": DATASET_SCHEMA_VERSION,
            "n_sessions": len(all_entries),
            "n_users": len({e["user_id"] for e in all_entries}),
            "sessions_per_user": _partition_user_counts(all_entries),
            "source": "locally collected real behavioral sessions (Phase 2 collector)",
        },
        "measurement": {
            "statement": RECALIBRATION_STATEMENT,
            "synthetic_vs_real": SYNTHETIC_VS_REAL_STATEMENT,
            "recalibration_performed": False,
            "synthetic_presented_as_real": False,
        },
        "threshold": {
            "mode": "fixed deployed threshold (no recalibration)",
            "value": DEPLOYED_THRESHOLD,
            "provenance": THRESHOLD_PROVENANCE,
        },
        "checkpoint": {
            "checkpoint_id": verifier.checkpoint_id,
            "epoch": int(verifier.epoch),
            "seed": int(verifier.seed),
            "embedding_dim": int(verifier.embedding_dim),
            "encoder_type": "BehavioralEncoder(CNN+GRU)",
        },
        "split": {
            "method": (
                "existing user-aware per-user deterministic split "
                "(dataset.split.split_train_test via ml.training.prepare_training_data)"
            ),
            "target": "70% train / 20% development / 10% test (project convention)",
            "train_fraction": float(train_fraction),
            "val_fraction": float(val_fraction),
            "n_train_sessions": len(data.train_sessions),
            "n_dev_sessions": len(data.val_sessions),
            "n_test_sessions": len(data.test_sessions),
            "per_user_partition_counts": _per_user_partition_counts(
                data.train_sessions, data.val_sessions, data.test_sessions
            ),
        },
        "scaler": {
            "fitted_on": "training partition only",
            "reused_for": ["dev", "test"],
        },
        "dev": dev_block,
        "test": test_block,
        "leakage_checks": _leakage_checks(
            data.train_sessions, data.val_sessions, data.test_sessions, dev_pairs, test_pairs
        ),
    }

    if output_dir:
        json_path, txt_path = write_report(report, output_dir)
        report["output_files"] = {"json": json_path, "text": txt_path}

    return report


def render_text(report: Mapping[str, Any]) -> str:
    """Human-readable rendering of the report (no raw events, no wall-clock)."""
    lines: List[str] = []
    lines.append("Phase 16A real-behavioral evaluation report")
    lines.append("=" * 58)
    lines.append("")
    lines.append("THRESHOLD: fixed deployed = {:.12f} (no recalibration)".format(
        report["threshold"]["value"]
    ))
    lines.append("STATEMENT: {}".format(report["measurement"]["statement"]))
    lines.append("MODEL/TRAINING: {}".format(report["measurement"]["synthetic_vs_real"]))
    lines.append("")
    ds = report["dataset"]
    lines.append("dataset: {} | schema v{} | {} sessions | {} users".format(
        ds["id"], ds["schema_version"], ds["n_sessions"], ds["n_users"]
    ))
    sp = report["split"]
    lines.append("split: {} | train {train}/dev {dev}/test {test} sessions".format(
        sp["method"], train=sp["n_train_sessions"], dev=sp["n_dev_sessions"], test=sp["n_test_sessions"]
    ))
    for user, counts in sp["per_user_partition_counts"].items():
        lines.append("  {}: train {} / dev {} / test {}".format(
            user, counts["train"], counts["dev"], counts["test"]
        ))
    lines.append("")
    for partition in ("dev", "test"):
        block = report[partition]
        pairs = block["pairs"]
        m = block["metrics_at_fixed_threshold"]
        lines.append("{} partition: {} genuine + {} impostor pairs".format(
            partition, pairs["genuine"], pairs["impostor"]
        ))
        lines.append("  @ fixed deployed threshold: TAR={tar:.4f} FAR={far:.4f} FRR={frr:.4f} "
                     "accuracy={accuracy:.4f} precision={precision:.4f} recall={recall:.4f} "
                     "f1={f1:.4f}".format(**m))
        lines.append("  confusion: TP={tp:.0f} TN={tn:.0f} FP={fp:.0f} FN={fn:.0f}".format(
            tp=m["confusion_matrix"]["tp"], tn=m["confusion_matrix"]["tn"],
            fp=m["confusion_matrix"]["fp"], fn=m["confusion_matrix"]["fn"],
        ))
        lines.append("  ROC-AUC: {auc:.4f} | EER: {eer:.4f} at threshold {eet:.6f}".format(
            auc=block["auc"], eer=block["eer"]["eer"], eet=block["eer"]["threshold"]
        ))
        for kind, stat in block["distributions"].items():
            lines.append("  {} distances: n={count} mean={mean:.4f} std={std:.4f} "
                         "min={min:.4f} p5={p5:.4f} med={median:.4f} p95={p95:.4f} "
                         "max={max:.4f}".format(kind, **stat))
    lines.append("")
    lines.append("leakage: {}".format(report["leakage_checks"]))
    lines.append("")
    lines.append("IMPORTANT: this measures the EXISTING system trained/calibrated on "
                 "synthetic data, evaluated on real data WITHOUT recalibration.")
    return "\n".join(lines)


def write_report(report: Mapping[str, Any], output_dir: str) -> tuple:
    """Write ``real_behavior_report.json`` + ``.txt`` under ``output_dir``."""
    os.makedirs(output_dir, exist_ok=True)
    json_path = os.path.join(output_dir, "real_behavior_report.json")
    txt_path = os.path.join(output_dir, "real_behavior_report.txt")
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)
        f.write("\n")
    with open(txt_path, "w") as f:
        f.write(render_text(report))
        f.write("\n")
    return json_path, txt_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the EXISTING checkpoint on real behavioral data at the fixed "
            "deployed threshold (no recalibration, no retraining)."
        )
    )
    parser.add_argument("--dataset", default=DEFAULT_REAL_DATASET,
                        help="real dataset path (default: %(default)s)")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT,
                        help="Phase 7 checkpoint (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42,
                        help="deterministic seed (default: %(default)s)")
    parser.add_argument("--train-fraction", type=float, default=0.7,
                        help="per-user outer train fraction (default: %(default)s)")
    parser.add_argument("--val-fraction", type=float, default=2 / 7,
                        help="fraction of train used as development (default: %(default)s)")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="where to write the reports (default: %(default)s)")
    parser.add_argument("--embed-batch-size", type=int, default=16,
                        help="embedding batch size (default: %(default)s)")
    parser.add_argument("--max-dev-genuine", type=int, default=None,
                        help="cap dev genuine pairs (None = all)")
    parser.add_argument("--max-dev-impostor", type=int, default=None,
                        help="cap dev impostor pairs (None = all)")
    parser.add_argument("--max-test-genuine", type=int, default=None,
                        help="cap test genuine pairs (None = all)")
    parser.add_argument("--max-test-impostor", type=int, default=None,
                        help="cap test impostor pairs (None = all)")
    parser.add_argument("--no-write", action="store_true",
                        help="evaluate and print only; do not write report files")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    if os.path.isdir(args.dataset):
        print("error: --dataset must be a dataset JSON file, got a directory", file=sys.stderr)
        return 2

    try:
        report = run_evaluation(
            args.dataset,
            checkpoint_path=args.checkpoint,
            seed=args.seed,
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            output_dir=None if args.no_write else args.output_dir,
            embed_batch_size=args.embed_batch_size,
            max_dev_genuine=args.max_dev_genuine,
            max_dev_impostor=args.max_dev_impostor,
            max_test_genuine=args.max_test_genuine,
            max_test_impostor=args.max_test_impostor,
        )
    except Exception as exc:  # noqa: BLE001 - report at CLI level
        print("error: evaluation failed: {!r}".format(exc), file=sys.stderr)
        return 1

    print(RECALIBRATION_STATEMENT)
    print(render_text(report))
    if not args.no_write:
        print("evaluation report written: {}".format(report["output_files"]["json"]))
        print("text report written: {}".format(report["output_files"]["text"]))
    print("DONE -- real-data evaluation of the EXISTING system (no retraining, no recalibration).")
    return 0


if __name__ == "__main__":
    sys.exit(main())