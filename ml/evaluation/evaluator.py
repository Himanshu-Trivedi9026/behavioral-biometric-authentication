"""
Phase 8 — Verification evaluation orchestrator.

Puts the whole enrollment/verification story together on the persisted
synthetic dataset and the Phase 7 checkpoint:

1. Materialise partitions + scalers with :func:`ml.training.dataset.prepare_training_data`
   (deterministic: same seed -> same 25/10/15 session split, same scaler stats).
2. Load the trained verifier (reused as-is, never modified).
3. Build **dev** genuine/impostor pairs (validation partition — "development"
   data) and **test** pairs (test partition, untouched during calibration).
   Pairs never mix sessions from different partitions, so there is no
   train/validation/test leakage by construction.
4. Embed every session with the train-partition scalers, cache embeddings and
   compute Phase 6 L2 pair distances.
5. Calibrate the decision threshold on dev distances against ``target_far``.
6. Evaluate the calibrated threshold + full ROC/AUC/EER on test distances.

Everything is JSON-serialisable (``numpy``/``torch`` values coerced to
``float``/``int``). The dataset contains synthetic users, so the numbers are
**not** evidence about real-world biometric performance.
"""

from __future__ import annotations

import json
import os
import random
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from ml.evaluation.calibration import CalibrationResult, calibrate_threshold
from ml.evaluation.metrics import auc, eer, evaluate_at_threshold, roc_points
from ml.evaluation.model import LoadedVerifier, load_verifier
from ml.training import TrainingConfig, prepare_training_data
from ml.verification import (
    VerificationPair,
    genuine_candidates,
    impostor_candidates,
)

TARGET_FAR = 0.05


def build_partition_pairs(
    entries: Sequence[Mapping[str, Any]],
    *,
    n_genuine: Optional[int] = None,
    n_impostor: Optional[int] = None,
    seed: int = 0,
) -> List[VerificationPair]:
    """All genuine/impostor candidates in a partition, optionally subsampled.

    ``None`` means "use every candidate". Subsampling is deterministic for a
    given seed. Genuine pairs come first, then impostor pairs.
    """
    gen = genuine_candidates(entries)
    imp = impostor_candidates(entries)
    if not gen or not imp:
        raise ValueError(
            "partition needs both genuine and impostor candidates "
            "(genuine={}, impostor={}); check session/user counts".format(len(gen), len(imp))
        )

    def pick(
        candidates: List[VerificationPair], n: Optional[int], kind: str
    ) -> List[VerificationPair]:
        if n is None:
            return candidates
        if isinstance(n, bool) or not isinstance(n, int) or n < 0:
            raise ValueError("{} must be a non-negative int or None, got {!r}".format(kind, n))
        if n > len(candidates):
            raise ValueError(
                "requested {} {} pairs but only {} candidates exist in this partition".format(
                    n, kind, len(candidates)
                )
            )
        if n == len(candidates):
            return candidates
        return random.Random(seed).sample(candidates, n)

    return pick(gen, n_genuine, "genuine") + pick(imp, n_impostor, "impostor")


def _embed_and_pool(
    pairs: Sequence[VerificationPair],
    verifier: LoadedVerifier,
    keyboard_scaler: Any,
    mouse_scaler: Any,
    embed_batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, torch.Tensor]]:
    """Compute pair distances, labels, and the session->embedding cache."""
    session_ids: List[str] = []
    session_by_id: Dict[str, Mapping[str, Any]] = {}
    for pair in pairs:
        a_id = str(pair.session_id_a)
        b_id = str(pair.session_id_b)
        if a_id not in session_by_id:
            session_by_id[a_id] = pair.session_a
            session_ids.append(a_id)
        if b_id not in session_by_id:
            session_by_id[b_id] = pair.session_b
            session_ids.append(b_id)

    cache: Dict[str, torch.Tensor] = {}
    for start in range(0, len(session_ids), embed_batch_size):
        chunk_ids = session_ids[start : start + embed_batch_size]
        chunk = [session_by_id[i] for i in chunk_ids]
        embeddings = verifier.embed_sessions(
            chunk, keyboard_scaler, mouse_scaler, batch_size=embed_batch_size
        )
        if embeddings.shape[0] != len(chunk_ids):
            raise RuntimeError("embed_sessions returned the wrong number of rows")
        for i, sid in enumerate(chunk_ids):
            cache[sid] = embeddings[i]

    n = len(pairs)
    dists = np.empty(n, dtype=np.float64)
    labels = np.empty(n, dtype=np.int64)
    for i, pair in enumerate(pairs):
        ea = cache[str(pair.session_id_a)].unsqueeze(0)
        eb = cache[str(pair.session_id_b)].unsqueeze(0)
        d = verifier.pair_distance(ea, eb)
        dists[i] = float(d[0].detach().cpu().item())
        labels[i] = int(pair.label)
    return dists, labels, cache


def evaluate_verification(
    training_config: TrainingConfig,
    *,
    target_far: float = TARGET_FAR,
    max_dev_genuine: Optional[int] = None,
    max_dev_impostor: Optional[int] = None,
    max_test_genuine: Optional[int] = None,
    max_test_impostor: Optional[int] = None,
    embed_batch_size: int = 16,
) -> Dict[str, Any]:
    """Run the full Phase 8 evaluation and return a JSON-serialisable dict."""
    if not isinstance(training_config, TrainingConfig):
        raise TypeError("training_config must be a TrainingConfig")
    if not (0.0 <= target_far <= 1.0):
        raise ValueError("target_far must be in [0, 1]")

    data = prepare_training_data(training_config)
    verifier = load_verifier(training_config.checkpoint_path)

    n_users = len({s["user_id"] for s in data.train_sessions})

    dev_pairs = build_partition_pairs(
        data.val_sessions,
        n_genuine=max_dev_genuine,
        n_impostor=max_dev_impostor,
        seed=training_config.seed + 1000,
    )
    test_pairs = build_partition_pairs(
        data.test_sessions,
        n_genuine=max_test_genuine,
        n_impostor=max_test_impostor,
        seed=training_config.seed + 2000,
    )

    dev_dist, dev_labels, _ = _embed_and_pool(
        dev_pairs, verifier, data.keyboard_scaler, data.mouse_scaler, embed_batch_size
    )
    test_dist, test_labels, _ = _embed_and_pool(
        test_pairs, verifier, data.keyboard_scaler, data.mouse_scaler, embed_batch_size
    )

    dev_gen = dev_dist[dev_labels == 1]
    dev_imp = dev_dist[dev_labels == 0]
    test_gen = test_dist[test_labels == 1]
    test_imp = test_dist[test_labels == 0]

    calibration: CalibrationResult = calibrate_threshold(dev_gen, dev_imp, target_far=target_far)
    threshold = calibration.threshold

    dev_at_t = evaluate_at_threshold(dev_dist, dev_labels, threshold)
    test_at_t = evaluate_at_threshold(test_dist, test_labels, threshold)
    roc = roc_points(test_gen, test_imp)
    roc_auc = auc(roc)
    eer_result = eer(test_gen, test_imp)

    report: Dict[str, Any] = {
        "dataset_id": os.path.basename(training_config.dataset_path),
        "checkpoint_id": verifier.checkpoint_id,
        "checkpoint_epoch": verifier.epoch,
        "seed": training_config.seed,
        "split": {
            "train_fraction": training_config.train_fraction,
            "val_fraction": training_config.val_fraction,
            "n_users": n_users,
            "train_sessions": len(data.train_sessions),
            "dev_sessions": len(data.val_sessions),
            "test_sessions": len(data.test_sessions),
        },
        "encoder": {
            "embedding_dim": verifier.embedding_dim,
            "encoder_type": "BehavioralEncoder(CNN+GRU)",
        },
        "dev_pairs": {
            "genuine": int((dev_labels == 1).sum()),
            "impostor": int((dev_labels == 0).sum()),
            "total": int(len(dev_labels)),
        },
        "test_pairs": {
            "genuine": int((test_labels == 1).sum()),
            "impostor": int((test_labels == 0).sum()),
            "total": int(len(test_labels)),
        },
        "calibration": {
            **calibration.as_dict(),
            "data": "development (validation) partition only",
        },
        "dev_metrics_at_threshold": {
            k: (float(v) if isinstance(v, (np.floating, float)) else v)
            for k, v in dev_at_t.items()
        },
        "test_metrics_at_threshold": {
            k: (float(v) if isinstance(v, (np.floating, float)) else v)
            for k, v in test_at_t.items()
        },
        "roc": roc,
        "auc": float(roc_auc),
        "eer": {k: float(v) for k, v in eer_result.items()},
        "leakage_guard": "partitions isolated by construction; dev/test pairs never share sessions",
    }
    return report


def save_evaluation_report(report: Mapping[str, Any], path: str) -> str:
    """Write the report (plus ROC points) to JSON at ``path``."""
    if not isinstance(path, str) or not path:
        raise ValueError("path must be a non-empty string")
    directory = os.path.dirname(path)
    if directory and not os.path.isdir(directory):
        os.makedirs(directory, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(dict(report), fh, indent=2, sort_keys=True)
    return os.path.abspath(path)


__all__ = ["TARGET_FAR", "build_partition_pairs", "evaluate_verification", "save_evaluation_report"]