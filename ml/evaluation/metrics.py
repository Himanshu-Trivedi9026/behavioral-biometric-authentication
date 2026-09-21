"""
Phase 8 — Verification metrics: FAR / TAR / FRR, confusion-matrix counts,
ROC points, ROC AUC and EER.

Distance convention (Phase 4/6/7): a verification decision is ``VERIFIED`` when
``distance <= threshold``, lower distance means stronger match. Labels are
``1`` = genuine pair (same user), ``0`` = impostor pair (different users).

Metrics at a threshold
    * TP / TN / FP / FN  confusion-matrix counts
    * TAR = TP / n_genuine   (genuine accepted)
    * FAR = FP / n_impostor  (impostor accepted)
    * FRR = 1 - TAR          (genuine rejected)
    * accuracy = (TP + TN) / total

All functions operate on plain sequences of floats so the same code covers
direct neural pair distances and derived score columns.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Union

import numpy as np

Number = Union[int, float]


def _to_array(a: Sequence[Number]) -> np.ndarray:
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError("expected a 1-D sequence of distances/labels")
    if not bool(np.isfinite(arr).all()):
        raise ValueError("distances/labels must be finite")
    return arr


def evaluate_at_threshold(
    distances: Sequence[Number],
    labels: Sequence[Number],
    threshold: Number,
) -> Dict[str, float]:
    """Full confusion-matrix metrics for one decision threshold.

    Works with a single class present (genuine-only or impostor-only): the
    absent class contributes zero counts / rate 0.0.
    """
    d = _to_array(distances)
    y = np.round(np.asarray(labels, dtype=np.float64))

    tp = float(((d <= threshold) & (y == 1)).sum())
    fp = float(((d <= threshold) & (y == 0)).sum())
    fn = float(((d > threshold) & (y == 1)).sum())
    tn = float(((d > threshold) & (y == 0)).sum())

    n_gen = float((y == 1).sum())
    n_imp = float((y == 0).sum())
    total = float(len(y))

    tar = tp / n_gen if n_gen > 0 else 0.0
    far = fp / n_imp if n_imp > 0 else 0.0
    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "n_genuine": n_gen,
        "n_impostor": n_imp,
        "tar": tar,
        "far": far,
        "frr": 1.0 - tar,
        "accuracy": (tp + tn) / total if total > 0 else 0.0,
        "threshold": float(threshold),
    }


def _require_both_classes(gen: np.ndarray, imp: np.ndarray) -> None:
    if gen.size == 0 or imp.size == 0:
        raise ValueError(
            "ROC/AUC/EER need both genuine and impostor distances "
            "(got genuine={}, impostor={}); single-class evaluation is "
            "only meaningful with evaluate_at_threshold".format(gen.size, imp.size)
        )


def roc_points(
    genuine_distances: Sequence[Number],
    impostor_distances: Sequence[Number],
) -> Dict[str, List]:
    """ROC curve points over the unique candidate thresholds.

    Returns dict with ``roc_curve`` (list of dicts), plus parallel
    ``thresholds`` / ``fpr`` / ``tpr`` arrays. Endpoints use ``threshold=None``
    so the result stays JSON-friendly.

    ``fpr`` is non-decreasing and ``tpr`` may jump within an fpr step; that is
    expected for a distance-score ROC.
    """
    gen = _to_array(genuine_distances)
    imp = _to_array(impostor_distances)
    _require_both_classes(gen, imp)

    candidates = np.unique(np.concatenate([gen, imp]))
    points: List[Dict[str, Union[float, None]]] = [
        {"threshold": None, "fpr": 0.0, "tpr": 0.0}
    ]
    for t in candidates:
        fpr = float((imp <= t).mean())
        tpr = float((gen <= t).mean())
        points.append({"threshold": float(t), "fpr": fpr, "tpr": tpr})
    points.append({"threshold": None, "fpr": 1.0, "tpr": 1.0})

    return {
        "roc_curve": points,
        "thresholds": [p["threshold"] for p in points],
        "fpr": [p["fpr"] for p in points],
        "tpr": [p["tpr"] for p in points],
    }


def auc(roc: Dict[str, List]) -> float:
    """ROC AUC by staircase integration of the ROC points.

    Equivalent to the two-sample U-statistic P(distance_gen < distance_imp)
    (+0.5 ties), which is the empirical AUC for a distance score; the test
    suite cross-checks this against the rank statistic.
    """
    fpr = np.asarray(roc["fpr"], dtype=np.float64)
    tpr = np.asarray(roc["tpr"], dtype=np.float64)
    order = np.lexsort((tpr, fpr))
    fpr = fpr[order]
    tpr = tpr[order]
    area = 0.0
    for i in range(len(fpr) - 1):
        # Trapezoid rule over the ROC points. When fpr and tpr both jump in
        # the same threshold bin, the trapezoid is the average of the two
        # possible staircase orderings, which is exactly the 0.5-tie-handling
        # of the two-sample U-statistic AUC.
        area += (fpr[i + 1] - fpr[i]) * (tpr[i + 1] + tpr[i]) / 2.0
    return float(area)


def eer(genuine_distances: Sequence[Number], impostor_distances: Sequence[Number]) -> Dict[str, float]:
    """Equal Error Rate over unique candidate thresholds.

    EER is the point where FAR == FRR (ties favour the lower threshold)
    """
    gen = _to_array(genuine_distances)
    imp = _to_array(impostor_distances)
    _require_both_classes(gen, imp)

    candidates = np.unique(np.concatenate([gen, imp]))
    best: Dict[str, float] = {}
    for t in candidates:
        far = float((imp <= t).mean())
        frr = 1.0 - float((gen <= t).mean())
        gap = abs(far - frr)
        if not best or gap < best["_gap"] or (gap == best["_gap"] and t < best["threshold"]):
            best = {"threshold": float(t), "far": far, "frr": frr, "tar": 1.0 - frr, "_gap": gap}
    best.pop("_gap", None)
    best["eer"] = (best["far"] + best["frr"]) / 2.0
    return best


__all__ = ["evaluate_at_threshold", "roc_points", "auc", "eer"]