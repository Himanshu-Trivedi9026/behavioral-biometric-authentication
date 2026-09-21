"""
Phase 8 — unit tests for verification metrics (FAR/TAR/FRR, confusion-matrix
counts, ROC points, ROC AUC, EER).

These tests are pure :mod:`ml.evaluation.metrics` logic on tiny hand-made
distance arrays — no model, no dataset, no checkpoint required.

Run:
    .venv/bin/python -m pytest tests/test_metrics.py -v
"""

import numpy as np
import pytest

from ml.evaluation.metrics import auc, eer, evaluate_at_threshold, roc_points


class TestEvaluateAtThreshold:
    def test_all_four_confusion_counts(self):
        metrics = evaluate_at_threshold(distances=[1, 2, 3, 4], labels=[1, 0, 1, 0], threshold=2.5)
        assert metrics["tp"] == 1
        assert metrics["fp"] == 1
        assert metrics["fn"] == 1
        assert metrics["tn"] == 1
        assert metrics["n_genuine"] == 2
        assert metrics["n_impostor"] == 2
        assert metrics["tar"] == pytest.approx(0.5)
        assert metrics["far"] == pytest.approx(0.5)
        assert metrics["frr"] == pytest.approx(0.5)
        assert metrics["accuracy"] == pytest.approx(0.5)

    def test_positive_only_labels_one(self):
        metrics = evaluate_at_threshold(distances=[1, 2, 3], labels=[1, 1, 1], threshold=2)
        assert metrics["tar"] == pytest.approx(2 / 3)
        assert metrics["far"] == 0.0
        assert metrics["frr"] == pytest.approx(1 / 3)

    def test_negative_only_labels_zero(self):
        metrics = evaluate_at_threshold(distances=[1, 2, 3], labels=[0, 0, 0], threshold=2)
        assert metrics["tar"] == 0.0
        assert metrics["frr"] == 1.0
        assert metrics["far"] == pytest.approx(2 / 3)

    def test_empty_input_all_zeroes(self):
        metrics = evaluate_at_threshold(distances=[], labels=[], threshold=0.5)
        assert metrics["tar"] == 0.0
        assert metrics["far"] == 0.0
        assert metrics["accuracy"] == 0.0
        assert metrics["n_genuine"] == 0

    def test_1d_only(self):
        with pytest.raises(ValueError):
            evaluate_at_threshold(distances=[[1, 2], [3, 4]], labels=[1, 0], threshold=1)

    def test_non_finite_rejected(self):
        with pytest.raises(ValueError):
            evaluate_at_threshold(distances=[1.0, float("nan")], labels=[1, 1], threshold=1)


class TestRocPoints:
    def test_structure_and_endpoints(self):
        roc = roc_points(genuine_distances=[0, 1, 2], impostor_distances=[5, 6, 7])
        assert roc["roc_curve"][0] == {"threshold": None, "fpr": 0.0, "tpr": 0.0}
        assert roc["roc_curve"][-1] == {"threshold": None, "fpr": 1.0, "tpr": 1.0}
        assert len(roc["thresholds"]) == len(roc["fpr"]) == len(roc["tpr"]) == len(roc["roc_curve"])
        assert len(roc["roc_curve"]) == 2 + 6  # endpoints + 6 unique candidate thresholds

    def test_perfect_genuine_first(self):
        roc = roc_points(genuine_distances=[0, 1, 2], impostor_distances=[5, 6, 7])
        fpr, tpr = roc["fpr"], roc["tpr"]
        assert fpr == sorted(fpr)
        assert tpr == sorted(tpr)
        # perfect separation: every interior point has fpr == 0 or tpr == 1
        for point in roc["roc_curve"][1:-1]:
            assert point["fpr"] == 0.0 or point["tpr"] == 1.0
        assert auc(roc) == pytest.approx(1.0)

    def test_single_class_rejected(self):
        with pytest.raises(ValueError):
            roc_points([1, 2, 3], [])
        with pytest.raises(ValueError):
            roc_points([], [1, 2, 3])


class TestAuc:
    @staticmethod
    def _rank_auc(gen, imp):
        gen = np.asarray(gen, dtype=np.float64)
        imp = np.asarray(imp, dtype=np.float64)
        less = gen[:, None] < imp[None, :]
        equal = gen[:, None] == imp[None, :]
        return float((less.sum() + 0.5 * equal.sum()) / (gen.size * imp.size))

    def test_perfect_separation_is_one(self):
        roc = roc_points([0, 1, 2], [5, 6, 7])
        assert auc(roc) == pytest.approx(1.0)

    def test_reversed_separation_is_zero(self):
        roc = roc_points([5, 6, 7], [0, 1, 2])
        assert auc(roc) == pytest.approx(0.0)

    def test_all_ties_is_half(self):
        roc = roc_points([1, 1, 1], [1, 1, 1])
        assert auc(roc) == pytest.approx(0.5)

    def test_auc_equals_rank_statistic(self):
        rng = np.random.default_rng(7)
        for _ in range(20):
            gen = rng.uniform(0, 1, size=6)
            imp = rng.uniform(0, 1, size=9)
            roc = roc_points(gen, imp)
            assert auc(roc) == pytest.approx(self._rank_auc(gen, imp), abs=1e-12)

    def test_values_in_unit_interval(self):
        rng = np.random.default_rng(11)
        for _ in range(50):
            roc = roc_points(rng.uniform(size=4), rng.uniform(size=5))
            assert 0.0 <= auc(roc) <= 1.0


class TestEer:
    def test_interleaved_distances(self):
        result = eer(genuine_distances=[0, 2, 4, 6], impostor_distances=[1, 3, 5, 7])
        assert result["threshold"] == pytest.approx(3)
        assert result["far"] == pytest.approx(0.5)
        assert result["frr"] == pytest.approx(0.5)
        assert result["eer"] == pytest.approx(0.5)
        assert result["tar"] == pytest.approx(0.5)

    def test_balanced_point(self):
        result = eer(genuine_distances=[0.5, 0.9, 1.0], impostor_distances=[0.2, 0.8, 1.2])
        assert result["far"] == pytest.approx(result["frr"])
        assert abs(result["far"] - result["frr"]) < 1e-12

    def test_single_class_rejected(self):
        with pytest.raises(ValueError):
            eer([1, 2, 3], [])
        with pytest.raises(ValueError):
            eer([], [1, 2])


class TestAucEerShareHelpers:
    def test_roc_points_thresholds_match_threshold_metrics(self):
        gen = [0.5, 1.0, 1.5]
        imp = [0.2, 0.8, 1.2]
        roc = roc_points(gen, imp)
        for point in roc["roc_curve"]:
            t = point["threshold"]
            if t is None:
                continue
            m = evaluate_at_threshold(gen + imp, [1, 1, 1, 0, 0, 0], threshold=t)
            assert point["tpr"] == pytest.approx(m["tar"])
            assert point["fpr"] == pytest.approx(m["far"])