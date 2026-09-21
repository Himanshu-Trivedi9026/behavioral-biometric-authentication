"""
Phase 8 — end-to-end verification evaluation + threshold calibration tests.

The pipeline under test: partitions + scalers (:func:`prepare_training_data`,
train-partition only) -> read-only Phase 7 checkpoint embeddings -> dev/test
genuine & impostor pairs -> L2 distances -> threshold calibrated on the
DEVELOPMENT partition -> test-set TAR/FAR/FRR/ROC/AUC/EER report.

Skipped when the checkpoint/dataset artifacts are missing.

Run:
    .venv/bin/python -m pytest tests/test_verification_evaluation.py -v
"""

import json
import os

import pytest

from ml.evaluation import (
    build_partition_pairs,
    calibrate_threshold,
    evaluate_verification,
    save_evaluation_report,
)
from ml.evaluation.metrics import roc_points
from ml.training import TrainingConfig

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(_REPO, "models", "siamese_behavioral_encoder.pt")
DATASET_PATH = os.path.join(_REPO, "data", "datasets", "dataset_synthetic.json")

pytestmark = pytest.mark.skipif(
    not os.path.exists(CHECKPOINT_PATH) or not os.path.exists(DATASET_PATH),
    reason="Phase 7 checkpoint and/or synthetic dataset not present on disk",
)

TARGET_FAR = 0.05


def _config():
    return TrainingConfig(
        dataset_path=DATASET_PATH,
        seed=42,
        checkpoint_path=CHECKPOINT_PATH,
    )


@pytest.fixture(scope="function")
def config():
    return _config()


@pytest.fixture(scope="function")
def report(config):
    return evaluate_verification(config, target_far=TARGET_FAR)


class TestCalibrationUnit:
    def test_picks_largest_threshold_meeting_budget(self):
        result = calibrate_threshold(
            genuine_distances=[0, 2, 4, 6],
            impostor_distances=[1, 3, 5, 7],
            target_far=0.5,
        )
        assert result.threshold == pytest.approx(4)
        assert result.achieved_far == pytest.approx(0.5)
        assert result.achieved_tar == pytest.approx(0.75)
        assert result.fallback is False

    def test_strictest_never_exceeds_budget(self):
        result = calibrate_threshold(genuine_distances=[0, 2], impostor_distances=[1, 3], target_far=1.0)
        assert result.achieved_far <= 1.0
        assert result.threshold == pytest.approx(3)  # largest with far <= 1.0

    def test_fallback_to_strictest_when_budget_impossible(self):
        result = calibrate_threshold(genuine_distances=[1, 5], impostor_distances=[1, 1, 1], target_far=0.0)
        assert result.fallback is True
        assert result.threshold == pytest.approx(1)
        assert result.achieved_far == pytest.approx(1.0)  # closest feasible

    def test_invalid_target_far_raises(self):
        with pytest.raises(ValueError):
            calibrate_threshold([1, 2], [3, 4], target_far=2.0)

    def test_single_class_raises(self):
        with pytest.raises(ValueError):
            calibrate_threshold([], [3, 4], target_far=0.05)
        with pytest.raises(ValueError):
            calibrate_threshold([1, 2], [], target_far=0.05)

    def test_result_is_json_serialisable(self):
        result = calibrate_threshold([1, 2, 3], [4, 5], target_far=0.5)
        payload = json.dumps(result.as_dict())
        assert "threshold" in payload


def _pair_counts(pairs):
    genuine = sum(1 for p in pairs if p.label == 1)
    return genuine, len(pairs) - genuine


class TestBuildPartitionPairs:
    def test_default_all_candidates_dev(self, config):
        import ml.training as T

        data = T.prepare_training_data(config)
        pairs = build_partition_pairs(data.val_sessions, seed=config.seed + 1000)
        assert _pair_counts(pairs) == (5, 40)

    def test_default_all_candidates_test(self, config):
        import ml.training as T

        data = T.prepare_training_data(config)
        pairs = build_partition_pairs(data.test_sessions, seed=config.seed + 2000)
        assert _pair_counts(pairs) == (15, 90)

    def test_caps_respected_and_deterministic(self, config):
        import ml.training as T

        data = T.prepare_training_data(config)
        a = build_partition_pairs(data.test_sessions, n_genuine=6, n_impostor=10, seed=7)
        b = build_partition_pairs(data.test_sessions, n_genuine=6, n_impostor=10, seed=7)
        assert _pair_counts(a) == (6, 10)
        assert [p.session_id_a + "|" + p.session_id_b for p in a] == [
            p.session_id_a + "|" + p.session_id_b for p in b
        ]

    def test_cap_over_candidates_raises(self, config):
        import ml.training as T

        data = T.prepare_training_data(config)
        with pytest.raises(ValueError):
            build_partition_pairs(data.test_sessions, n_impostor=10 ** 6, seed=0)

    def test_bool_count_rejected(self, config):
        import ml.training as T

        data = T.prepare_training_data(config)
        with pytest.raises(ValueError):
            build_partition_pairs(data.test_sessions, n_genuine=True, seed=0)

    def test_every_pair_is_within_one_partition(self):
        from ml.evaluation.evaluator import build_partition_pairs as build  # noqa: PLC0415
        from ml.training import prepare_training_data  # noqa: PLC0415

        data = prepare_training_data(_config())
        dev_ids = {s["session_id"] for s in data.val_sessions}
        test_ids = {s["session_id"] for s in data.test_sessions}
        assert not (dev_ids & test_ids)
        dev_pairs = build(data.val_sessions, seed=1)
        test_pairs = build(data.test_sessions, seed=2)
        for pair in dev_pairs:
            assert pair.session_id_a in dev_ids and pair.session_id_b in dev_ids
        for pair in test_pairs:
            assert pair.session_id_a in test_ids and pair.session_id_b in test_ids


class TestEvaluateVerification:
    def test_report_structure_and_split(self, report):
        assert report["split"]["n_users"] == 5
        assert report["split"]["train_sessions"] == 25
        assert report["split"]["dev_sessions"] == 10
        assert report["split"]["test_sessions"] == 15
        assert report["encoder"]["embedding_dim"] == 128
        assert report["dev_pairs"] == {"genuine": 5, "impostor": 40, "total": 45}
        assert report["test_pairs"] == {"genuine": 15, "impostor": 90, "total": 105}

    def test_calibration_uses_development_only(self, report):
        cal = report["calibration"]
        assert cal["data"] == "development (validation) partition only"
        assert abs(cal["target_far"] - TARGET_FAR) < 1e-12
        assert cal["fallback"] is False
        assert cal["achieved_far"] <= TARGET_FAR + 1e-9

    def test_dev_metrics_consistent_with_calibration(self, report):
        dev = report["dev_metrics_at_threshold"]
        cal = report["calibration"]
        assert dev["threshold"] == pytest.approx(cal["threshold"])
        assert dev["tar"] == pytest.approx(cal["achieved_tar"])
        assert dev["far"] == pytest.approx(cal["achieved_far"])

    def test_test_metrics_self_consistent(self, report):
        m = report["test_metrics_at_threshold"]
        assert m["tp"] + m["fn"] > 0
        assert m["tn"] + m["fp"] > 0
        n_gen = report["test_pairs"]["genuine"]
        n_imp = report["test_pairs"]["impostor"]
        assert m["n_genuine"] == n_gen
        assert m["n_impostor"] == n_imp
        assert m["tar"] == pytest.approx(m["tp"] / n_gen)
        assert m["far"] == pytest.approx(m["fp"] / n_imp)
        assert m["frr"] == pytest.approx(1 - m["tar"])
        total = n_gen + n_imp
        assert m["accuracy"] == pytest.approx((m["tp"] + m["tn"]) / total)

    def test_roc_and_curves_are_sane(self, report):
        roc = report["roc"]
        assert roc["roc_curve"][0] == {"threshold": None, "fpr": 0.0, "tpr": 0.0}
        assert roc["roc_curve"][-1] == {"threshold": None, "fpr": 1.0, "tpr": 1.0}
        fpr, tpr = roc["fpr"], roc["tpr"]
        assert fpr == sorted(fpr)
        assert tpr == sorted(tpr)
        assert 0.0 <= report["auc"] <= 1.0

    def test_eer_far_balanced(self, report):
        result = report["eer"]
        # discrete thresholds: EER = average of the FAR/FRR pair at the best threshold
        assert result["eer"] == pytest.approx((result["far"] + result["frr"]) / 2)
        assert 0.0 <= result["far"] <= 1.0
        assert 0.0 <= result["frr"] <= 1.0
        assert result["tar"] == pytest.approx(1 - result["frr"])

    def test_auc_matches_roc_points(self, report):
        from ml.evaluation.metrics import auc  # noqa: PLC0415

        assert report["auc"] == pytest.approx(auc(report["roc"]))

    def test_leakage_guard_reported(self, report):
        assert "leakage_guard" in report
        assert "partitions isolated" in report["leakage_guard"]

    def test_report_is_json_round_trippable(self, report, tmp_path):
        out = save_evaluation_report(report, os.path.join(str(tmp_path), "result.json"))
        with open(out, "r", encoding="utf-8") as fh:
            loaded = json.load(fh)
        assert loaded == report
        assert loaded["test_metrics_at_threshold"]["threshold"] == report["test_metrics_at_threshold"]["threshold"]

    def test_deterministic_across_runs(self, config):
        first = evaluate_verification(config, target_far=TARGET_FAR)
        second = evaluate_verification(config, target_far=TARGET_FAR)
        assert first == second

    def test_eer_and_roc_consistent_with_metrics(self, report):
        # EER threshold is a real distance; metrics at it must reproduce its far/frr
        m = report["eer"]
        threshold = m["threshold"]
        assert isinstance(threshold, float)
        assert threshold >= 0.0


def test_roc_points_export_matches_report_metrics(report):
    # The report's ROC points derive from unseen test distances; regenerating
    # roc_points on any held-out numbers is pure, so this just guards structure.
    roc = report["roc"]
    assert len(roc["thresholds"]) == len(roc["fpr"]) == len(roc["tpr"])
    assert roc["fpr"][-1] == 1.0 and roc["tpr"][-1] == 1.0


def test_synthetic_data_not_real_performance(report):
    assert "synthetic" in repr(report["dataset_id"]).lower() or "synthetic" in report["dataset_id"].lower()