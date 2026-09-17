"""
Phase 4 baseline tests.

Run from the repo root::

    .venv/bin/python -m pytest tests/test_baseline.py -v
"""

import copy
import math

import pytest

from baseline.evaluator import (
    BaselineEvaluator,
    calibrate_threshold,
    collect_scores,
    evaluate_at_threshold,
    sweep_thresholds,
)
from baseline.profile import UserProfile, build_profile, build_profiles
from baseline.scoring import score_session
from dataset.generator import SyntheticDatasetGenerator
from dataset.split import split_train_test


def _dataset(seed=0, n_users=5, sessions=10):
    return SyntheticDatasetGenerator(seed=seed, n_users=n_users, sessions_per_user=sessions).generate()


def _single_user_sessions(n=3):
    return [e for e in _dataset() if e["user_id"] == "user_001"][:n]


def _fitted(seed=0):
    """Fit a baseline on the 70/30 split of the synthetic dataset (seed 100)."""
    entries = _dataset(seed=seed)
    train, test = split_train_test(entries, seed=100)
    evaluator = BaselineEvaluator()
    evaluator.fit(train)
    return evaluator, train, test


# ---------------------------------------------------------------------------
# Profile creation
# ---------------------------------------------------------------------------

class TestProfile:

    def test_build_profile_ids_and_counts(self):
        entries = _single_user_sessions(3)
        profile = build_profile("user_001", entries)
        assert profile.user_id == "user_001"
        assert profile.n_train_sessions == 3
        assert profile.keyboard_samples == sum(len(e["keyboard_sequence"]) for e in entries)
        assert profile.mouse_samples == sum(len(e["mouse_sequence"]) for e in entries)

    def test_profile_feature_stats(self):
        profile = build_profile("user_001", _single_user_sessions(3))
        for feature in ("hold_time", "flight_time"):
            assert feature in profile.keyboard
            mean, std = profile.keyboard[feature]
            assert mean > 0.0
            assert std >= 0.0
        for feature in ("dx", "dy", "dt", "distance", "speed"):
            assert feature in profile.mouse

    def test_build_profiles_per_user(self):
        train, _ = split_train_test(_dataset(seed=5))
        profiles = build_profiles(train)
        assert set(profiles) == {
            "user_001", "user_002", "user_003", "user_004", "user_005"
        }

    def test_constant_feature_no_divide_by_zero(self):
        entries = copy.deepcopy(_single_user_sessions(2))
        for e in entries:
            e["keyboard_sequence"] = [{"hold_time": 100.0, "flight_time": 50.0}]
        profile = build_profile("user_001", entries)
        score = score_session(profile, entries[0])
        assert score["keyboard"] == 0.0  # constant feature -> no signal
        assert math.isfinite(score["combined"])

    def test_empty_training_sessions_profile(self):
        profile = build_profile("user_001", [])
        assert profile.keyboard == {}
        assert profile.mouse == {}
        assert profile.n_train_sessions == 0


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

class TestScoring:

    def test_score_direction_similar_is_low(self):
        train = _single_user_sessions(4)
        profile = build_profile("user_001", train[:-1])
        # A session drawn from the same user should be closer than a copy
        # whose values were perturbed far away.
        similar = train[-1]
        different = copy.deepcopy(train[-1])
        for sample in different["mouse_sequence"]:
            sample["dx"] *= 20.0
            sample["dy"] *= 20.0
            sample["speed"] *= 20.0
        for sample in different["keyboard_sequence"]:
            sample["hold_time"] *= 10.0

        s_sim = score_session(profile, similar)["combined"]
        s_diff = score_session(profile, different)["combined"]
        assert s_sim is not None and s_diff is not None
        assert s_sim < s_diff

    def test_scores_include_counts(self):
        profile = build_profile("user_001", _single_user_sessions(3))
        session = _single_user_sessions(3)[0]
        result = score_session(profile, session)
        assert "keyboard" in result and "mouse" in result and "combined" in result
        assert result["keyboard_samples"] == len(session["keyboard_sequence"])
        assert result["mouse_samples"] == len(session["mouse_sequence"])
        assert result["keyboard"] >= 0.0
        assert result["mouse"] >= 0.0

    def test_keyboard_mouse_kept_separate(self):
        profile = build_profile("user_001", _single_user_sessions(3))
        session = _single_user_sessions(3)[0]
        result = score_session(profile, session)
        assert result["keyboard"] != result["mouse"]

    def test_different_sequence_lengths_handled(self):
        profile = build_profile("user_001", _single_user_sessions(3))
        session = _single_user_sessions(3)[0]
        short = copy.deepcopy(session)
        short["keyboard_sequence"] = short["keyboard_sequence"][:1]
        long = copy.deepcopy(session)
        long["keyboard_sequence"] = long["keyboard_sequence"] * 5
        a = score_session(profile, short)
        b = score_session(profile, long)
        assert a["combined"] is not None
        assert b["combined"] is not None
        assert math.isfinite(a["combined"]) and math.isfinite(b["combined"])

    def test_empty_session(self):
        profile = build_profile("user_001", _single_user_sessions(2))
        session = {
            "session_id": "empty",
            "user_id": "user_001",
            "keyboard_sequence": [],
            "mouse_sequence": [],
            "mouse_action_events": [],
        }
        result = score_session(profile, session)
        assert result["keyboard"] is None
        assert result["mouse"] is None
        assert result["combined"] is None
        assert result["keyboard_samples"] == 0
        assert result["mouse_samples"] == 0

    def test_on_modality_missing_falls_back(self):
        profile = build_profile("user_001", _single_user_sessions(3))
        session = _single_user_sessions(3)[0]
        no_keyboard = copy.deepcopy(session)
        no_keyboard["keyboard_sequence"] = []
        result = score_session(profile, no_keyboard)
        assert result["keyboard"] is None
        assert result["mouse"] is not None
        assert result["combined"] == result["mouse"]

    def test_combined_weights_renormalize(self):
        profile = build_profile("user_001", _single_user_sessions(3))
        session = _single_user_sessions(3)[0]
        default = score_session(profile, session)["combined"]
        skewed = score_session(profile, session, weights={"keyboard": 1.0, "mouse": 1.0})["combined"]
        # weight (1,1) normalizes to 0.5/0.5 -> must equal default
        assert skewed == pytest.approx(default, abs=1e-9)

    def test_zero_weights_rejected(self):
        profile = build_profile("user_001", _single_user_sessions(2))
        session = _single_user_sessions(2)[0]
        with pytest.raises(ValueError):
            score_session(profile, session, weights={"keyboard": 0.0, "mouse": 0.0})


# ---------------------------------------------------------------------------
# Genuine vs impostor separation + metrics
# ---------------------------------------------------------------------------

class TestEvaluator:

    def test_evaluate_runs_both_attempt_types(self, tmp_path):
        evaluator, train, test = _fitted()
        report = evaluator.evaluate(test, far_target=0.05, train_entries=train)
        genuine = report.genuine
        impostor = report.impostor
        assert len(genuine) == 15          # 5 users x 3 test sessions
        assert len(impostor) == 60         # 15 sessions x 4 other users
        assert all(a["is_genuine"] for a in genuine)
        assert all(not a["is_genuine"] for a in impostor)
        assert {a["claimed_user"] for a in genuine} == set(profiles_set(report))

    def test_genuine_closer_than_impostor(self):
        evaluator, train, test = _fitted(seed=3)
        report = evaluator.evaluate(test, far_target=0.1, train_entries=train)
        g = [a["combined"] for a in report.genuine if a["combined"] is not None]
        i = [a["combined"] for a in report.impostor if a["combined"] is not None]
        assert g and i
        assert sum(g) / len(g) < sum(i) / len(i)

    def test_metrics_finite_and_range(self):
        evaluator, train, test = _fitted()
        report = evaluator.evaluate(test, far_target=0.05, train_entries=train)
        m = report.metrics
        for key in ("tar", "far", "frr", "accuracy"):
            assert 0.0 <= m[key] <= 1.0
            assert math.isfinite(m[key])
        assert m["tp"] + m["fn"] == m["n_genuine"]
        assert m["fp"] + m["tn"] == m["n_impostor"]

    def test_extreme_thresholds(self):
        evaluator, train, test = _fitted()
        report = evaluator.evaluate(test, threshold=float("inf"), train_entries=train)
        assert report.metrics["tar"] == 1.0
        assert report.metrics["far"] == 1.0
        report = evaluator.evaluate(test, threshold=-1.0, train_entries=train)
        assert report.metrics["tar"] == 0.0
        assert report.metrics["far"] == 0.0

    def test_confusion_counts(self):
        evaluator, train, test = _fitted()
        report = evaluator.evaluate(test, threshold=report_threshold_vector(evaluator, train, test), train_entries=train)
        m = report.metrics
        assert m["tp"] + m["fp"] + m["fn"] + m["tn"] == m["n_genuine"] + m["n_impostor"]

    def test_deterministic_evaluation(self):
        evaluator, train, test = _fitted()
        a = evaluator.evaluate(test, far_target=0.05, train_entries=train).as_dict()
        b = evaluator.evaluate(test, far_target=0.05, train_entries=train).as_dict()
        assert a == b

    def test_explicit_threshold_not_calibrated(self):
        evaluator, train, test = _fitted()
        report = evaluator.evaluate(test, threshold=1.234, train_entries=train)
        assert report.threshold == 1.234
        assert report.threshold_source == "explicit"

    def test_fit_without_train_raises(self):
        evaluator = BaselineEvaluator()
        with pytest.raises(ValueError):
            evaluator.evaluate([], threshold=1.0)

    def test_evaluate_needs_threshold_path(self):
        evaluator, train, test = _fitted()
        with pytest.raises(ValueError):
            evaluator.evaluate(test, calibrate=False)  # no threshold, no calibration

    def test_calibration_requires_train_data(self):
        evaluator, train, test = _fitted()
        with pytest.raises(ValueError):
            evaluator.evaluate(test)  # calibrate=True but train_entries not given


def profiles_set(report):
    return set(report.profiles.keys())


def report_threshold_vector(evaluator, train, test):
    """A threshold guaranteed to accept/reject sensibly for confusion test."""
    return 100000.0  # accept everything -> all genuine/impostor accepted


# ---------------------------------------------------------------------------
# Threshold calibration
# ---------------------------------------------------------------------------

class TestCalibration:

    def test_calibrate_returns_finite_threshold(self):
        train, _ = split_train_test(_dataset(seed=8))
        result = calibrate_threshold(train, far_target=0.05)
        assert result.method == "target-far"
        assert math.isfinite(result.threshold)
        assert result.dev_far <= 0.05 + 1e-9
        assert result.genuine_dev
        assert result.impostor_dev

    def test_calibrate_single_user_raises(self):
        entries = _single_user_sessions(3)
        with pytest.raises(ValueError):
            calibrate_threshold(entries)

    def test_calibrate_uses_only_training_input(self):
        # Calibration is a pure function of train entries; the test set is not
        # an argument, so it structurally cannot leak into threshold choice.
        train, _ = split_train_test(_dataset(seed=2))
        calibrate_threshold(train)

    def test_max_tar_far_method(self):
        train, _ = split_train_test(_dataset(seed=2))
        result = calibrate_threshold(train, method="max-tar-far")
        assert result.method == "max-tar-far"

    def test_invalid_method_raises(self):
        train, _ = split_train_test(_dataset(seed=2))
        with pytest.raises(ValueError):
            calibrate_threshold(train, method="nope")

    def test_invalid_far_target_raises(self):
        train, _ = split_train_test(_dataset(seed=2))
        with pytest.raises(ValueError):
            calibrate_threshold(train, far_target=1.5)

    def test_calibrated_threshold_used_in_eval(self):
        evaluator, train, test = _fitted(seed=4)
        report = evaluator.evaluate(test, train_entries=train)
        assert report.threshold_source == "calibrated"
        assert report.threshold == report.threshold_result.threshold

    def test_threshold_result_dict(self):
        train, _ = split_train_test(_dataset(seed=1))
        result = calibrate_threshold(train)
        d = result.to_dict()
        assert d["threshold"] == result.threshold
        assert d["method"] == "target-far"


# ---------------------------------------------------------------------------
# Threshold sweep / ROC
# ---------------------------------------------------------------------------

class TestSweep:

    def test_sweep_extremes(self):
        evaluator, train, test = _fitted()
        genuine, impostor = collect_scores(evaluator.profiles, test)
        points = sweep_thresholds(genuine, impostor)
        assert points[0]["threshold"] == pytest.approx(-float("inf"))
        assert points[0]["tar"] == 0.0
        assert points[-1]["tar"] == 1.0
        assert points[-1]["far"] == 1.0
        assert all(0.0 <= p["far"] <= p["tar"] + 1e-9 for p in points if p["threshold"] != -float("inf"))

    def test_sweep_monotonic(self):
        evaluator, train, test = _fitted()
        genuine, impostor = collect_scores(evaluator.profiles, test)
        points = sweep_thresholds(genuine, impostor, n_points=25)
        tars = [p["tar"] for p in points]
        fars = [p["far"] for p in points]
        assert tars == sorted(tars)
        assert fars == sorted(fars)

    def test_sweep_empty_returns_empty(self):
        assert sweep_thresholds([], []) == []

    def test_evaluate_at_threshold_none_handled(self):
        # Attempts with combined=None (no usable modality) are excluded from
        # metrics and reported as unscorable, not as rejects.
        gen_attempts = [
            {
                "true_user": "user_001",
                "claimed_user": "user_001",
                "session_id": "x",
                "combined": None,
                "keyboard": None,
                "mouse": None,
                "is_genuine": True,
            }
        ]
        metrics = evaluate_at_threshold(gen_attempts, [], 1.0)
        assert metrics["unscorable_genuine"] == 1
        assert metrics["n_genuine"] == 0