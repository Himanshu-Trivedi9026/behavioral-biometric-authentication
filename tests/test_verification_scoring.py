"""
Phase 6 verification-scoring tests.

Run from the repo root::

    .venv/bin/python -m pytest tests/test_verification_scoring.py -v
"""

import inspect

import pytest
import torch

from ml.verification.scoring import (
    VerificationDecision,
    decide,
    decide_batch,
    decision,
    default_threshold,
)


class TestDecisionRule:

    def test_below_threshold_accepted(self):
        assert decide(0.4, 1.0) is True

    def test_equal_threshold_accepted(self):
        assert decide(1.0, 1.0) is True

    def test_above_threshold_rejected(self):
        assert decide(1.1, 1.0) is False

    def test_smaller_distance_more_likely_accepted(self):
        # smaller distance = more similar -> both accepted here but monotone
        assert decide(0.1, 0.5) and not decide(0.9, 0.5)

    def test_tensor_scalar_supported(self):
        assert decide(torch.tensor(0.5), torch.tensor(1.0)) is True


class TestThresholdValidation:

    def test_threshold_required_and_finite(self):
        with pytest.raises(ValueError, match="finite"):
            decide(0.5, float("nan"))

    def test_negative_threshold_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            decide(0.5, -1.0)

    def test_non_numeric_threshold_rejected(self):
        with pytest.raises(TypeError):
            decide(0.5, "high")

    def test_non_finite_distance_rejected(self):
        with pytest.raises(ValueError, match="finite"):
            decide(float("inf"), 1.0)

    def test_negative_distance_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            decide(-0.1, 1.0)


class TestDecisionRecord:

    def test_decision_dataclass_fields(self):
        d = decision(0.7, 1.0)
        assert isinstance(d, VerificationDecision)
        assert d.distance == 0.7
        assert d.threshold == 1.0
        assert d.accepted is True

    def test_to_dict(self):
        d = decision(2.0, 1.0).to_dict()
        assert d == {"distance": 2.0, "threshold": 1.0, "accepted": False}


class TestBatch:

    def test_decide_batch(self):
        result = decide_batch([0.1, 1.0, 1.5], 1.0)
        assert result == [True, True, False]

    def test_decide_batch_accepts_tensors(self):
        distances = torch.tensor([0.2, 2.0])
        result = decide_batch(distances, 1.0)
        assert result == [True, False]


class TestNoAdaptiveThresholding:

    def test_decide_is_pure_function(self):
        """decide depends only on (distance, threshold) — no data-driven or
        adaptive threshold computation exists in this module."""
        sig = inspect.signature(decide)
        assert list(sig.parameters) == ["distance", "threshold"]

    def test_module_exposes_no_calibration(self):
        import ml.verification.scoring as scoring

        public = {name.lower() for name in dir(scoring) if not name.startswith("_")}
        assert "calibrate" not in public
        assert "roc" not in public
        assert "eer" not in public

    def test_repeated_calls_identical(self):
        # same input -> same output, always (no state/adaptation)
        for _ in range(3):
            assert decide(0.5, 1.0) is decide(0.5, 1.0)

    def test_default_threshold_is_constant_and_documented(self):
        assert default_threshold() == 1.0
        assert default_threshold() == default_threshold()