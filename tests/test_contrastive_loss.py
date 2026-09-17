"""
Phase 6 contrastive-loss tests.

Run from the repo root::

    .venv/bin/python -m pytest tests/test_contrastive_loss.py -v
"""

import math

import pytest
import torch

from ml.verification.loss import ContrastiveLoss, contrastive_loss


class TestLabelAcceptance:

    def test_genuine_labels_accepted(self):
        loss = contrastive_loss(torch.tensor([0.5]), torch.tensor([1]))
        assert loss == pytest.approx(0.25, abs=1e-7)

    def test_impostor_labels_accepted(self):
        loss = contrastive_loss(torch.tensor([0.5]), torch.tensor([0]), margin=1.0)
        assert loss == pytest.approx(0.25, abs=1e-7)

    def test_mixed_labels_accepted(self):
        distances = torch.tensor([0.5, 0.5])
        loss = contrastive_loss(distances, torch.tensor([1, 0]), margin=1.0)
        assert loss == pytest.approx(0.25, abs=1e-7)

    def test_bool_labels_accepted(self):
        distances = torch.tensor([0.5, 0.5])
        labels = torch.tensor([True, False])
        a = contrastive_loss(distances, labels, margin=1.0)
        b = contrastive_loss(distances, torch.tensor([1, 0]), margin=1.0)
        assert a == pytest.approx(b, abs=1e-7)


class TestValidation:

    def test_invalid_labels_rejected(self):
        with pytest.raises(ValueError, match="binary"):
            contrastive_loss(torch.tensor([0.5]), torch.tensor([2]))

    def test_shape_mismatch_rejected(self):
        with pytest.raises(ValueError, match="batch"):
            contrastive_loss(torch.tensor([0.5, 0.5]), torch.tensor([1]))

    def test_2d_distance_rejected(self):
        with pytest.raises(ValueError, match="1-D"):
            contrastive_loss(torch.ones(2, 2), torch.tensor([1, 0]))

    def test_invalid_labels_type_rejected(self):
        with pytest.raises(TypeError, match="labels"):
            contrastive_loss(torch.tensor([0.5]), [1])

    def test_invalid_distance_type_rejected(self):
        with pytest.raises(TypeError, match="distance"):
            contrastive_loss([0.5], torch.tensor([1]))

    def test_negative_margin_rejected_functional(self):
        with pytest.raises(ValueError, match="margin"):
            contrastive_loss(torch.tensor([0.5]), torch.tensor([1]), margin=-1.0)

    def test_zero_margin_rejected_functional(self):
        with pytest.raises(ValueError, match="margin"):
            contrastive_loss(torch.tensor([0.5]), torch.tensor([1]), margin=0.0)

    def test_negative_margin_rejected_module(self):
        with pytest.raises(ValueError, match="margin"):
            ContrastiveLoss(margin=-1.0)

    def test_invalid_reduction_rejected(self):
        with pytest.raises(ValueError, match="reduction"):
            contrastive_loss(torch.tensor([0.5]), torch.tensor([1]), reduction="median")

    def test_empty_distance_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            contrastive_loss(torch.empty(0), torch.empty(0))


class TestAnalyticValues:

    def test_genuine_loss_is_distance_squared(self):
        for d, expected in ((0.0, 0.0), (0.5, 0.25), (1.0, 1.0), (2.0, 4.0)):
            loss = contrastive_loss(torch.tensor([d]), torch.tensor([1]))
            assert loss.item() == pytest.approx(expected, abs=1e-7)

    def test_impostor_below_margin_squared(self):
        # margin=2, d=1 -> (2-1)^2 = 1
        loss = contrastive_loss(torch.tensor([1.0]), torch.tensor([0]), margin=2.0)
        assert loss.item() == pytest.approx(1.0, abs=1e-7)

    def test_impostor_at_margin_zero(self):
        d = torch.tensor([2.0])
        assert contrastive_loss(d, torch.tensor([0]), margin=2.0).item() == pytest.approx(0.0, abs=1e-7)

    def test_impostor_above_margin_zero(self):
        loss = contrastive_loss(torch.tensor([3.0]), torch.tensor([0]), margin=2.0)
        assert loss.item() == pytest.approx(0.0, abs=1e-7)

    def test_batch_mean(self):
        # genuine d=0.5 -> 0.25 ; impostor d=0.75 margin=1 -> 0.0625 ; mean=0.15625
        distances = torch.tensor([0.5, 0.75])
        labels = torch.tensor([1, 0])
        loss = contrastive_loss(distances, labels, margin=1.0)
        assert loss.item() == pytest.approx(0.15625, abs=1e-7)

    def test_batch_sum(self):
        distances = torch.tensor([0.5, 0.75])
        labels = torch.tensor([1, 0])
        loss = contrastive_loss(distances, labels, margin=1.0, reduction="sum")
        assert loss.item() == pytest.approx(0.3125, abs=1e-7)


class TestPenaltyBehaviour:

    def test_genuine_penalized_when_distance_large(self):
        small = contrastive_loss(torch.tensor([0.1]), torch.tensor([1]))
        large = contrastive_loss(torch.tensor([2.0]), torch.tensor([1]))
        assert large.item() > small.item()

    def test_impostor_penalized_below_margin(self):
        below = contrastive_loss(torch.tensor([0.5]), torch.tensor([0]), margin=2.0)
        above = contrastive_loss(torch.tensor([2.0]), torch.tensor([0]), margin=2.0)
        assert below.item() > above.item() == pytest.approx(0.0, abs=1e-7)

    def test_impostor_at_above_margin_zero_contribution(self):
        for d in (2.0, 5.0, 100.0):
            loss = contrastive_loss(torch.tensor([d]), torch.tensor([0]), margin=2.0)
            assert loss.item() == pytest.approx(0.0, abs=1e-7)


class TestModule:

    def test_module_matches_functional(self):
        module = ContrastiveLoss(margin=1.5)
        d = torch.tensor([0.4, 1.2, 2.0])
        y = torch.tensor([1, 0, 0])
        assert module(d, y).item() == pytest.approx(
            contrastive_loss(d, y, margin=1.5).item(), abs=1e-9
        )

    def test_module_margin_exposed(self):
        module = ContrastiveLoss(margin=2.5)
        assert module.margin == 2.5


class TestGradients:

    def test_gradients_exist_and_finite(self):
        d = torch.tensor([1.0, 0.5], requires_grad=True)
        y = torch.tensor([0, 1])
        loss = contrastive_loss(d, y, margin=1.0)
        assert loss.requires_grad
        loss.backward()
        assert d.grad is not None
        assert bool(torch.isfinite(d.grad).all())

    def test_impostor_above_margin_zero_gradient(self):
        d = torch.tensor([5.0], requires_grad=True)
        loss = contrastive_loss(d, torch.tensor([0]), margin=1.0)
        loss.backward()
        assert d.grad.item() == pytest.approx(0.0, abs=1e-9)

    def test_loss_scalar_finite(self):
        d = torch.tensor([0.3, 2.0], requires_grad=True)
        loss = contrastive_loss(d, torch.tensor([1, 0]), margin=1.0)
        assert loss.ndim == 0
        assert math.isfinite(loss.item())

    def test_backprop_through_module(self):
        module = ContrastiveLoss(margin=1.0)
        d = torch.tensor([0.2, 3.0], requires_grad=True)
        loss = module(d, torch.tensor([1, 0]))
        loss.backward()
        assert bool(torch.isfinite(d.grad).all())