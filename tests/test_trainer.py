"""
Phase 7 — trainer tests.

Covers (per Phase 7 spec):
  1.  A tiny training run completes successfully.
  2.  Loss is finite (train + validation).
  3.  Gradients are finite.
  4.  Parameters actually update after optimizer.step().
  5.  History contains train and validation losses.
  6.  Same seed + same configuration is reproducible (history + weights).
  7.  A different seed can produce different training behaviour.
  8.  The Phase 5 encoder output contract remains intact.
  9.  The Siamese branches still share parameters.
  10. Contrastive loss receives correct labels / distances.

Run:
    .venv/bin/python -m pytest tests/test_trainer.py -v
"""

import math
import os

import pytest
import torch
from torch import nn

from ml.verification import ContrastiveLoss
from ml.training import (
    TrainingConfig,
    create_criterion,
    create_optimizer,
    create_verifier,
    prepare_training_data,
    run_train_epoch,
    run_validation,
)
from ml.training.dataset import SiamesePairDataset, siamese_collate
from ml.training.trainer import SiameseTrainer, TrainingHistory
from torch.utils.data import DataLoader

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "datasets",
    "dataset_synthetic.json",
)


def _fast_config(**overrides):
    base = dict(
        dataset_path=DATASET_PATH,
        seed=42,
        n_genuine_train=8,
        n_impostor_train=8,
        n_genuine_val=2,
        n_impostor_val=4,
        n_genuine_test=3,
        n_impostor_test=3,
        batch_size=4,
        epochs=3,
    )
    base.update(overrides)
    return TrainingConfig(**base)


class _CriterionSpy:
    """Records the last distance/label tensors forwarded to the loss."""

    def __init__(self, inner):
        self.inner = inner
        self.last_distances = None
        self.last_labels = None
        self.last_loss = None

    def __call__(self, distance, labels):
        self.last_distances = distance.detach().clone()
        self.last_labels = labels.clone()
        self.last_loss = self.inner(distance, labels)
        return self.last_loss


class _ModelRecordingSpy(nn.Module):
    """nn.Module wrapper that records the last forward's sessions and output."""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner
        self.last_sessions_a = None
        self.last_sessions_b = None
        self.last_out = None

    def forward(self, sessions_a, sessions_b, keyboard_scaler=None, mouse_scaler=None):
        self.last_sessions_a = list(sessions_a)
        self.last_sessions_b = list(sessions_b)
        self.last_out = self.inner(
            sessions_a, sessions_b,
            keyboard_scaler=keyboard_scaler,
            mouse_scaler=mouse_scaler,
        )
        return self.last_out


def _build(config, data):
    model = create_verifier(config)
    criterion = create_criterion(config)
    optimizer = create_optimizer(model, config)
    return model, criterion, optimizer


@pytest.fixture()
def data():
    return prepare_training_data(_fast_config())


def _train_loader(data, config, generator=None):
    return DataLoader(
        SiamesePairDataset(data.train_pairs),
        batch_size=config.batch_size,
        shuffle=True,
        generator=generator,
        collate_fn=siamese_collate,
    )


def _val_loader(data, config):
    return DataLoader(
        SiamesePairDataset(data.val_pairs),
        batch_size=config.batch_size,
        shuffle=False,
        collate_fn=siamese_collate,
    )


class TestTrainingLoop:

    def test_tiny_run_completes(self, data):
        config = _fast_config(epochs=1)
        model, criterion, optimizer = _build(config, data)
        trainer = SiameseTrainer(
            model, criterion, optimizer, config,
            keyboard_scaler=data.keyboard_scaler,
            mouse_scaler=data.mouse_scaler,
        )
        history = trainer.fit(_train_loader(data, config), _val_loader(data, config))
        assert len(history) == 1
        assert history.records[0]["epoch"] == 1

    def test_losses_are_finite(self, data):
        config = _fast_config(epochs=2)
        model, criterion, optimizer = _build(config, data)
        trainer = SiameseTrainer(
            model, criterion, optimizer, config,
            keyboard_scaler=data.keyboard_scaler,
            mouse_scaler=data.mouse_scaler,
        )
        history = trainer.fit(_train_loader(data, config), _val_loader(data, config))
        for record in history.records:
            assert math.isfinite(record["train_loss"])
            assert math.isfinite(record["val_loss"])
            assert record["train_loss"] >= 0.0
            assert record["val_loss"] >= 0.0

    def test_gradients_are_finite_after_epoch(self, data):
        config = _fast_config(epochs=1)
        model, criterion, optimizer = _build(config, data)
        run_train_epoch(
            model, criterion, optimizer,
            _train_loader(data, config),
            keyboard_scaler=data.keyboard_scaler,
            mouse_scaler=data.mouse_scaler,
            gradient_clip=config.gradient_clip,
        )
        for name, param in model.shared_encoder.named_parameters():
            assert param.grad is not None, name
            assert bool(torch.isfinite(param.grad).all()), name

    def test_parameters_actually_update(self, data):
        config = _fast_config(epochs=1)
        model, criterion, optimizer = _build(config, data)
        before = {n: p.detach().clone() for n, p in model.shared_encoder.named_parameters()}
        run_train_epoch(
            model, criterion, optimizer,
            _train_loader(data, config),
            keyboard_scaler=data.keyboard_scaler,
            mouse_scaler=data.mouse_scaler,
            gradient_clip=config.gradient_clip,
        )
        deltas = {
            n: float((before[n] - p.detach()).abs().sum())
            for n, p in model.shared_encoder.named_parameters()
        }
        assert sum(deltas.values()) > 0.0, "no parameter moved after optimizer.step()"

    def test_history_contains_train_and_val_losses(self, data):
        config = _fast_config(epochs=3)
        model, criterion, optimizer = _build(config, data)
        trainer = SiameseTrainer(
            model, criterion, optimizer, config,
            keyboard_scaler=data.keyboard_scaler,
            mouse_scaler=data.mouse_scaler,
        )
        history = trainer.fit(_train_loader(data, config), _val_loader(data, config))
        assert [int(r["epoch"]) for r in history.records] == [1, 2, 3]
        assert all("train_loss" in r and "val_loss" in r for r in history.records)
        serialized = history.to_list()
        revived = TrainingHistory.from_list(serialized)
        assert revived.approx_equal(history)


class TestReproducibility:

    def _train(self, config, data):
        model, criterion, optimizer = _build(config, data)
        trainer = SiameseTrainer(
            model, criterion, optimizer, config,
            keyboard_scaler=data.keyboard_scaler,
            mouse_scaler=data.mouse_scaler,
        )
        history = trainer.fit(_train_loader(data, config), _val_loader(data, config))
        return history, model.state_dict()

    def test_same_seed_same_config_is_reproducible(self, data):
        config = _fast_config(epochs=3)
        h1, s1 = self._train(config, data)
        h2, s2 = self._train(config, data)
        assert h1.approx_equal(h2)
        assert sorted(s1.keys()) == sorted(s2.keys())
        assert all(bool(torch.equal(s1[k], s2[k])) for k in s1)

    def test_different_seed_changes_behaviour(self, data):
        h1, _ = self._train(_fast_config(seed=42, epochs=3), data)
        h2, _ = self._train(_fast_config(seed=43, epochs=3), data)
        assert not h1.approx_equal(h2, atol=1e-9)


class TestContracts:

    def test_phase5_encoder_contract_intact_after_training(self, data):
        config = _fast_config(epochs=2)
        model = create_verifier(config)
        criterion = create_criterion(config)
        optimizer = create_optimizer(model, config)
        trainer = SiameseTrainer(
            model, criterion, optimizer, config,
            keyboard_scaler=data.keyboard_scaler,
            mouse_scaler=data.mouse_scaler,
        )
        trainer.fit(_train_loader(data, config), _val_loader(data, config))
        encoded = model.shared_encoder.encode_entries(
            data.test_sessions[:3],
            keyboard_scaler=data.keyboard_scaler,
            mouse_scaler=data.mouse_scaler,
        )
        single = model.shared_encoder.encode_session(
            data.test_sessions[0],
            keyboard_scaler=data.keyboard_scaler,
            mouse_scaler=data.mouse_scaler,
        )
        assert encoded.shape == (3, 128)
        assert single.shape == (128,)

    def test_siamese_branches_share_single_encoder(self, data):
        config = _fast_config(epochs=1)
        model = create_verifier(config)
        # Only one BehavioralEncoder lives inside the verifier, and it backs
        # both branches — one backward pass assigns gradients to ALL of its
        # parameters (i.e. both branch paths contributed to the shared weights).
        encoder_modules = [
            m for m in model.modules()
            if m.__class__.__name__ == "BehavioralEncoder"
        ]
        assert len(encoder_modules) == 1
        criterion = create_criterion(config)
        optimizer = create_optimizer(model, config)
        run_train_epoch(
            model, criterion, optimizer,
            _train_loader(data, config),
            keyboard_scaler=data.keyboard_scaler,
            mouse_scaler=data.mouse_scaler,
            gradient_clip=config.gradient_clip,
        )
        grads = [p.grad for n, p in model.shared_encoder.named_parameters() if p.requires_grad]
        assert all(g is not None and bool(torch.isfinite(g).all()) for g in grads)

    def test_contrastive_loss_receives_correct_labels_and_distances(self, data):
        config = _fast_config(epochs=1)
        inner = create_verifier(config)
        model = _ModelRecordingSpy(inner)
        spy = _CriterionSpy(ContrastiveLoss(margin=config.margin))
        optimizer = create_optimizer(inner, config)
        loader = _train_loader(data, config)

        run_train_epoch(
            model, spy, optimizer, loader,
            keyboard_scaler=data.keyboard_scaler,
            mouse_scaler=data.mouse_scaler,
            gradient_clip=None,
        )
        assert spy.last_labels is not None
        assert spy.last_labels.shape == torch.Size([config.batch_size])
        # every captured label must equal the actual user relationship of the
        # sessions that the verifier produced its distance tensor for
        for i, (a, b) in enumerate(zip(model.last_sessions_a, model.last_sessions_b)):
            same_user = a["user_id"] == b["user_id"]
            assert int(spy.last_labels[i]) == (1 if same_user else 0)
            assert a["session_id"] != b["session_id"]
        # the criterion saw exactly the distances of THAT forward pass
        captured = model.last_out.distance.detach()
        assert torch.equal(spy.last_distances, captured)
        assert bool((spy.last_distances >= 0).all())

    def test_validation_does_not_accumulate_gradients(self, data):
        config = _fast_config(epochs=1)
        model, criterion, optimizer = _build(config, data)
        loss = run_validation(
            model, criterion, _val_loader(data, config),
            keyboard_scaler=data.keyboard_scaler,
            mouse_scaler=data.mouse_scaler,
        )
        assert loss >= 0.0
        assert all(p.grad is None for p in model.shared_encoder.parameters())


class TestTrainerGuards:

    def test_empty_train_loader_raises(self, data):
        config = _fast_config()
        model, criterion, optimizer = _build(config, data)
        empty = DataLoader(SiamesePairDataset([]), batch_size=config.batch_size)
        with pytest.raises(ValueError):
            run_train_epoch(model, criterion, optimizer, empty)

    def test_invalid_epochs_rejected(self, data):
        with pytest.raises(ValueError):
            _fast_config(epochs=0)

    def test_pairs_never_leave_their_partition_in_loaders(self, data):
        config = _fast_config()
        val_ids = {p.session_id_a for p in data.val_pairs} | {p.session_id_b for p in data.val_pairs}
        loader = _val_loader(data, config)
        for sessions_a, sessions_b, _labels in loader:
            for sid in [s["session_id"] for s in sessions_a + sessions_b]:
                assert sid in val_ids