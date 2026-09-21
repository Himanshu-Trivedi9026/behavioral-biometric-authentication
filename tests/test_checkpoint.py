"""
Phase 7 — checkpoint tests.

Covers (per Phase 7 spec):
  1.  Checkpoint file is created.
  2.  Checkpoint contains all required state.
  3.  A fresh model reloads it.
  4.  Reloaded model reproduces the saved model's output.
  5.  Optimizer state can be restored.
  6.  Corrupt / missing checkpoints fail clearly (no untrusted pickles).

Run:
    .venv/bin/python -m pytest tests/test_checkpoint.py -v
"""

import os

import pytest
import torch

from ml.training import (
    CheckpointError,
    TrainingConfig,
    build_verifier_from_checkpoint,
    config_from_checkpoint,
    create_criterion,
    create_optimizer,
    create_verifier,
    history_from_checkpoint,
    load_checkpoint,
    prepare_training_data,
    restore_optimizer,
    save_checkpoint,
    verify_reload,
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
        n_genuine_train=6,
        n_impostor_train=6,
        n_genuine_val=2,
        n_impostor_val=4,
        n_genuine_test=3,
        n_impostor_test=3,
        batch_size=4,
        epochs=2,
    )
    base.update(overrides)
    return TrainingConfig(**base)


@pytest.fixture()
def data():
    return prepare_training_data(_fast_config())


class TestCheckpoint:

    def test_checkpoint_is_created(self, data, tmp_path):
        path = str(tmp_path / "model.pt")
        config = _fast_config()
        model = create_verifier(config)
        optimizer = create_optimizer(model, config)
        history = TrainingHistory()
        history.record(1, 0.5, 0.4)
        save_checkpoint(path, model=model, optimizer=optimizer, epoch=1, history=history, config=config)
        assert os.path.exists(path)
        assert not os.path.exists(path + ".tmp")

    def test_checkpoint_contains_required_state(self, data, tmp_path):
        path = str(tmp_path / "model.pt")
        config = _fast_config()
        model = create_verifier(config)
        optimizer = create_optimizer(model, config)
        history = TrainingHistory()
        history.record(2, 0.25, 0.33)
        save_checkpoint(path, model=model, optimizer=optimizer, epoch=2, history=history, config=config)
        payload = load_checkpoint(path)
        for key in (
            "format",
            "seed",
            "epoch",
            "history",
            "training_config",
            "encoder_config",
            "model_state_dict",
            "optimizer_state_dict",
        ):
            assert key in payload
        assert payload["epoch"] == 2
        assert config_from_checkpoint(payload) == config

    def test_checkpoint_reload_is_deterministic_across_files(self, data, tmp_path):
        """Two indepently-produced (seeded) checkpoints reload to identical
        weights — training determinism, at the checkpoint-artifact level."""
        config = _fast_config()

        def _produce(path):
            model = create_verifier(config)
            save_checkpoint(path, model=model, optimizer=create_optimizer(model, config),
                            epoch=1, history=TrainingHistory(), config=config)
            return build_verifier_from_checkpoint(load_checkpoint(path)).state_dict()

        a = _produce(str(tmp_path / "a.pt"))
        b = _produce(str(tmp_path / "b.pt"))
        assert list(a.keys()) == list(b.keys())
        assert all(bool(torch.equal(a[k], b[k])) for k in a)


class TestReload:

    @pytest.fixture()
    def trained(self, data):
        config = _fast_config(epochs=2)
        model = create_verifier(config)
        criterion = create_criterion(config)
        optimizer = create_optimizer(model, config)
        train_loader = DataLoader(
            SiamesePairDataset(data.train_pairs), batch_size=config.batch_size,
            shuffle=True, collate_fn=siamese_collate,
        )
        val_loader = DataLoader(
            SiamesePairDataset(data.val_pairs), batch_size=config.batch_size,
            shuffle=False, collate_fn=siamese_collate,
        )
        trainer = SiameseTrainer(
            model, criterion, optimizer, config,
            keyboard_scaler=data.keyboard_scaler,
            mouse_scaler=data.mouse_scaler,
        )
        history = trainer.fit(train_loader, val_loader)
        return config, model, optimizer, history

    def test_fresh_model_reloads_checkpoint(self, trained, data, tmp_path):
        config, model, optimizer, history = trained
        path = str(tmp_path / "trained.pt")
        save_checkpoint(path, model=model, optimizer=optimizer, epoch=config.epochs,
                        history=history, config=config)
        payload = load_checkpoint(path)
        fresh = build_verifier_from_checkpoint(payload)
        assert list(fresh.state_dict().keys()) == list(model.state_dict().keys())
        for key in model.state_dict():
            assert torch.equal(fresh.state_dict()[key], model.state_dict()[key])

    def test_reloaded_model_reproduces_output(self, trained, data, tmp_path):
        config, model, optimizer, history = trained
        path = str(tmp_path / "trained.pt")
        save_checkpoint(path, model=model, optimizer=optimizer, epoch=config.epochs,
                        history=history, config=config)
        payload = load_checkpoint(path)
        a = data.test_pairs[0].session_a
        b = data.test_pairs[0].session_b
        deviations = verify_reload(
            payload, model, [a], [b],
            keyboard_scaler=data.keyboard_scaler,
            mouse_scaler=data.mouse_scaler,
        )
        assert max(deviations.values()) <= 1e-5
        assert deviations["distance_max_abs_diff"] < 1e-6

    def test_reloaded_history_matches(self, trained, data, tmp_path):
        config, model, optimizer, history = trained
        path = str(tmp_path / "trained.pt")
        save_checkpoint(path, model=model, optimizer=optimizer, epoch=config.epochs,
                        history=history, config=config)
        payload = load_checkpoint(path)
        revived = history_from_checkpoint(payload)
        assert revived.to_list() == history.to_list()

    def test_optimizer_state_can_be_restored(self, trained, data, tmp_path):
        config, model, optimizer, history = trained
        path = str(tmp_path / "trained.pt")
        save_checkpoint(path, model=model, optimizer=optimizer, epoch=config.epochs,
                        history=history, config=config)
        payload = load_checkpoint(path)
        fresh = build_verifier_from_checkpoint(payload)
        restored = restore_optimizer(payload, fresh, config=config)
        saved_state = optimizer.state_dict()
        restored_state = restored.state_dict()
        assert set(saved_state["param_groups"][0]) == set(restored_state["param_groups"][0])
        assert saved_state["param_groups"][0]["lr"] == restored_state["param_groups"][0]["lr"]
        for param_key, values in saved_state["state"].items():
            restored_values = restored_state["state"].get(param_key)
            assert restored_values is not None
            for tensor_key in ("exp_avg", "exp_avg_sq"):
                assert torch.equal(values[tensor_key], restored_values[tensor_key])


class TestFailures:

    def test_missing_checkpoint_fails_clearly(self, tmp_path):
        with pytest.raises(CheckpointError):
            load_checkpoint(str(tmp_path / "does-not-exist.pt"))

    def test_corrupt_checkpoint_fails_clearly(self, tmp_path):
        path = str(tmp_path / "corrupt.pt")
        with open(path, "w") as f:
            f.write("this is not a torch checkpoint")
        with pytest.raises(CheckpointError):
            load_checkpoint(path)

    def test_wrong_format_fails_clearly(self, tmp_path):
        path = str(tmp_path / "wrong.pt")
        torch.save({"format": "something-else", "epoch": 5}, path)
        with pytest.raises(CheckpointError):
            load_checkpoint(path)

    def test_missing_keys_fail_clearly(self, tmp_path):
        path = str(tmp_path / "missing.pt")
        torch.save({"format": "siamese-behavioral-checkpoint-v1", "epoch": 5}, path)
        with pytest.raises(CheckpointError):
            load_checkpoint(path)

    def test_save_rejects_zero_epoch(self, data, tmp_path):
        config = _fast_config()
        model = create_verifier(config)
        with pytest.raises(ValueError):
            save_checkpoint(str(tmp_path / "x.pt"), model=model,
                            optimizer=create_optimizer(model, config),
                            epoch=0, history=TrainingHistory(), config=config)