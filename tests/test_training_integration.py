"""
Phase 7 — end-to-end training smoke test on the persisted synthetic dataset.

Pipeline under test:

    persisted dataset (dataset_synthetic.json)
        -> user-aware train/val/test split (7/3 -> 5/2/3 per user)
        -> per-partition genuine/impostor pairs
        -> DataLoader (collate -> SiameseVerifier -> L2 -> contrastive loss)
        -> optimizer -> training epoch -> validation loss
        -> checkpoint save -> fresh model reload -> output parity

The test set is isolated: the trainer never sees it, and it is only touched
once, after training, to (a) reload-verify output parity and (b) compute a
clearly-labelled synthetic-data test loss.

Run:
    .venv/bin/python -m pytest tests/test_training_integration.py -v
"""

import os

import torch

from ml.training import (
    CheckpointError,
    create_criterion,
    create_optimizer,
    create_verifier,
    load_checkpoint,
    prepare_loaders,
    prepare_training_data,
    run_validation,
    save_checkpoint,
    verify_reload,
)
from ml.training.trainer import SiameseTrainer

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "datasets",
    "dataset_synthetic.json",
)


def _isfinite(value):
    try:
        return value == value and value not in (float("inf"), float("-inf"))
    except TypeError:
        return False


def test_full_training_smoke(tmp_path):
    config = _config()
    data = prepare_training_data(config)

    assert data.session_count_summary() == {
        "train_sessions": 25,
        "val_sessions": 10,
        "test_sessions": 15,
    }
    assert data.pair_count_summary() == {
        "train": {"genuine": 6, "impostor": 6, "total": 12},
        "val": {"genuine": 2, "impostor": 4, "total": 6},
        "test": {"genuine": 3, "impostor": 3, "total": 6},
    }
    # leakage guard
    train_ids = {e["session_id"] for e in data.train_sessions}
    val_ids = {e["session_id"] for e in data.val_sessions}
    test_ids = {e["session_id"] for e in data.test_sessions}
    assert not (train_ids & val_ids | train_ids & test_ids | val_ids & test_ids)
    for pair in data.train_pairs:
        assert pair.session_id_a in train_ids and pair.session_id_b in train_ids

    model = create_verifier(config)
    criterion = create_criterion(config)
    optimizer = create_optimizer(model, config)
    train_loader, val_loader, test_loader, train_generator = prepare_loaders(config, data)
    trainer = SiameseTrainer(
        model, criterion, optimizer, config,
        keyboard_scaler=data.keyboard_scaler,
        mouse_scaler=data.mouse_scaler,
        train_generator=train_generator,
    )
    history = trainer.fit(train_loader, val_loader)
    assert len(history) == config.epochs
    for record in history.records:
        assert record["train_loss"] >= 0.0
        assert record["val_loss"] >= 0.0
        assert _isfinite(record["train_loss"]) and _isfinite(record["val_loss"])

    # ---- post-training, isolated test-set evaluation (synthetic-data only) --
    test_loss = run_validation(
        model, criterion, test_loader,
        keyboard_scaler=data.keyboard_scaler,
        mouse_scaler=data.mouse_scaler,
    )
    assert bool(torch.isfinite(torch.tensor(test_loss)))

    # ---- checkpoint + reload parity ----
    path = str(tmp_path / "integration.pt")
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

    # a deliberately-corrupted file must be refused
    bad = str(tmp_path / "bad.pt")
    torch.save({"format": "unknown"}, bad)
    try:
        load_checkpoint(bad)
        raised = False
    except CheckpointError:
        raised = True
    assert raised


def _config():
    from ml.training import TrainingConfig

    return TrainingConfig(
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