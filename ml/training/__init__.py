"""
Behavioral Biometric Authentication — Phase 7 Siamese training pipeline.

Trains the Phase 5 CNN+GRU :class:`~ml.encoder.BehavioralEncoder` through the
Phase 6 Siamese verification architecture + :class:`~ml.verification.
ContrastiveLoss` on the persisted Phase 4 synthetic dataset, with a strict
session-disjoint train/validation/test split, deterministic pair generation,
deterministic CPU training, atomic checkpoints, and reload verification.

This is an ENGINEERING/TRAINING-PIPELINE validation: sessions come from
**synthetic generated identities**, not real participants. Training metrics on
this dataset must not be reported as real-world biometric performance.

Public API (re-exported from ``ml.training``):

- :class:`TrainingConfig` — immutable run configuration.
- :func:`prepare_training_data` — load dataset, split, pair, fit scalers.
- :class:`SiamesePairDataset` / :func:`siamese_collate` / :func:`build_data_loaders`.
- :class:`SiameseTrainer` / :class:`TrainingHistory`.
- :func:`create_verifier` / :func:`create_criterion` / :func:`create_optimizer`.
- :func:`save_checkpoint` / :func:`load_checkpoint` / :func:`verify_reload`.
"""

from .checkpoint import (
    CHECKPOINT_FORMAT,
    CheckpointError,
    build_verifier_from_checkpoint,
    checkpoint_summary,
    config_from_checkpoint,
    history_from_checkpoint,
    load_checkpoint,
    restore_optimizer,
    save_checkpoint,
    verify_reload,
)
from .config import DEFAULT_CHECKPOINT, DEFAULT_DATASET, TrainingConfig
from .dataset import (
    SiamesePairDataset,
    TrainingData,
    build_data_loaders,
    make_torch_generator,
    prepare_training_data,
    siamese_collate,
)
from .trainer import (
    SiameseTrainer,
    TrainingHistory,
    create_criterion,
    create_optimizer,
    create_verifier,
    prepare_loaders,
    run_train_epoch,
    run_validation,
)

__version__ = "1.0.0"

__all__ = [
    "CHECKPOINT_FORMAT",
    "CheckpointError",
    "DEFAULT_CHECKPOINT",
    "DEFAULT_DATASET",
    "SiamesePairDataset",
    "SiameseTrainer",
    "TrainingConfig",
    "TrainingData",
    "TrainingHistory",
    "build_data_loaders",
    "build_verifier_from_checkpoint",
    "checkpoint_summary",
    "config_from_checkpoint",
    "create_criterion",
    "create_optimizer",
    "create_verifier",
    "history_from_checkpoint",
    "load_checkpoint",
    "make_torch_generator",
    "prepare_loaders",
    "prepare_training_data",
    "restore_optimizer",
    "run_train_epoch",
    "run_validation",
    "save_checkpoint",
    "siamese_collate",
    "verify_reload",
]