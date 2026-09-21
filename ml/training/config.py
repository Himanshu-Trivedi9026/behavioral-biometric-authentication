"""
Phase 7 training configuration (frozen dataclass with validation).

Everything the training pipeline needs is captured in one immutable object so
that a checkpoint can store the exact configuration that produced it and any
run can be reproduced by re-serializing the same values.

Defaults are tuned for the **synthetic** engineering dataset
(``data/datasets/dataset_synthetic.json``: 5 users, 50 sessions) and for CPU
execution. They are deliberately modest: no hyperparameter search is performed
in Phase 7 and no claim is made that synthetic results transfer to real
biometric performance.

Split strategy (documented, deterministic)
------------------------------------------
1. Outer split: ``train_fraction`` of every user's sessions -> train, the rest
   -> test (default ``0.7`` -> **7 train / 3 test** per user; 25 / 15 sessions
   globally).
2. Inner split: ``val_fraction`` of the *outer train* partition -> validation,
   the rest -> training (default ``2/7`` -> **5 train / 2 val / 3 test** per
   user; 25 / 10 / 15 sessions globally). ``val_fraction`` is the fraction of
   the 7 training sessions that become validation: ``round(7 * 2/7) = 2``.

Every user stays represented in each partition (each user has >= 1 session in
every required split), so no user-level leakage or drop-out occurs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from ml.encoder import EncoderConfig

DEFAULT_DATASET = "data/datasets/dataset_synthetic.json"
DEFAULT_CHECKPOINT = "models/siamese_behavioral_encoder.pt"


@dataclass(frozen=True)
class TrainingConfig:
    """Immutable configuration for a Phase 7 Siamese training run."""

    dataset_path: str = DEFAULT_DATASET
    seed: int = 42
    train_fraction: float = 0.7
    val_fraction: float = 2 / 7

    n_genuine_train: int = 40
    n_impostor_train: int = 40
    n_genuine_val: int = 5
    n_impostor_val: int = 10
    n_genuine_test: int = 15
    n_impostor_test: int = 15

    epochs: int = 10
    batch_size: int = 8
    learning_rate: float = 1e-3
    margin: float = 1.0
    weight_decay: float = 0.0
    gradient_clip: Optional[float] = 1.0
    num_workers: int = 0

    checkpoint_path: str = DEFAULT_CHECKPOINT
    encoder_config: Optional[EncoderConfig] = field(default=None)
    device: str = "cpu"

    def __post_init__(self) -> None:
        if not self.dataset_path or not isinstance(self.dataset_path, str):
            raise ValueError("dataset_path must be a non-empty string")
        if not isinstance(self.seed, int):
            raise TypeError("seed must be an integer")
        if not (0.0 < self.train_fraction < 1.0):
            raise ValueError("train_fraction must be in (0, 1), got {!r}".format(self.train_fraction))
        if not (0.0 <= self.val_fraction < 1.0):
            raise ValueError("val_fraction must be in [0, 1), got {!r}".format(self.val_fraction))
        if self.val_fraction == 0:
            raise ValueError("val_fraction must be > 0 so a validation partition exists")
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1, got {!r}".format(self.epochs))
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1, got {!r}".format(self.batch_size))
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive, got {!r}".format(self.learning_rate))
        if self.margin <= 0:
            raise ValueError("margin must be positive, got {!r}".format(self.margin))
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0, got {!r}".format(self.weight_decay))
        if self.gradient_clip is not None and self.gradient_clip <= 0:
            raise ValueError("gradient_clip must be > 0 or None, got {!r}".format(self.gradient_clip))
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0, got {!r}".format(self.num_workers))
        for name in (
            "n_genuine_train",
            "n_impostor_train",
            "n_genuine_val",
            "n_impostor_val",
            "n_genuine_test",
            "n_impostor_test",
        ):
            if getattr(self, name) < 0:
                raise ValueError("{}.{} must be >= 0, got {!r}".format(type(self).__name__, name, getattr(self, name)))
        if not self.checkpoint_path or not isinstance(self.checkpoint_path, str):
            raise ValueError("checkpoint_path must be a non-empty string")

    @property
    def encoder(self) -> EncoderConfig:
        """The encoder configuration used to build the shared model."""
        return self.encoder_config or EncoderConfig()

    def pair_counts(self) -> dict:
        """Name -> requested pair count (for reproducible logging/docs)."""
        return {
            "n_genuine_train": self.n_genuine_train,
            "n_impostor_train": self.n_impostor_train,
            "n_genuine_val": self.n_genuine_val,
            "n_impostor_val": self.n_impostor_val,
            "n_genuine_test": self.n_genuine_test,
            "n_impostor_test": self.n_impostor_test,
        }


__all__ = ["DEFAULT_CHECKPOINT", "DEFAULT_DATASET", "TrainingConfig"]