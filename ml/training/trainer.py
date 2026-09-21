"""
Phase 7 training loop (Siamese + contrastive loss).

The trainer owns the shared :class:`~ml.verification.SiameseVerifier`, the
Phase 6 :class:`~ml.verification.ContrastiveLoss`, and a plain Adam optimizer.
Each epoch it:

1. sets the model to **train** mode,
2. iterates the training batches: zero-grad -> Siamese forward -> contrastive
   loss -> backward -> optional gradient clipping -> ``step()``,
3. sets the model to **eval** mode and computes the **validation** loss
   (``torch.no_grad``).

The test set is never passed to the trainer. Reproducibility rules:

- the encoder is constructed under the run ``seed`` (see
  :func:`create_verifier`), so initialization is deterministic,
- the training DataLoader receives a :class:`torch.Generator` that is reseeded
  to ``seed + epoch`` at the start of every epoch, giving a deterministic but
  epoch-varying shuffle,
- no uncontrolled global randomness is consumed anywhere in the loop.

Phase 7 is validated on CPU (``num_workers=0``); nothing here performs
hyperparameter search.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Sequence

import torch
from torch import nn
from torch.utils.data import DataLoader

from ml.encoder import set_seed
from ml.verification import ContrastiveLoss, SiameseVerifier

from .config import TrainingConfig
from .dataset import TrainingData, build_data_loaders, make_torch_generator


@dataclass
class TrainingHistory:
    """Per-epoch train/validation loss records (kept in memory, JSON-able)."""

    records: List[Dict[str, float]] = field(default_factory=list)

    def record(self, epoch: int, train_loss: float, val_loss: float) -> None:
        self.records.append(
            {
                "epoch": float(epoch),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
            }
        )

    def __len__(self) -> int:
        return len(self.records)

    @property
    def epochs(self) -> List[int]:
        return [int(r["epoch"]) for r in self.records]

    def train_losses(self) -> List[float]:
        return [float(r["train_loss"]) for r in self.records]

    def val_losses(self) -> List[float]:
        return [float(r["val_loss"]) for r in self.records]

    def to_list(self) -> List[Dict[str, float]]:
        return [dict(r) for r in self.records]

    @classmethod
    def from_list(cls, records: Sequence[Dict[str, Any]]) -> "TrainingHistory":
        history = cls()
        for r in records:
            history.record(
                int(r["epoch"]),
                float(r["train_loss"]),
                float(r["val_loss"]),
            )
        return history

    def approx_equal(self, other: "TrainingHistory", atol: float = 1e-12) -> bool:
        if len(self.records) != len(other.records):
            return False
        for a, b in zip(self.records, other.records):
            for key in ("epoch", "train_loss", "val_loss"):
                if abs(float(a[key]) - float(b[key])) > atol:
                    return False
        return True


def create_verifier(config: TrainingConfig, seed: Optional[int] = None) -> SiameseVerifier:
    """Build a fresh shared Siamese verifier under a fixed RNG seed.

    Identical ``config`` + ``seed`` reproduces identical initialization (CPU).
    """
    actual_seed = config.seed if seed is None else seed
    set_seed(actual_seed)
    return SiameseVerifier(config=config.encoder, seed=actual_seed)


def create_criterion(config: TrainingConfig) -> ContrastiveLoss:
    """Phase 6 contrastive loss with the configured margin."""
    return ContrastiveLoss(margin=config.margin)


def create_optimizer(model: nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    """Adam over the shared encoder's parameters (no engine RNG needed)."""
    return torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )


def run_train_epoch(
    model: nn.Module,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    loader: DataLoader,
    keyboard_scaler: Any = None,
    mouse_scaler: Any = None,
    gradient_clip: Optional[float] = None,
) -> float:
    """Run one full training epoch; returns the mean contrastive loss."""
    model.train()
    total_loss = 0.0
    total_items = 0
    for sessions_a, sessions_b, labels in loader:
        optimizer.zero_grad()
        output = model(
            sessions_a,
            sessions_b,
            keyboard_scaler=keyboard_scaler,
            mouse_scaler=mouse_scaler,
        )
        loss = criterion(output.distance, labels)
        loss.backward()
        if gradient_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        batch_size = labels.numel()
        total_loss += float(loss.detach().cpu().item()) * batch_size
        total_items += batch_size
    if total_items == 0:
        raise ValueError("training epoch ran over an empty DataLoader")
    return total_loss / total_items


def run_validation(
    model: nn.Module,
    criterion: nn.Module,
    loader: DataLoader,
    keyboard_scaler: Any = None,
    mouse_scaler: Any = None,
) -> float:
    """Compute the mean contrastive loss over a (val/test) loader; no backward."""
    model.eval()
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for sessions_a, sessions_b, labels in loader:
            output = model(
                sessions_a,
                sessions_b,
                keyboard_scaler=keyboard_scaler,
                mouse_scaler=mouse_scaler,
            )
            loss = criterion(output.distance, labels)
            total_loss += float(loss.detach().cpu().item()) * labels.numel()
            total_items += labels.numel()
    if total_items == 0:
        raise ValueError("validation ran over an empty DataLoader")
    return total_loss / total_items


class SiameseTrainer:
    """Small deterministic trainer for the Siamese + contrastive objective.

    Only the **training** and **validation** DataLoaders enter ``fit``; the
    test partition stays isolated until explicit post-training evaluation.
    """

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        keyboard_scaler: Any = None,
        mouse_scaler: Any = None,
        train_generator: Optional[torch.Generator] = None,
    ) -> None:
        if config.epochs < 1:
            raise ValueError("epochs must be >= 1")
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.keyboard_scaler = keyboard_scaler
        self.mouse_scaler = mouse_scaler
        self.history = TrainingHistory()
        # This generator MUST be the one backing the training DataLoader's
        # shuffle (see build_data_loaders); reseeding it per epoch keeps the
        # shuffle deterministic yet epoch-varying.
        self.train_generator = train_generator or make_torch_generator(config.seed)

    def iter_train_epochs(
        self, train_loader: DataLoader, val_loader: DataLoader
    ) -> Iterator[Dict[str, float]]:
        """Yield ``{"epoch", "train_loss", "val_loss"}`` one epoch at a time.

        Reseeding ``train_loader``'s generator to ``seed + epoch`` keeps the
        shuffle deterministic while varying between epochs.
        """
        for epoch in range(1, self.config.epochs + 1):
            self.train_generator.manual_seed(self.config.seed + epoch)
            train_loss = run_train_epoch(
                self.model,
                self.criterion,
                self.optimizer,
                train_loader,
                keyboard_scaler=self.keyboard_scaler,
                mouse_scaler=self.mouse_scaler,
                gradient_clip=self.config.gradient_clip,
            )
            val_loss = run_validation(
                self.model,
                self.criterion,
                val_loader,
                keyboard_scaler=self.keyboard_scaler,
                mouse_scaler=self.mouse_scaler,
            )
            record = {"epoch": float(epoch), "train_loss": train_loss, "val_loss": val_loss}
            self.history.record(epoch, train_loss, val_loss)
            yield record

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        history: Optional[TrainingHistory] = None,
    ) -> TrainingHistory:
        """Train for ``config.epochs`` epochs; records and returns history."""
        if history is None:
            history = TrainingHistory()
        else:
            history.records.clear()
        self.history = history
        for _ in self.iter_train_epochs(train_loader, val_loader):
            pass
        return self.history


def prepare_loaders(
    config: TrainingConfig, data: TrainingData
) -> tuple[DataLoader, DataLoader, DataLoader, torch.Generator]:
    """Build train/val/test loaders plus the per-epoch-shuffle generator.

    Returns ``(train_loader, val_loader, test_loader, train_generator)`` —
    pass ``train_generator`` to :class:`SiameseTrainer` so per-epoch reseeding
    actually controls the train shuffle, and use the returned loaders only.
    """
    generator = make_torch_generator(config.seed)
    train_loader, val_loader, test_loader = build_data_loaders(data, config, generator)
    return train_loader, val_loader, test_loader, generator


__all__ = [
    "SiameseTrainer",
    "TrainingHistory",
    "create_criterion",
    "create_optimizer",
    "create_verifier",
    "prepare_loaders",
    "run_train_epoch",
    "run_validation",
]