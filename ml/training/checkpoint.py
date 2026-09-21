"""
Phase 7 checkpoint persistence (save / safe load / reload verification).

Checkpoint format (``~``-prefix header, standard ``torch.state_dict`` payload):

.. code-block:: python

    {
        "format": "siamese-behavioral-checkpoint-v1",
        "seed": int,
        "epoch": int,
        "history": [{"epoch","train_loss","val_loss"}, ...],
        "training_config": {...},      # JSON-able, includes encoder_config
        "encoder_config": {...},       # EncoderConfig fields
        "model_state_dict": {...},     # SiameseVerifier state (shared encoder)
        "optimizer_state_dict": {...}, # Adam state
    }

Safety rules
------------
* Loading uses ``torch.load(..., weights_only=True)`` and rejects any payload
  that does not carry the expected ``format`` header, so an arbitrary pickle
  file is refused early.
* Saving is atomic: written to ``<path>.tmp`` then ``os.replace``.
* Reloading reconstructs a **fresh** :class:`~ml.verification.SiameseVerifier`
  from the stored ``encoder_config`` and loads the saved ``model_state_dict``,
  then callers can prove output parity with the original model.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from typing import Any, Dict, List, Mapping, Optional, Sequence

import torch

from ml.encoder import EncoderConfig, set_seed
from ml.verification import SiameseVerifier

from .config import TrainingConfig
from .trainer import TrainingHistory, create_optimizer

CHECKPOINT_FORMAT = "siamese-behavioral-checkpoint-v1"


class CheckpointError(Exception):
    """Raised when a checkpoint file is missing, corrupt, or unexpected."""


def _validate_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("format") != CHECKPOINT_FORMAT:
        raise CheckpointError(
            "not a Phase 7 checkpoint (expected format {!r}, got {!r})".format(
                CHECKPOINT_FORMAT, payload.get("format")
            )
        )
    for key in (
        "seed",
        "epoch",
        "history",
        "training_config",
        "encoder_config",
        "model_state_dict",
        "optimizer_state_dict",
    ):
        if key not in payload:
            raise CheckpointError("checkpoint is missing required key {!r}".format(key))


def save_checkpoint(
    path: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    history: TrainingHistory,
    config: TrainingConfig,
) -> str:
    """Write a Phase 7 checkpoint atomically; returns ``path``.

    Raises:
        ValueError: if ``epoch`` is not a positive int.
    """
    if epoch < 1:
        raise ValueError("cannot save a checkpoint for epoch {} (< 1)".format(epoch))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    payload = {
        "format": CHECKPOINT_FORMAT,
        "seed": config.seed,
        "epoch": int(epoch),
        "history": history.to_list(),
        "training_config": asdict(config),
        "encoder_config": asdict(config.encoder),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }
    temporary = "{}.tmp".format(path)
    torch.save(payload, temporary)
    os.replace(temporary, path)
    return path


def load_checkpoint(path: str) -> Dict[str, Any]:
    """Load and validate a checkpoint payload (weights_only, CPU).

    Raises:
        CheckpointError: on any of (a) a missing path, (b) an unreadable or
            unsafe file, or (c) a payload without the expected header/keys.
    """
    if not os.path.exists(path):
        raise CheckpointError("checkpoint file does not exist: {}".format(path))
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # noqa: BLE001 - convert every load failure
        raise CheckpointError("failed to load checkpoint {}: {!r}".format(path, exc)) from exc
    if not isinstance(payload, dict):
        raise CheckpointError("checkpoint payload must be a dict, got {!r}".format(type(payload).__name__))
    _validate_payload(payload)
    return payload


def config_from_checkpoint(payload: Mapping[str, Any]) -> TrainingConfig:
    """Reconstruct the :class:`TrainingConfig` that produced a checkpoint."""
    cfg = dict(payload["training_config"])
    raw = cfg.get("encoder_config") or {}
    if raw:
        try:
            encoder_config = EncoderConfig(**dict(raw))
        except TypeError as exc:
            raise CheckpointError("checkpoint encoder_config is not serializable: {!r}".format(exc)) from exc
    else:
        encoder_config = None
    cfg["encoder_config"] = encoder_config
    return TrainingConfig(**cfg)


def build_verifier_from_checkpoint(
    payload: Mapping[str, Any], seed: Optional[int] = None
) -> SiameseVerifier:
    """Build a fresh Siamese verifier and load the saved weights into it.

    The reconstructed verifier shares no state with the model that produced the
    checkpoint; it is initialized from the stored ``encoder_config`` and then
    its weights are overwritten by ``model_state_dict``.
    """
    _validate_payload(payload)
    actual_seed = payload.get("seed") if seed is None else seed
    set_seed(actual_seed)
    encoder_config = EncoderConfig(**payload["encoder_config"])
    verifier = SiameseVerifier(config=encoder_config, seed=actual_seed)
    result = verifier.load_state_dict(payload["model_state_dict"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise CheckpointError(
            "state_dict mismatch: missing={} unexpected={}".format(
                result.missing_keys, result.unexpected_keys
            )
        )
    return verifier


def restore_optimizer(
    payload: Mapping[str, Any],
    model: torch.nn.Module,
    config: Optional[TrainingConfig] = None,
) -> torch.optim.Optimizer:
    """Recreate the Adam optimizer and restore its saved state."""
    _validate_payload(payload)
    if config is None:
        config = config_from_checkpoint(payload)
    optimizer = create_optimizer(model, config)
    optimizer.load_state_dict(payload["optimizer_state_dict"])
    return optimizer


def history_from_checkpoint(payload: Mapping[str, Any]) -> TrainingHistory:
    """Recover the recorded train/val loss history stored in a checkpoint."""
    _validate_payload(payload)
    return TrainingHistory.from_list(payload["history"])


def verify_reload(
    payload: Mapping[str, Any],
    reference_model: torch.nn.Module,
    sessions_a: List[Dict[str, Any]],
    sessions_b: List[Dict[str, Any]],
    keyboard_scaler: Any = None,
    mouse_scaler: Any = None,
    atol: float = 1e-5,
) -> Dict[str, float]:
    """Prove a reloaded model reproduces the saved model's behaviour.

    Runs ``reference_model`` and a fresh verifier reconstructed from the
    checkpoint over the same inputs and compares the Siamese outputs.

    Returns the maximum per-component embedding / distance deviations.

    Raises:
        CheckpointError: if the reloaded output deviates beyond ``atol``.
    """
    reloaded = build_verifier_from_checkpoint(payload)
    with torch.no_grad():
        expected = reference_model(
            sessions_a,
            sessions_b,
            keyboard_scaler=keyboard_scaler,
            mouse_scaler=mouse_scaler,
        )
        actual = reloaded(
            sessions_a,
            sessions_b,
            keyboard_scaler=keyboard_scaler,
            mouse_scaler=mouse_scaler,
        )
    deviations = {
        "embedding_a_max_abs_diff": float(
            (expected.embedding_a - actual.embedding_a).abs().max().item()
        ),
        "embedding_b_max_abs_diff": float(
            (expected.embedding_b - actual.embedding_b).abs().max().item()
        ),
        "distance_max_abs_diff": float((expected.distance - actual.distance).abs().max().item()),
    }
    if max(deviations.values()) > atol:
        raise CheckpointError(
            "reloaded model output deviates from saved model: {}".format(deviations)
        )
    return deviations


def checkpoint_summary(payload: Mapping[str, Any]) -> Dict[str, Any]:
    """Human-readable summary of a loaded checkpoint (for CLI logs)."""
    _validate_payload(payload)
    history = payload.get("history") or []
    return {
        "format": payload["format"],
        "seed": payload["seed"],
        "epoch": payload["epoch"],
        "train_loss_last": history[-1]["train_loss"] if history else None,
        "val_loss_last": history[-1]["val_loss"] if history else None,
    }


__all__ = [
    "CHECKPOINT_FORMAT",
    "CheckpointError",
    "build_verifier_from_checkpoint",
    "checkpoint_summary",
    "config_from_checkpoint",
    "history_from_checkpoint",
    "load_checkpoint",
    "restore_optimizer",
    "save_checkpoint",
    "verify_reload",
]