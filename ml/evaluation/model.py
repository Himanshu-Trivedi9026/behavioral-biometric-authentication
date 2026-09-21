"""
Phase 8 — trained-model loading + embedding extraction.

Loads the Phase 7 checkpoint (`model_state_dict` + encoder configuration) into
a fresh :class:`~ml.verification.SiameseVerifier`, puts it in evaluation mode,
and wraps it in a read-only inference interface:

* ``embed_session``  ->  ``[128]`` torch.Tensor (one session)
* ``embed_sessions`` ->  ``[N, 128]`` torch.Tensor
* ``pair_distance``  ->  Phase 6 Euclidean L2 distance over cached embeddings

Inference rules
---------------
* The model is never re-trained and the Phase 7 checkpoint is never modified.
* All forward passes run under ``torch.no_grad()``; calling a wrapper method
  never changes model parameters or gradient state.
* The scale parameters are **not** manufactured here: the caller supplies the
  ``FeatureScaler`` objects that came from the training/enrollment pipeline,
  and the same scalers are reused for every probe/test session (never fitted
  on probe data). With ``keyboard_scaler=None`` the encoder consumes raw
  feature values (documented: that is off the training-time distribution).

Checkpoint safety: loading goes through ``ml.training.checkpoint.load_checkpoint``
(``weights_only=True`` + format header check), so a missing/corrupt file fails
loudly with :class:`~ml.training.CheckpointError`.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Mapping, Optional, Sequence

import torch

from ml.encoder import EncoderConfig
from ml.encoder.input import has_behavior
from ml.training import (
    CheckpointError,
    build_verifier_from_checkpoint,
    config_from_checkpoint,
    load_checkpoint,
)
from ml.verification import SiameseVerifier


@dataclass(frozen=True)
class _LoadedCheckpointInfo:
    """Read-only metadata about the checkpoint a verifier was built from."""

    checkpoint_path: str
    epoch: int
    seed: int
    training_config: Any


class LoadedVerifier:
    """Inference-only wrapper around the trained Siamese behavioral encoder.

    The wrapper never trains; it just gives callers a clean surface for
    embedding extraction and Phase 6-style pair distances.
    """

    def __init__(
        self,
        verifier: SiameseVerifier,
        *,
        checkpoint_path: str,
        epoch: int,
        seed: int,
        training_config: Any,
    ) -> None:
        if not isinstance(verifier, SiameseVerifier):
            raise TypeError("verifier must be a SiameseVerifier")
        self._verifier = verifier
        self._verifier.eval()
        self.checkpoint_path = checkpoint_path
        self.epoch = int(epoch)
        self.seed = int(seed)
        self.training_config = training_config

    # -- introspection -------------------------------------------------------

    @property
    def embedding_dim(self) -> int:
        """Embedding dimension (Phase 5 default 128)."""
        return self._verifier.embedding_dim

    @property
    def encoder_config(self) -> EncoderConfig:
        return self._verifier.shared_encoder.config

    @property
    def checkpoint_id(self) -> str:
        return os.path.basename(self.checkpoint_path)

    @property
    def model(self) -> SiameseVerifier:
        """Access to the underlying module (already in ``eval()`` mode)."""
        return self._verifier

    # -- embedding extraction ------------------------------------------------

    @torch.no_grad()
    def embed_session(
        self,
        session: Mapping[str, Any],
        keyboard_scaler: Any = None,
        mouse_scaler: Any = None,
    ) -> torch.Tensor:
        """Embed one session -> ``[embedding_dim]`` finite torch.Tensor."""
        if not isinstance(session, Mapping):
            raise TypeError("session must be a session/entry dict")
        if not has_behavior(session):
            raise ValueError(
                "session {} has no behavioural content; cannot embed".format(
                    session.get("session_id", "<unknown>")
                )
            )
        embedding = self._verifier.shared_encoder.encode_session(
            session,
            keyboard_scaler=keyboard_scaler,
            mouse_scaler=mouse_scaler,
        )
        if embedding.shape != (self.embedding_dim,):
            raise RuntimeError(
                "session embedding has shape {}, expected ({},)".format(
                    tuple(embedding.shape), self.embedding_dim
                )
            )
        if not bool(torch.isfinite(embedding).all()):
            raise RuntimeError("session embedding contains non-finite values")
        return embedding

    @torch.no_grad()
    def embed_sessions(
        self,
        sessions: Sequence[Mapping[str, Any]],
        keyboard_scaler: Any = None,
        mouse_scaler: Any = None,
        batch_size: int = 16,
    ) -> torch.Tensor:
        """Embed many sessions -> ``[N, embedding_dim]`` finite tensor.

        Sessions are encoded in sub-batches of ``batch_size``; the same scalers
        are applied to every session.
        """
        if not isinstance(sessions, (list, tuple)):
            raise TypeError("sessions must be a list/tuple of session dicts")
        for i, session in enumerate(sessions):
            if not has_behavior(session):
                raise ValueError("sessions[{}] has no behavioural content".format(i))
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")

        chunks: List[torch.Tensor] = []
        for start in range(0, len(sessions), batch_size):
            sub = list(sessions[start : start + batch_size])
            encoded = self._verifier.shared_encoder.encode_entries(
                sub,
                keyboard_scaler=keyboard_scaler,
                mouse_scaler=mouse_scaler,
            )
            if not bool(torch.isfinite(encoded).all()):
                raise RuntimeError("embedding chunk contains non-finite values")
            chunks.append(encoded)
        embeddings = torch.cat(chunks, dim=0) if chunks else torch.zeros(
            0, self.embedding_dim
        )
        return embeddings

    # -- distances -----------------------------------------------------------

    @torch.no_grad()
    def pair_distance(
        self, embeddings_a: torch.Tensor, embeddings_b: torch.Tensor
    ) -> torch.Tensor:
        """Phase 6 Euclidean L2 distances between two embedding batches."""
        return self._verifier.distance(embeddings_a, embeddings_b)

    @torch.no_grad()
    def verify_pair(
        self,
        session_a: Mapping[str, Any],
        session_b: Mapping[str, Any],
        keyboard_scaler: Any = None,
        mouse_scaler: Any = None,
    ) -> float:
        """Distance between two sessions (batch of 1) as a Python float."""
        out = self._verifier.verify(
            session_a,
            session_b,
            keyboard_scaler=keyboard_scaler,
            mouse_scaler=mouse_scaler,
        )
        return float(out.distance[0].detach().cpu().item())


def load_verifier(checkpoint_path: str) -> LoadedVerifier:
    """Load the Phase 7 checkpoint into an evaluation-mode wrapper.

    Args:
        checkpoint_path: path to the Phase 7 ``.pt`` checkpoint.

    Returns:
        A :class:`LoadedVerifier` ready for embedding extraction.

    Raises:
        CheckpointError: if the file is missing, corrupt, or not a Phase 7
            checkpoint (never tries to load an arbitrary pickle).
    """
    if not isinstance(checkpoint_path, str) or not checkpoint_path:
        raise ValueError("checkpoint_path must be a non-empty string")
    if not os.path.exists(checkpoint_path):
        raise CheckpointError("checkpoint file does not exist: {}".format(checkpoint_path))
    payload = load_checkpoint(checkpoint_path)
    verifier = build_verifier_from_checkpoint(payload)
    verifier.eval()
    config = config_from_checkpoint(payload)
    return LoadedVerifier(
        verifier,
        checkpoint_path=checkpoint_path,
        epoch=int(payload["epoch"]),
        seed=int(payload["seed"]),
        training_config=config,
    )


__all__ = ["LoadedVerifier", "load_verifier"]