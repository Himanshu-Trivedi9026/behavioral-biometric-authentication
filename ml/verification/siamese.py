"""
Siamese verification module (Phase 6).

Architecture
------------
Two input sessions are pushed through a **single shared** Phase 5
:class:`ml.encoder.BehavioralEncoder` (the same module instance is used for
both branches — parameters are literally shared, not copies):

.. code-block:: text

    session A          session B
       |                  |
       v                  v
    shared BehavioralEncoder (one instance, one set of weights)
       |                  |
       v                  v
   embedding A [B,128]  embedding B [B,128]
              \\         //
               \\       //
                \\     //
                   (+-+)
                    (-)
            L2 distance [B]

Distance function
-----------------
Euclidean (L2) distance between the two behavioral embeddings, computed with a
tiny epsilon floor so the derivative stays finite even when embeddings are
identical::

    diff   = embedding_a - embedding_b
    distance[i] = sqrt( sum_j diff[i, j]^2 )   (stable, differentiable)

Expected shapes
---------------
* ``embedding_a`` / ``embedding_b``: ``[B, 128]``
* ``distance``: ``[B]``

Batching
--------
Both branches are encoded in one call: the ``B`` A-sessions and ``B`` B-sessions
are collated together, run through the shared encoder once, then split back.
This keeps the two branches on exactly the same parameters by construction and
is batch-independent (padding never leaks across items, Phase 5 contract).

No arbitrary extra embedding network is introduced in Phase 6: the 128-dim
Phase 5 representation IS the representation the verifier compares.

Identity / privacy
------------------
The verifier reads ONLY the behavioural sequences (``keyboard_sequence`` /
``mouse_sequence``) out of each session dict. ``user_id`` / ``session_id`` are
never part of the model computation (verified by tests).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Union

import torch
from torch import nn

from ml.encoder import BehavioralEncoder, EncoderConfig
from ml.encoder.input import has_behavior


@dataclass
class SiameseOutput:
    """Batched Siamese forward result.

    Attributes:
        embedding_a: behavioral embedding of the A side, ``[B, embedding_dim]``.
        embedding_b: behavioral embedding of the B side, ``[B, embedding_dim]``.
        distance:    per-pair Euclidean distance, ``[B]`` (non-negative).
    """

    embedding_a: torch.Tensor
    embedding_b: torch.Tensor
    distance: torch.Tensor


class SiameseVerifier(nn.Module):
    """Shared-weight Siamese verifier built around the Phase 5 encoder.

    Args:
        encoder: an existing :class:`~ml.encoder.BehavioralEncoder` to share
            between both branches. If ``None`` one is constructed from
            ``config`` (optionally under ``seed``). Passing an existing encoder
            is the recommended way to re-use trained Phase 5 weights.
        config: :class:`~ml.encoder.EncoderConfig` used only when ``encoder``
            is ``None``.
        seed:   torch seed used only when constructing a fresh encoder.

    Exactly **one** :class:`~ml.encoder.BehavioralEncoder` exists inside this
    module; ``embedding_dim`` therefore equals the encoder's ``embedding_dim``
    (default 128).
    """

    def __init__(
        self,
        encoder: Optional[BehavioralEncoder] = None,
        config: Optional[EncoderConfig] = None,
        seed: Optional[int] = None,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if encoder is not None and not isinstance(encoder, BehavioralEncoder):
            raise TypeError(
                "encoder must be a BehavioralEncoder or None, got {!r}".format(
                    type(encoder).__name__
                )
            )
        if not isinstance(eps, (int, float)) or eps <= 0.0:
            raise ValueError("eps must be a positive number, got {!r}".format(eps))
        self.shared_encoder = encoder or BehavioralEncoder(config=config, seed=seed)
        self.eps = float(eps)
        self.embedding_dim = self.shared_encoder.embedding_dim

    # ------------------------------------------------------------------
    # Distance
    # ------------------------------------------------------------------

    def distance(self, embedding_a: torch.Tensor, embedding_b: torch.Tensor) -> torch.Tensor:
        """Euclidean (L2) distance between two embedding batches -> ``[B]``.

        ``distance[i] = sqrt(sum_j (a[i,j] - b[i,j])^2)`` with an epsilon floor
        so the gradient remains finite at zero distance (stable sqrt).
        """
        if not isinstance(embedding_a, torch.Tensor) or not isinstance(embedding_b, torch.Tensor):
            raise TypeError("embeddings must be torch.Tensor")
        if embedding_a.ndim != 2 or embedding_b.ndim != 2:
            raise ValueError(
                "embeddings must be 2-D [B, D], got shapes {} and {}".format(
                    tuple(embedding_a.shape), tuple(embedding_b.shape)
                )
            )
        if embedding_a.shape[0] != embedding_b.shape[0]:
            raise ValueError(
                "embedding batch sizes differ: {} vs {}".format(
                    embedding_a.shape[0], embedding_b.shape[0]
                )
            )
        if embedding_a.shape[1] != embedding_b.shape[1]:
            raise ValueError(
                "embedding dims differ: {} vs {}".format(
                    embedding_a.shape[1], embedding_b.shape[1]
                )
            )
        diff = embedding_a - embedding_b
        return torch.sqrt(diff.square().sum(dim=1) + self.eps)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        sessions_a: Sequence[Mapping[str, Any]],
        sessions_b: Sequence[Mapping[str, Any]],
        keyboard_scaler: Any = None,
        mouse_scaler: Any = None,
        dtype: torch.dtype = torch.float32,
    ) -> SiameseOutput:
        """Encode two aligned lists of sessions with the shared encoder.

        Args:
            sessions_a: list of ``B`` dataset entries (the "A" side).
            sessions_b: list of ``B`` dataset entries (the "B" side).
            keyboard_scaler: optional fitted Phase 3 ``FeatureScaler``.
            mouse_scaler:    optional fitted Phase 3 ``FeatureScaler``.
            dtype:           tensor dtype for input preparation.

        Returns:
            :class:`SiameseOutput` with ``embedding_a [B,128]``,
            ``embedding_b [B,128]`` and ``distance [B]``.

        Raises:
            ValueError: if the lists have unequal length or a session has no
                behavioural content at all.
        """
        if not isinstance(sessions_a, (list, tuple)) or not isinstance(sessions_b, (list, tuple)):
            raise TypeError("sessions_a and sessions_b must be lists of session dicts")
        if len(sessions_a) != len(sessions_b):
            raise ValueError(
                "sessions_a and sessions_b must have equal length, got {} and {}".format(
                    len(sessions_a), len(sessions_b)
                )
            )
        batch = len(sessions_a)
        if batch == 0:
            embeddings = torch.zeros(0, self.embedding_dim, dtype=dtype)
            empty = torch.zeros(0, dtype=dtype)
            return SiameseOutput(embedding_a=embeddings, embedding_b=embeddings, distance=empty)

        for i in range(batch):
            if not has_behavior(sessions_a[i]):
                raise ValueError(
                    "sessions_a[{}] has no behavioural content; filter with "
                    "ml.encoder.input.has_behavior before verifying".format(i)
                )
            if not has_behavior(sessions_b[i]):
                raise ValueError(
                    "sessions_b[{}] has no behavioural content; filter with "
                    "ml.encoder.input.has_behavior before verifying".format(i)
                )

        # One collate + one shared-encoder pass over the full 2B stack, then
        # split; guarantees both branches used identical parameters and no
        # padding leaked across items.
        stack = list(sessions_a) + list(sessions_b)
        encoded = self.shared_encoder.encode_entries(
            stack,
            keyboard_scaler=keyboard_scaler,
            mouse_scaler=mouse_scaler,
            dtype=dtype,
        )
        embedding_a = encoded[:batch]
        embedding_b = encoded[batch:]
        distance = self.distance(embedding_a, embedding_b)
        return SiameseOutput(
            embedding_a=embedding_a,
            embedding_b=embedding_b,
            distance=distance,
        )

    # ------------------------------------------------------------------
    # Single-pair convenience
    # ------------------------------------------------------------------

    def verify(
        self,
        session_a: Mapping[str, Any],
        session_b: Mapping[str, Any],
        keyboard_scaler: Any = None,
        mouse_scaler: Any = None,
        dtype: torch.dtype = torch.float32,
    ) -> SiameseOutput:
        """Encode a single pair ``(session_a, session_b)`` -> batch of 1."""
        return self.forward(
            [session_a],
            [session_b],
            keyboard_scaler=keyboard_scaler,
            mouse_scaler=mouse_scaler,
            dtype=dtype,
        )


__all__ = ["SiameseOutput", "SiameseVerifier"]