"""
Contrastive loss for the Phase 6 Siamese verifier.

Label convention (shared with ``ml.verification.pairs``)
---------------------------------------------------------
* ``label = 1`` -> genuine  (same user)  → push distance toward ``0``
* ``label = 0`` -> impostor (different users) → push distance toward/above the margin

Loss formulation
----------------
For a batch of pair distances ``D`` with binary labels ``y``::

    L_genuine   = y        * D^2
    L_impostor  = (1 - y) * max(0, margin - D)^2
    L           = mean( L_genuine + L_impostor )

Properties
----------
* Genuine pairs are penalized quadratically the farther apart they are.
* Impostor pairs are penalized only while they are **below** the margin; at/above
  the margin they contribute exactly ``0``.
* ``margin > 0`` is required (validated at construction time).
* The whole computation stays in PyTorch tensors — no NumPy, fully
  differentiable, produces a finite scalar.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from .pairs import GENUINE, IMPOSTOR, VALID_LABELS


def contrastive_loss(
    distance: torch.Tensor,
    labels,
    margin: float = 1.0,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> torch.Tensor:
    """Functional contrastive loss over a batch of Siamese distances.

    Args:
        distance: ``[B]`` non-negative pair distances (floats).
        labels:   ``[B]`` binary labels (``1`` genuine / ``0`` impostor) as a
                  tensor (long, float, or bool).
        margin:   positive scalar; impostor pairs are only penalized while
                  ``distance < margin``.
        reduction: ``"mean"`` (default) or ``"sum"``.
        eps:      small constant added under the impostor hinge (not used for
                  the genuine term); keeps everything finite and well-defined.

    Returns:
        A scalar ``torch.Tensor`` loss (requires grad on ``distance``).

    Raises:
        ValueError: on shape mismatches, non-binary labels, or an invalid
            ``margin`` / ``reduction``.
        TypeError:  on non-tensor inputs.
    """
    if not isinstance(distance, torch.Tensor):
        raise TypeError("distance must be a torch.Tensor, got {!r}".format(type(distance).__name__))
    if distance.ndim != 1:
        raise ValueError("distance must be 1-D [B], got {}".format(tuple(distance.shape)))
    if distance.numel() == 0:
        raise ValueError("distance must contain at least one pair")
    if not isinstance(labels, torch.Tensor):
        raise TypeError("labels must be a torch.Tensor, got {!r}".format(type(labels).__name__))
    if labels.ndim != 1:
        raise ValueError("labels must be 1-D [B], got {}".format(tuple(labels.shape)))
    if labels.shape[0] != distance.shape[0]:
        raise ValueError(
            "distance batch {} != labels batch {}".format(
                distance.shape[0], labels.shape[0]
            )
        )
    if labels.dtype == torch.bool:
        labels = labels.to(torch.long)
    unique = torch.unique(labels)
    if unique.numel() == 0 or not bool(torch.all((unique == 0) | (unique == 1))):
        raise ValueError(
            "labels must be binary (0/1), got values {}".format(unique.tolist())
        )
    if margin <= 0:
        raise ValueError("margin must be positive, got {!r}".format(margin))
    if reduction not in ("mean", "sum"):
        raise ValueError("reduction must be 'mean' or 'sum', got {!r}".format(reduction))

    y = labels.to(torch.float32)
    distance = distance.to(torch.float32)

    genuine = y * distance.square()
    hinge = torch.clamp(margin - distance, min=0.0)
    impostor = (1.0 - y) * hinge.square()

    losses = genuine + impostor
    if reduction == "sum":
        return losses.sum()
    return losses.mean()


class ContrastiveLoss(nn.Module):
    """Module wrapper around :func:`contrastive_loss`.

    Args:
        margin: positive margin (impostor pairs must reach at least this
            distance to stop contributing loss).
        reduction: ``"mean"`` (default) or ``"sum"``.
        eps: small constant for numeric stability of the hinge term.
    """

    def __init__(self, margin: float = 1.0, reduction: str = "mean", eps: float = 1e-8) -> None:
        super().__init__()
        if margin <= 0:
            raise ValueError("margin must be positive, got {!r}".format(margin))
        if reduction not in ("mean", "sum"):
            raise ValueError("reduction must be 'mean' or 'sum', got {!r}".format(reduction))
        if eps <= 0:
            raise ValueError("eps must be positive, got {!r}".format(eps))
        self.margin = float(margin)
        self.reduction = reduction
        self.eps = float(eps)

    def forward(self, distance: torch.Tensor, labels) -> torch.Tensor:
        """Compute the contrastive loss (see :func:`contrastive_loss`)."""
        return contrastive_loss(
            distance,
            labels,
            margin=self.margin,
            reduction=self.reduction,
            eps=self.eps,
        )


__all__ = ["ContrastiveLoss", "contrastive_loss", "GENUINE", "IMPOSTOR", "VALID_LABELS"]