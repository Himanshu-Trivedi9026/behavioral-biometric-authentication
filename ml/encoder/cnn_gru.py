"""
CNN + GRU behavioral encoder (Phase 5).

Architecture (per modality, then fused)
--------------------------------------
For each modality (keyboard / mouse) a dedicated encoder runs:

1. **Convolution**  ``Conv1d`` over the feature channels (each feature = one
   channel over time), producing a local-feature field ``[B, T, C]``.
2. **GRU**         ``pack_padded_sequence`` / ``pad_packed_sequence`` over the
   convolved time axis, so variable-length sessions are processed without any
   influence from padding. The GRU's final hidden state summarises the
   sequence.
3. **Head**        ``Linear(gru_hidden -> embedding_dim)`` yields the modality
   embedding.

The final behavioral embedding concatenates the keyboard and mouse
embeddings (default ``64 + 64 = 128`` dimensions).

Variable-length & empty-sequence semantics (documented)
--------------------------------------------------------
* Padding never leaks into the GRU output: ``pack_padded_sequence`` runs the
  RNN only over each item's true length.
* An item with an **empty keyboard or mouse sequence** contributes a *zero
  embedding* for that modality — a well-defined, "no evidence" representation
  distinct from any learned behaviour.
* An item with **no behaviour at all** (both modalities empty) is rejected with
  :class:`ValueError`; use ``ml.encoder.input.has_behavior`` to filter first.

Determinism
-----------
The module respects ``torch.manual_seed``. Construct :class:`BehavioralEncoder`
under a fixed seed (or pass ``seed``) and re-seed before inference to reproduce
identically. ``set_seed`` is provided as a convenience wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Union

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence

from ml.preprocessing import KEYBOARD_FEATURE_COLUMNS, MOUSE_FEATURE_COLUMNS

from .input import ModalityBatch, PaddedBatch, collate_sessions

_ModalityLike = Union[PaddedBatch, None]


def set_seed(seed: int) -> None:
    """Seed ``torch``'s RNG (the encoder is CPU-focused and respects this)."""
    torch.manual_seed(seed)


@dataclass(frozen=True)
class EncoderConfig:
    """Configuration for the :class:`BehavioralEncoder`.

    Attributes:
        keyboard_features: feature count of keyboard modality (default 2).
        mouse_features:    feature count of mouse modality (default 5).
        conv_channels:     channels produced by each modality's ``Conv1d``.
        conv_kernel:       temporal kernel size (odd kernels keep length stable).
        gru_hidden:        GRU hidden size per modality.
        embedding_dim:     embedding size per modality (final = 2 * this).
        dropout:           dropout between CNN and GRU (default 0.0).
    """

    keyboard_features: int = 2
    mouse_features: int = 5
    conv_channels: int = 32
    conv_kernel: int = 3
    gru_hidden: int = 64
    embedding_dim: int = 64
    dropout: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "keyboard_features",
            "mouse_features",
            "conv_channels",
            "conv_kernel",
            "gru_hidden",
            "embedding_dim",
        ):
            if getattr(self, name) < 1:
                raise ValueError(
                    "{}.{} must be >= 1, got {}".format(
                        type(self).__name__, name, getattr(self, name)
                    )
                )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                "EncoderConfig.dropout must be in [0, 1), got {}".format(self.dropout)
            )
        if self.keyboard_features != len(KEYBOARD_FEATURE_COLUMNS):
            raise ValueError(
                "keyboard_features {} != Phase 3 column count {} "
                "(KEYBOARD_FEATURE_COLUMNS)".format(
                    self.keyboard_features, len(KEYBOARD_FEATURE_COLUMNS)
                )
            )
        if self.mouse_features != len(MOUSE_FEATURE_COLUMNS):
            raise ValueError(
                "mouse_features {} != Phase 3 column count {} "
                "(MOUSE_FEATURE_COLUMNS)".format(
                    self.mouse_features, len(MOUSE_FEATURE_COLUMNS)
                )
            )


class ConvGruEncoder(nn.Module):
    """Single-modality CNN → GRU → linear encoder with variable-length support.

    Args:
        input_features: number of feature channels (e.g. 2 for keyboard).
        embedding_dim:  output embedding width.
        conv_channels:  Conv1d output channels.
        conv_kernel:    Conv1d temporal kernel.
        gru_hidden:     GRU hidden size.
        dropout:        dropout probability after the CNN (default 0.0).
    """

    def __init__(
        self,
        input_features: int,
        embedding_dim: int,
        conv_channels: int = 32,
        conv_kernel: int = 3,
        gru_hidden: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if input_features < 1 or embedding_dim < 1:
            raise ValueError("input_features and embedding_dim must be >= 1")
        self.conv = nn.Conv1d(
            input_features,
            conv_channels,
            kernel_size=conv_kernel,
            padding=conv_kernel // 2,
        )
        self.dropout = nn.Dropout(p=dropout)
        self.gru = nn.GRU(conv_channels, gru_hidden, batch_first=True)
        self.head = nn.Linear(gru_hidden, embedding_dim)
        self.embedding_dim = embedding_dim

    def forward(
        self, inputs: torch.Tensor, lengths: torch.Tensor
    ) -> torch.Tensor:
        """Encode a padded batch to per-item embeddings ``[B, embedding_dim]``.

        Args:
            inputs:  ``[B, T_max, F]`` padded feature tensor.
            lengths: ``[B]`` int64 true lengths (``0`` = empty item).

        Empty items (length ``0``) yield a **zero embedding** by construction —
        they never enter the CNN/GRU, so no padding or missing data can affect
        any other item.
        """
        batch = inputs.size(0)
        present = lengths > 0
        out = inputs.new_zeros(batch, self.embedding_dim)
        if not bool(present.any()):
            return out

        present_indices = present.nonzero(as_tuple=False).view(-1)
        x = inputs.index_select(0, present_indices)
        present_lengths = lengths[present_indices]

        x = x.permute(0, 2, 1)  # [P, F, T]
        x = torch.relu(self.conv(x))  # [P, C, T]
        x = self.dropout(x)
        x = x.permute(0, 2, 1)  # [P, T, C]

        packed = pack_padded_sequence(
            x, present_lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)  # hidden: [1, P, gru_hidden]
        hidden = hidden[0]

        out[present_indices] = self.head(hidden)
        return out


class BehavioralEncoder(nn.Module):
    """fused keyboard + mouse encoder producing ``[B, 2 * embedding_dim]``."""

    def __init__(self, config: Optional[EncoderConfig] = None, seed: Optional[int] = None) -> None:
        super().__init__()
        cfg = config or EncoderConfig()
        if seed is not None:
            torch.manual_seed(seed)

        self.config = cfg
        self.keyboard_encoder = ConvGruEncoder(
            cfg.keyboard_features,
            cfg.embedding_dim,
            cfg.conv_channels,
            cfg.conv_kernel,
            cfg.gru_hidden,
            cfg.dropout,
        )
        self.mouse_encoder = ConvGruEncoder(
            cfg.mouse_features,
            cfg.embedding_dim,
            cfg.conv_channels,
            cfg.conv_kernel,
            cfg.gru_hidden,
            cfg.dropout,
        )
        self.keyboard_embedding_dim = cfg.embedding_dim
        self.mouse_embedding_dim = cfg.embedding_dim
        self.embedding_dim = cfg.embedding_dim * 2

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self,
        keyboard: _ModalityLike = None,
        mouse: _ModalityLike = None,
    ) -> torch.Tensor:
        """Encode a batch of sessions to behavioral embeddings ``[B, embedding_dim]``.

        Args:
            keyboard: keyboard :class:`~ml.encoder.input.PaddedBatch` or ``None``.
            mouse:    mouse :class:`~ml.encoder.input.PaddedBatch` or ``None``.

        Raises:
            ValueError: if an item has no behavioural content at all, or if the
                supplied modalities disagree on batch size.
        """
        if keyboard is None and mouse is None:
            raise ValueError(
                "BehavioralEncoder.forward requires at least one modality "
                "(keyboard and/or mouse)"
            )
        if keyboard is not None and not isinstance(keyboard, PaddedBatch):
            raise TypeError(
                "keyboard must be a PaddedBatch or None, got {!r}".format(
                    type(keyboard).__name__
                )
            )
        if mouse is not None and not isinstance(mouse, PaddedBatch):
            raise TypeError(
                "mouse must be a PaddedBatch or None, got {!r}".format(
                    type(mouse).__name__
                )
            )

        if keyboard is not None:
            batch = keyboard.inputs.size(0)
            keyboard_have_behavior = keyboard.lengths > 0
            keyboard_output = self.keyboard_encoder(
                keyboard.inputs, keyboard.lengths
            )
        else:
            batch = mouse.inputs.size(0)
            keyboard_have_behavior = torch.zeros(batch, dtype=torch.bool)
            keyboard_output = mouse.inputs.new_zeros(
                batch, self.keyboard_embedding_dim
            )

        if mouse is not None:
            if mouse.inputs.size(0) != batch:
                raise ValueError(
                    "keyboard batch size {} != mouse batch size {}".format(
                        batch, mouse.inputs.size(0)
                    )
                )
            mouse_have_behavior = mouse.lengths > 0
            mouse_output = self.mouse_encoder(mouse.inputs, mouse.lengths)
        else:
            mouse_have_behavior = torch.zeros(batch, dtype=torch.bool)
            mouse_output = keyboard_output.new_zeros(batch, self.mouse_embedding_dim)

        have_behavior = keyboard_have_behavior | mouse_have_behavior
        if not bool(have_behavior.all()):
            missing = have_behavior.logical_not().nonzero(as_tuple=False).view(-1)
            raise ValueError(
                "entries at batch indices {} have no behavioural content "
                "(both keyboard and mouse sequences are empty); filter with "
                "ml.encoder.input.has_behavior before encoding".format(
                    missing.tolist()
                )
            )

        return torch.cat([keyboard_output, mouse_output], dim=1)

    # ------------------------------------------------------------------
    # Convenience wrappers over Phase 3/4 sessions
    # ------------------------------------------------------------------

    def encode_session(
        self,
        session: Mapping[str, object],
        keyboard_scaler: Any = None,
        mouse_scaler: Any = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Embed a single processed session / dataset entry -> ``[embedding_dim]``."""
        batch = collate_sessions(
            [session],
            keyboard_scaler=keyboard_scaler,
            mouse_scaler=mouse_scaler,
            dtype=dtype,
        )
        return self._encode_batch(batch)[0]

    def encode_entries(
        self,
        sessions: Sequence[Mapping[str, object]],
        keyboard_scaler: Any = None,
        mouse_scaler: Any = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Embed a list of sessions/entries -> ``[len(sessions), embedding_dim]``."""
        batch = collate_sessions(
            sessions,
            keyboard_scaler=keyboard_scaler,
            mouse_scaler=mouse_scaler,
            dtype=dtype,
        )
        return self._encode_batch(batch)

    def _encode_batch(self, batch: ModalityBatch) -> torch.Tensor:
        if batch.keyboard is None and batch.mouse is None:
            raise ValueError(
                "sessions have no behavioural content (both keyboard and mouse "
                "sequences are empty); filter with ml.encoder.input.has_behavior "
                "before encoding"
            )
        return self(keyboard=batch.keyboard, mouse=batch.mouse)


__all__ = [
    "BehavioralEncoder",
    "ConvGruEncoder",
    "EncoderConfig",
    "set_seed",
]