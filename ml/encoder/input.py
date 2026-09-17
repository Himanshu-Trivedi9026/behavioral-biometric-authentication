"""
Sequence preparation for the behavioral encoder (Phase 5).

Turns Phase 3 / Phase 4 feature sequences (plain lists of feature dicts) into
the ``torch`` tensors the CNN+GRU encoder consumes. This module is **pure
numeric preparation** — it contains no model weights and performs no training.
It intentionally reuses Phase 3's feature columns and ``FeatureScaler`` so the
whole pipeline shares one source of truth for feature definitions and
normalization statistics.

Size contract
-------------
* keyboard sequence : ``[T_keyboard, 2]``   (hold_time, flight_time)
* mouse sequence    : ``[T_mouse, 5]``      (dx, dy, dt, distance, speed)
* collated batch    : ``inputs [B, T_max, F]`` (zero-padded) + ``lengths [B]``

Variables length & empty sequences
----------------------------------
Sessions have naturally variable-length sequences. Batches are built by
zero-padding every sequence to the longest length in the batch and carrying
each item's true ``length`` alongside. **Empty sequences are legal at the
batch level** (the encoder turns an absent modality into a documented zero
embedding); a fully empty session (no keyboard AND no mouse behaviour) is
rejected by the encoder, and :func:`has_behavior` is provided so callers can
filter such sessions up front.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence as TypingSequence

import torch

from ml.preprocessing import (
    KEYBOARD_FEATURE_COLUMNS,
    MOUSE_FEATURE_COLUMNS,
)


@dataclass
class PaddedBatch:
    """A zero-padded variable-length batch.

    Attributes:
        inputs:  ``[B, T_max, F]`` float tensor (the padded features).
        lengths: ``[B]`` int64 tensor (each item's true sequence length;
                 ``0`` marks an empty sequence).
    """

    inputs: torch.Tensor
    lengths: torch.Tensor

    @property
    def batch_size(self) -> int:
        return self.inputs.size(0)

    @property
    def feature_count(self) -> int:
        return self.inputs.size(2)


@dataclass
class ModalityBatch:
    """Per-modality batches for one collated set of sessions.

    A modality is ``None`` only when **every** item in the batch has an empty
    sequence for it (the encoder then uses a zero embedding for that modality).
    Individual items with an empty modality are represented by a ``0`` in that
    item's ``lengths`` instead.
    """

    keyboard: Optional[PaddedBatch]
    mouse: Optional[PaddedBatch]


def _validated_rows(
    sequence: TypingSequence[Mapping[str, Any]],
    feature_columns: TypingSequence[str],
) -> list[list[float]]:
    """Validate a Phase 3 feature sequence and return rows in column order.

    Every sample must be a dict-like with exactly the declared feature columns
    and finite numeric values (mirrors the Phase 4 schema contract).
    """
    columns = tuple(feature_columns)
    if not isinstance(sequence, (list, tuple)):
        raise TypeError(
            "sequence must be a list of feature dicts, got {!r}".format(
                type(sequence).__name__
            )
        )
    rows: list[list[float]] = []
    for index, sample in enumerate(sequence):
        if not isinstance(sample, Mapping):
            raise TypeError(
                "sequence[{}] must be a dict-like mapping, got {!r}".format(
                    index, type(sample).__name__
                )
            )
        missing = [column for column in columns if column not in sample]
        if missing:
            raise ValueError(
                "sequence[{}] is missing feature(s): {}".format(
                    index, ", ".join(missing)
                )
            )
        extra = [key for key in sample if key not in columns]
        if extra:
            raise ValueError(
                "sequence[{}] has unexpected feature(s): {}".format(
                    index, ", ".join(sorted(extra))
                )
            )
        row: list[float] = []
        for column in columns:
            value = sample[column]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(
                    "sequence[{}].{} must be a finite number, got {!r}".format(
                        index, column, value
                    )
                )
            number = float(value)
            if not math.isfinite(number):
                raise ValueError(
                    "sequence[{}].{} must be a finite number, got {!r}".format(
                        index, column, value
                    )
                )
            row.append(number)
        rows.append(row)
    return rows


def sequence_tensor(
    sequence: TypingSequence[Mapping[str, Any]],
    feature_columns: TypingSequence[str],
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Convert one feature sequence (list of dicts) into a ``[T, F]`` tensor.

    An empty sequence yields a ``[0, len(feature_columns)]`` tensor.
    """
    rows = _validated_rows(sequence, feature_columns)
    if not rows:
        return torch.zeros(0, len(feature_columns), dtype=dtype)
    return torch.tensor(rows, dtype=dtype)


def collate_rows(
    rows_list: TypingSequence[TypingSequence[TypingSequence[float]]],
    n_features: int,
    dtype: torch.dtype = torch.float32,
) -> PaddedBatch:
    """Collate pre-validated rows (list of ``[T_i, F]`` row lists) into a batch.

    Empty sequences (``T_i == 0``) are allowed and carry length ``0``; when
    every item is empty the padded tensor has shape ``[B, 0, F]``.
    """
    if not isinstance(rows_list, (list, tuple)):
        raise TypeError(
            "rows_list must be a list of row lists, got {!r}".format(
                type(rows_list).__name__
            )
        )
    batch = len(rows_list)
    lengths = [
        len(rows) if isinstance(rows, (list, tuple)) else 0 for rows in rows_list
    ]
    max_length = max(lengths) if lengths else 0
    inputs = torch.zeros(batch, max_length, n_features, dtype=dtype)
    for index, rows in enumerate(rows_list):
        if not isinstance(rows, (list, tuple)):
            raise TypeError(
                "rows_list[{}] must be a list of rows, got {!r}".format(
                    index, type(rows).__name__
                )
            )
        if rows and len(rows[0]) != n_features:
            raise ValueError(
                "rows_list[{}] rows have {} features, expected {}".format(
                    index, len(rows[0]), n_features
                )
            )
        if rows:
            inputs[index, : len(rows), :] = torch.tensor(rows, dtype=dtype)
    return PaddedBatch(
        inputs=inputs,
        lengths=torch.tensor(lengths, dtype=torch.long),
    )


def collate_sequences(
    sequences: TypingSequence[TypingSequence[Mapping[str, Any]]],
    feature_columns: TypingSequence[str],
    dtype: torch.dtype = torch.float32,
) -> PaddedBatch:
    """Collate a list of Phase 3 feature sequences into one padded batch."""
    columns = tuple(feature_columns)
    rows_list = [_validated_rows(sequence, columns) for sequence in sequences]
    return collate_rows(rows_list, len(columns), dtype=dtype)


def _scaled_rows(
    source_sequence: TypingSequence[Mapping[str, Any]],
    feature_columns: TypingSequence[str],
    scaler: Any,
) -> list[list[float]]:
    rows = _validated_rows(source_sequence, feature_columns)
    if scaler is not None:
        rows = scaler.transform(rows)
    return rows


def collate_sessions(
    sessions: TypingSequence[Mapping[str, Any]],
    keyboard_scaler: Any = None,
    mouse_scaler: Any = None,
    dtype: torch.dtype = torch.float32,
) -> ModalityBatch:
    """Collate processed sessions / dataset entries into a :class:`ModalityBatch`.

    Reads ``keyboard_sequence`` / ``mouse_sequence`` (Phase 3/4 schema). When a
    ``FeatureScaler`` is given for a modality it is applied before tensor
    conversion, so validation/test data can reuse training statistics.
    """
    if not isinstance(sessions, (list, tuple)):
        raise TypeError(
            "sessions must be a list of sessions/entries, got {!r}".format(
                type(sessions).__name__
            )
        )
    keyboard_rows = [
        _scaled_rows(
            session.get("keyboard_sequence", []),
            KEYBOARD_FEATURE_COLUMNS,
            keyboard_scaler,
        )
        for session in sessions
    ]
    mouse_rows = [
        _scaled_rows(
            session.get("mouse_sequence", []),
            MOUSE_FEATURE_COLUMNS,
            mouse_scaler,
        )
        for session in sessions
    ]

    keyboard_batch = collate_rows(
        keyboard_rows, len(KEYBOARD_FEATURE_COLUMNS), dtype=dtype
    )
    mouse_batch = collate_rows(mouse_rows, len(MOUSE_FEATURE_COLUMNS), dtype=dtype)

    return ModalityBatch(
        keyboard=keyboard_batch if bool(keyboard_batch.lengths.any()) else None,
        mouse=mouse_batch if bool(mouse_batch.lengths.any()) else None,
    )


def has_behavior(session: Mapping[str, Any]) -> bool:
    """Return ``True`` if a session/entry has any keyboard or mouse behaviour.

    Sessions with neither sequence can never be authenticated; the encoder
    rejects them, so callers should filter with this helper first.
    """
    return bool(session.get("keyboard_sequence")) or bool(session.get("mouse_sequence"))


__all__ = [
    "ModalityBatch",
    "PaddedBatch",
    "collate_rows",
    "collate_sequences",
    "collate_sessions",
    "has_behavior",
    "sequence_tensor",
]