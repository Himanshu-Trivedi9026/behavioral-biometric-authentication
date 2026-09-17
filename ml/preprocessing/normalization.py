"""
Numerical normalization (standardization).

Implemented separately from feature extraction so it can be reused when
training a model: you ``fit`` the scaler on TRAINING data only, persist its
state, then ``transform`` validation/test/inference data with the SAME
statistics. Normalization statistics are never hard-coded.

Method
------
Per-feature **standardization (z-score)**:

    standardized = (value - mean_feature) / std_feature

This is the standard choice for behavioral timing/trajectory features
(raw durations and pixel deltas are roughly symmetric and unit-relative,
so centering by the mean and scaling by the standard deviation is
well-behaved and robust).

Constant features
-----------------
If a feature has zero standard deviation (all training values identical),
dividing would produce ``0/0``. That is handled by storing the standard
deviation as ``1.0`` for such features, so every transformed value is
``(value - mean) / 1 = 0``. No division by zero can occur.

Interface
---------
``FeatureScaler`` provides ``fit`` / ``transform`` / ``fit_transform`` and a
JSON-serializable ``state_dict()`` / ``load_state_dict()`` pair so fitted
statistics can later be reused on new data without re-fitting.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Sequence

from .keyboard import KEYBOARD_FEATURE_COLUMNS
from .mouse import MOUSE_FEATURE_COLUMNS


def sequence_to_rows(
    sequence: Sequence[Dict[str, float]], feature_columns: Sequence[str]
) -> List[List[float]]:
    """Convert a sequence of feature dicts into an (n_samples, n_features) row list.

    Args:
        sequence: e.g. a ``keyboard_sequence`` or ``mouse_sequence``.
        feature_columns: column order, e.g. ``KEYBOARD_FEATURE_COLUMNS``.

    Returns:
        A list of rows, each row a list of the feature values in the given
        column order (safe for later tensor conversion).
    """
    return [
        [float(row[column]) for column in feature_columns]
        for row in sequence
    ]


def rows_to_sequence(
    rows: Sequence[Sequence[float]], feature_columns: Sequence[str]
) -> List[Dict[str, float]]:
    """Invert :func:`sequence_to_rows`: rows back to feature dicts."""
    return [
        {column: float(row[i]) for i, column in enumerate(feature_columns)}
        for row in rows
    ]


class FeatureScaler:
    """Per-feature standardization with fit/transform and reusable state."""

    def __init__(self, feature_columns: Sequence[str] | None = None) -> None:
        self.feature_columns = list(feature_columns) if feature_columns else None
        self.feature_means_: List[float] | None = None
        self.feature_stds_: List[float] | None = None
        self.n_features_: int | None = None
        self._fitted = False

    def _validate_rows(self, rows: Sequence[Sequence[float]]) -> List[List[float]]:
        if not rows:
            raise ValueError("FeatureScaler requires at least one row")
        n = len(rows[0])
        if n == 0:
            raise ValueError("FeatureScaler rows must have at least one feature")
        cleaned: List[List[float]] = []
        for i, row in enumerate(rows):
            if len(row) != n:
                raise ValueError(
                    "FeatureScaler rows must all have the same number of "
                    "features (row {} has {}, expected {})".format(i, len(row), n)
                )
            values = []
            for value in row:
                value = float(value)
                if not math.isfinite(value):
                    raise ValueError(
                        "FeatureScaler requires finite values, got {!r} in row {}".format(
                            value, i
                        )
                    )
                values.append(value)
            cleaned.append(values)
        return cleaned

    def fit(self, rows: Sequence[Sequence[float]]) -> "FeatureScaler":
        """Compute per-feature mean/std from the provided rows (training data only)."""
        cleaned = self._validate_rows(rows)
        n_features = len(cleaned[0])
        col_values = [
            [row[j] for row in cleaned] for j in range(n_features)
        ]
        means = [sum(col) / len(col) for col in col_values]
        # Population std (divide by n) to match the familiar z-score form.
        stds = [
            math.sqrt(sum((v - mean) ** 2 for v in col) / len(col))
            for col, mean in zip(col_values, means)
        ]
        # Constant-feature guard: never divide by zero.
        stds = [std if std > 0 else 1.0 for std in stds]

        self.feature_means_ = means
        self.feature_stds_ = stds
        self.n_features_ = n_features
        self._fitted = True
        return self

    def transform(self, rows: Sequence[Sequence[float]]) -> List[List[float]]:
        """Standardize rows using the fitted statistics."""
        if not self._fitted:
            raise RuntimeError("FeatureScaler must be fitted before transform()")

        cleaned = self._validate_rows(rows)
        if len(cleaned[0]) != self.n_features_:
            raise ValueError(
                "FeatureScaler was fitted on {} features, got {}".format(
                    self.n_features_, len(cleaned[0])
                )
            )

        means = self.feature_means_
        stds = self.feature_stds_
        return [
            [(value - means[j]) / stds[j] for j, value in enumerate(row)]
            for row in cleaned
        ]

    def fit_transform(self, rows: Sequence[Sequence[float]]) -> List[List[float]]:
        """Fit on the given rows, then return the standardized rows."""
        return self.fit(rows).transform(rows)

    def state_dict(self) -> Dict[str, Any]:
        """JSON-serializable snapshot of the fitted statistics (for reuse later)."""
        if not self._fitted:
            raise RuntimeError("FeatureScaler must be fitted before state_dict()")
        return {
            "method": "z-score",
            "feature_columns": self.feature_columns,
            "feature_means": self.feature_means_,
            "feature_stds": self.feature_stds_,
        }

    @classmethod
    def load_state_dict(cls, state: Dict[str, Any]) -> "FeatureScaler":
        """Recreate a fitted scaler from a ``state_dict()`` produced earlier."""
        scaler = cls(feature_columns=state.get("feature_columns"))
        scaler.feature_means_ = [float(v) for v in state["feature_means"]]
        scaler.feature_stds_ = [float(v) for v in state["feature_stds"]]
        scaler.n_features_ = len(scaler.feature_means_)
        scaler._fitted = True
        return scaler


__all__ = [
    "FeatureScaler",
    "rows_to_sequence",
    "sequence_to_rows",
]