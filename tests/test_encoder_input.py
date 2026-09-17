"""
Phase 5 encoder input-prep unit tests.

Covers ``ml.encoder.input``: feature-sequence validation, tensor conversion,
variable-length collation, scaler reuse, and per-modality batch assembly.

Run from the repo root::

    .venv/bin/python -m pytest tests/test_encoder_input.py -v
"""

import math

import pytest
import torch

from ml.encoder.input import (
    ModalityBatch,
    PaddedBatch,
    collate_rows,
    collate_sequences,
    collate_sessions,
    has_behavior,
    sequence_tensor,
)
from ml.preprocessing import (
    FeatureScaler,
    KEYBOARD_FEATURE_COLUMNS,
    MOUSE_FEATURE_COLUMNS,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _kb(hold=80.0, flight=40.0):
    return {"hold_time": hold, "flight_time": flight}


def _mo(dx=1.0, dy=2.0, dt=16.0, distance=2.236, speed=0.14):
    return {
        "dx": dx, "dy": dy, "dt": dt,
        "distance": distance, "speed": speed,
    }


# =====================================================================
# sequence_tensor
# =====================================================================

class TestSequenceTensor:

    def test_basic_shape_and_dtype(self):
        seq = [_kb(), _kb(hold=90.0, flight=30.0)]
        tensor = sequence_tensor(seq, KEYBOARD_FEATURE_COLUMNS)
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (2, 2)
        assert tensor.dtype == torch.float32

    def test_values_in_column_order(self):
        seq = [_kb(hold=80.0, flight=40.0), _kb(hold=90.0, flight=30.0)]
        tensor = sequence_tensor(seq, KEYBOARD_FEATURE_COLUMNS)
        assert tensor[0, 0].item() == 80.0
        assert tensor[0, 1].item() == 40.0
        assert tensor[1, 0].item() == 90.0
        assert tensor[1, 1].item() == 30.0

    def test_empty_sequence_produces_zero_rows_shape(self):
        tensor = sequence_tensor([], KEYBOARD_FEATURE_COLUMNS)
        assert tensor.shape == (0, 2)

    def test_mouse_columns(self):
        seq = [_mo()]
        tensor = sequence_tensor(seq, MOUSE_FEATURE_COLUMNS)
        assert tensor.shape == (1, 5)
        assert tensor[0, 0].item() == 1.0
        assert tensor[0, 4].item() == pytest.approx(0.14, abs=1e-6)

    def test_rejects_non_sequence(self):
        with pytest.raises(TypeError):
            sequence_tensor(None, KEYBOARD_FEATURE_COLUMNS)

    def test_rejects_non_mapping_sample(self):
        with pytest.raises(TypeError):
            sequence_tensor([1, 2, 3], KEYBOARD_FEATURE_COLUMNS)

    def test_rejects_missing_feature(self):
        with pytest.raises(ValueError, match="missing feature"):
            sequence_tensor([{"hold_time": 80.0}], KEYBOARD_FEATURE_COLUMNS)

    def test_rejects_extra_feature(self):
        with pytest.raises(ValueError, match="unexpected feature"):
            sequence_tensor([_kb(), {"hold_time": 1.0, "flight_time": 2.0, "key": "a"}],
                            KEYBOARD_FEATURE_COLUMNS)

    def test_rejects_non_numeric_value(self):
        with pytest.raises(ValueError, match="finite number"):
            sequence_tensor([{"hold_time": "fast", "flight_time": 40.0}],
                            KEYBOARD_FEATURE_COLUMNS)

    def test_rejects_non_finite_value(self):
        with pytest.raises(ValueError, match="finite number"):
            sequence_tensor([_kb(hold=float("nan"))], KEYBOARD_FEATURE_COLUMNS)

    def test_rejects_bool_value(self):
        with pytest.raises(ValueError, match="finite number"):
            sequence_tensor([_kb(hold=True)], KEYBOARD_FEATURE_COLUMNS)


# =====================================================================
# collate_rows / collate_sequences
# =====================================================================

class TestCollate:

    def test_uniform_lengths(self):
        batch = collate_rows(
            [ [[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]] ], 2
        )
        assert batch.inputs.shape == (2, 2, 2)
        assert batch.lengths.tolist() == [2, 2]
        assert batch.batch_size == 2
        assert batch.feature_count == 2

    def test_mixed_lengths_pad_to_longest(self):
        rows = [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0]],
            [],
        ]
        batch = collate_rows(rows, 2)
        assert batch.inputs.shape == (3, 3, 2)
        assert batch.lengths.tolist() == [3, 1, 0]
        # padding is exactly zero
        assert float(batch.inputs[0, 2, 0]) == 5.0
        assert float(batch.inputs[1, 1, 0]) == 0.0
        assert float(batch.inputs[2, 0, 0]) == 0.0

    def test_all_empty_shapes(self):
        batch = collate_rows([[], []], 2)
        assert batch.inputs.shape == (2, 0, 2)
        assert batch.lengths.tolist() == [0, 0]

    def test_row_width_mismatch_rejected(self):
        with pytest.raises(ValueError, match="features"):
            collate_rows([[[1.0, 2.0]], [[1.0, 2.0, 3.0]]], 2)

    def test_collate_sequences_uses_feature_columns(self):
        seqs = [
            [_kb(hold=80.0, flight=40.0), _kb(hold=90.0, flight=30.0)],
            [],
            [_kb(hold=100.0, flight=60.0)],
        ]
        batch = collate_sequences(seqs, KEYBOARD_FEATURE_COLUMNS)
        assert isinstance(batch, PaddedBatch)
        assert batch.inputs.shape == (3, 2, 2)
        assert batch.lengths.tolist() == [2, 0, 1]

    def test_collate_validates_sequences(self):
        with pytest.raises(ValueError, match="missing feature"):
            collate_sequences([[{"hold_time": 1.0}]], KEYBOARD_FEATURE_COLUMNS)


# =====================================================================
# collate_sessions / has_behavior
# =====================================================================

class TestSessions:

    def _sessions(self):
        return [
            {
                "session_id": "s1",
                "keyboard_sequence": [_kb(), _kb(hold=90.0)],
                "mouse_sequence": [_mo()],
            },
            {
                "session_id": "s2",
                "keyboard_sequence": [],
                "mouse_sequence": [_mo(), _mo(dx=-2.0)],
            },
            {
                "session_id": "s3",
                "keyboard_sequence": [_kb(hold=120.0)],
                "mouse_sequence": [],
            },
            {
                "session_id": "s4",
                "keyboard_sequence": [],
                "mouse_sequence": [],
            },
        ]

    def test_modality_batches_and_absence_flags(self):
        batch = collate_sessions(self._sessions())
        assert isinstance(batch, ModalityBatch)
        assert batch.keyboard is not None
        assert batch.mouse is not None
        assert batch.keyboard.lengths.tolist() == [2, 0, 1, 0]
        assert batch.mouse.lengths.tolist() == [1, 2, 0, 0]
        assert batch.keyboard.inputs.shape == (4, 2, 2)
        assert batch.mouse.inputs.shape == (4, 2, 5)

    def test_modality_none_when_every_item_empty(self):
        empty = [
            {"keyboard_sequence": [], "mouse_sequence": []},
            {"keyboard_sequence": [], "mouse_sequence": []},
        ]
        batch = collate_sessions(empty)
        assert batch.keyboard is None
        assert batch.mouse is None

    def test_scaler_applied_before_tensor_conversion(self):
        kb_rows = [[80.0, 40.0], [90.0, 30.0], [120.0, 60.0]]
        scaler = FeatureScaler(KEYBOARD_FEATURE_COLUMNS).fit(kb_rows)
        session = {"keyboard_sequence": [_kb(hold=80.0, flight=40.0)],
                   "mouse_sequence": []}
        batch = collate_sessions([session], keyboard_scaler=scaler)
        assert batch.keyboard is not None
        standardized = scaler.transform([[80.0, 40.0]])[0]
        got = batch.keyboard.inputs[0, 0].tolist()
        assert got == pytest.approx(standardized)

    def test_scaler_missing_transform_called_only_when_given(self):
        session = {"keyboard_sequence": [_kb(hold=80.0)]}
        batch = collate_sessions([session])
        assert batch.keyboard.inputs[0, 0, 0].item() == 80.0

    def test_dtype_respected(self):
        sessions = [{"keyboard_sequence": [_kb()], "mouse_sequence": []}]
        batch = collate_sessions(sessions, dtype=torch.float64)
        assert batch.keyboard.inputs.dtype == torch.float64

    def test_has_behavior(self):
        assert has_behavior({"keyboard_sequence": [_kb()], "mouse_sequence": []}) is True
        assert has_behavior({"keyboard_sequence": [], "mouse_sequence": [_mo()]}) is True
        assert has_behavior({"keyboard_sequence": [], "mouse_sequence": []}) is False
        assert has_behavior({}) is False

    def test_collate_sessions_requires_sequence(self):
        with pytest.raises(TypeError, match="sessions must be a list"):
            collate_sessions({"keyboard_sequence": []})