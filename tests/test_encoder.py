"""
Phase 5 CNN + GRU behavioral encoder tests.

Covers: architecture/dims, variable-length + padding non-influence, batch
forward, empty-modality semantics, determinism, gradient flow, identity
independence, sensitivity to feature values, and end-to-end dataset scaling.

Run from the repo root::

    .venv/bin/python -m pytest tests/test_encoder.py -v
"""

import pytest
import torch

from dataset.generator import SyntheticDatasetGenerator
from ml.encoder import BehavioralEncoder, EncoderConfig, set_seed
from ml.encoder import collate_sequences
from ml.encoder.cnn_gru import ConvGruEncoder
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


def _session(kb=None, mo=None, session_id="s", user_id="u"):
    return {
        "session_id": session_id,
        "user_id": user_id,
        "keyboard_sequence": list(kb or []),
        "mouse_sequence": list(mo or []),
    }


def _dataset(seed=0, n_users=2, sessions=4):
    return SyntheticDatasetGenerator(
        seed=seed, n_users=n_users, sessions_per_user=sessions
    ).generate()


def _scalers(entries):
    kb_scaler = FeatureScaler(KEYBOARD_FEATURE_COLUMNS).fit(
        [row for e in entries for row in _rows(e["keyboard_sequence"])]
    )
    mo_scaler = FeatureScaler(MOUSE_FEATURE_COLUMNS).fit(
        [row for e in entries for row in _rows(e["mouse_sequence"], MOUSE_FEATURE_COLUMNS)]
    )
    return kb_scaler, mo_scaler


def _rows(sequence, columns=None):
    from ml.encoder.input import _validated_rows

    return _validated_rows(sequence, columns or KEYBOARD_FEATURE_COLUMNS)


# =====================================================================
# Architecture & configuration
# =====================================================================

class TestArchitecture:

    def test_default_embedding_dims(self):
        enc = BehavioralEncoder()
        assert enc.embedding_dim == 128
        assert enc.keyboard_embedding_dim == 64
        assert enc.mouse_embedding_dim == 64

    def test_custom_config_embedding_dims(self):
        cfg = EncoderConfig(embedding_dim=32, conv_channels=16, gru_hidden=32)
        enc = BehavioralEncoder(config=cfg)
        assert enc.embedding_dim == 64
        assert enc.keyboard_embedding_dim == 32
        assert enc.mouse_embedding_dim == 32

    def test_config_rejects_wrong_keyboard_feature_count(self):
        with pytest.raises(ValueError, match="keyboard_features"):
            EncoderConfig(keyboard_features=3)

    def test_config_rejects_wrong_mouse_feature_count(self):
        with pytest.raises(ValueError, match="mouse_features"):
            EncoderConfig(mouse_features=7)

    def test_config_rejects_bad_dropout(self):
        with pytest.raises(ValueError, match="dropout"):
            EncoderConfig(dropout=1.0)

    def test_config_rejects_non_positive_width(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            EncoderConfig(conv_channels=-1)

    def test_convgru_rejects_bad_width(self):
        with pytest.raises(ValueError, match="must be >= 1"):
            ConvGruEncoder(input_features=0, embedding_dim=8)

    def test_submodule_shapes(self):
        enc = BehavioralEncoder()
        assert isinstance(enc.keyboard_encoder, ConvGruEncoder)
        assert isinstance(enc.mouse_encoder, ConvGruEncoder)
        assert enc.keyboard_encoder.conv.in_channels == 2
        assert enc.mouse_encoder.conv.in_channels == 5


# =====================================================================
# Forward pass
# =====================================================================

class TestForward:

    def test_single_session_shape(self):
        enc = BehavioralEncoder()
        kb = collate_sequences([[_kb(), _kb(hold=95.0)]], KEYBOARD_FEATURE_COLUMNS)
        mo = collate_sequences([[_mo(), _mo()]], MOUSE_FEATURE_COLUMNS)
        out = enc(keyboard=kb, mouse=mo)
        assert out.shape == (1, 128)

    def test_batch_shape(self):
        enc = BehavioralEncoder()
        kb = collate_sequences(
            [[_kb()], [_kb(), _kb(hold=95.0)]], KEYBOARD_FEATURE_COLUMNS
        )
        mo = collate_sequences(
            [[_mo()], [_mo(), _mo(), _mo()]], MOUSE_FEATURE_COLUMNS
        )
        out = enc(keyboard=kb, mouse=mo)
        assert out.shape == (2, 128)

    def test_variable_length_padding_does_not_affect_values(self):
        """Encoding an item alone must equal encoding it inside a mixed batch."""
        enc = BehavioralEncoder()
        a = _session(kb=[_kb(), _kb(hold=95.0), _kb(hold=77.0)], mo=[_mo()])
        b = _session(kb=[_kb(hold=120.0)], mo=[_mo(), _mo(), _mo(), _mo()])

        alone = enc.encode_entries([a])
        mixed = enc.encode_entries([a, b])
        assert torch.allclose(alone, mixed[0], atol=1e-6)

    def test_padding_regions_do_not_change_same_length_input(self):
        enc = BehavioralEncoder()
        kb = collate_sequences(
            [[_kb(hold=90.0), _kb(hold=70.0)], [_kb(hold=90.0), _kb(hold=70.0)]],
            KEYBOARD_FEATURE_COLUMNS,
        )
        mo = collate_sequences(
            [[_mo()], [_mo()]], MOUSE_FEATURE_COLUMNS
        )
        out = enc(keyboard=kb, mouse=mo)
        assert torch.allclose(out[0], out[1], atol=1e-6)

    def test_forward_rejects_unsupported_types(self):
        enc = BehavioralEncoder()
        with pytest.raises(TypeError, match="PaddedBatch"):
            enc(keyboard="nope", mouse="nope")

    def test_forward_rejects_both_none(self):
        enc = BehavioralEncoder()
        with pytest.raises(ValueError, match="at least one modality"):
            enc()

    def test_forward_rejects_batch_size_mismatch(self):
        enc = BehavioralEncoder()
        kb = collate_sequences([[_kb()]], KEYBOARD_FEATURE_COLUMNS)
        mo = collate_sequences([[_mo()], [_mo()]], MOUSE_FEATURE_COLUMNS)
        with pytest.raises(ValueError, match="batch size"):
            enc(keyboard=kb, mouse=mo)


# =====================================================================
# Output contract (regression: "too many indices ... dimension 2")
# =====================================================================

class TestOutputContract:
    """Pin the return contract: plain torch.Tensor, not a container/3-D.

    Regression: a manual verification script indexed the forward result with
    three axes (``out[0, :, :]``), which mistook the 2-D ``[B, embedding_dim]``
    output for something else and raised
    ``IndexError: too many indices for tensor of dimension 2``.
    """

    def test_forward_returns_plain_2d_tensor(self):
        enc = BehavioralEncoder(seed=0)
        kb = collate_sequences([[_kb(), _kb(hold=95.0)]], KEYBOARD_FEATURE_COLUMNS)
        mo = collate_sequences([[_mo()]], MOUSE_FEATURE_COLUMNS)
        out = enc(keyboard=kb, mouse=mo)
        assert isinstance(out, torch.Tensor)
        assert type(out).__name__ == "Tensor"
        assert out.ndim == 2
        assert out.shape == (1, 128)

    def test_encode_entries_returns_plain_2d_tensor(self):
        enc = BehavioralEncoder(seed=0)
        sessions = [
            _session(kb=[_kb()], mo=[_mo()]),
            _session(kb=[_kb(), _kb(hold=95.0)], mo=[]),
            _session(kb=[], mo=[_mo(dx=-3.0)]),
        ]
        out = enc.encode_entries(sessions)
        assert isinstance(out, torch.Tensor)
        assert out.ndim == 2
        assert out.shape == (3, 128)

    def test_encode_session_returns_1d_vector(self):
        enc = BehavioralEncoder(seed=0)
        session = _session(kb=[_kb(), _kb(hold=95.0)], mo=[_mo()])
        out = enc.encode_session(session)
        assert isinstance(out, torch.Tensor)
        assert out.ndim == 1
        assert out.shape == (128,)

    def test_documented_first_row_indexing_works(self):
        enc = BehavioralEncoder(seed=0)
        kb = collate_sequences([[_kb()]], KEYBOARD_FEATURE_COLUMNS)
        mo = collate_sequences([[_mo()]], MOUSE_FEATURE_COLUMNS)
        out = enc(keyboard=kb, mouse=mo)
        first = out[0]
        assert first.shape == (128,)


# =====================================================================
# Empty-modality semantics (documented safe representation)
# =====================================================================

class TestEmptyModality:

    def test_keyboard_only_equals_keyboard_plus_empty_mouse(self):
        enc = BehavioralEncoder()
        kb = collate_sequences([[_kb(), _kb(hold=95.0)]], KEYBOARD_FEATURE_COLUMNS)
        empty_mo = collate_sequences([[]], MOUSE_FEATURE_COLUMNS)
        with_ke = enc(keyboard=kb, mouse=empty_mo)
        ke_only = enc(keyboard=kb)
        assert torch.allclose(with_ke, ke_only, atol=1e-6)

    def test_mouse_only_equals_mouse_plus_empty_keyboard(self):
        enc = BehavioralEncoder()
        mo = collate_sequences([[_mo(), _mo()]], MOUSE_FEATURE_COLUMNS)
        empty_kb = collate_sequences([[]], KEYBOARD_FEATURE_COLUMNS)
        assert torch.allclose(enc(mouse=mo), enc(keyboard=empty_kb, mouse=mo), atol=1e-6)

    def test_per_item_empty_modality_in_mixed_batch(self):
        enc = BehavioralEncoder()
        sessions = [
            _session(kb=[_kb()], mo=[_mo()]),
            _session(kb=[_kb(hold=90.0)], mo=[]),  # empty mouse for this item
            _session(kb=[], mo=[_mo(dx=-5.0)]),    # empty keyboard for this item
        ]
        out = enc.encode_entries(sessions)
        assert out.shape == (3, 128)
        # item1 (empty mouse) must equal a pure-keyboard encoding of item1
        alone = enc.encode_entries([sessions[1]])
        assert torch.allclose(out[1], alone[0], atol=1e-6)

    def test_session_with_no_behaviour_rejected(self):
        enc = BehavioralEncoder()
        with pytest.raises(ValueError, match="no behavioural content"):
            enc.encode_entries([_session(kb=[], mo=[])])

    def test_has_behavior_filters_the_rejection(self):
        from ml.encoder.input import has_behavior

        enc = BehavioralEncoder()
        sessions = [
            _session(kb=[_kb()], mo=[]),
            _session(kb=[], mo=[]),  # must be filtered first
        ]
        keep = [s for s in sessions if has_behavior(s)]
        out = enc.encode_entries(keep)
        assert out.shape == (1, 128)


# =====================================================================
# Determinism
# =====================================================================

class TestDeterminism:

    def test_same_seed_reproduces_forward(self):
        sessions = [_session(kb=[_kb(), _kb(hold=95.0)], mo=[_mo()])]
        set_seed(123)
        a = BehavioralEncoder().encode_entries(sessions)
        set_seed(123)
        b = BehavioralEncoder().encode_entries(sessions)
        assert torch.equal(a, b)

    def test_seed_constructor_reproduces_forward(self):
        sessions = [_session(kb=[_kb(hold=88.0)], mo=[_mo(), _mo()])]
        a = BehavioralEncoder(seed=7).encode_entries(sessions)
        b = BehavioralEncoder(seed=7).encode_entries(sessions)
        assert torch.equal(a, b)

    def test_different_seeds_usually_differ(self):
        sessions = [_session(kb=[_kb(), _kb(hold=95.0)], mo=[_mo()])]
        a = BehavioralEncoder(seed=1).encode_entries(sessions)
        b = BehavioralEncoder(seed=2).encode_entries(sessions)
        assert not torch.allclose(a, b, atol=1e-3)

    def test_reseed_before_inference_is_reproducible(self):
        enc = BehavioralEncoder()
        sessions = [_session(kb=[_kb()], mo=[_mo()])]
        set_seed(42)
        first = enc.encode_entries(sessions)
        set_seed(42)
        second = enc.encode_entries(sessions)
        assert torch.equal(first, second)


# =====================================================================
# Gradient flow
# =====================================================================

class TestGradients:

    def test_gradients_flow_to_all_leaf_parameters(self):
        enc = BehavioralEncoder()
        kb = collate_sequences([[_kb(), _kb(hold=95.0)]], KEYBOARD_FEATURE_COLUMNS)
        mo = collate_sequences([[_mo(), _mo()]], MOUSE_FEATURE_COLUMNS)
        out = enc(keyboard=kb, mouse=mo)
        out.sum().backward()
        leaf_params = [p for p in enc.parameters() if p.requires_grad]
        assert len(leaf_params) > 0
        missing = [name for name, p in enc.named_parameters()
                   if p.requires_grad and p.grad is None]
        assert missing == []

    def test_gradients_flow_with_mixed_modalities(self):
        """Both encoders execute when each modality is used by some item."""
        enc = BehavioralEncoder()
        sessions = [
            _session(kb=[_kb()], mo=[]),       # keyboard-only item
            _session(kb=[], mo=[_mo()]),       # mouse-only item
        ]
        out = enc.encode_entries(sessions)
        out.sum().backward()
        for name, p in enc.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, "no gradient for {}".format(name)


# =====================================================================
# Identity independence & feature sensitivity (privacy-facing)
# =====================================================================

class TestIdentityIndependence:

    def test_embedding_depends_only_on_feature_values(self):
        """user_id/session_id labels must never influence the embedding."""
        shared_kb = [_kb(hold=80.0, flight=40.0), _kb(hold=90.0, flight=30.0)]
        shared_mo = [_mo()]
        enc = BehavioralEncoder()
        a = enc.encode_entries([_session(kb=shared_kb, mo=shared_mo, session_id="x1",
                                         user_id="alice")])
        b = enc.encode_entries([_session(kb=shared_kb, mo=shared_mo, session_id="x2",
                                         user_id="bob")])
        assert torch.equal(a, b)

    def test_changed_feature_values_change_embedding(self):
        enc = BehavioralEncoder()
        a = enc.encode_entries([_session(kb=[_kb(hold=80.0)], mo=[_mo()])])
        b = enc.encode_entries([_session(kb=[_kb(hold=500.0)], mo=[_mo()])])
        assert not torch.allclose(a, b, atol=1e-3)


# =====================================================================
# End-to-end integration with the Phase 4 dataset
# =====================================================================

class TestIntegration:

    def test_scaled_dataset_embeddings(self):
        entries = _dataset(seed=0, n_users=2, sessions=4)
        kb_scaler, mo_scaler = _scalers(entries)

        enc = BehavioralEncoder(seed=0)
        out = enc.encode_entries(entries, keyboard_scaler=kb_scaler,
                                 mouse_scaler=mo_scaler)
        assert out.shape == (8, 128)
        assert bool(torch.isfinite(out).all())
        # both users' sessions are distinct on average
        user0 = out[0:4].mean(dim=0)
        user1 = out[4:8].mean(dim=0)
        assert float(torch.dist(user0.detach(), user1.detach())) > 0.0

    def test_encode_session_matches_encode_entries_single(self):
        entry = _dataset(seed=0, n_users=1, sessions=2)[0]
        enc = BehavioralEncoder(seed=3)
        single = enc.encode_entries([entry])
        session = enc.encode_session(entry)
        assert session.shape == (128,)
        assert torch.equal(single[0], session)

    def test_end_to_end_raw_fixture(self):
        """Phase 2 raw fixture -> Phase 3 processing -> encoder embedding."""
        import json
        import os

        from ml.preprocessing import process_session

        fixture = os.path.join(
            os.path.dirname(__file__), "fixtures", "session_valid.json"
        )
        with open(fixture) as f:
            raw = json.load(f)
        processed = process_session(raw)
        enc = BehavioralEncoder(seed=0)
        emb = enc.encode_session(processed)
        assert emb.shape == (128,)
        assert bool(torch.isfinite(emb).all())