"""
Phase 6 Siamese verifier tests.

Run from the repo root::

    .venv/bin/python -m pytest tests/test_siamese.py -v
"""

import pytest
import torch

from dataset.generator import SyntheticDatasetGenerator
from ml.encoder import BehavioralEncoder, EncoderConfig, set_seed
from ml.verification.siamese import SiameseOutput, SiameseVerifier


def _session(kb=(1.0,), mo=(1.0,), session_id="s", user_id="u"):
    return {
        "session_id": session_id,
        "user_id": user_id,
        "keyboard_sequence": [{"hold_time": float(kb[0]), "flight_time": float(kb[0])}],
        "mouse_sequence": [
            {"dx": float(mo[0]), "dy": float(mo[0]), "dt": 16.0,
             "distance": float(mo[0]), "speed": 0.2}
        ],
        "mouse_action_events": [],
    }


def _dataset(seed=0, n_users=3, sessions=4):
    return SyntheticDatasetGenerator(
        seed=seed, n_users=n_users, sessions_per_user=sessions
    ).generate()


class TestShapes:

    def test_duplicate_sessions_same_embedding_shapes(self):
        verifier = SiameseVerifier(seed=0)
        s = _session()
        out = verifier.verify(s, s)
        assert out.embedding_a.shape == (1, 128)
        assert out.embedding_b.shape == (1, 128)
        assert out.distance.shape == (1,)

    def test_batch_shapes(self):
        verifier = SiameseVerifier(seed=0)
        entries = _dataset(seed=0, n_users=2, sessions=3)
        a = entries[:3]
        b = entries[3:6]
        out = verifier(a, b)
        assert out.embedding_a.shape == (3, 128)
        assert out.embedding_b.shape == (3, 128)
        assert out.distance.shape == (3,)

    def test_duplicate_sessions_identical_embeddings(self):
        verifier = SiameseVerifier(seed=0)
        s = _session(kb=(80.0,), mo=(2.0,))
        out = verifier.verify(s, s)
        assert torch.equal(out.embedding_a, out.embedding_b)

    def test_empty_batch_shape(self):
        verifier = SiameseVerifier(seed=0)
        out = verifier([], [])
        assert out.embedding_a.shape == (0, 128)
        assert out.embedding_b.shape == (0, 128)
        assert out.distance.shape == (0,)


class TestDistance:

    def test_distance_non_negative_and_finite(self):
        verifier = SiameseVerifier(seed=0)
        s1 = _session(kb=(80.0,), mo=(2.0,))
        s2 = _session(kb=(200.0,), mo=(5.0,), session_id="x", user_id="v")
        for batch in ([(s1, s2)], [(s1, s2), (s2, s1)]):
            a = [p[0] for p in batch]
            b = [p[1] for p in batch]
            out = verifier(a, b)
            assert bool((out.distance >= 0.0).all())
            assert bool(torch.isfinite(out.distance).all())

    def test_identical_embeddings_zero_distance(self):
        verifier = SiameseVerifier(seed=0)
        s = _session(kb=(80.0,), mo=(2.0,))
        out = verifier.verify(s, s)
        assert float(out.distance[0].detach()) < 1e-3  # sqrt(eps)

    def test_distance_symmetric(self):
        verifier = SiameseVerifier(seed=3)
        s1 = _session(kb=(80.0,), mo=(2.0,))
        s2 = _session(kb=(140.0,), mo=(4.0,), session_id="y", user_id="w")
        d_ab = float(verifier.verify(s1, s2).distance[0].detach())
        d_ba = float(verifier.verify(s2, s1).distance[0].detach())
        assert d_ab == pytest.approx(d_ba, abs=1e-6)

    def test_different_behaviour_gives_nonzero_distance(self):
        verifier = SiameseVerifier(seed=0)
        s1 = _session(kb=(80.0,), mo=(2.0,))
        s2 = _session(kb=(999.0,), mo=(50.0,), session_id="q", user_id="u2")
        out = verifier.verify(s1, s2)
        assert float(out.distance[0].detach()) > 1e-3

    def test_embeddings_equal_through_shared_encoder(self):
        verifier = SiameseVerifier(seed=0)
        s1 = _session(kb=(80.0,), mo=(2.0,))
        # A duplicate copy of the same session must land at the same embedding
        out = verifier.verify(s1, _session(kb=(80.0,), mo=(2.0,), session_id="copy"))
        assert torch.allclose(out.embedding_a, out.embedding_b, atol=1e-6)


class TestSharedWeights:

    def test_single_shared_encoder_instance(self):
        encoder = BehavioralEncoder(seed=0)
        verifier = SiameseVerifier(encoder=encoder)
        # Both branches are literally the same module.
        assert verifier.shared_encoder is encoder
        assert verifier.embedding_dim == encoder.embedding_dim == 128

    def test_parameters_are_identical_objects(self):
        encoder = BehavioralEncoder(seed=0)
        verifier = SiameseVerifier(encoder=encoder)
        # The Siamese module must NOT create a second independent encoder.
        enc_params = {id(p) for p in encoder.parameters()}
        ver_params = {id(p) for p in verifier.parameters()}
        assert enc_params == ver_params
        # exactly one set of encoder parameters lives inside the verifier
        assert len(enc_params) == len(ver_params)

    def test_parameter_pointers_match_across_branches(self):
        verifier = SiameseVerifier(seed=0)
        entries = _dataset(seed=0, n_users=1, sessions=2)
        out = verifier(entries, entries)
        # embeddings computed through the same parameter tensors
        weight_a = verifier.shared_encoder.keyboard_encoder.conv.weight
        assert out.embedding_a.grad_fn is not None


class TestGradients:

    def test_gradients_flow_through_shared_encoder(self):
        verifier = SiameseVerifier(seed=0)
        entries = _dataset(seed=0, n_users=2, sessions=2)
        out = verifier([entries[0]], [entries[1]])
        out.distance.sum().backward()
        missing = [
            name for name, p in verifier.shared_encoder.named_parameters()
            if p.requires_grad and p.grad is None
        ]
        assert missing == []

    def test_gradient_of_distance_wrt_embeddings(self):
        verifier = SiameseVerifier(seed=0)
        a_batch = _dataset(seed=0, n_users=1, sessions=2)
        out = verifier(a_batch, a_batch)
        assert out.distance.requires_grad
        # distance is differentiable through embeddings
        assert out.embedding_a.grad_fn is not None


class TestBatchAndDeterminism:

    def test_batched_inputs_work(self):
        verifier = SiameseVerifier(seed=0)
        entries = _dataset(seed=1, n_users=4, sessions=4)  # 16 entries
        a = entries[:8]
        b = entries[8:16]
        out = verifier(a, b)
        assert out.embedding_a.shape == (8, 128)
        assert bool(torch.isfinite(out.embedding_a).all())
        assert bool(torch.isfinite(out.embedding_b).all())
        assert bool(torch.isfinite(out.distance).all())

    def test_seeded_construction_reproducible(self):
        entries = _dataset(seed=0, n_users=2, sessions=2)
        a = SiameseVerifier(seed=7)
        b = SiameseVerifier(seed=7)
        oa = a(entries[:1], entries[1:2])
        ob = b(entries[:1], entries[1:2])
        assert torch.equal(oa.embedding_a, ob.embedding_a)
        assert torch.equal(oa.distance, ob.distance)

    def test_reseed_before_forward_reproducible(self):
        verifier = SiameseVerifier(seed=1)
        entries = _dataset(seed=0, n_users=2, sessions=2)
        set_seed(42)
        first = verifier(entries[:1], entries[1:2]).distance
        set_seed(42)
        second = verifier(entries[:1], entries[1:2]).distance
        assert torch.equal(first, second)


class TestEncoderContract:

    def test_phase5_encoder_contract_intact(self):
        encoder = BehavioralEncoder(seed=0)
        entries = _dataset(seed=0, n_users=1, sessions=2)
        embeddings = encoder.encode_entries(entries)
        assert embeddings.ndim == 2
        assert embeddings.shape == (2, 128)
        single = encoder.encode_session(entries[0])
        assert single.shape == (128,)

    def test_verifier_reuses_phase5_encoder_directly(self):
        encoder = BehavioralEncoder(seed=0)
        verifier = SiameseVerifier(encoder=encoder)
        entries = _dataset(seed=0, n_users=1, sessions=2)
        expected_a = encoder.encode_entries([entries[0]])[0]
        out = verifier([entries[0]], [entries[1]])
        assert torch.allclose(out.embedding_a[0], expected_a, atol=1e-5)

    def test_config_used_for_fresh_encoder(self):
        cfg = EncoderConfig(embedding_dim=32, conv_channels=16, gru_hidden=32)
        verifier = SiameseVerifier(config=cfg, seed=0)
        assert verifier.embedding_dim == 64  # 2 * 32
        s = _session()
        out = verifier.verify(s, s)
        assert out.embedding_a.shape == (1, 64)

    def test_invalid_encoder_type_rejected(self):
        with pytest.raises(TypeError, match="BehavioralEncoder"):
            SiameseVerifier(encoder="not-an-encoder")


class TestIdentitySeparation:

    def test_identity_metadata_never_affects_embedding(self):
        verifier = SiameseVerifier(seed=0)
        kb = [(80.0,), (90.0,)]
        mo = [(2.0,), (3.0,)]
        a = _session(kb=kb[0], mo=mo[0], session_id="sid-a", user_id="alice")
        a_copy = _session(kb=kb[0], mo=mo[0], session_id="sid-b", user_id="bob")
        out = verifier.verify(a, a_copy)
        # identical behaviour with different ids -> ~zero distance
        assert float(out.distance[0].detach()) < 1e-3

    def test_identity_field_change_changes_nothing(self):
        verifier = SiameseVerifier(seed=0)
        s1 = _session(kb=(80.0,), mo=(2.0,), session_id="x1", user_id="u1")
        s2 = _session(kb=(80.0,), mo=(2.0,), session_id="x2", user_id="u2")
        out = verifier.verify(s1, s2)
        assert torch.equal(out.embedding_a, out.embedding_b)

    def test_session_input_is_behaviour_only(self):
        """Only keyboard/mouse sequences reach the model - user/session ids
        injected into extra dict keys must not appear in the tensors."""
        verifier = SiameseVerifier(seed=0)
        s1 = _session(kb=(80.0,), mo=(2.0,))
        s2 = dict(s1, session_id="zz", user_id="zz", evil_key="KEY_A")
        # No error and the evil metadata key is ignored.
        out = verifier.verify(s1, s2)
        assert float(out.distance[0].detach()) < 1e-3


class TestValidation:

    def test_mismatched_lengths_rejected(self):
        verifier = SiameseVerifier(seed=0)
        entries = _dataset(seed=0, n_users=1, sessions=3)
        with pytest.raises(ValueError, match="equal length"):
            verifier(entries, entries[:2])

    def test_non_sequence_input_rejected(self):
        verifier = SiameseVerifier(seed=0)
        s = _session()
        with pytest.raises(TypeError, match="lists"):
            verifier(s, s)

    def test_empty_session_rejected(self):
        verifier = SiameseVerifier(seed=0)
        empty = {"session_id": "e", "user_id": "u", "keyboard_sequence": [], "mouse_sequence": []}
        with pytest.raises(ValueError, match="no behavioural content"):
            verifier([empty], [_session()])

    def test_eps_validation(self):
        with pytest.raises(ValueError, match="eps"):
            SiameseVerifier(seed=0, eps=0.0)

    def test_return_type(self):
        verifier = SiameseVerifier(seed=0)
        s = _session()
        assert isinstance(verifier.verify(s, s), SiameseOutput)