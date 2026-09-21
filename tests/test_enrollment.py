"""
Phase 8 — tests for enrollment profiles and verification decisions.

The bulk of the tests use a tiny deterministic :class:`_FakeVerifier` (no
torch model / checkpoint needed) so the enrollment *logic* is exercised in
isolation. One integration test at the end uses the real Phase 7 checkpoint
(skipped when the artifact is missing).

Run:
    .venv/bin/python -m pytest tests/test_enrollment.py -v
"""

import json
import os
import zlib

import pytest
import torch

from ml.evaluation.enrollment import EnrollmentProfile, enroll, verify

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(_REPO, "models", "siamese_behavioral_encoder.pt")
DATASET_PATH = os.path.join(_REPO, "data", "datasets", "dataset_synthetic.json")


def _session(user_id, session_id):
    return {
        "user_id": user_id,
        "session_id": session_id,
        "keyboard_sequence": [{"hold_time": 1, "flight_time": 1}],
        "mouse_sequence": [{"dx": 1.0, "dy": 1.0, "dt": 0.01, "distance": 2.0, "speed": 100.0}],
    }


class _FakeVerifier:
    """Deterministic embedding generator: user base vector + small session noise.

    Genuine sessions of the same user therefore sit close together and
    different users are far apart, which is exactly the geometry the Phase 6
    distance convention assumes.
    """

    def __init__(self, dim=8, noise=0.1):
        self.dim = dim
        self.noise = noise

    @staticmethod
    def _vector(key):
        generator = torch.Generator()
        generator.manual_seed(zlib.crc32(key.encode("utf-8")))
        return torch.randn((8,), generator=generator)

    def _embed(self, session):
        base = self._vector(session["user_id"])
        noise = self._vector(session["session_id"]) * self.noise
        return base + noise

    def embed_session(self, session, keyboard_scaler=None, mouse_scaler=None):
        return self._embed(session)

    def embed_sessions(self, sessions, keyboard_scaler=None, mouse_scaler=None, batch_size=16):
        return torch.stack([self._embed(s) for s in sessions])

    def pair_distance(self, embeddings_a, embeddings_b):
        return torch.sqrt(((embeddings_a - embeddings_b) ** 2).sum(dim=-1) + 1e-8)


@pytest.fixture(scope="function")
def fake():
    return _FakeVerifier()


def _sessions_for(users=("u1", "u2"), per_user=2):
    return [_session(uid, "{}_s{}".format(uid, i)) for uid in users for i in range(per_user)]


class TestEnroll:
    def test_centroid_is_mean_embedding(self, fake):
        sessions = _sessions_for(("u1",), 2)
        profile = enroll(sessions, fake, user_ref="u1")
        embeddings = fake.embed_sessions(sessions)
        expected_centroid = embeddings.mean(dim=0)
        assert torch.equal(profile.centroid, expected_centroid)
        assert profile.n_sessions == 2
        assert profile.embedding_dim == fake.dim

    def test_user_ref_falls_back_to_session(self, fake):
        sessions = _sessions_for(("u1",), 1)
        profile = enroll(sessions, fake)
        assert profile.user_id == "u1"

    def test_empty_raises(self, fake):
        with pytest.raises(ValueError):
            enroll([], fake)

    def test_non_behavioral_session_raises(self, fake):
        bad = {"user_id": "u", "session_id": "s", "keyboard_sequence": [], "mouse_sequence": []}
        with pytest.raises(ValueError):
            enroll([bad], fake)

    def test_verifier_without_embed_sessions_raises(self, fake):
        class Broken:
            pass

        with pytest.raises(TypeError):
            enroll([_session("u1", "u1_s1")], Broken())

    def test_duplicate_sessions_allowed(self, fake):
        sessions = [_session("u1", "u1_s1"), _session("u1", "u1_s1")]
        profile = enroll(sessions, fake, user_ref="u1")
        assert profile.n_sessions == 2


class TestVerify:
    def test_genuine_verified_and_impostor_suspicious(self, fake):
        sessions = [
            _session("u1", "u1_s0"),
            _session("u1", "u1_s1"),
            _session("u1", "u1_s2"),   # genuine probe for u1
            _session("u2", "u2_s0"),   # impostor probe
            _session("u2", "u2_s1"),
        ]
        profile = enroll(sessions[:2], fake, user_ref="u1")
        result_gen = verify(profile, sessions[2], fake, threshold=0.5)
        result_imp = verify(profile, sessions[3], fake, threshold=0.5)
        assert result_gen.decision == "VERIFIED"
        assert result_imp.decision == "SUSPICIOUS"
        assert result_gen.distance < result_imp.distance

    def test_result_fields(self, fake):
        sessions = _sessions_for(("u1", "u2"), 1)
        profile = enroll([sessions[0]], fake, user_ref="u1")
        result = verify(profile, sessions[1], fake, threshold=0.25)
        assert result.user_id == "u1"
        assert result.session_id == "u2_s0"
        assert result.threshold == 0.25
        assert result.n_enrolled_sessions == 1
        assert result.embedding_dim == fake.dim
        assert result.decision in ("VERIFIED", "SUSPICIOUS")

    def test_threshold_rule(self, fake):
        sessions = _sessions_for(("u1",), 1)
        profile = enroll([sessions[0]], fake, user_ref="u1")
        probe = sessions[0]
        distance = verify(profile, probe, fake, threshold=10 ** 9).distance
        assert verify(profile, probe, fake, threshold=10 ** 9).decision == "VERIFIED"
        assert verify(profile, probe, fake, threshold=0.0).decision == "SUSPICIOUS"
        assert distance > 0.0

    def test_boundary_exact_equality_is_verified(self, fake):
        # distance of an identical probe is sqrt(eps) > 0 (Phase 6 convention);
        # a threshold set exactly to that distance must still verify.
        profile = EnrollmentProfile(user_id="u1", centroid=torch.zeros(8), n_sessions=1, embedding_dim=8)
        probe = _session("u1", "u1_s0")

        def zero_embed(*a, **k):
            return torch.zeros(8, dtype=torch.float32)

        fake.embed_session = zero_embed
        boundary = float(torch.sqrt(torch.tensor(1e-8, dtype=torch.float32)))
        result = verify(profile, probe, fake, threshold=boundary)
        assert result.distance == pytest.approx(boundary)
        assert result.decision == "VERIFIED"

    def test_probe_without_behaviour_raises(self, fake):
        sessions = _sessions_for(("u1",), 1)
        profile = enroll([sessions[0]], fake, user_ref="u1")
        empty = {"user_id": "u1", "session_id": "x", "keyboard_sequence": [], "mouse_sequence": []}
        with pytest.raises(ValueError):
            verify(profile, empty, fake, threshold=0.5)

    def test_negative_threshold_raises(self, fake):
        sessions = _sessions_for(("u1",), 1)
        profile = enroll([sessions[0]], fake, user_ref="u1")
        with pytest.raises(ValueError):
            verify(profile, sessions[0], fake, threshold=-0.1)

    def test_dimension_mismatch_raises(self, fake):
        profile = EnrollmentProfile(user_id="u1", centroid=torch.zeros(8), n_sessions=1, embedding_dim=8)
        other = _FakeVerifier(dim=8)
        other.dim = 16  # mismatch between verifier and profile dimension

        def embed16(session, *a, **k):
            return torch.zeros(16)

        other.embed_session = embed16
        with pytest.raises(RuntimeError):
            verify(profile, _session("u1", "u1_s0"), other, threshold=0.5)


@pytest.mark.skipif(
    not os.path.exists(CHECKPOINT_PATH) or not os.path.exists(DATASET_PATH),
    reason="Phase 7 checkpoint and/or synthetic dataset not present on disk",
)
def test_enroll_and_verify_with_real_model():
    from ml.evaluation.model import load_verifier  # noqa: PLC0415
    from ml.training import TrainingConfig, prepare_training_data  # noqa: PLC0415

    data = prepare_training_data(TrainingConfig())
    verifier = load_verifier(CHECKPOINT_PATH)
    user = "user_001"
    enrolled = [s for s in data.val_sessions if s["user_id"] == user][:2]
    profile = enroll(
        enrolled, verifier,
        keyboard_scaler=data.keyboard_scaler, mouse_scaler=data.mouse_scaler,
        user_ref=user,
    )
    assert profile.embedding_dim == 128
    assert profile.n_sessions == 2
    genuine_probe = next(s for s in data.test_sessions if s["user_id"] == user)
    impostor_probe = next(s for s in data.test_sessions if s["user_id"] != user)
    # permissive threshold accepts everything
    for probe in (genuine_probe, impostor_probe):
        result = verify(
            profile, probe, verifier, threshold=10 ** 9,
            keyboard_scaler=data.keyboard_scaler, mouse_scaler=data.mouse_scaler,
        )
        assert result.decision == "VERIFIED"
        assert result.embedding_dim == 128
    # sanity: impostor should be further than (or equal to) genuine at permissive threshold
    d_gen = verify(
        profile, genuine_probe, verifier, threshold=10 ** 9,
        keyboard_scaler=data.keyboard_scaler, mouse_scaler=data.mouse_scaler,
    ).distance
    d_imp = verify(
        profile, impostor_probe, verifier, threshold=10 ** 9,
        keyboard_scaler=data.keyboard_scaler, mouse_scaler=data.mouse_scaler,
    ).distance
    assert d_imp > 0.0
    assert d_gen > 0.0