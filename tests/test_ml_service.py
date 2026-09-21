"""
Phase 9B — tests for the inference-only ML service layer.

The service is exercised with a deterministic fake verifier + injected artifact
loaders so the *wiring* (preprocessing -> embedding -> enrollment -> distance
-> calibrated threshold) is tested in isolation from the real checkpoint. One
scalar-reuse test asserts the injected FeatureScalers are threaded through.

No production checkpoint or dataset is touched here.
"""

import zlib

import pytest
import torch

from backend.app.config import Settings
from backend.app.services.ml_service import (
    BehavioralMLService,
    InferenceError,
    InvalidSessionError,
    ModelLoadError,
    PreprocessingArtifactError,
    VerificationConfigError,
)
from ml.preprocessing import FeatureScaler

SETTINGS = Settings(environment="test", debug=False)


def _raw_session(session_id, *, kb=4, mouse=4):
    t = 0.0
    keyboard_events = []
    for i in range(kb):
        keyboard_events.append({"event_type": "keyboard", "event": "keydown", "timestamp": t})
        t += 50.0
        keyboard_events.append({"event_type": "keyboard", "event": "keyup", "timestamp": t})
        t += 100.0
    x = y = mt = 0.0
    mouse_events = []
    for i in range(mouse):
        x += 5.0 + i
        y += 3.0
        mt += 16.0
        mouse_events.append(
            {"event_type": "mouse", "event": "mousemove", "x": x, "y": y, "timestamp": mt}
        )
    return {
        "session_id": session_id,
        "started_at": "2025-01-01T00:00:00Z",
        "ended_at": "2025-01-01T00:00:05Z",
        "timestamp_source": "test",
        "keyboard_events": keyboard_events,
        "mouse_events": mouse_events,
    }


class _FakeVerifier:
    """Deterministic embeddings: close within a user, far across users.

    ``user`` is derived from the ``<user>_<session>`` session_id convention;
    the real service strips identity fields, so the processed dicts only carry
    session_id (lowercase comparison: the derived group is the prefix).
    """

    def __init__(self, dim=8, noise=0.1):
        self.dim = dim
        self.embedding_dim = dim
        self.noise = noise
        self.seen_scalers = []

    @staticmethod
    def _vec(key):
        generator = torch.Generator()
        generator.manual_seed(zlib.crc32(key.encode("utf-8")))
        return torch.randn((8,), generator=generator)

    def _user(self, session):
        sid = str(session.get("session_id", ""))
        return sid.split("_", 1)[0] if "_" in sid else sid

    def embed_session(self, session, keyboard_scaler=None, mouse_scaler=None):
        self.seen_scalers.append((keyboard_scaler, mouse_scaler))
        return self._vec(self._user(session)) + self.noise * self._vec(
            "noise:" + str(session.get("session_id"))
        )

    def embed_sessions(self, sessions, keyboard_scaler=None, mouse_scaler=None, batch_size=16):
        self.seen_scalers.append((keyboard_scaler, mouse_scaler))
        return torch.stack(
            [self.embed_session(s, keyboard_scaler, mouse_scaler) for s in sessions]
        )

    def pair_distance(self, a, b):
        return torch.sqrt(((a - b) ** 2).sum(dim=-1) + 1e-8)


def _fill_scalers():
    keyboard = FeatureScaler(["hold_time", "flight_time"]).fit([[1, 2], [2, 1], [3, 1]])
    mouse = FeatureScaler(["dx", "dy", "dt", "distance", "speed"]).fit(
        [[1, 1, 1, 1, 1], [2, 1, 1, 2, 2], [1, 2, 1, 2, 2]]
    )
    return keyboard, mouse


def make_service(verifier=None, threshold=1.0, prep_payload=None, config_payload=None, **kw):
    verifier = verifier or _FakeVerifier()
    return BehavioralMLService(
        SETTINGS,
        verifier_factory=lambda s: verifier,
        preprocessing_loader=lambda s: (
            None,
            None,
            prep_payload or {"artifact": "behavioral-preprocessing", "artifact_version": 1},
        ),
        verification_config_loader=lambda s: config_payload
        or {"artifact": "verification-config", "calibration": {"threshold": threshold}},
        **kw,
    )


class TestLoad:
    def test_unloaded_by_default(self):
        service = make_service()
        assert not service.is_loaded

    def test_load_initialises_and_is_idempotent(self):
        calls = []
        verifier = _FakeVerifier()

        def factory(settings):
            calls.append(settings)
            return verifier

        service = BehavioralMLService(
            SETTINGS,
            verifier_factory=factory,
            preprocessing_loader=lambda s: (None, None, {"artifact": "behavioral-preprocessing"}),
            verification_config_loader=lambda s: {
                "artifact": "verification-config",
                "calibration": {"threshold": 2.5},
            },
        )
        assert not service.is_loaded
        service.load()
        service.load()
        assert service.is_loaded
        assert len(calls) == 1
        assert service.threshold == 2.5
        assert service.embedding_dim == verifier.embedding_dim

    def test_embedding_dim_matches_verifier(self):
        service = make_service()
        assert service.embedding_dim == 8

    def test_scalers_from_artifact_are_threaded_through(self):
        keyboard, mouse = _fill_scalers()
        verifier = _FakeVerifier()
        service = BehavioralMLService(
            SETTINGS,
            verifier_factory=lambda s: verifier,
            preprocessing_loader=lambda s: (
                keyboard,
                mouse,
                {"artifact": "behavioral-preprocessing"},
            ),
            verification_config_loader=lambda s: {
                "artifact": "verification-config",
                "calibration": {"threshold": 1.0},
            },
        )
        stored = service.enroll_to_store("u1", [_raw_session("u1_s1")])
        service.verify_stored("u1", stored, _raw_session("u1_s2"))
        seen = verifier.seen_scalers
        assert seen and all(kb is keyboard for kb, _ in seen)
        assert seen and all(ms is mouse for _, ms in seen)


class TestEnrollVerify:
    def test_genuine_verified_impostor_suspicious(self):
        service = make_service(threshold=1.0)
        profile = service.enroll_sessions(
            "u1", [_raw_session("u1_s1"), _raw_session("u1_s2")]
        )
        assert profile.user_id == "u1"
        assert profile.n_sessions == 2
        assert profile.embedding_dim == 8

        stored = service.enroll_to_store("u1", [_raw_session("u1_s1"), _raw_session("u1_s2")])
        assert stored.session_count == 2
        genuine = service.verify_stored("u1", stored, _raw_session("u1_s3"))
        impostor = service.verify_stored("u1", stored, _raw_session("u2_s1"))
        assert genuine.decision == "VERIFIED"
        assert impostor.decision == "SUSPICIOUS"
        assert genuine.distance < impostor.distance

    def test_decision_respects_configured_threshold(self):
        service = make_service(threshold=0.0)
        stored = service.enroll_to_store("u1", [_raw_session("u1_s1")])
        result = service.verify_stored("u1", stored, _raw_session("u1_s2"))
        assert result.decision == "SUSPICIOUS"
        assert result.threshold == 0.0

    def test_enrollment_deterministic(self):
        service = make_service()
        first = service.enroll_sessions("u1", [_raw_session("u1_s1"), _raw_session("u1_s2")])
        second = service.enroll_sessions("u1", [_raw_session("u1_s1"), _raw_session("u1_s2")])
        assert torch.allclose(first.centroid, second.centroid)

    def test_verify_result_carries_decision_fields(self):
        service = make_service()
        stored = service.enroll_to_store("u1", [_raw_session("u1_s1")])
        result = service.verify_stored("u1", stored, _raw_session("u1_s1"))
        assert result.user_id == "u1"
        assert result.decision in ("VERIFIED", "SUSPICIOUS")
        assert result.distance >= 0
        assert result.threshold == 1.0
        assert result.n_enrolled_sessions == 1
        assert result.embedding_dim == 8


class TestInvalidSessions:
    def test_missing_events_rejected(self):
        service = make_service()
        bad = _raw_session("u1_s1")
        del bad["mouse_events"]
        with pytest.raises(InvalidSessionError) as info:
            service.enroll_sessions("u1", [bad])
        assert info.value.code == "invalid_session"
        assert info.value.status_code == 422

    def test_session_without_behaviour_rejected(self):
        service = make_service()
        empty = _raw_session("u1_s1")
        empty["keyboard_events"] = []  # only keydown without keyup yields no features
        empty["mouse_events"] = []
        with pytest.raises(InvalidSessionError):
            service.enroll_sessions("u1", [empty])

    def test_empty_enrollment_rejected(self):
        service = make_service()
        with pytest.raises(InvalidSessionError):
            service.enroll_sessions("u1", [])

    def test_unmatched_keydown_only_session_rejected(self):
        service = make_service()
        stored = service.enroll_to_store("u1", [_raw_session("u1_s1")])
        session = _raw_session("u1_s1")
        session["keyboard_events"] = [
            {"event_type": "keyboard", "event": "keydown", "timestamp": 1.0}
        ]
        session["mouse_events"] = []
        with pytest.raises(InvalidSessionError):
            service.verify_stored("u1", stored, session)


class TestFailures:
    def test_model_load_error_on_missing_checkpoint(self):
        service = BehavioralMLService(
            SETTINGS,
            verifier_factory=lambda s: (_ for _ in ()).throw(RuntimeError("missing")),
            preprocessing_loader=lambda s: (None, None, {}),
            verification_config_loader=lambda s: {},
        )
        with pytest.raises(ModelLoadError) as info:
            service.enroll_sessions("u1", [_raw_session("u1_s1")])
        assert info.value.code == "model_load_error"
        assert info.value.status_code == 503

    def test_preprocessing_artifact_error(self):
        service = BehavioralMLService(
            SETTINGS,
            verifier_factory=lambda s: _FakeVerifier(),
            preprocessing_loader=lambda s: (_ for _ in ()).throw(PreprocessingArtifactError()),
            verification_config_loader=lambda s: {},
        )
        with pytest.raises(PreprocessingArtifactError) as info:
            service.enroll_sessions("u1", [_raw_session("u1_s1")])
        assert info.value.code == "preprocessing_artifact_error"

    def test_verification_config_error(self):
        service = BehavioralMLService(
            SETTINGS,
            verifier_factory=lambda s: _FakeVerifier(),
            preprocessing_loader=lambda s: (None, None, {}),
            verification_config_loader=lambda s: (_ for _ in ()).throw(VerificationConfigError()),
        )
        with pytest.raises(VerificationConfigError) as info:
            service.enroll_sessions("u1", [_raw_session("u1_s1")])
        assert info.value.code == "verification_config_error"

    def test_inference_failure_is_reported(self):
        class _BrokenVerifier(_FakeVerifier):
            def embed_sessions(self, sessions, keyboard_scaler=None, mouse_scaler=None, batch_size=16):
                raise RuntimeError("boom")

        service = make_service(verifier=_BrokenVerifier())
        with pytest.raises(InferenceError) as info:
            service.enroll_sessions("u1", [_raw_session("u1_s1")])
        assert info.value.code == "inference_error"