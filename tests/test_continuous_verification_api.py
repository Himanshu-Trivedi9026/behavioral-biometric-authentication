"""
Phase 14A — POST /api/v1/continuous-verification API tests.

The endpoint is a thin integration layer over the existing ML pipeline
(Phase 3 preprocessing / Phase 7 encoder / Phase 8 verification decision /
existing calibrated threshold) and the authenticated user's stored profile.
These tests use dependency overrides (fake verifier service + fresh in-memory
store) exactly like the Phase 9B/11 enrollment/verification tests, so only
the continuous-verification contract is under test.
"""

import zlib

import pytest
import torch
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.dependencies import get_ml_service, get_profile_store
from backend.app.main import create_app
from backend.app.services.ml_service import BehavioralMLService
from backend.app.services.profile_store import InMemoryProfileStore
from tests.auth_testing import TEST_JWT_SECRET, set_current_user

ENDPOINT = "/api/v1/continuous-verification"
ENROLLMENT_ENDPOINT = "/api/v1/enrollment"
VERIFICATION_ENDPOINT = "/api/v1/verification"


def _raw_session(session_id, *, kb=3, mouse=3):
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
        x += 4.0 + i
        y += 2.0
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
    def __init__(self, dim=8):
        self.dim = dim
        self.embedding_dim = dim
        self.checkpoint_id = "phase14a-fake-verifier"

    @staticmethod
    def _vec(key):
        generator = torch.Generator()
        generator.manual_seed(zlib.crc32(key.encode("utf-8")))
        return torch.randn((8,), generator=generator)

    def embed_session(self, session, keyboard_scaler=None, mouse_scaler=None):
        return self._vec(str(session.get("session_id", "unknown")))

    def embed_sessions(self, sessions, keyboard_scaler=None, mouse_scaler=None, batch_size=16):
        return torch.stack([self.embed_session(s) for s in sessions])

    def pair_distance(self, a, b):
        return torch.sqrt(((a - b) ** 2).sum(dim=-1) + 1e-8)


def _make_client(threshold, store=None):
    store = store or InMemoryProfileStore()
    service = BehavioralMLService(
        Settings(environment="test", debug=False),
        verifier_factory=lambda s: _FakeVerifier(),
        preprocessing_loader=lambda s: (None, None, {"artifact": "behavioral-preprocessing", "artifact_version": 1}),
        verification_config_loader=lambda s: {"artifact": "verification-config", "calibration": {"threshold": threshold}},
    )
    app = create_app(Settings(environment="test", debug=False))
    app.dependency_overrides[get_ml_service] = lambda: service
    app.dependency_overrides[get_profile_store] = lambda: store
    set_current_user(app, username="alice")
    return TestClient(app), service, store


def _enroll(client, user_ref="alice", session=None):
    response = client.post(
        ENROLLMENT_ENDPOINT,
        json={"user_ref": user_ref, "sessions": [session or _raw_session(user_ref + "_s1")]},
    )
    assert response.status_code == 201, response.text


@pytest.fixture(scope="function")
def default_client():
    client, _, _ = _make_client(threshold=1.0)
    with client as c:
        yield c


class TestAuthentication:
    def test_unauthenticated_request_is_401(self):
        app = create_app(Settings(environment="test", jwt_secret_key=TEST_JWT_SECRET))
        with TestClient(app) as client:
            response = client.post(ENDPOINT, json=_raw_session("x_s1"))
            assert response.status_code == 401
            assert response.json()["error"]["code"] == "authentication_required"

    def test_valid_body_without_credentials_is_401(self):
        app = create_app(Settings(environment="test", jwt_secret_key=TEST_JWT_SECRET))
        with TestClient(app) as client:
            response = client.post(ENDPOINT, json=_raw_session("x_s1"))
            assert response.status_code == 401
            assert response.json()["error"]["code"] == "authentication_required"

    def test_invalid_bearer_token_is_401(self):
        app = create_app(Settings(environment="test", jwt_secret_key=TEST_JWT_SECRET))
        with TestClient(app) as client:
            response = client.post(
                ENDPOINT,
                json=_raw_session("x_s1"),
                headers={"Authorization": "Bearer not-a-real-jwt"},
            )
            assert response.status_code == 401
            assert response.json()["error"]["code"] == "invalid_credentials"

    def test_unknown_profile_is_404(self, default_client):
        set_current_user(default_client.app, username="ghost")
        response = default_client.post(ENDPOINT, json=_raw_session("ghost_s1"))
        assert response.status_code == 404
        assert response.json() == {
            "error": {
                "code": "profile_not_found",
                "message": "no enrollment profile found for user 'ghost'",
            }
        }


class TestValidation:
    def test_malformed_window_is_422(self, default_client):
        _enroll(default_client)
        window = _raw_session("alice_s1")
        del window["mouse_events"]
        assert default_client.post(ENDPOINT, json=window).status_code == 422

    def test_empty_window_is_422(self, default_client):
        _enroll(default_client)
        window = _raw_session("alice_s1", kb=0, mouse=0)
        response = default_client.post(ENDPOINT, json=window)
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "invalid_session"

    def test_missing_session_id_is_422(self, default_client):
        window = _raw_session("alice_s1")
        del window["session_id"]
        assert default_client.post(ENDPOINT, json=window).status_code == 422

    def test_identity_field_in_body_is_rejected_422(self, default_client):
        _enroll(default_client)
        window = _raw_session("alice_s1")
        window["user_ref"] = "bob"
        response = default_client.post(ENDPOINT, json=window)
        assert response.status_code == 422


class TestDecisions:
    def test_same_session_window_is_verified(self, default_client):
        _enroll(default_client)
        response = default_client.post(ENDPOINT, json=_raw_session("alice_s1"))
        assert response.status_code == 200
        body = response.json()
        assert body["decision"] == "VERIFIED"
        assert body["distance"] < 0.5
        assert body["threshold"] == 1.0

    def test_zero_threshold_yields_suspicious(self):
        client, _, _ = _make_client(threshold=0.0)
        with client as c:
            _enroll(c)
            response = c.post(ENDPOINT, json=_raw_session("alice_s1"))
            body = response.json()
            assert response.status_code == 200
            assert body["decision"] == "SUSPICIOUS"
            assert body["threshold"] == 0.0
            assert body["distance"] >= 0.0

    def test_decision_consistent_with_threshold(self, default_client):
        _enroll(default_client)
        for session_id in ("alice_s1", "alice_s2", "alice_s3"):
            response = default_client.post(ENDPOINT, json=_raw_session(session_id))
            body = response.json()
            expected = "VERIFIED" if body["distance"] <= body["threshold"] else "SUSPICIOUS"
            assert body["decision"] == expected


class TestIdentityIsolation:
    def test_jwt_identity_determines_profile(self, default_client):
        _enroll(default_client, "alice")
        set_current_user(default_client.app, username="bob")
        _enroll(default_client, "bob")
        set_current_user(default_client.app, username="alice")
        service = default_client.app.dependency_overrides[get_ml_service]()
        seen = []
        original = service.verify_stored

        def recording(user_ref, stored, window):
            seen.append((user_ref, stored.user_ref))
            return original(user_ref, stored, window)

        service.verify_stored = recording
        response = default_client.post(ENDPOINT, json=_raw_session("alice_s1"))
        assert response.status_code == 200
        assert seen == [("alice", "alice")]

    def test_client_cannot_select_another_users_profile(self, default_client):
        _enroll(default_client, "alice")
        set_current_user(default_client.app, username="bob")
        _enroll(default_client, "bob")
        set_current_user(default_client.app, username="alice")
        window = _raw_session("alice_s1")
        window["user_ref"] = "bob"
        response = default_client.post(ENDPOINT, json=window)
        assert response.status_code == 422


class TestResponsePrivacy:
    def test_response_shape_is_minimal(self, default_client):
        _enroll(default_client)
        body = default_client.post(ENDPOINT, json=_raw_session("alice_s1")).json()
        # Phase 14B additions: the server-authoritative session state that the
        # decision just produced. Everything else stays minimal.
        assert set(body) == {
            "decision",
            "distance",
            "threshold",
            "session_state",
            "consecutive_suspicious",
            "last_verified_at",
        }

    def test_response_leaks_no_centroid_embedding_or_raw_events(self, default_client):
        _enroll(default_client)
        body = default_client.post(ENDPOINT, json=_raw_session("alice_s1")).json()
        raw = str(body)
        for forbidden in ("centroid", "embedding", "keyboard_events", "mouse_events", "user_ref", "password"):
            assert forbidden not in raw

    def test_response_has_no_identity_field(self, default_client):
        _enroll(default_client)
        body = default_client.post(ENDPOINT, json=_raw_session("alice_s1")).json()
        assert "user_ref" not in body

    def test_session_state_matches_decision(self, default_client):
        _enroll(default_client)
        body = default_client.post(ENDPOINT, json=_raw_session("alice_s1")).json()
        expected = (
            "verified" if body["decision"] == "VERIFIED" else "reverification_required"
        )
        assert body["session_state"] == expected
        assert body["consecutive_suspicious"] >= 0


class TestProfileIntegrity:
    def test_stored_profile_is_not_modified_by_verification(self, default_client):
        _enroll(default_client)
        store = default_client.app.dependency_overrides[get_profile_store]()
        before = store.get("alice")
        default_client.post(ENDPOINT, json=_raw_session("alice_s1"))
        after = store.get("alice")
        assert after is not None
        assert after.centroid == before.centroid
        assert after.embedding_dim == before.embedding_dim
        assert after.session_count == before.session_count
        assert after.created_at == before.created_at

    def test_no_new_profiles_created_by_verification(self, default_client):
        _enroll(default_client)
        store = default_client.app.dependency_overrides[get_profile_store]()
        assert store.count() == 1
        default_client.post(ENDPOINT, json=_raw_session("alice_s1"))
        default_client.post(ENDPOINT, json=_raw_session("alice_s2"))
        assert store.count() == 1


class TestThresholdIntegrity:
    def test_threshold_is_the_configured_threshold(self, default_client):
        _enroll(default_client)
        body = default_client.post(ENDPOINT, json=_raw_session("alice_s1")).json()
        assert body["threshold"] == 1.0

    def test_client_supplied_threshold_is_rejected(self, default_client):
        _enroll(default_client)
        window = _raw_session("alice_s1")
        window["threshold"] = 100.0
        assert default_client.post(ENDPOINT, json=window).status_code == 422


class TestExistingEndpointsUnchanged:
    def test_enrollment_and_verification_still_work(self, default_client):
        _enroll(default_client)
        verification = default_client.post(
            VERIFICATION_ENDPOINT,
            json={"user_ref": "alice", "session": _raw_session("alice_s1")},
        )
        assert verification.status_code == 200
        assert verification.json()["decision"] in ("VERIFIED", "SUSPICIOUS")
        assert set(verification.json()) == {"user_ref", "decision", "distance", "threshold"}