"""
Phase 9B (Phase 11 protected) — POST /api/v1/verification API tests.

Uses dependency overrides (fake verifier service + fresh in-memory store);
the calibrated threshold is controlled via the injected config loader so the
"threshold is never client-supplied" rule can be asserted precisely. Every
request runs as an authenticated user matching its ``user_ref``.
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
from tests.auth_testing import set_current_user

ENDPOINT = "/api/v1/verification"


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


def _make_client(threshold):
    store = InMemoryProfileStore()
    verifier = _FakeVerifier()
    service = BehavioralMLService(
        Settings(environment="test", debug=False),
        verifier_factory=lambda s: verifier,
        preprocessing_loader=lambda s: (None, None, {"artifact": "behavioral-preprocessing", "artifact_version": 1}),
        verification_config_loader=lambda s: {"artifact": "verification-config", "calibration": {"threshold": threshold}},
    )
    app = create_app(Settings(environment="test", debug=False))
    app.dependency_overrides[get_ml_service] = lambda: service
    app.dependency_overrides[get_profile_store] = lambda: store
    set_current_user(app, username="alice")
    return TestClient(app), store


@pytest.fixture(scope="function")
def default_client():
    client, store = _make_client(threshold=1.0)
    with client as c:
        yield c


def _enroll(client, user_ref="alice", session=None):
    response = client.post(
        "/api/v1/enrollment",
        json={"user_ref": user_ref, "sessions": [session or _raw_session("alice_s1")]},
    )
    assert response.status_code == 201


class TestVerificationDecisions:
    def test_same_session_is_verified(self, default_client):
        _enroll(default_client)
        response = default_client.post(
            ENDPOINT, json={"user_ref": "alice", "session": _raw_session("alice_s1")}
        )
        assert response.status_code == 200
        body = response.json()
        assert body["user_ref"] == "alice"
        assert body["decision"] == "VERIFIED"
        assert body["distance"] < 0.5
        assert body["threshold"] == 1.0

    def test_threshold_zero_yields_suspicious(self):
        client, _ = _make_client(threshold=0.0)
        with client as c:
            _enroll(c)
            response = c.post(
                ENDPOINT, json={"user_ref": "alice", "session": _raw_session("alice_s2")}
            )
            body = response.json()
            assert response.status_code == 200
            assert body["decision"] == "SUSPICIOUS"
            assert body["threshold"] == 0.0
            assert body["distance"] > 0.0

    def test_decision_consistent_with_threshold(self, default_client):
        _enroll(default_client)
        for index, session_id in enumerate(("alice_s1", "alice_s2", "alice_s3")):
            response = default_client.post(
                ENDPOINT, json={"user_ref": "alice", "session": _raw_session(session_id)}
            )
            body = response.json()
            expected = "VERIFIED" if body["distance"] <= body["threshold"] else "SUSPICIOUS"
            assert body["decision"] == expected


class TestIdentityLookup:
    def test_unknown_user_404(self, default_client):
        # Verified identity "ghost" has no profile -> 404 (not a 403).
        set_current_user(default_client.app, username="ghost")
        response = default_client.post(
            ENDPOINT, json={"user_ref": "ghost", "session": _raw_session("ghost_s1")}
        )
        assert response.status_code == 404
        assert response.json() == {
            "error": {
                "code": "profile_not_found",
                "message": "no enrollment profile found for user 'ghost'",
            }
        }


class TestThresholdIntegrity:
    def test_client_supplied_threshold_is_ignored(self, default_client):
        _enroll(default_client)
        payload = {"user_ref": "alice", "session": _raw_session("alice_s2"), "threshold": 100.0}
        response = default_client.post(ENDPOINT, json=payload)
        body = response.json()
        assert body["threshold"] == 1.0
        expected = "VERIFIED" if body["distance"] <= 1.0 else "SUSPICIOUS"
        assert body["decision"] == expected

    def test_verification_response_shape(self, default_client):
        _enroll(default_client)
        response = default_client.post(
            ENDPOINT, json={"user_ref": "alice", "session": _raw_session("alice_s1")}
        )
        assert set(response.json()) == {"user_ref", "decision", "distance", "threshold"}
        assert response.json()["distance"] >= 0


class TestValidationAndRouting:
    def test_malformed_session_422(self, default_client):
        _enroll(default_client)
        session = _raw_session("alice_s1")
        del session["mouse_events"]
        response = default_client.post(ENDPOINT, json={"user_ref": "alice", "session": session})
        assert response.status_code == 422

    def test_blank_user_ref_422(self, default_client):
        response = default_client.post(
            ENDPOINT, json={"user_ref": " ", "session": _raw_session("x_s1")}
        )
        assert response.status_code == 422

    def test_get_method_is_405(self, default_client):
        assert default_client.get(ENDPOINT).status_code == 405
        assert default_client.get("/api/v1/enrollment").status_code == 405