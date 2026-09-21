"""
Phase 9B (Phase 11 protected) — POST /api/v1/enrollment API tests.

Uses dependency overrides (fake verifier service + fresh in-memory store) so no
checkpoint/artifacts are required and each test gets isolated state. Every
request runs as an authenticated user; the authenticated identity is either the
matching ``user_ref`` or the default ``alice``.
"""

import json
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

ENDPOINT = "/api/v1/enrollment"


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


@pytest.fixture(scope="function")
def store():
    return InMemoryProfileStore()


@pytest.fixture(scope="function")
def client(store):
    verifier = _FakeVerifier()
    service = BehavioralMLService(
        Settings(environment="test", debug=False),
        verifier_factory=lambda s: verifier,
        preprocessing_loader=lambda s: (None, None, {"artifact": "behavioral-preprocessing", "artifact_version": 1}),
        verification_config_loader=lambda s: {"artifact": "verification-config", "calibration": {"threshold": 1.0}},
    )
    app = create_app(Settings(environment="test", debug=False))
    app.dependency_overrides[get_ml_service] = lambda: service
    app.dependency_overrides[get_profile_store] = lambda: store
    set_current_user(app, username="alice")
    with TestClient(app) as test_client:
        yield test_client


def _payload(user_ref="alice", sessions=None):
    return {"user_ref": user_ref, "sessions": sessions or [_raw_session("alice_s1")]}


class TestSuccessfulEnrollment:
    def test_single_session_returns_201(self, client):
        response = client.post(ENDPOINT, json=_payload())
        assert response.status_code == 201
        assert response.json() == {
            "user_ref": "alice",
            "status": "enrolled",
            "embedding_dimension": 8,
            "session_count": 1,
        }

    def test_multi_session_counts_sessions(self, client):
        set_current_user(client.app, username="bob")
        payload = {
            "user_ref": "bob",
            "sessions": [_raw_session("bob_s1"), _raw_session("bob_s2")],
        }
        response = client.post(ENDPOINT, json=payload)
        assert response.status_code == 201
        body = response.json()
        assert body["session_count"] == 2
        assert body["embedding_dimension"] == 8
        assert body["status"] == "enrolled"

    def test_extra_unknown_fields_are_ignored(self, client):
        payload = _payload()
        payload["extra_marker"] = "ignored"
        response = client.post(ENDPOINT, json=payload)
        assert response.status_code == 201

    def test_profile_persisted_in_store(self, client, store):
        set_current_user(client.app, username="carol")
        client.post(ENDPOINT, json=_payload("carol"))
        assert store.contains("carol")
        stored = store.get("carol")
        assert stored.session_count == 1
        assert stored.embedding_dim == 8
        assert isinstance(stored.centroid, tuple)
        assert all(isinstance(v, float) for v in stored.centroid)


class TestDuplicateEnrollment:
    def test_duplicate_returns_409(self, client):
        assert client.post(ENDPOINT, json=_payload()).status_code == 201
        response = client.post(ENDPOINT, json=_payload())
        assert response.status_code == 409
        assert response.json() == {
            "error": {
                "code": "profile_exists",
                "message": "a profile already exists for user 'alice'",
            }
        }

    def test_duplicate_does_not_overwrite(self, client, store):
        set_current_user(client.app, username="dave")
        client.post(ENDPOINT, json=_payload("dave"))
        client.post(
            ENDPOINT,
            json={
                "user_ref": "dave",
                "sessions": [_raw_session("dave_s1"), _raw_session("dave_s2")],
            },
        )
        assert store.get("dave").session_count == 1


class TestValidationFailures:
    def test_empty_sessions_422(self, client):
        response = client.post(ENDPOINT, json={"user_ref": "alice", "sessions": []})
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    def test_blank_user_ref_422(self, client):
        response = client.post(ENDPOINT, json=_payload(user_ref="   "))
        assert response.status_code == 422

    def test_missing_session_field_422(self, client):
        payload = _payload()
        del payload["sessions"][0]["started_at"]
        response = client.post(ENDPOINT, json=payload)
        assert response.status_code == 422

    def test_unknown_event_type_422(self, client):
        payload = _payload()
        payload["sessions"][0]["keyboard_events"][0]["event"] = "keypress?"
        response = client.post(ENDPOINT, json=payload)
        assert response.status_code == 422

    def test_non_finite_timestamp_422(self, client):
        payload = _payload()
        payload["sessions"][0]["keyboard_events"][0]["timestamp"] = float("inf")
        body = json.dumps(payload)
        response = client.post(
            ENDPOINT, content=body, headers={"content-type": "application/json"}
        )
        assert response.status_code == 422

    def test_too_many_sessions_422(self, client):
        sessions = [_raw_session("many_{}".format(i)) for i in range(17)]
        response = client.post(ENDPOINT, json={"user_ref": "alice", "sessions": sessions})
        assert response.status_code == 422