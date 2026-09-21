"""
Phase 9B (Phase 11 updated) — security/privacy guarantees around the ML API.

Verifies that:
* ``user_ref`` is metadata only and never reaches the model as a feature.
* raw events/key identity/characters/passwords are never persisted or echoed.
* centroids and per-session embeddings are never returned by the API.
* responses contain no secrets, paths or internal markers.
* the API REQUIRES a bearer token for the ML endpoints (Phase 11); the token
  is never echoed and the OpenAPI spec declares bearer auth.

The Phase 9B tests use a fixed authenticated identity (``alice``) standing in
for real token verification; the token/authorization mechanics themselves are
covered exhaustively in ``test_auth_authorization.py`` and ``test_auth_jwt.py``.
"""

import zlib

import torch
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.dependencies import get_ml_service, get_profile_store
from backend.app.main import create_app
from backend.app.services.ml_service import BehavioralMLService
from backend.app.services.profile_store import InMemoryProfileStore, stored_fields
from tests.auth_testing import TEST_JWT_SECRET, set_current_user, settings

ENROLLMENT = "/api/v1/enrollment"
VERIFICATION = "/api/v1/verification"
_BANNED_IDENTITY_KEYS = {
    "user_ref", "user_id", "key", "key_code", "key_name", "character", "char",
    "password", "passwd", "secret", "credentials", "token", "jwt",
}
_BANNED_RESPONSE_MARKERS = (
    "centroid", "embedding_vector", "embeddings", "keyboard_sequence",
    "mouse_sequence", "keyboard_events", "mouse_events", "hold_time",
    "flight_time",
)


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


class _RecordingVerifier:
    def __init__(self, dim=8):
        self.dim = dim
        self.embedded = []
        self.leak_hits = []

    @staticmethod
    def _vec(key):
        generator = torch.Generator()
        generator.manual_seed(zlib.crc32(key.encode("utf-8")))
        return torch.randn((8,), generator=generator)

    def _record(self, session):
        self.embedded.append(dict(session))
        hits = sorted(_BANNED_IDENTITY_KEYS & set(session))
        if hits:
            self.leak_hits.append(hits)

    def embed_session(self, session, keyboard_scaler=None, mouse_scaler=None):
        self._record(session)
        return self._vec(str(session.get("session_id", "unknown")))

    def embed_sessions(self, sessions, keyboard_scaler=None, mouse_scaler=None, batch_size=16):
        for session in sessions:
            self._record(session)
        return torch.stack([self.embed_session(s) for s in sessions])

    def pair_distance(self, a, b):
        return torch.sqrt(((a - b) ** 2).sum(dim=-1) + 1e-8)


def _build():
    verb = _RecordingVerifier()
    store = InMemoryProfileStore()
    service = BehavioralMLService(
        Settings(environment="test", debug=False),
        verifier_factory=lambda s: verb,
        preprocessing_loader=lambda s: (None, None, {"artifact": "behavioral-preprocessing", "artifact_version": 1}),
        verification_config_loader=lambda s: {"artifact": "verification-config", "calibration": {"threshold": 1.0}},
    )
    app = create_app(settings())
    app.dependency_overrides[get_ml_service] = lambda: service
    app.dependency_overrides[get_profile_store] = lambda: store
    set_current_user(app, username="alice")
    client = TestClient(app)
    return client, verb, store


class TestModelInputsNeverCarryIdentity:
    def test_user_ref_never_reaches_the_model(self):
        client, verb, _ = _build()
        with client:
            client.post(
                ENROLLMENT,
                json={
                    "user_ref": "alice",
                    "sessions": [
                        _raw_session("alice_s1"),
                        _raw_session("alice_s2"),
                    ],
                },
            )
            client.post(
                VERIFICATION,
                json={"user_ref": "alice", "session": _raw_session("alice_s3")},
            )
        assert verb.embedded, "the verifier should have seen processed sessions"
        assert verb.leak_hits == [], (
            "identity/privacy fields leaked into model input: {!r}".format(verb.leak_hits)
        )
        for session in verb.embedded:
            assert set(session).isdisjoint(_BANNED_IDENTITY_KEYS)
            assert "user_id" not in session
            assert "user_ref" not in session


class TestResponsesAreSanitized:
    def test_enrollment_response_has_only_documented_fields(self):
        client, _, _ = _build()
        with client:
            response = client.post(
                ENROLLMENT,
                json={"user_ref": "alice", "sessions": [_raw_session("alice_s1")]},
            )
        body = response.json()
        assert set(body) == {"user_ref", "status", "embedding_dimension", "session_count"}
        assert response.status_code == 201

    def test_verification_response_has_only_documented_fields(self):
        client, _, _ = _build()
        with client:
            client.post(
                ENROLLMENT,
                json={"user_ref": "alice", "sessions": [_raw_session("alice_s1")]},
            )
            response = client.post(
                VERIFICATION,
                json={"user_ref": "alice", "session": _raw_session("alice_s1")},
            )
        assert set(response.json()) == {"user_ref", "decision", "distance", "threshold"}

    def test_no_raw_or_internal_markers_in_any_response(self):
        client, _, _ = _build()
        with client:
            responses = [
                client.post(ENROLLMENT, json={"user_ref": "alice", "sessions": [_raw_session("alice_s1")]}),
                client.post(VERIFICATION, json={"user_ref": "alice", "session": _raw_session("alice_s1")}),
                client.get("/health"),
                client.post(ENROLLMENT, json={"user_ref": "alice", "sessions": [_raw_session("alice_s1")]}),
            ]
        for response in responses:
            text = response.text.lower()
            for marker in _BANNED_RESPONSE_MARKERS + ("models/", ".pt", "/home/", "feature_means"):
                assert marker not in text, marker


class TestStoreNeverPersistsRawData:
    def test_profile_store_keeps_only_aggregates(self):
        client, _, store = _build()
        with client:
            client.post(
                ENROLLMENT,
                json={"user_ref": "alice", "sessions": [_raw_session("alice_s1")]},
            )
        assert set(stored_fields()) == {
            "user_ref", "centroid", "embedding_dim", "session_count", "created_at", "updated_at",
        }
        stored = store.get("alice")
        assert isinstance(stored.centroid, tuple)
        assert all(isinstance(v, float) for v in stored.centroid)
        rows = store.list_profiles()
        assert "centroid" not in rows[0]

    def test_no_raw_events_survive_in_store(self):
        client, _, store = _build()
        with client:
            client.post(
                ENROLLMENT,
                json={"user_ref": "alice", "sessions": [_raw_session("alice_s1")]},
            )
            payload = repr(store)
        for banned in ("keyboard_events", "mouse_events", "hold_time", "alice_s1"):
            assert banned not in payload


class TestNoSecretsAndAuthDependency:
    def test_no_secret_markers_in_responses(self, monkeypatch):
        monkeypatch.setenv("BBA_DO_NOT_LEAK", "super-secret-value")
        client, _, _ = _build()
        with client:
            response = client.post(
                ENROLLMENT,
                json={"user_ref": "alice", "sessions": [_raw_session("alice_s1")]},
            )
        assert response.status_code == 201
        assert "super-secret-value" not in response.text

    def test_api_requires_authentication(self):
        # Phase 11: the ML endpoints are protected. A request without a bearer
        # token is rejected with 401 before any ML work happens.
        app = create_app(settings())
        with TestClient(app) as client:
            response = client.post(
                ENROLLMENT,
                json={"user_ref": "alice", "sessions": [_raw_session("alice_s1")]},
            )
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "authentication_required"

    def test_openapi_declares_bearer_auth(self):
        client, _, _ = _build()
        with client:
            spec = client.get("/openapi.json").json()
        schemes = spec.get("components", {}).get("securitySchemes", {})
        assert "HTTPBearer" in schemes
        assert schemes["HTTPBearer"].get("scheme") == "bearer"
        # Protected ML endpoints require the bearer scheme...
        for endpoint in (ENROLLMENT, VERIFICATION):
            operation = spec["paths"][endpoint]["post"]
            assert operation.get("security") == [{"HTTPBearer": []}]
        # ...while the public register/login endpoints stay open.
        for operation in spec["paths"]["/api/v1/auth/register"].values():
            assert not operation.get("security")
        for operation in spec["paths"]["/api/v1/auth/login"].values():
            assert not operation.get("security")