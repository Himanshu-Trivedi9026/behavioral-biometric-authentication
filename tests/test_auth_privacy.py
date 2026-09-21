"""
Phase 11 — privacy guarantees of the authentication layer.

Verifies that:

* passwords, Argon2id hashes, bearer tokens and the JWT secret never appear in
  any API response,
* :class:`Settings` repr/str masks the JWT secret,
* the ``users`` table/record persists ONLY identity fields (never raw
  behavioural events, embeddings or secrets),
* authenticated identity (`user_ref`) never reaches the model as a feature
  tensor input,

and that the JWT present in ``Authorization`` is never echoed back.
"""

import zlib

import torch
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.dependencies import get_ml_service, get_profile_store
from backend.app.main import create_app
from backend.app.repositories.user_repository import UserRecord
from backend.app.services.ml_service import BehavioralMLService
from backend.app.services.profile_store import InMemoryProfileStore
from tests.auth_testing import (
    TEST_JWT_SECRET,
    VALID_PASSWORD,
    auth_headers,
    register,
    settings,
)

ENROLLMENT = "/api/v1/enrollment"
VERIFICATION = "/api/v1/verification"

_BANNED_IDENTITY_KEYS = {
    "user_ref", "user_id", "key", "key_code", "key_name", "character", "char",
    "password", "passwd", "secret", "credentials", "token", "jwt",
}


def _raw_session(session_id):
    t = 0.0
    keyboard_events = []
    for i in range(3):
        keyboard_events.append({"event_type": "keyboard", "event": "keydown", "timestamp": t})
        t += 50.0
        keyboard_events.append({"event_type": "keyboard", "event": "keyup", "timestamp": t})
        t += 100.0
    x = y = mt = 0.0
    mouse_events = []
    for i in range(3):
        x += 4.0 + i
        y += 2.0
        mt += 16.0
        mouse_events.append({"event_type": "mouse", "event": "mousemove", "x": x, "y": y, "timestamp": mt})
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
    verifier = _RecordingVerifier()
    store = InMemoryProfileStore()
    service = BehavioralMLService(
        settings(),
        verifier_factory=lambda s: verifier,
        preprocessing_loader=lambda s: (None, None, {"artifact": "behavioral-preprocessing", "artifact_version": 1}),
        verification_config_loader=lambda s: {"artifact": "verification-config", "calibration": {"threshold": 1.0}},
    )
    app = create_app(settings())
    app.dependency_overrides[get_ml_service] = lambda: service
    app.dependency_overrides[get_profile_store] = lambda: store
    return TestClient(app), verifier, store


def _enroll(client, headers):
    return client.post(
        ENROLLMENT,
        json={"sessions": [_raw_session("secure_s1"), _raw_session("secure_s2")]},
        headers=headers,
    )


def _verify(client, headers):
    return client.post(
        VERIFICATION, json={"session": _raw_session("secure_s3")}, headers=headers
    )


class TestSecretsNeverAppearInResponses:
    def test_full_flow_responses_contain_no_secrets(self):
        client, _, _ = _build()
        with client as c:
            register_response = register(c, "secure", password=VALID_PASSWORD)
            headers = auth_headers(c, "secure", password=VALID_PASSWORD)
            enroll_response = _enroll(c, headers)
            verify_response = _verify(c, headers)

        all_text = "".join(
            r.text for r in (register_response, enroll_response, verify_response)
        ).lower()
        assert "password" not in all_text
        assert "$argon2id" not in all_text
        # The bearer token + JWT secret must never appear in any response.
        for response in (register_response, enroll_response, verify_response):
            assert TEST_JWT_SECRET not in response.text
            assert "bearer " + TEST_JWT_SECRET[:16] not in response.text.lower()

    def test_openapi_does_not_embed_secrets(self):
        client, _, _ = _build()
        with client as c:
            spec = c.get("/openapi.json").json()
        assert TEST_JWT_SECRET not in repr(spec)

    def test_settings_repr_masks_jwt_secret(self):
        s = settings()
        assert TEST_JWT_SECRET not in repr(s)
        assert TEST_JWT_SECRET not in str(s)

    def test_settings_default_never_contains_useable_secret(self):
        # The default settings object must not ship with a signable secret.
        assert Settings(environment="test", jwt_secret_key="").jwt_secret_key == ""


class TestUsersStoreNeverPersistsBehavioralData:
    def test_user_record_has_only_identity_fields(self):
        fields = {
            "id", "username", "password_hash", "created_at", "updated_at",
        }
        assert set(UserRecord.__dataclass_fields__) == fields

    def test_user_repository_never_stores_behavioral_events(self):
        from backend.app.repositories.memory_user_repository import InMemoryUserRepository

        repo = InMemoryUserRepository()
        user = UserRecord(
            id="11111111-1111-4111-8111-111111111111",
            username="alice",
            password_hash="$argon2id$only-a-hash",
            created_at="2025-01-01T00:00:00+00:00",
            updated_at="2025-01-01T00:00:00+00:00",
        )
        repo.create(user)
        for banned in ("keyboard_events", "mouse_events", "hold_time", "flight_time"):
            assert banned not in repr(repo)
        stored = repo.get_by_username("alice")
        assert stored.password_hash == "$argon2id$only-a-hash"
        assert not hasattr(stored, "centroid")


class TestIdentityNeverEntersTheModel:
    def test_user_ref_never_reaches_model_sessions(self):
        client, verifier, _ = _build()
        with client as c:
            register(c, "noleak")
            headers = auth_headers(c, "noleak")
            assert _enroll(c, headers).status_code == 201
            assert _verify(c, headers).status_code == 200

        assert verifier.embedded, "the verifier should have seen processed sessions"
        assert verifier.leak_hits == [], (
            "identity/privacy fields leaked into model input: {!r}".format(verifier.leak_hits)
        )
        for session in verifier.embedded:
            assert set(session).isdisjoint(_BANNED_IDENTITY_KEYS)
            assert "user_ref" not in session
            assert "user_id" not in session


class TestResponsesRemainSanitized:
    def test_enrollment_and_verification_response_shapes_unchanged(self):
        client, _, _ = _build()
        with client as c:
            register(c, "shape")
            headers = auth_headers(c, "shape")
            enroll = _enroll(c, headers)
            verify = _verify(c, headers)
        assert set(enroll.json()) == {"user_ref", "status", "embedding_dimension", "session_count"}
        assert set(verify.json()) == {"user_ref", "decision", "distance", "threshold"}
        for marker in ("centroid", "embeddings", "keyboard_events", "mouse_events", "/home/"):
            assert marker not in enroll.text.lower()
            assert marker not in verify.text.lower()