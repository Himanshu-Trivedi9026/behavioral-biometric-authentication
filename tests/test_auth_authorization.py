"""
Phase 11 — authorization tests for the protected ML endpoints.

Verifies the central security guarantee: the authenticated identity is the
ONLY source of authorization for enrollment/verification.

* missing / malformed / invalid / expired bearer tokens -> 401,
* an authenticated user can enroll and verify their OWN profile (with or
  without a body ``user_ref``),
* a request-body ``user_ref`` can never select another user's profile
  (mismatch -> 403 ``forbidden``),
* a second user cannot operate on the first user's profile in any direction,
* register -> login -> authenticated enrollment/verification works end to end
  with real JWTs.

The ML service is a deterministic fake so no checkpoint is required; identity
never reaches the model (that is asserted in test_auth_privacy.py).
"""

from __future__ import annotations

import datetime

import jwt as pyjwt
import pytest
import torch
import zlib
from fastapi.testclient import TestClient

from backend.app.dependencies import get_ml_service, get_profile_store
from backend.app.main import create_app
from backend.app.services.ml_service import BehavioralMLService
from backend.app.services.profile_store import InMemoryProfileStore
from tests.auth_testing import (
    VALID_PASSWORD_2,
    access_token,
    auth_headers,
    login,
    register,
    settings,
)

ENROLLMENT = "/api/v1/enrollment"
VERIFICATION = "/api/v1/verification"


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
def client():
    store = InMemoryProfileStore()
    verifier = _FakeVerifier()
    service = BehavioralMLService(
        settings(),
        verifier_factory=lambda s: verifier,
        preprocessing_loader=lambda s: (None, None, {"artifact": "behavioral-preprocessing", "artifact_version": 1}),
        verification_config_loader=lambda s: {"artifact": "verification-config", "calibration": {"threshold": 1.0}},
    )
    app = create_app(settings())
    app.dependency_overrides[get_ml_service] = lambda: service
    app.dependency_overrides[get_profile_store] = lambda: store
    with TestClient(app) as test_client:
        yield test_client, store


def _enroll_payload(user_ref, session_id="s1"):
    payload = {"sessions": [_raw_session(session_id)]}
    if user_ref is not None:
        payload["user_ref"] = user_ref
    return payload


def _verify_payload(user_ref, session_id="s1"):
    payload = {"session": _raw_session(session_id)}
    if user_ref is not None:
        payload["user_ref"] = user_ref
    return payload


class TestAuthenticationRequired:
    @pytest.mark.parametrize("endpoint", [ENROLLMENT, VERIFICATION])
    def test_missing_authorization_401(self, client, endpoint):
        c, _ = client
        payload = _enroll_payload("alice") if endpoint == ENROLLMENT else _verify_payload("alice")
        response = c.post(endpoint, json=payload)
        assert response.status_code == 401
        assert response.json() == {
            "error": {"code": "authentication_required", "message": "Authentication is required."}
        }

    @pytest.mark.parametrize("endpoint", [ENROLLMENT, VERIFICATION])
    @pytest.mark.parametrize(
        "header",
        [
            {"Authorization": "Token abc123"},
            {"Authorization": "Bearer"},
            {"Authorization": ""},
            {"Authorization": "Basic dXNlcjpwYXNz"},
        ],
    )
    def test_malformed_authorization_401(self, client, endpoint, header):
        c, _ = client
        payload = _enroll_payload("alice") if endpoint == ENROLLMENT else _verify_payload("alice")
        response = c.post(endpoint, json=payload, headers=header)
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "authentication_required"

    @pytest.mark.parametrize("endpoint", [ENROLLMENT, VERIFICATION])
    def test_invalid_bearer_token_401(self, client, endpoint):
        c, _ = client
        payload = _enroll_payload("alice") if endpoint == ENROLLMENT else _verify_payload("alice")
        response = c.post(endpoint, json=payload, headers={"Authorization": "Bearer not-a-jwt"})
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "invalid_credentials"

    @pytest.mark.parametrize("endpoint", [ENROLLMENT, VERIFICATION])
    def test_expired_token_401(self, client, endpoint):
        c, _ = client
        register(c, "timed")
        expired = pyjwt.encode(
            {
                "sub": "whoever",
                "exp": datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=5),
            },
            settings().jwt_secret_key,
            algorithm="HS256",
        )
        payload = _enroll_payload("timed") if endpoint == ENROLLMENT else _verify_payload("timed")
        response = c.post(endpoint, json=payload, headers={"Authorization": "Bearer " + expired})
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "invalid_credentials"

    def test_valid_token_for_unknown_user_401(self, client):
        c, _ = client
        token = pyjwt.encode(
            {
                "sub": "00000000-0000-4000-8000-000000000000",
                "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=5),
            },
            settings().jwt_secret_key,
            algorithm="HS256",
        )
        response = c.post(ENROLLMENT, json=_enroll_payload(None), headers={"Authorization": "Bearer " + token})
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "invalid_credentials"


class TestOwnProfileAccess:
    def test_register_login_enrollment_verification(self, client):
        c, store = client
        assert register(c, "alice").status_code == 201
        headers = auth_headers(c, "alice")

        enroll = c.post(ENROLLMENT, json=_enroll_payload("alice"), headers=headers)
        assert enroll.status_code == 201
        assert enroll.json()["user_ref"] == "alice"
        assert store.contains("alice")

        verify = c.post(VERIFICATION, json=_verify_payload("alice"), headers=headers)
        assert verify.status_code == 200
        assert verify.json()["user_ref"] == "alice"

    def test_user_ref_omitted_uses_authenticated_identity(self, client):
        c, store = client
        register(c, "bob")
        headers = auth_headers(c, "bob")

        enroll = c.post(ENROLLMENT, json=_enroll_payload(None), headers=headers)
        assert enroll.status_code == 201
        assert enroll.json()["user_ref"] == "bob"
        assert store.contains("bob")

        verify = c.post(VERIFICATION, json=_verify_payload(None), headers=headers)
        assert verify.status_code == 200
        assert verify.json()["user_ref"] == "bob"

    def test_user_ref_matching_identity_is_allowed(self, client):
        c, _ = client
        register(c, "carol")
        headers = auth_headers(c, "carol")
        assert c.post(ENROLLMENT, json=_enroll_payload("carol"), headers=headers).status_code == 201
        assert c.post(VERIFICATION, json=_verify_payload("carol"), headers=headers).status_code == 200


class TestCannotOperateOnAnotherUsersProfile:
    def test_cannot_enroll_another_users_ref(self, client):
        c, store = client
        register(c, "alice")
        register(c, "bob")
        bob = auth_headers(c, "bob")

        response = c.post(ENROLLMENT, json=_enroll_payload("alice"), headers=bob)
        assert response.status_code == 403
        assert response.json() == {
            "error": {
                "code": "forbidden",
                "message": "You are not authorized to access this profile.",
            }
        }
        assert not store.contains("alice")
        assert not store.contains("bob")

    def test_cannot_verify_another_users_ref(self, client):
        c, _ = client
        register(c, "alice")
        register(c, "bob")
        alice = auth_headers(c, "alice")
        assert c.post(ENROLLMENT, json=_enroll_payload("alice"), headers=alice).status_code == 201

        bob = auth_headers(c, "bob")
        response = c.post(VERIFICATION, json=_verify_payload("alice"), headers=bob)
        assert response.status_code == 403
        assert response.json()["error"]["code"] == "forbidden"

    def test_second_user_cannot_operate_on_first_users_profile(self, client):
        # bob is registered first; alice must still not be able to touch bob's profile.
        c, store = client
        register(c, "bob")
        register(c, "alice")
        headers = auth_headers(c, "bob")
        assert c.post(ENROLLMENT, json=_enroll_payload(None), headers=headers).status_code == 201
        assert store.contains("bob")

        alice = auth_headers(c, "alice")
        assert c.post(ENROLLMENT, json=_enroll_payload("bob"), headers=alice).status_code == 403
        assert c.post(VERIFICATION, json=_verify_payload("bob"), headers=alice).status_code == 403

    def test_verified_user_cannot_probe_anothers_profile(self, client):
        c, _ = client
        register(c, "alice")
        register(c, "bob")
        bob = auth_headers(c, "bob")
        assert c.post(ENROLLMENT, json=_enroll_payload(None), headers=bob).status_code == 201

        # alice (not bob) cannot read bob's profile by name.
        alice = auth_headers(c, "alice")
        response = c.post(VERIFICATION, json=_verify_payload("bob"), headers=alice)
        assert response.status_code == 403
        assert "VERIFIED" not in response.text and "SUSPICIOUS" not in response.text


class TestRequestBodyCannotOverrideIdentity:
    def test_enrollment_override_rejected(self, client):
        c, _ = client
        register(c, "alice")
        register(c, "mallory")
        mallory = auth_headers(c, "mallory")
        response = c.post(ENROLLMENT, json=_enroll_payload("alice"), headers=mallory)
        assert response.status_code == 403

    def test_verification_override_rejected(self, client):
        c, _ = client
        register(c, "alice")
        register(c, "mallory")
        alice = auth_headers(c, "alice")
        assert c.post(ENROLLMENT, json=_enroll_payload("alice"), headers=alice).status_code == 201
        mallory = auth_headers(c, "mallory")
        response = c.post(VERIFICATION, json=_verify_payload("alice"), headers=mallory)
        assert response.status_code == 403

    def test_own_user_ref_still_accepted(self, client):
        c, _ = client
        register(c, "alice")
        headers = auth_headers(c, "alice")
        assert c.post(ENROLLMENT, json=_enroll_payload("alice"), headers=headers).status_code == 201


class TestUnauthenticatedRequests:
    def test_unauthenticated_enrollment_401(self, client):
        c, _ = client
        assert c.post(ENROLLMENT, json=_enroll_payload("ghost")).status_code == 401

    def test_unauthenticated_verification_401(self, client):
        c, _ = client
        assert c.post(VERIFICATION, json=_verify_payload("ghost")).status_code == 401

    def test_empty_bearer_value_401(self, client):
        c, _ = client
        response = c.post(ENROLLMENT, json=_enroll_payload("ghost"), headers={"Authorization": "Bearer "})
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "authentication_required"


class TestLoginFlowUsedByProtectedEndpoints:
    def test_authenticated_flow_uses_real_jwt(self, client):
        c, _ = client
        register(c, "realflow")
        token = access_token(c, "realflow")
        headers = {"Authorization": "Bearer " + token}
        assert c.post(ENROLLMENT, json=_enroll_payload("realflow"), headers=headers).status_code == 201
        assert c.post(VERIFICATION, json=_verify_payload("realflow"), headers=headers).status_code == 200

    def test_wrong_password_never_grants_access(self, client):
        c, _ = client
        register(c, "locked")
        login(c, "locked", password=VALID_PASSWORD_2)
        response = c.post(ENROLLMENT, json=_enroll_payload("locked"))
        assert response.status_code == 401
        assert response.json()["error"]["code"] == "authentication_required"