"""
Phase 14B — session-scoped, fail-closed re-verification gating (API tests).

Exercises the Phase 14B rules through the real FastAPI application (with the
Phase 14A fake-verifier + in-memory store dependency overrides):

* a fresh session starts clean (``verified``) and is NEVER gated;
* after a SUSPICIOUS continuous window the session's server state becomes
  ``reverification_required`` and POST /enrollment + POST /verification fail
  closed with 403 ``reverification_required``;
* the continuous-verification endpoint stays AVAILABLE while gated (the
  recovery path), and a VERIFIED continuous window clears the gate for that
  session only;
* the gate is per-session (keyed by the JWT-derived session id), never an
  account-wide lockout;
* ``GET /continuous-verification/state`` reports the server-authoritative state
  (and is still authenticated).

The fake verifier embeds a window purely by ``session_id``, so within ONE app
(a single threshold, a single in-memory store) the window whose id matches the
enrolled session is VERIFIED (distance ~0) while any other id is SUSPICIOUS
(distance of two seeded random vectors ~4). That lets a single test drive both
gate-raising and gate-clearing windows without any configuration toggling.
"""

from __future__ import annotations

import json
import zlib

import pytest
import torch
from fastapi.testclient import TestClient

from backend.app.auth.dependencies import CurrentUser, _session_id_from_token, get_current_user
from backend.app.config import Settings
from backend.app.dependencies import get_ml_service, get_profile_store
from backend.app.main import create_app
from backend.app.services.ml_service import BehavioralMLService
from backend.app.services.profile_store import InMemoryProfileStore
from tests.auth_testing import REGISTER, TEST_JWT_SECRET, VALID_PASSWORD, access_token, set_current_user

CONTINUOUS_ENDPOINT = "/api/v1/continuous-verification"
STATE_ENDPOINT = "/api/v1/continuous-verification/state"
ENROLLMENT_ENDPOINT = "/api/v1/enrollment"
VERIFICATION_ENDPOINT = "/api/v1/verification"

# A window id that never matches an enrolled profile -> SUSPICIOUS at
# threshold 1.0 (distance of two independent seeded vectors is ~4 >> 1.0).
FAR_SESSION = "far_from_enrolled_session"


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
        self.checkpoint_id = "phase14b-fake-verifier"

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


def _make_client(threshold=1.0, store=None):
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
    return TestClient(app), store


def _set_session(client, username="alice", session_id="login-1"):
    """Force the authenticated identity + explicit session id."""
    client.app.dependency_overrides[get_current_user] = lambda: CurrentUser(
        id="11111111-1111-4111-8111-111111111111",
        username=username,
        session_id=session_id,
    )


def _auth(token):
    return {"Authorization": "Bearer " + token}


def _make_real_auth_client(threshold=1.0, store=None, verifier=None):
    """App with the REAL (untouched) ``get_current_user`` + fake ML verifier.

    Identity, ``session_id`` and the bearer-token mechanics all come from the
    production dependency graph (no ``set_current_user`` override), which is
    what makes replay/fresh-login fingerprint tests meaningful.
    """
    store = store if store is not None else InMemoryProfileStore()
    if verifier is None:
        verifier = _FakeVerifier()
    service = BehavioralMLService(
        Settings(environment="test", debug=False),
        verifier_factory=lambda s: verifier,
        preprocessing_loader=lambda s: (None, None, {"artifact": "behavioral-preprocessing", "artifact_version": 1}),
        verification_config_loader=lambda s: {"artifact": "verification-config", "calibration": {"threshold": threshold}},
    )
    app = create_app(Settings(environment="test", debug=False, jwt_secret_key=TEST_JWT_SECRET))
    app.dependency_overrides[get_ml_service] = lambda: service
    app.dependency_overrides[get_profile_store] = lambda: store
    return TestClient(app), store


def _enroll_token(client, token, user_ref="alice", session=None):
    return client.post(
        ENROLLMENT_ENDPOINT,
        headers=_auth(token),
        json={"user_ref": user_ref, "sessions": [session or _raw_session(user_ref + "_s1")]},
    )


def _verify_token(client, token, user_ref="alice", session=None):
    return client.post(
        VERIFICATION_ENDPOINT,
        headers=_auth(token),
        json={"user_ref": user_ref, "session": session or _raw_session(user_ref + "_s1")},
    )


class _RecordingVerifier(_FakeVerifier):
    """Fake verifier that records every window handed to the ML service."""

    def __init__(self):
        super().__init__(dim=8)
        self.seen = []

    def embed_session(self, session, keyboard_scaler=None, mouse_scaler=None):
        self.seen.append(session)
        return super().embed_session(session, keyboard_scaler, mouse_scaler)

    def embed_sessions(self, sessions, keyboard_scaler=None, mouse_scaler=None, batch_size=16):
        self.seen.extend(sessions)
        return super().embed_sessions(sessions, keyboard_scaler, mouse_scaler, batch_size)


def _enroll(client, user_ref="alice", session=None):
    return client.post(
        ENROLLMENT_ENDPOINT,
        json={"user_ref": user_ref, "sessions": [session or _raw_session(user_ref + "_s1")]},
    )


def _continuous(client, window, user_ref="alice"):
    return client.post(CONTINUOUS_ENDPOINT, json=window)


def _verify(client, user_ref="alice", session=None):
    return client.post(
        VERIFICATION_ENDPOINT,
        json={"user_ref": user_ref, "session": session or _raw_session(user_ref + "_s1")},
    )


@pytest.fixture(scope="function")
def clean_session():
    """One app + one explicit session (threshold 1.0), fully unblocked."""
    client, _ = _make_client()
    with client as c:
        _set_session(c, session_id="login-1")
        yield c


class TestFreshSessionStartsClean:
    def test_state_defaults_to_verified_with_no_tracking(self, clean_session):
        assert clean_session.get(STATE_ENDPOINT).json() == {
            "session_state": "verified",
            "consecutive_suspicious": 0,
            "last_verified_at": None,
        }

    def test_fresh_session_can_enroll(self, clean_session):
        assert _enroll(clean_session).status_code == 201

    def test_fresh_session_can_verify(self, clean_session):
        assert _enroll(clean_session).status_code == 201
        assert _verify(clean_session).status_code == 200


class TestFailClosedGating:
    @pytest.fixture
    def blocked_session(self):
        client, _ = _make_client()
        with client as c:
            _set_session(c, session_id="login-1")
            _enroll(c)
            response = _continuous(c, _raw_session(FAR_SESSION))
            assert response.status_code == 200
            assert response.json()["decision"] == "SUSPICIOUS"
            yield c

    def test_enrollment_blocked_with_structured_403(self, blocked_session):
        response = _enroll(blocked_session)
        assert response.status_code == 403
        assert response.json() == {
            "error": {
                "code": "reverification_required",
                "message": "Behavioral re-verification is required before this action.",
            }
        }

    def test_verification_blocked_with_structured_403(self, blocked_session):
        response = _verify(blocked_session)
        assert response.status_code == 403
        body = response.json()["error"]
        assert body["code"] == "reverification_required"

    def test_state_reports_blocked_with_counter(self, blocked_session):
        body = blocked_session.get(STATE_ENDPOINT).json()
        assert body["session_state"] == "reverification_required"
        assert body["consecutive_suspicious"] == 1

    def test_continuous_verification_stays_available_while_gated(self, blocked_session):
        response = _continuous(blocked_session, _raw_session(FAR_SESSION + "2"))
        assert response.status_code == 200
        assert response.json()["decision"] == "SUSPICIOUS"


class TestRecoveryViaVerifiedWindow:
    def test_verified_window_clears_the_gate(self):
        client, _ = _make_client()
        with client as c:
            _set_session(c, session_id="login-1")
            _enroll(c)
            assert _continuous(c, _raw_session(FAR_SESSION)).json()["decision"] == "SUSPICIOUS"
            assert _verify(c).status_code == 403

            # a matching window re-verifies the session and clears the gate
            recovery = _continuous(c, _raw_session("alice_s1"))
            assert recovery.status_code == 200
            assert recovery.json()["decision"] == "VERIFIED"
            assert recovery.json()["session_state"] == "verified"

            assert _verify(c).status_code == 200
            assert c.get(STATE_ENDPOINT).json()["session_state"] == "verified"


class TestGateIsPerSession:
    def test_fresh_login_for_same_user_is_unaffected(self):
        client, _ = _make_client()
        with client as c:
            # session login-1 becomes blocked
            _set_session(c, session_id="login-1")
            _enroll(c)
            assert _continuous(c, _raw_session(FAR_SESSION)).json()["decision"] == "SUSPICIOUS"
            assert _verify(c).status_code == 403

            # a fresh login (new session_id) starts clean and is NOT gated
            _set_session(c, session_id="login-2")
            assert _verify(c).status_code == 200
            assert c.get(STATE_ENDPOINT).json()["session_state"] == "verified"

    def test_different_user_unaffected(self):
        client, _ = _make_client()
        with client as c:
            _set_session(c, username="alice", session_id="login-1")
            _enroll(c, "alice")
            assert _continuous(c, _raw_session(FAR_SESSION)).json()["decision"] == "SUSPICIOUS"

            _set_session(c, username="bob", session_id="login-2")
            _enroll(c, "bob")
            assert _verify(c, "bob").status_code == 200


class TestContinuousResponseStateFields:
    def test_suspicious_window_reports_state_and_counter(self):
        client, _ = _make_client()
        with client as c:
            _set_session(c, session_id="login-1")
            _enroll(c)
            body = _continuous(c, _raw_session(FAR_SESSION)).json()
            assert body["session_state"] == "reverification_required"
            assert body["consecutive_suspicious"] == 1
            assert body["last_verified_at"] is None

    def test_counter_increments_then_resets_after_verified(self):
        client, _ = _make_client()
        with client as c:
            _set_session(c, session_id="login-1")
            _enroll(c)
            assert _continuous(c, _raw_session(FAR_SESSION)).json()["consecutive_suspicious"] == 1
            assert _continuous(c, _raw_session(FAR_SESSION + "2")).json()["consecutive_suspicious"] == 2
            cleared = _continuous(c, _raw_session("alice_s1")).json()
            assert cleared["decision"] == "VERIFIED"
            assert cleared["consecutive_suspicious"] == 0
            assert cleared["last_verified_at"] is not None
            assert c.get(STATE_ENDPOINT).json()["session_state"] == "verified"

    def test_verified_window_reports_state(self):
        client, _ = _make_client()
        with client as c:
            _set_session(c, session_id="login-1")
            _enroll(c)
            body = _continuous(c, _raw_session("alice_s1")).json()
            assert body["session_state"] == "verified"
            assert body["consecutive_suspicious"] == 0
            assert body["last_verified_at"] is not None


class TestStateEndpoint:
    def test_unauthenticated_state_is_401(self):
        app = create_app(Settings(environment="test", jwt_secret_key=TEST_JWT_SECRET))
        with TestClient(app) as client:
            response = client.get(STATE_ENDPOINT)
            assert response.status_code == 401
            assert response.json()["error"]["code"] == "authentication_required"

    def test_invalid_token_state_is_401(self):
        app = create_app(Settings(environment="test", jwt_secret_key=TEST_JWT_SECRET))
        with TestClient(app) as client:
            response = client.get(
                STATE_ENDPOINT, headers={"Authorization": "Bearer not-a-real-jwt"}
            )
            assert response.status_code == 401
            assert response.json()["error"]["code"] == "invalid_credentials"

    def test_state_does_not_require_a_profile(self, clean_session):
        set_current_user(clean_session.app, username="ghost")
        body = clean_session.get(STATE_ENDPOINT).json()
        assert body["session_state"] == "verified"

    def test_state_response_leaks_no_raw_identity(self, clean_session):
        raw = str(clean_session.get(STATE_ENDPOINT).json())
        for forbidden in ("keyboard_events", "mouse_events", "centroid", "embedding", "password"):
            assert forbidden not in raw


class TestRealTokenSessionIdentity:
    """End-to-end session identity through the REAL bearer-token dependency.

    Uses no ``set_current_user`` override: ``session_id`` is the SHA-256 digest
    of the actual JWT string, so these tests exercise requirement: same token →
    same session, fresh token → fresh session, even with identical claims.
    """

    def _register_and_login(self, client, username):
        assert client.post(
            REGISTER, json={"username": username, "password": VALID_PASSWORD}
        ).status_code == 201
        return access_token(client, username)

    def test_same_token_replayed_retains_state_and_fresh_login_starts_clean(self):
        client, _ = _make_real_auth_client()
        username = "token_fingerprint_alice"
        with client as c:
            token_a = self._register_and_login(c, username)
            assert _enroll_token(c, token_a, username).status_code == 201
            suspicious = c.post(
                CONTINUOUS_ENDPOINT, headers=_auth(token_a), json=_raw_session(FAR_SESSION)
            )
            assert suspicious.status_code == 200
            assert suspicious.json()["decision"] == "SUSPICIOUS"

            # replaying the SAME token re-derives the SAME session id -> the
            # blocked state survives the replay.
            replayed = c.get(STATE_ENDPOINT, headers=_auth(token_a)).json()
            assert replayed["session_state"] == "reverification_required"
            assert replayed["consecutive_suspicious"] == 1
            assert _verify_token(c, token_a, username).status_code == 403

            # a fresh login (login-only; the account already exists) issues a
            # DISTINCT token (header nonce) even in the same iat/exp second ->
            # a distinct session id -> clean state, and the old session's gate
            # does not leak onto the new one.
            token_b = access_token(c, username)
            assert token_a != token_b
            assert c.get(STATE_ENDPOINT, headers=_auth(token_b)).json()[
                "session_state"
            ] == "verified"
            assert _verify_token(c, token_b, username).status_code == 200

    def test_state_survives_a_refreshed_page_read(self):
        """A browser refresh re-uses the SAME token -> the state endpoint must
        re-report the persisted server state (idempotent reads, no 404)."""
        client, _ = _make_real_auth_client()
        username = "refresh_fingerprint_alice"
        with client as c:
            token = self._register_and_login(c, username)
            assert _enroll_token(c, token, username).status_code == 201

            first = c.get(STATE_ENDPOINT, headers=_auth(token)).json()
            assert first["session_state"] == "verified"
            second = c.get(STATE_ENDPOINT, headers=_auth(token)).json()
            assert second == first  # re-reading must not mutate or vanish

            c.post(
                CONTINUOUS_ENDPOINT, headers=_auth(token), json=_raw_session(FAR_SESSION)
            )
            refreshed = c.get(STATE_ENDPOINT, headers=_auth(token)).json()
            assert refreshed["session_state"] == "reverification_required"
            assert refreshed["consecutive_suspicious"] == 1

            recovery = c.post(
                CONTINUOUS_ENDPOINT,
                headers=_auth(token),
                json=_raw_session(username + "_s1"),
            ).json()
            assert recovery["decision"] == "VERIFIED"
            assert c.get(STATE_ENDPOINT, headers=_auth(token)).json()[
                "session_state"
            ] == "verified"

    def test_derived_session_id_never_reaches_the_ml_service(self):
        recorder = _RecordingVerifier()
        client, _ = _make_real_auth_client(verifier=recorder)
        username = "privacy_fingerprint_alice"
        with client as c:
            token = self._register_and_login(c, username)
            derived = _session_id_from_token(token)
            assert _enroll_token(c, token, username).status_code == 201
            c.post(
                CONTINUOUS_ENDPOINT, headers=_auth(token), json=_raw_session(username + "_s1")
            )
            _verify_token(c, token, username)
            assert recorder.seen, "the ML service must have seen the windows"
            for payload in recorder.seen:
                dumped = json.dumps(payload)
                assert derived not in dumped  # the session id is not a feature

    def test_api_responses_never_contain_the_derived_session_id(self):
        client, _ = _make_real_auth_client()
        username = "noleak_fingerprint_alice"
        with client as c:
            token = self._register_and_login(c, username)
            derived = _session_id_from_token(token)
            enroll_body = c.post(
                ENROLLMENT_ENDPOINT,
                headers=_auth(token),
                json={"user_ref": username, "sessions": [_raw_session(username + "_s1")]},
            ).json()
            verify_body = c.post(
                VERIFICATION_ENDPOINT,
                headers=_auth(token),
                json={"user_ref": username, "session": _raw_session(username + "_s1")},
            ).json()
            continuous_body = c.post(
                CONTINUOUS_ENDPOINT, headers=_auth(token), json=_raw_session(username + "_s1")
            ).json()
            state_body = c.get(STATE_ENDPOINT, headers=_auth(token)).json()
            for body in (enroll_body, verify_body, continuous_body, state_body):
                assert "session_id" not in body
                assert derived not in json.dumps(body)