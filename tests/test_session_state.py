"""
Phase 14B — per-session behavioral verification state: domain + store unit tests.

Covers the pure logic that the routes and the PostgreSQL repository build on:

* the frozen :class:`SessionVerificationState` value object and its validation;
* the clean :func:`default_session_state` default for a brand-new session;
* the :class:`InMemorySessionStateStore` contract semantics (get_or_create,
  mark_verified, mark_suspicious) including the counter and timestamp rules;
* the token-derived :func:`_session_id_from_token` session identity (stable per
  login, distinct across fresh logins even with identical claims, derived ONLY
  from the raw JWT bytes — never from a payload claim).

No database and no ML are involved anywhere in this module.
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone

import jwt as pyjwt
import pytest

from backend.app.auth.dependencies import _session_id_from_token
from backend.app.repositories import InMemorySessionStateStore
from backend.app.repositories.session_state_repository import (
    SESSION_STATE_REVERIFICATION_REQUIRED,
    SESSION_STATE_VERIFIED,
    SessionVerificationState,
    default_session_state,
)

T0 = "2025-06-01T00:00:00.000000+00:00"
T1 = "2025-06-01T00:00:30.000000+00:00"
T2 = "2025-06-01T00:01:00.000000+00:00"
T3 = "2025-06-01T00:01:30.000000+00:00"


class TestSessionVerificationState:
    def test_default_state_is_clean_and_unblocked(self):
        state = default_session_state("alice", "s1", now=T0)
        assert state.user_ref == "alice"
        assert state.session_id == "s1"
        assert state.state == SESSION_STATE_VERIFIED
        assert state.consecutive_suspicious == 0
        assert state.last_verified_at is None
        assert state.created_at == T0
        assert state.updated_at == T0

    def test_frozen_value_object_rejects_bad_state(self):
        with pytest.raises(ValueError, match="state"):
            SessionVerificationState(
                user_ref="alice",
                session_id="s1",
                state="blocked",
                consecutive_suspicious=0,
                last_verified_at=None,
                created_at=T0,
                updated_at=T0,
            )

    def test_frozen_value_object_rejects_negative_counter(self):
        with pytest.raises(ValueError, match="consecutive_suspicious"):
            SessionVerificationState(
                user_ref="alice",
                session_id="s1",
                state=SESSION_STATE_VERIFIED,
                consecutive_suspicious=-1,
                last_verified_at=None,
                created_at=T0,
                updated_at=T0,
            )

    def test_value_object_is_immutable(self):
        state = default_session_state("alice", "s1", now=T0)
        with pytest.raises(AttributeError):
            state.state = SESSION_STATE_REVERIFICATION_REQUIRED


class TestInMemoryStore:
    def test_get_or_create_is_idempotent_and_race_safe(self):
        store = InMemorySessionStateStore()
        first = store.get_or_create("alice", "s1", now=T0)
        second = store.get_or_create("alice", "s1", now=T1)
        assert first is second  # same object -> atomic create-if-absent
        assert first.state == SESSION_STATE_VERIFIED
        assert first.created_at == T0
        assert store.count() == 1

    def test_get_returns_none_for_untouched_session(self):
        store = InMemorySessionStateStore()
        assert store.get("alice", "ghost") is None

    def test_mark_suspicious_blocks_and_counts_from_scratch(self):
        store = InMemorySessionStateStore()
        state = store.mark_suspicious("alice", "s1", now=T0)
        assert state.state == SESSION_STATE_REVERIFICATION_REQUIRED
        assert state.consecutive_suspicious == 1
        assert state.last_verified_at is None

    def test_consecutive_windows_increment_the_counter(self):
        store = InMemorySessionStateStore()
        store.mark_suspicious("alice", "s1", now=T0)
        state = store.mark_suspicious("alice", "s1", now=T1)
        assert state.consecutive_suspicious == 2
        # another session for the same user is independent
        other = store.mark_suspicious("alice", "s2", now=T1)
        assert other.consecutive_suspicious == 1

    def test_mark_verified_clears_the_gate_and_stamps_time(self):
        store = InMemorySessionStateStore()
        store.mark_suspicious("alice", "s1", now=T0)
        state = store.mark_verified("alice", "s1", now=T1)
        assert state.state == SESSION_STATE_VERIFIED
        assert state.consecutive_suspicious == 0
        assert state.last_verified_at == T1
        assert state.created_at == T0  # preserved across transitions
        assert state.updated_at == T1

    def test_mark_verified_on_fresh_session_just_creates_verified_row(self):
        store = InMemorySessionStateStore()
        state = store.mark_verified("alice", "s1", now=T0)
        assert state.state == SESSION_STATE_VERIFIED
        assert state.consecutive_suspicious == 0
        assert state.last_verified_at == T0

    def test_mark_suspicious_preserves_last_verified_at(self):
        store = InMemorySessionStateStore()
        store.mark_verified("alice", "s1", now=T0)
        state = store.mark_suspicious("alice", "s1", now=T1)
        assert state.last_verified_at == T0
        assert state.consecutive_suspicious == 1

    def test_state_is_keyed_per_session_not_per_user(self):
        store = InMemorySessionStateStore()
        store.mark_suspicious("alice", "login-1", now=T0)
        fresh = store.get_or_create("alice", "login-2", now=T1)
        assert fresh.state == SESSION_STATE_VERIFIED  # fresh login starts clean

    def test_clear_removes_all_sessions(self):
        store = InMemorySessionStateStore()
        store.get_or_create("alice", "s1", now=T0)
        store.get_or_create("bob", "s2", now=T0)
        assert store.clear() == 2
        assert store.count() == 0


class TestSessionIdDerivation:
    TOKEN_A = "eyJhbGciOiJIUzI1NiIsIm5vbmNlIjoiYSJ9.eyJzdWIiOiJ1c2VyLXV1aWQiLCJpYXQiOjE3Mjc3ODQwMDAsImV4cCI6MTcyNzc4NTgwMH0.sig-a"
    TOKEN_B = "eyJhbGciOiJIUzI1NiIsIm5vbmNlIjoiYiJ9.eyJzdWIiOiJ1c2VyLXV1aWQiLCJpYXQiOjE3Mjc3ODQwMDAsImV4cCI6MTcyNzc4NTgwMH0.sig-b"

    def test_same_token_maps_to_the_same_session(self):
        # A replayed token MUST recover the same behavioral session state.
        assert _session_id_from_token(self.TOKEN_A) == _session_id_from_token(self.TOKEN_A)
        assert _session_id_from_token(self.TOKEN_B) == _session_id_from_token(self.TOKEN_B)

    def test_different_tokens_map_to_different_sessions(self):
        assert _session_id_from_token(self.TOKEN_A) != _session_id_from_token(self.TOKEN_B)

    def test_identical_claims_yet_distinct_tokens_map_to_different_sessions(self):
        # Two tokens that share sub/iat/exp (the same-second login collision
        # that used to break sub@iat isolation) but differ as byte strings MUST
        # NOT collide: the session id is a digest of the token itself.
        secret = "phase11-test-secret-" + "X9" * 24
        issued = datetime.now(timezone.utc).replace(microsecond=0)
        payload = {"sub": "user-uuid", "iat": issued, "exp": issued + timedelta(minutes=30)}
        token_a = pyjwt.encode(payload, secret, algorithm="HS256", headers={"nonce": "aaaa"})
        token_b = pyjwt.encode(payload, secret, algorithm="HS256", headers={"nonce": "bbbb"})
        assert token_a != token_b
        decoded_a = pyjwt.decode(token_a, secret, algorithms=["HS256"])
        decoded_b = pyjwt.decode(token_b, secret, algorithms=["HS256"])
        assert decoded_a == decoded_b == {
            "sub": "user-uuid",
            "iat": int(issued.timestamp()),
            "exp": int((issued + timedelta(minutes=30)).timestamp()),
        }
        assert _session_id_from_token(token_a) != _session_id_from_token(token_b)

    def test_session_id_is_a_digest_and_never_exposes_claims(self):
        sid = _session_id_from_token("abc.def.ghi")
        assert sid == hashlib.sha256(b"abc.def.ghi").hexdigest()
        assert len(sid) == 64
        # no claim/identity bytes leak into the key
        assert "user-uuid" not in _session_id_from_token(self.TOKEN_A)
        assert self.TOKEN_A not in _session_id_from_token(self.TOKEN_A)