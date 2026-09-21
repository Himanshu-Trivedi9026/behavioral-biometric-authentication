"""
Phase 11 — JWT token tests.

Covers the ``create_access_token`` / ``decode_access_token`` contract:

* valid tokens decode back to the intended ``sub``,
* signatures, expiration and required claims are enforced,
* expired / malformed / wrong-signature / missing-subject tokens are rejected
  with the same structured error,
* the token payload is deliberately minimal: ``sub``/``iat``/``exp`` and never
  passwords, hashes, behavioural data or secrets,
* an unconfigured JWT secret makes token creation fail closed.
"""

import datetime

import jwt
import pytest

from backend.app.auth.dependencies import CurrentUser  # noqa: F401 - re-export
from backend.app.auth.errors import AuthUnavailableError, InvalidCredentialsError
from backend.app.auth.security import create_access_token, decode_access_token
from backend.app.config import Settings
from tests.auth_testing import TEST_JWT_SECRET, settings

_SUBJECT_ID = "aaaaaaaa-bbbb-4ccc-8ddd-eeeeeeeeeeee"

_BANNED_CLAIMS = {
    "password",
    "password_hash",
    "hash",
    "embeddings",
    "centroid",
    "keyboard_events",
    "mouse_events",
    "user_ref",
}


def _settings(**overrides) -> Settings:
    return settings(**overrides)


class TestValidTokens:
    def test_create_and_decode_round_trip(self):
        s = _settings()
        token = create_access_token(s, _SUBJECT_ID)
        payload = decode_access_token(s, token)
        assert payload["sub"] == _SUBJECT_ID

    def test_payload_contains_only_sub_iat_exp(self):
        s = _settings()
        token = create_access_token(s, _SUBJECT_ID)
        # Inspect without verifying to assert on the encoded claims directly.
        raw = jwt.decode(
            token,
            s.jwt_secret_key,
            algorithms=[s.jwt_algorithm],
            options={"verify_signature": False},
        )
        assert set(raw) == {"sub", "iat", "exp"}
        assert raw["sub"] == _SUBJECT_ID
        assert isinstance(raw["exp"], (int, float))

    def test_token_never_carries_sensitive_claims(self):
        s = _settings()
        token = create_access_token(s, _SUBJECT_ID)
        raw = jwt.decode(
            token,
            s.jwt_secret_key,
            algorithms=[s.jwt_algorithm],
            options={"verify_signature": False},
        )
        assert set(raw).isdisjoint(_BANNED_CLAIMS)

    def test_multiple_tokens_for_same_user_are_identical_claims(self):
        s = _settings()
        a = create_access_token(s, _SUBJECT_ID)
        b = create_access_token(s, _SUBJECT_ID)
        assert decode_access_token(s, a)["sub"] == decode_access_token(s, b)["sub"]

    def test_fresh_issuance_produces_distinct_tokens_with_identical_claims(self):
        # Phase 14B: every issuance must be a distinct token STRING (header
        # nonce) while the claims stay exactly {sub, iat, exp} — this is what
        # lets two same-second logins carry two independent session ids.
        s = _settings()
        a = create_access_token(s, _SUBJECT_ID)
        b = create_access_token(s, _SUBJECT_ID)
        assert a != b
        assert "nonce" in jwt.get_unverified_header(a)
        assert "nonce" in jwt.get_unverified_header(b)
        assert set(decode_access_token(s, a)) == {"sub", "iat", "exp"}
        assert set(decode_access_token(s, b)) == {"sub", "iat", "exp"}

    def test_identical_claims_encode_to_distinct_tokens(self):
        # The strongest unit-level version: two tokens signed from ONE payload
        # dict (same user, same iat, same exp) still differ as byte strings
        # because of the header nonce — and verify to identical claims only.
        s = _settings()
        now = datetime.datetime.now(datetime.timezone.utc)
        payload = {
            "sub": _SUBJECT_ID,
            "iat": now,
            "exp": now + datetime.timedelta(minutes=s.jwt_access_token_expire_minutes),
        }
        a = jwt.encode(
            payload, s.jwt_secret_key, algorithm=s.jwt_algorithm, headers={"nonce": "n1"}
        )
        b = jwt.encode(
            payload, s.jwt_secret_key, algorithm=s.jwt_algorithm, headers={"nonce": "n2"}
        )
        assert a != b
        assert jwt.get_unverified_header(a)["nonce"] == "n1"
        assert jwt.get_unverified_header(b)["nonce"] == "n2"
        claims_a = jwt.decode(a, s.jwt_secret_key, algorithms=[s.jwt_algorithm])
        claims_b = jwt.decode(b, s.jwt_secret_key, algorithms=[s.jwt_algorithm])
        assert claims_a == claims_b
        assert set(claims_a) == {"sub", "iat", "exp"}


class TestExpiration:
    def test_expired_token_rejected(self):
        s = _settings()
        in_the_past = (
            datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(minutes=5)
        )
        token = jwt.encode(
            {"sub": _SUBJECT_ID, "exp": in_the_past},
            s.jwt_secret_key,
            algorithm=s.jwt_algorithm,
        )
        with pytest.raises(InvalidCredentialsError):
            decode_access_token(s, token)


class TestInvalidTokens:
    def test_malformed_token_rejected(self):
        s = _settings()
        for bad in ("not-a-jwt", "", "a.b", "a.b.c.d.e"):
            with pytest.raises(InvalidCredentialsError):
                decode_access_token(s, bad)

    def test_wrong_signature_rejected(self):
        s = _settings()
        other_secret = "another-test-only-secret-" + "Y!91" * 12
        token = jwt.encode(
            {"sub": _SUBJECT_ID, "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=5)},
            other_secret,
            algorithm="HS256",
        )
        with pytest.raises(InvalidCredentialsError):
            decode_access_token(s, token)

    def test_missing_sub_rejected(self):
        s = _settings()
        token = jwt.encode(
            {
                "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=5),
            },
            s.jwt_secret_key,
            algorithm=s.jwt_algorithm,
        )
        with pytest.raises(InvalidCredentialsError):
            decode_access_token(s, token)

    def test_non_string_sub_rejected(self):
        s = _settings()
        token = jwt.encode(
            {
                "sub": 12345,
                "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=5),
            },
            s.jwt_secret_key,
            algorithm=s.jwt_algorithm,
        )
        with pytest.raises(InvalidCredentialsError):
            decode_access_token(s, token)

    def test_missing_exp_rejected(self):
        s = _settings()
        token = jwt.encode({"sub": _SUBJECT_ID}, s.jwt_secret_key, algorithm=s.jwt_algorithm)
        with pytest.raises(InvalidCredentialsError):
            decode_access_token(s, token)

    def test_empty_sub_rejected(self):
        s = _settings()
        token = jwt.encode(
            {
                "sub": "   ",
                "exp": datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(minutes=5),
            },
            s.jwt_secret_key,
            algorithm=s.jwt_algorithm,
        )
        with pytest.raises(InvalidCredentialsError):
            decode_access_token(s, token)


class TestUnconfiguredSecret:
    def test_create_fails_closed_when_secret_empty(self):
        s = Settings(environment="test", jwt_secret_key="")
        with pytest.raises(AuthUnavailableError):
            create_access_token(s, _SUBJECT_ID)

    def test_decode_fails_closed_when_secret_empty(self):
        s = Settings(environment="test", jwt_secret_key="")
        token = create_access_token(settings(), _SUBJECT_ID)
        with pytest.raises(InvalidCredentialsError):
            decode_access_token(s, token)


class TestSettingsValidation:
    def test_short_secret_rejected(self):
        with pytest.raises(ValueError, match="jwt_secret_key"):
            Settings(environment="test", jwt_secret_key="short")

    def test_known_placeholder_secret_rejected(self):
        # The weak dependency-smoke-test value must never be accepted.
        for weak in ("s3cret", "secret", "changeme", "jwtsecret"):
            with pytest.raises(ValueError, match="jwt_secret_key"):
                Settings(environment="test", jwt_secret_key=weak)

    def test_production_requires_secret(self):
        with pytest.raises(ValueError, match="BBA_JWT_SECRET_KEY"):
            Settings(environment="production")
        # A strong secret satisfies production validation.
        assert Settings(environment="production", jwt_secret_key=TEST_JWT_SECRET)

    def test_algorithm_is_hs256_only(self):
        with pytest.raises(ValueError, match="jwt_algorithm"):
            Settings(environment="test", jwt_secret_key=TEST_JWT_SECRET, jwt_algorithm="none")
        assert settings().jwt_algorithm == "HS256"

    def test_secret_hidden_from_repr(self):
        s = settings()
        assert TEST_JWT_SECRET not in repr(s)
        assert TEST_JWT_SECRET not in str(s)
        assert "jwt_secret_key=<redacted>" in repr(s)

    def test_env_loading(self, monkeypatch):
        monkeypatch.setenv("BBA_JWT_SECRET_KEY", TEST_JWT_SECRET)
        monkeypatch.setenv("BBA_JWT_ALGORITHM", "HS256")
        monkeypatch.setenv("BBA_JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "7")
        s = Settings.from_env()
        assert s.jwt_secret_key == TEST_JWT_SECRET
        assert s.jwt_algorithm == "HS256"
        assert s.jwt_access_token_expire_minutes == 7