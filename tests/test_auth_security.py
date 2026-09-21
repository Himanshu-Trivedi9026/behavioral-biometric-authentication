"""
Phase 11 — password hashing (Argon2id) tests.

Covers the ``hash_password`` / ``verify_password`` contract:

* hashes are Argon2id strings and salted (different for the same password),
* the correct password verifies,
* the wrong password fails,
* malformed / non-hash input fails safely (``False``, never a raise),
* a plaintext password or hash never ends up in the module's repr or logs.

No real cryptography is implemented in this repository; only argon2-cffi is
used to hash and verify.
"""

import argon2
from argon2 import PasswordHasher

from backend.app.auth.security import hash_password, verify_password

_PLAINTEXT = "The.Password-Is: Not In The Hash!1"


def _is_argon2id(hash_value: str) -> bool:
    return hash_value.startswith("$argon2id$") and len(hash_value) > 30


class TestHashPassword:
    def test_hash_returns_argon2id_string(self):
        hashed = hash_password(_PLAINTEXT)
        assert isinstance(hashed, str)
        assert _is_argon2id(hashed)

    def test_hashes_are_salted_and_different(self):
        first = hash_password(_PLAINTEXT)
        second = hash_password(_PLAINTEXT)
        assert first != second
        assert _is_argon2id(first)
        assert _is_argon2id(second)

    def test_hash_never_contains_the_plaintext(self):
        assert _PLAINTEXT not in hash_password(_PLAINTEXT)

    def test_rejects_non_string_or_empty_password(self):
        for bad in ("", "   ", None, 1234, b"bytes"):
            try:
                hash_password(bad)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                pass
            else:
                raise AssertionError("hash_password should reject {!r}".format(bad))


class TestVerifyPassword:
    def test_correct_password_verifies(self):
        hashed = hash_password(_PLAINTEXT)
        assert verify_password(_PLAINTEXT, hashed) is True

    def test_wrong_password_fails(self):
        hashed = hash_password(_PLAINTEXT)
        assert verify_password(_PLAINTEXT + "-nope", hashed) is False

    def test_malformed_hash_fails_safely(self):
        assert verify_password(_PLAINTEXT, "not a valid argon2 hash") is False
        assert verify_password(_PLAINTEXT, "") is False
        assert verify_password(_PLAINTEXT, "$argon2id$v=19$m=1") is False

    def test_argon2_wrong_type_hash_fails_safely(self):
        # A syntactically valid argon2i hash (not argon2id) must not verify.
        other_type = PasswordHasher(type=argon2.low_level.Type.I).hash(_PLAINTEXT)
        assert other_type.startswith("$argon2i$")
        assert verify_password(_PLAINTEXT, other_type) is False

    def test_non_string_inputs_fail_safely(self):
        hashed = hash_password(_PLAINTEXT)
        assert verify_password(None, hashed) is False  # type: ignore[arg-type]
        assert verify_password(b"bytes", hashed) is False  # type: ignore[arg-type]
        assert verify_password(_PLAINTEXT, None) is False  # type: ignore[arg-type]
        assert verify_password(_PLAINTEXT, b"hash") is False  # type: ignore[arg-type]

    def test_repeated_verification_is_stable(self):
        # Multiple logins against the same stored hash must keep working.
        hashed = hash_password(_PLAINTEXT)
        for _ in range(3):
            assert verify_password(_PLAINTEXT, hashed) is True


class TestDefaultParametersAreArgon2idStandard:
    def test_default_hasher_matches_documented_parameters(self):
        hasher = PasswordHasher()
        assert hasher.time_cost == 3
        assert hasher.memory_cost == 65536
        assert hasher.parallelism == 4
        assert hasher.hash_len == 32
        assert hasher.salt_len == 16