"""
Phase 9A/10 — typed application configuration.

:class:`Settings` is a frozen dataclass (matching the repository's config
conventions — see :class:`ml.training.config.TrainingConfig`).  Every field
has a sensible default; the only thing read at construction time is the
environment via :meth:`Settings.from_env`.

Supported environment variables
-----------------------------
``BBA_APP_NAME``                  application title shown in OpenAPI docs
``BBA_APP_VERSION``               application version string
``BBA_ENVIRONMENT``               ``development`` | ``test`` | ``production``
``BBA_DEBUG``                     ``true`` / ``1`` / ``yes`` to enable debug mode
``BBA_API_PREFIX``                API prefix (default ``/api/v1``)
``BBA_CORS_ORIGINS``              comma-separated list of allowed origins
``BBA_CHECKPOINT_PATH``           Phase 7 checkpoint path (default ``models/...pt``)
``BBA_PREPROCESSING_ARTIFACT_PATH``  train-only scaler artifact (JSON)
``BBA_VERIFICATION_CONFIG_PATH``  calibrated-threshold artifact (JSON)
``BBA_DATABASE_URL``              PostgreSQL connection string (Phase 10), e.g.
                                  ``postgresql://user:pass@localhost:5432/db``.
                                  Empty by default: no database is configured,
                                  so profile storage is reported unavailable
                                  until a URL is provided.
``BBA_JWT_SECRET_KEY``            HMAC secret used to sign access tokens
                                  (Phase 11). Empty by default: the auth
                                  endpoints then fail closed (503
                                  ``auth_unavailable``) rather than signing
                                  with a weak/default secret.
``BBA_JWT_ALGORITHM``             JWT signing algorithm (Phase 11; ``HS256``).
``BBA_JWT_ACCESS_TOKEN_EXPIRE_MINUTES``  access-token lifetime in minutes
                                  (Phase 11; default ``30``).
``BBA_LOG_LEVEL``                root log level (Phase 15B; one of
                                  ``DEBUG``/``INFO``/``WARNING``/``ERROR``/
                                  ``CRITICAL``; default ``INFO``). Invalid
                                  values are rejected at construction.

All fields are optional; the defaults are appropriate for local development.

Security
--------
Database credentials live only in the environment (``BBA_DATABASE_URL``).
They are never written to source code, never returned by the API, and the
``__repr__``/``__str__`` of :class:`Settings` masks the connection string and
the JWT secret. ``BBA_CORS_ORIGINS`` containing ``"*"`` is rejected (unsafe
with credentials). The JWT algorithm is restricted to ``HS256`` and a
production environment is rejected unless a strong (>= 32 character) JWT
secret is configured — there is deliberately no usable default secret.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Tuple

DEFAULT_CHECKPOINT_PATH = "models/siamese_behavioral_encoder.pt"
DEFAULT_PREPROCESSING_ARTIFACT_PATH = "models/behavioral_preprocessing.json"
DEFAULT_VERIFICATION_CONFIG_PATH = "models/verification_config.json"
DEFAULT_JWT_ALGORITHM = "HS256"
DEFAULT_JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 30
DEFAULT_LOG_LEVEL = "INFO"

_VALID_ENVIRONMENTS = frozenset({"development", "test", "production"})
_VALID_LOG_LEVELS = frozenset(
    {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL", "NOTSET"}
)
_TRUE_LIKE = frozenset({"true", "1", "yes"})
_DATABASE_SCHEMES = ("postgres://", "postgresql://")
# Only HMAC-SHA256 is supported: a single, well-understood signing algorithm
# avoids algorithm-confusion/misconfiguration. There is no justified reason in
# this architecture to accept anything else.
_VALID_JWT_ALGORITHMS = frozenset({DEFAULT_JWT_ALGORITHM})
# Known placeholder/weak secrets that must never be accepted, even in tests.
_WEAK_JWT_SECRETS = frozenset(
    {
        "s3cret",
        "secret",
        "password",
        "changeme",
        "change-me",
        "replace-me",
        "replace_me",
        "jwtsecret",
        "jwt-secret",
        "your-secret-key",
        "test-secret",
    }
)
_MIN_JWT_SECRET_LENGTH = 32


def _parse_bool(value: str) -> bool:
    return value.lower().strip() in _TRUE_LIKE


def _is_weak_jwt_secret(value: str) -> bool:
    """True when ``value`` is a known placeholder or simply too short.

    An empty value is treated as "not configured" (auth fails closed at call
    time); it does not raise here so that health/ML-only deployments can run
    without auth configuration.
    """
    if not value:
        return False
    if len(value) < _MIN_JWT_SECRET_LENGTH:
        return True
    return value.strip().lower() in _WEAK_JWT_SECRETS


@dataclass(frozen=True, repr=False)
class Settings:
    """Immutable, validation-enforced application settings."""

    app_name: str = "Behavioral Biometric Authentication API"
    app_version: str = "0.1.0"
    environment: str = "development"
    debug: bool = False
    api_prefix: str = "/api/v1"
    cors_origins: Tuple[str, ...] = ("http://localhost:5173",)
    checkpoint_path: str = DEFAULT_CHECKPOINT_PATH
    preprocessing_artifact_path: str = DEFAULT_PREPROCESSING_ARTIFACT_PATH
    verification_config_path: str = DEFAULT_VERIFICATION_CONFIG_PATH
    database_url: str = ""
    jwt_secret_key: str = ""
    jwt_algorithm: str = DEFAULT_JWT_ALGORITHM
    jwt_access_token_expire_minutes: int = DEFAULT_JWT_ACCESS_TOKEN_EXPIRE_MINUTES
    log_level: str = DEFAULT_LOG_LEVEL

    # -- validation ----------------------------------------------------------

    def __post_init__(self) -> None:
        if not isinstance(self.app_name, str) or not self.app_name.strip():
            raise ValueError("app_name must be a non-empty string")
        if not isinstance(self.app_version, str) or not self.app_version.strip():
            raise ValueError("app_version must be a non-empty string")
        if self.environment not in _VALID_ENVIRONMENTS:
            raise ValueError(
                "environment must be one of {}; got {!r}".format(
                    sorted(_VALID_ENVIRONMENTS), self.environment
                )
            )
        if not isinstance(self.api_prefix, str) or not self.api_prefix.startswith("/"):
            raise ValueError("api_prefix must start with /, got {!r}".format(self.api_prefix))
        if "*" in self.cors_origins:
            raise ValueError(
                'cors_origins must not contain "*"; '
                "configure explicit origins instead of a wildcard"
            )
        for name in (
            "checkpoint_path",
            "preprocessing_artifact_path",
            "verification_config_path",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError("{}.{} must be a non-empty string".format(type(self).__name__, name))
        if not isinstance(self.database_url, str):
            raise ValueError("Settings.database_url must be a string")
        if self.database_url and not self.database_url.lower().startswith(_DATABASE_SCHEMES):
            raise ValueError(
                "database_url must use the postgres:// or postgresql:// scheme"
            )
        if not isinstance(self.jwt_secret_key, str):
            raise ValueError("Settings.jwt_secret_key must be a string")
        if self.jwt_algorithm not in _VALID_JWT_ALGORITHMS:
            raise ValueError(
                "jwt_algorithm must be one of {}; got {!r}".format(
                    sorted(_VALID_JWT_ALGORITHMS), self.jwt_algorithm
                )
            )
        if (
            not isinstance(self.jwt_access_token_expire_minutes, int)
            or isinstance(self.jwt_access_token_expire_minutes, bool)
            or self.jwt_access_token_expire_minutes <= 0
        ):
            raise ValueError("jwt_access_token_expire_minutes must be a positive integer")
        if _is_weak_jwt_secret(self.jwt_secret_key):
            raise ValueError(
                "jwt_secret_key is too weak or a known placeholder; use a random "
                "secret of at least {} characters".format(_MIN_JWT_SECRET_LENGTH)
            )
        if not isinstance(self.log_level, str):
            raise ValueError("Settings.log_level must be a string")
        normalized_log_level = self.log_level.strip().upper()
        if normalized_log_level not in _VALID_LOG_LEVELS:
            raise ValueError(
                "log_level must be one of {}; got {!r}".format(
                    sorted(_VALID_LOG_LEVELS), self.log_level
                )
            )
        if self.environment == "production" and not self.jwt_secret_key:
            raise ValueError(
                "a production deployment must configure BBA_JWT_SECRET_KEY "
                "(at least {} characters)".format(_MIN_JWT_SECRET_LENGTH)
            )

    # -- env loading ---------------------------------------------------------

    @classmethod
    def from_env(cls) -> Settings:
        """Read ``BBA_*`` environment variables and return a ``Settings``.

        Missing variables fall back to the class defaults.  Invalid values
        raise :class:`ValueError` with a clear message.
        """
        fields = cls.__dataclass_fields__
        cors_raw = os.environ.get("BBA_CORS_ORIGINS", "")
        cors = tuple(
            origin.strip()
            for origin in cors_raw.split(",")
            if origin.strip()
        )
        return cls(
            app_name=os.environ.get("BBA_APP_NAME", fields["app_name"].default),
            app_version=os.environ.get("BBA_APP_VERSION", fields["app_version"].default),
            environment=os.environ.get("BBA_ENVIRONMENT", fields["environment"].default),
            debug=_parse_bool(os.environ.get("BBA_DEBUG", str(fields["debug"].default))),
            api_prefix=os.environ.get("BBA_API_PREFIX", fields["api_prefix"].default),
            cors_origins=cors if cors else fields["cors_origins"].default,
            checkpoint_path=os.environ.get("BBA_CHECKPOINT_PATH", fields["checkpoint_path"].default),
            preprocessing_artifact_path=os.environ.get(
                "BBA_PREPROCESSING_ARTIFACT_PATH",
                fields["preprocessing_artifact_path"].default,
            ),
            verification_config_path=os.environ.get(
                "BBA_VERIFICATION_CONFIG_PATH",
                fields["verification_config_path"].default,
            ),
            database_url=os.environ.get("BBA_DATABASE_URL", fields["database_url"].default),
            jwt_secret_key=os.environ.get(
                "BBA_JWT_SECRET_KEY", fields["jwt_secret_key"].default
            ),
            jwt_algorithm=os.environ.get(
                "BBA_JWT_ALGORITHM", fields["jwt_algorithm"].default
            ),
            jwt_access_token_expire_minutes=int(
                os.environ.get(
                    "BBA_JWT_ACCESS_TOKEN_EXPIRE_MINUTES",
                    str(fields["jwt_access_token_expire_minutes"].default),
                )
            ),
            log_level=os.environ.get("BBA_LOG_LEVEL", "").strip().upper()
            or fields["log_level"].default,
        )

    # -- privacy -------------------------------------------------------------

    def __repr__(self) -> str:
        """Dataclass-style repr with the database URL + JWT secret masked."""
        parts = []
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if name in ("database_url", "jwt_secret_key") and value:
                value = "<redacted>"
                parts.append("{}={}".format(name, value))
            else:
                parts.append("{}={!r}".format(name, value))
        return "{}({})".format(type(self).__name__, ", ".join(parts))

    __str__ = __repr__


__all__ = ["Settings"]