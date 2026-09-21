"""Phase 9A/9B/10/14B — dependency injection foundation.

FastAPI's dependency system resolves dependencies per-request:

* :func:`get_settings` — resolves the application :class:`Settings`.
* :func:`get_ml_service` — the (lazy) inference-only ML service.
* :func:`get_profile_store` — the per-application profile repository.
* :func:`get_user_repository` — the per-application account repository
  (Phase 11; same persistent/in-memory switching as the profile store).
* :func:`get_session_state_store` — the per-application per-session
  behavioral verification state (Phase 14B; persistent/in-memory switching).

Phase 10 wiring
---------------
:func:`get_profile_store` satisfies the same :class:`ProfileRepository`
contract routes have always depended on, but WHERE profiles live now depends
on configuration:

* ``database_url`` configured  -> :class:`PostgresProfileRepository` (persistent;
  PostgreSQL holds one row per ``user_ref``).
* no ``database_url``          -> :class:`InMemoryProfileStore` (the Phase 9B
  non-persistent fallback).

The PostgreSQL repository is created lazily (its connection pool only opens on
the first real database operation), so serving `/health`, running
``create_app()``, or resolving settings never touches the database — even when a
URL is configured.

Phase 14B wiring
----------------
:func:`get_session_state_store` follows the identical pattern: PostgreSQL (via
``db/migrations/003_create_session_states.sql``) when ``database_url`` is
configured, process-local memory otherwise. Like every other persistence dependency it is attached to
``app.state`` on first resolution, so each ``create_app()`` call gets its own
fresh store.

Both dependencies are attached to ``app.state`` on first resolution so each
``create_app()`` call gets its own service and store (fresh, isolated state per
application instance). Tests replace them with ``app.dependency_overrides``.
"""

from __future__ import annotations

from typing import Any

from fastapi import Request

from backend.app.config import Settings
from backend.app.repositories.memory_session_state_store import (
    InMemorySessionStateStore,
)
from backend.app.repositories.memory_user_repository import InMemoryUserRepository
from backend.app.repositories.postgres_profile_repository import (
    PostgresProfileRepository,
)
from backend.app.repositories.postgres_session_state_repository import (
    PostgresSessionStateRepository,
)
from backend.app.repositories.postgres_user_repository import PostgresUserRepository
from backend.app.repositories.profile_repository import ProfileRepository
from backend.app.repositories.session_state_repository import SessionStateRepository
from backend.app.repositories.user_repository import UserRepository
from backend.app.services.ml_service import BehavioralMLService
from backend.app.services.profile_store import InMemoryProfileStore


def get_settings(request: Request) -> Settings:
    """Resolve the application :class:`Settings` from ``app.state``.

    Falls back to a default :class:`Settings` if none was attached, which
    keeps the dependency usable even in minimal test harnesses.
    """
    settings: Any = getattr(request.app.state, "settings", None)
    if isinstance(settings, Settings):
        return settings
    return Settings()


def get_ml_service(request: Request) -> BehavioralMLService:
    """Resolve the shared (lazy) inference-only ML service for this app."""
    service = getattr(request.app.state, "ml_service", None)
    if not isinstance(service, BehavioralMLService):
        settings = get_settings(request)
        service = BehavioralMLService.from_settings(settings)
        request.app.state.ml_service = service
    return service


def get_profile_store(request: Request) -> ProfileRepository:
    """Resolve the per-application profile repository.

    Uses the persistent :class:`PostgresProfileRepository` when a
    ``database_url`` is configured; otherwise falls back to the Phase 9B
    in-memory store. Never touches the database on resolution — the
    PostgreSQL pool is created lazily on the first operation.
    """
    store = getattr(request.app.state, "profile_store", None)
    if store is None:
        settings = get_settings(request)
        if getattr(settings, "database_url", ""):
            store = PostgresProfileRepository.from_settings(settings)
        else:
            store = InMemoryProfileStore()
        request.app.state.profile_store = store
    return store


def get_user_repository(request: Request) -> UserRepository:
    """Resolve the per-application user repository (Phase 11).

    Uses the persistent :class:`PostgresUserRepository` when a
    ``database_url`` is configured; otherwise falls back to the in-memory
    repository. Like :func:`get_profile_store`, resolution never connects to
    the database — the PostgreSQL pool opens lazily on the first operation.
    """
    repository = getattr(request.app.state, "user_repository", None)
    if repository is None:
        settings = get_settings(request)
        if getattr(settings, "database_url", ""):
            repository = PostgresUserRepository.from_settings(settings)
        else:
            repository = InMemoryUserRepository()
        request.app.state.user_repository = repository
    return repository


def get_session_state_store(request: Request) -> SessionStateRepository:
    """Resolve the per-application per-session verification state store (Phase 14B).

    Uses the persistent :class:`PostgresSessionStateRepository` when a
    ``database_url`` is configured; otherwise falls back to the in-memory
    store. Resolution is lazily created on ``app.state`` and never connects to
    the database — the PostgreSQL pool opens only on the first operation.
    """
    store = getattr(request.app.state, "session_state_store", None)
    if store is None:
        settings = get_settings(request)
        if getattr(settings, "database_url", ""):
            store = PostgresSessionStateRepository.from_settings(settings)
        else:
            store = InMemorySessionStateStore()
        request.app.state.session_state_store = store
    return store


__all__ = [
    "get_ml_service",
    "get_profile_store",
    "get_session_state_store",
    "get_settings",
    "get_user_repository",
]