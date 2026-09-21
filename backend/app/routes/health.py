"""
Phase 9A / Phase 15B — health and readiness routes.

Two kinds of probes are exposed, each in a canonical (unversioned) and an
``<api_prefix>``-versioned form:

* ``GET /health``  — liveness probe. Returns a constant ``{"status": "ok"}``
  and performs no I/O: it never loads the ML model, never touches a database
  and never depends on an external service. This contract is unchanged by
  Phase 15B.
* ``GET /health/ready`` — readiness probe (Phase 15B). Checks the two things
  a load-balanced / orchestrated deployment needs before routing traffic:
  PostgreSQL availability (a lightweight ``SELECT 1``) and the ML service's
  current load state. The database check decides the status: 200 when the
  database is reachable (or deliberately not configured), otherwise the
  structured ``503 database_unavailable`` envelope. The ``ml`` field reports
  ``loaded``/``not_loaded`` from the (lazy, Phase 9B) ML service WITHOUT
  forcing a load — readiness gives operators visibility without changing the
  lazy-load semantics of the ML endpoints.

Unlike ``/health``, readiness is allowed to perform I/O. The probe never
leaks SQL, connection strings, credentials, passwords, stack traces or
secrets: every connection failure collapses to the constant
``database_unavailable`` envelope.
"""

from __future__ import annotations

from typing import Dict, Tuple

from fastapi import APIRouter, Request

from backend.app.config import Settings
from backend.app.dependencies import get_ml_service, get_settings
from backend.app.repositories.errors import DatabaseUnavailableError

router = APIRouter(tags=["health"])

HEALTH_BODY: Dict[str, str] = {"status": "ok"}


def default_database_probe(database_url: str) -> bool:
    """Return whether a short ``SELECT 1`` succeeds against ``database_url``.

    The psycopg driver is imported lazily (the repository convention), the
    connection uses a short connect timeout, and any failure returns ``False``
    so no driver exception, SQL, URL, credential or stack detail ever escapes.
    """
    if not database_url:
        return False
    import psycopg  # lazy driver import, matching the repository convention

    try:
        with psycopg.connect(database_url, connect_timeout=3.0) as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:  # noqa: BLE001 - any failure means "not reachable"
        return False


def read_health() -> Dict[str, str]:
    """Return the deterministic health payload (status always ``"ok"``)."""
    return dict(HEALTH_BODY)


def read_health_ready(request: Request) -> Dict[str, str]:
    """Return the readiness payload or raise the structured 503 envelope.

    ``database`` is ``ok`` when the configured PostgreSQL is reachable,
    ``not_configured`` when no ``BBA_DATABASE_URL`` is set (the in-memory dev
    mode is a supported configuration), and drives a structured
    ``503`` ``database_unavailable`` when configured but unreachable. ``ml``
    mirrors :attr:`BehavioralMLService.is_loaded` and never forces a load.
    """
    settings: Settings = get_settings(request)
    if not settings.database_url:
        database = "not_configured"
    else:
        database = "ok" if default_database_probe(settings.database_url) else "unavailable"
    if database == "unavailable":
        raise DatabaseUnavailableError()
    ml_service = get_ml_service(request)
    ml = "loaded" if ml_service.is_loaded else "not_loaded"
    return {"status": "ok", "database": database, "ml": ml}


@router.get("/health", summary="Health probe (canonical)", operation_id="read_health_canonical")
def _read_health() -> Dict[str, str]:
    return read_health()


@router.get(
    "/health/ready",
    summary="Readiness probe (canonical)",
    operation_id="read_health_ready_canonical",
)
def _read_health_ready(request: Request) -> Dict[str, str]:
    return read_health_ready(request)


def build_versioned_router(prefix: str) -> APIRouter:
    """Build the versioned health router under ``prefix`` (e.g. ``/api/v1``)."""
    api = APIRouter(prefix=prefix, tags=["health"])

    @api.get("/health", summary="Health probe (versioned)", operation_id="read_health_versioned")
    def _read_health_versioned() -> Dict[str, str]:
        return read_health()

    @api.get(
        "/health/ready",
        summary="Readiness probe (versioned)",
        operation_id="read_health_ready_versioned",
    )
    def _read_health_ready_versioned(request: Request) -> Dict[str, str]:
        return read_health_ready(request)

    return api


__all__ = [
    "build_versioned_router",
    "default_database_probe",
    "read_health",
    "read_health_ready",
    "router",
]