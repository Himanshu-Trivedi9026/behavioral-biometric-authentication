"""
Phase 9A — health routes.

Two routers expose the same lightweight health probe:

* ``router``               -> ``GET /health``         (canonical, unversioned;
  intended for infra/monitoring checks)
* ``build_versioned_router`` -> ``GET <api_prefix>/health`` (e.g.
  ``/api/v1/health``)

The handler returns a small constant JSON body and performs no I/O: it does
not load the ML model, does not touch a database and does not depend on any
external service.
"""

from __future__ import annotations

from typing import Dict

from fastapi import APIRouter

router = APIRouter(tags=["health"])

HEALTH_BODY: Dict[str, str] = {"status": "ok"}


def read_health() -> Dict[str, str]:
    """Return the deterministic health payload (status always ``"ok"``)."""
    return dict(HEALTH_BODY)


@router.get("/health", summary="Health probe (canonical)", operation_id="read_health_canonical")
def _read_health() -> Dict[str, str]:
    return read_health()


def build_versioned_router(prefix: str) -> APIRouter:
    """Build the versioned health router under ``prefix`` (e.g. ``/api/v1``)."""
    api = APIRouter(prefix=prefix, tags=["health"])

    @api.get("/health", summary="Health probe (versioned)", operation_id="read_health_versioned")
    def _read_health_versioned() -> Dict[str, str]:
        return read_health()

    return api


__all__ = ["build_versioned_router", "router"]