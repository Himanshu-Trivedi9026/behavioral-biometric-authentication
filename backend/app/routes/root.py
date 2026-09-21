"""
Phase 9A — root / API metadata routes.

``GET /`` returns minimal service information (service name, version, status)
and ``GET <api_prefix>/`` (e.g. ``/api/v1/``) returns API-version metadata.

These endpoints never disclose internal filesystem paths, model checkpoint
paths, environment variables or other internal details.
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Request

from backend.app.config import Settings

router = APIRouter(tags=["meta"])


def _settings_of(request: Request) -> Settings:
    settings = getattr(request.app.state, "settings", None)
    return settings if isinstance(settings, Settings) else Settings()


@router.get("/", summary="Service information", operation_id="read_root")
def read_root(request: Request) -> Dict[str, Any]:
    settings = _settings_of(request)
    return {
        "service": settings.app_name,
        "version": settings.app_version,
        "status": "ok",
    }


def build_versioned_router(prefix: str) -> APIRouter:
    """Build the versioned root router under ``prefix`` (e.g. ``/api/v1``)."""
    api = APIRouter(prefix=prefix, tags=["meta"])

    @api.get("/", summary="API version information", operation_id="read_api_version_root")
    def _read_api_root(request: Request) -> Dict[str, Any]:
        settings = _settings_of(request)
        return {
            "api": "v1",
            "version": settings.app_version,
            "status": "ok",
        }

    return api


__all__ = ["build_versioned_router", "router"]