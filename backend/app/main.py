"""
Phase 9A (Phase 11 extended) — FastAPI application factory.

:func:`create_app` builds a fresh FastAPI application:

1. Reads :class:`~backend.app.config.Settings` (defaults or ``BBA_*`` env).
2. Configures metadata (title / version / description, OpenAPI docs).
3. Attaches the settings to ``app.state`` (dependency injection source).
4. Adds CORS middleware (explicit, configurable origins; never ``*``).
5. Registers the centralized exception handlers.
6. Registers the foundation routes (``/``, ``/health``, ``/api/v1/...``),
   the Phase 11 auth routes (``/api/v1/auth/...``) and the protected
   enrollment/verification routes.

No global side effects happen at import time other than building the module
level ``app`` singleton used by the ASGI entrypoint
(``uvicorn backend.app.main:app``).  Nothing here loads the ML model, opens a
database, or connects to external services.  Health stays public and
independent of both authentication and the database.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.config import Settings
from backend.app.errors import install_exception_handlers
from backend.app.observability import RequestIdMiddleware, configure_logging
from backend.app.routes import auth, health, root, verification

logger = logging.getLogger("backend.app.main")

_DESCRIPTION = (
    "Backend for the Behavioral Biometric Authentication system "
    "(phases 1-8 complete). This service exposes a password-authenticated "
    "ML API: account registration and login (JWT bearer tokens, Argon2id "
    "password hashing) plus POST /api/v1/enrollment, "
    "POST /api/v1/verification and POST /api/v1/continuous-verification "
    "(Phase 14A) backed by the Phase 7 checkpoint. All protected endpoints "
    "require a valid bearer token and are scoped to the authenticated user's "
    "profile (request-body user_ref can never select another user's data). "
    "Profiles are kept in memory and lost on restart "
    "unless PostgreSQL (BBA_DATABASE_URL) is configured. This is a "
    "development service; it makes no claim of real-world biometric accuracy."
)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Lightweight lifespan hook.

    Logs startup/shutdown only. Purposely does NOT load the ML checkpoint,
    connect to a database, or initialise Redis — those integrations are later
    phases.
    """
    logger.info("%s v%s starting (environment=%s)", app.title, app.version, app.state.settings.environment)
    yield
    logger.info("%s v%s stopped", app.title, app.version)


def create_app(settings: Optional[Settings] = None) -> FastAPI:
    """Build and configure a new FastAPI application instance.

    Args:
        settings: optional :class:`Settings`; defaults to
            :meth:`Settings.from_env`.

    Returns:
        A fully configured, ready-to-serve FastAPI application. A new instance
        is created on every call (no shared mutable state).
    """
    settings = settings if isinstance(settings, Settings) else Settings.from_env()

    # Phase 15B: structured logging is configured for every environment (JSON
    # in production/test, the familiar human format in development), replacing
    # the development-only ``logging.basicConfig`` path. Idempotent.
    configure_logging(settings.log_level, json_format=settings.environment != "development")

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description=_DESCRIPTION,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(settings.cors_origins),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Phase 15B: request correlation + structured access logs (no bodies,
    # no headers, no query strings).
    app.add_middleware(RequestIdMiddleware)

    install_exception_handlers(app)

    app.include_router(root.router)
    app.include_router(health.router)
    app.include_router(root.build_versioned_router(settings.api_prefix))
    app.include_router(health.build_versioned_router(settings.api_prefix))
    app.include_router(auth.build_versioned_router(settings.api_prefix))
    app.include_router(verification.build_versioned_router(settings.api_prefix))

    return app


app = create_app()

__all__ = ["create_app", "lifespan"]