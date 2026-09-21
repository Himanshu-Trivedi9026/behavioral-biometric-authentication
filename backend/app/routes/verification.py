"""
Phase 9B (Phase 11 protected, Phase 14B gated) — enrollment + verification routes.

``build_versioned_router(prefix)`` registers the protected endpoints:

* ``POST <prefix>/enrollment``  - register one user's profile from raw sessions.
* ``POST <prefix>/verification`` - match one raw probe session to a profile.
* ``POST <prefix>/continuous-verification`` (Phase 14A) - match one bounded
  behavioral window to the authenticated user's profile.
* ``GET <prefix>/continuous-verification/state`` (Phase 14B) - read the
  authenticated session's server-authoritative behavioral verification state.

Authentication & authorization (Phase 11)
-----------------------------------------
All endpoints require a valid bearer access token. The authenticated identity
(:class:`~backend.app.auth.dependencies.CurrentUser`) is the ONLY source of
authorization:

* The behavioral profile is always keyed by the authenticated username.
* For enrollment/verification, if the request body includes ``user_ref`` it
  must equal the authenticated username, otherwise the request is rejected
  with 403 ``forbidden``. A request-body ``user_ref`` can therefore never
  select another user's profile.
* For continuous verification (Phase 14A) the request body is
  behavioral-data only — identity fields are rejected (422), so the JWT
  identity is the sole profile selector.

Phase 14B — session-scoped, fail-closed re-verification gating
--------------------------------------------------------------
Every login session carries a server-authoritative behavioral state
(``verified`` or ``reverification_required``) in
``behavioral_session_states``. ``POST /enrollment`` and ``POST /verification``
fail closed with 403 ``reverification_required`` while the session is not
``verified``. The continuous-verification POST stays AVAILABLE so the user can
re-verify behavior: a VERIFIED continuous window clears the gate for the
session (a SUSPICIOUS one sets the state to ``reverification_required`` for the
session — never an account-wide lockout). A fresh login carries a new JWT
``iat`` -> a new session id -> a clean unblocked state.

Behaviour (unchanged from Phase 9B/10 apart from auth/gating)
-------------------------------------------------------------
* Enrollment returns 201 with only ``user_ref`` / ``status`` /
  ``embedding_dimension`` / ``session_count``; 409 ``profile_exists`` when the
  authenticated user already has a profile (never silently overwritten).
* Verification returns the Phase 8 decision ``VERIFIED``/``SUSPICIOUS`` with
  the distance and the calibrated threshold; 404 ``profile_not_found`` when the
  user is unknown. The threshold always comes from the artifact — a
  client-supplied threshold is ignored.
* Raw events, embeddings and centroids are never echoed back. ``user_ref`` is
  identity metadata only and never becomes a model feature.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from backend.app.auth.dependencies import CurrentUser, get_current_user
from backend.app.auth.errors import ForbiddenError, ReverificationRequiredError
from backend.app.dependencies import (
    get_ml_service,
    get_profile_store,
    get_session_state_store,
)
from backend.app.repositories import ProfileExistsError, ProfileNotFoundError
from backend.app.repositories.profile_repository import ProfileRepository
from backend.app.repositories.session_state_repository import (
    SESSION_STATE_REVERIFICATION_REQUIRED,
    SessionStateRepository,
)
from backend.app.schemas import (
    ContinuousVerificationRequest,
    ContinuousVerificationResponse,
    EnrollmentRequest,
    EnrollmentResponse,
    SessionStateResponse,
    VerificationRequest,
    VerificationResponse,
)
from backend.app.services.ml_service import BehavioralMLService
from backend.app.services.profile_store import utc_now


def _authorized_user_ref(provided: object, current_user: CurrentUser) -> str:
    """Return the profile key, enforcing that identity is the authorization.

    ``user_ref`` omitted  -> the authenticated username is used.
    ``user_ref`` present  -> it must equal the authenticated username (403).
    """
    if provided is None:
        return current_user.username
    if provided != current_user.username:
        raise ForbiddenError()
    return current_user.username


def _require_reverified_session(
    current_user: CurrentUser, session_store: SessionStateRepository
) -> None:
    """Fail closed (403) when the session's behavioral state blocks the action.

    This gate is per-session and behavior-driven: it checks ONLY the
    authenticated login's own ``behavioral_session_states`` row and never any
    account-wide lockout. The continuous-verification endpoint is exempt so a
    user can always re-verify and clear the block.
    """
    state = session_store.get_or_create(
        current_user.username, current_user.session_id
    )
    if state.state == SESSION_STATE_REVERIFICATION_REQUIRED:
        raise ReverificationRequiredError()


def build_versioned_router(prefix: str) -> APIRouter:
    """Build the protected enrollment/verification router under ``prefix``."""
    api = APIRouter(prefix=prefix, tags=["enrollment", "verification"])

    @api.post(
        "/enrollment",
        response_model=EnrollmentResponse,
        status_code=201,
        summary="Enroll the authenticated user from one or more raw sessions",
        operation_id="create_enrollment",
    )
    def _create_enrollment(
        payload: EnrollmentRequest,
        current_user: CurrentUser = Depends(get_current_user),  # noqa: B008
        service: BehavioralMLService = Depends(get_ml_service),  # noqa: B008
        store: ProfileRepository = Depends(get_profile_store),  # noqa: B008
        session_store: SessionStateRepository = Depends(get_session_state_store),  # noqa: B008
    ) -> EnrollmentResponse:
        user_ref = _authorized_user_ref(payload.user_ref, current_user)
        # Phase 14B: enrollment is a protected action — a session that needs
        # re-verification cannot enroll (fail closed; verified windows unblock).
        _require_reverified_session(current_user, session_store)
        if store.contains(user_ref):
            raise ProfileExistsError(user_ref)
        sessions = [session.model_dump(mode="python") for session in payload.sessions]
        stored = service.enroll_to_store(user_ref, sessions)
        store.save(stored)
        return EnrollmentResponse(
            user_ref=stored.user_ref,
            status="enrolled",
            embedding_dimension=stored.embedding_dim,
            session_count=stored.session_count,
        )

    @api.post(
        "/verification",
        response_model=VerificationResponse,
        summary="Verify one raw probe session for the authenticated user",
        operation_id="verify_session",
    )
    def _verify_session(
        payload: VerificationRequest,
        current_user: CurrentUser = Depends(get_current_user),  # noqa: B008
        service: BehavioralMLService = Depends(get_ml_service),  # noqa: B008
        store: ProfileRepository = Depends(get_profile_store),  # noqa: B008
        session_store: SessionStateRepository = Depends(get_session_state_store),  # noqa: B008
    ) -> VerificationResponse:
        user_ref = _authorized_user_ref(payload.user_ref, current_user)
        # Phase 14B: verification is a protected action. A session in a
        # re-verification-required state is blocked here (fail closed) — only
        # a fresh continuous-verification window can clear it.
        _require_reverified_session(current_user, session_store)
        stored = store.get(user_ref)
        if stored is None:
            raise ProfileNotFoundError(user_ref)
        probe = payload.session.model_dump(mode="python")
        result = service.verify_stored(user_ref, stored, probe)
        return VerificationResponse(
            user_ref=stored.user_ref,
            decision=result.decision,
            distance=result.distance,
            threshold=result.threshold,
        )

    @api.post(
        "/continuous-verification",
        response_model=ContinuousVerificationResponse,
        summary="Verify a bounded behavioral window for the authenticated user",
        operation_id="verify_continuous_window",
    )
    def _verify_continuous_window(
        payload: ContinuousVerificationRequest,
        current_user: CurrentUser = Depends(get_current_user),  # noqa: B008
        service: BehavioralMLService = Depends(get_ml_service),  # noqa: B008
        store: ProfileRepository = Depends(get_profile_store),  # noqa: B008
        session_store: SessionStateRepository = Depends(get_session_state_store),  # noqa: B008
    ) -> ContinuousVerificationResponse:
        # The authenticated JWT identity is the ONLY profile selector. The
        # schema rejects identity fields in the body (extra="forbid"), so a
        # client can never verify against another user's profile.
        #
        # Phase 14B recovery path: this endpoint is intentionally NOT gated.
        # Each window updates the session's server-authoritative state —
        # VERIFIED windows clear the gate, SUSPICIOUS windows raise it — making
        # continuous verification the way back to a re-verifiable session.
        user_ref = current_user.username
        stored = store.get(user_ref)
        if stored is None:
            raise ProfileNotFoundError(user_ref)
        window = payload.model_dump(mode="python")
        result = service.verify_stored(user_ref, stored, window)
        now = utc_now()
        if result.decision == "VERIFIED":
            session_state = session_store.mark_verified(
                user_ref, current_user.session_id, now=now
            )
        else:
            session_state = session_store.mark_suspicious(
                user_ref, current_user.session_id, now=now
            )
        return ContinuousVerificationResponse(
            decision=result.decision,
            distance=result.distance,
            threshold=result.threshold,
            session_state=session_state.state,
            consecutive_suspicious=session_state.consecutive_suspicious,
            last_verified_at=session_state.last_verified_at,
        )

    @api.get(
        "/continuous-verification/state",
        response_model=SessionStateResponse,
        summary="Read the authenticated session's behavioral verification state",
        operation_id="get_continuous_session_state",
    )
    def _get_continuous_session_state(
        current_user: CurrentUser = Depends(get_current_user),  # noqa: B008
        session_store: SessionStateRepository = Depends(get_session_state_store),  # noqa: B008
    ) -> SessionStateResponse:
        """Read-only view of the session's server-authoritative state.

        Lets the frontend restore its UI truth after a refresh/login from the
        server instead of client storage. Aggregate flags/timestamps only.
        """
        state = session_store.get_or_create(
            current_user.username, current_user.session_id
        )
        return SessionStateResponse(
            session_state=state.state,
            consecutive_suspicious=state.consecutive_suspicious,
            last_verified_at=state.last_verified_at,
        )

    return api


__all__ = ["build_versioned_router"]