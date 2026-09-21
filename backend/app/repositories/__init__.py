"""Phase 10/11/14B — persistence repositories (profiles, users, session state)."""

from backend.app.repositories.errors import (
    DatabaseOperationError,
    DatabaseUnavailableError,
)
from backend.app.repositories.memory_session_state_store import (
    InMemorySessionStateStore,
)
from backend.app.repositories.memory_user_repository import InMemoryUserRepository
from backend.app.repositories.postgres_profile_repository import (
    PostgresProfileRepository,
    is_unique_violation,
    profile_to_values,
    raise_for_database_error,
    row_to_metadata,
    row_to_profile,
)
from backend.app.repositories.postgres_session_state_repository import (
    PostgresSessionStateRepository,
    row_to_session_state,
)
from backend.app.repositories.postgres_user_repository import (
    PostgresUserRepository,
    row_to_user,
    user_to_values,
)
from backend.app.repositories.profile_repository import (
    ProfileExistsError,
    ProfileNotFoundError,
    ProfileRepository,
    StoredProfile,
)
from backend.app.repositories.session_state_repository import (
    SESSION_STATE_REVERIFICATION_REQUIRED,
    SESSION_STATE_VERIFIED,
    VALID_SESSION_STATES,
    SessionStateRepository,
    SessionVerificationState,
    default_session_state,
)
from backend.app.repositories.user_repository import (
    UserRecord,
    UserRepository,
    UsernameExistsError,
)

__all__ = [
    "DatabaseOperationError",
    "DatabaseUnavailableError",
    "InMemorySessionStateStore",
    "InMemoryUserRepository",
    "PostgresProfileRepository",
    "PostgresSessionStateRepository",
    "PostgresUserRepository",
    "ProfileExistsError",
    "ProfileNotFoundError",
    "ProfileRepository",
    "SESSION_STATE_REVERIFICATION_REQUIRED",
    "SESSION_STATE_VERIFIED",
    "VALID_SESSION_STATES",
    "SessionStateRepository",
    "SessionVerificationState",
    "StoredProfile",
    "UserRecord",
    "UserRepository",
    "UsernameExistsError",
    "default_session_state",
    "is_unique_violation",
    "profile_to_values",
    "raise_for_database_error",
    "row_to_metadata",
    "row_to_profile",
    "row_to_session_state",
    "row_to_user",
    "user_to_values",
]
