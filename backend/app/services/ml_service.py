"""
Phase 9B — ML service layer (inference only).

Bridges the trained pipeline and the FastAPI routes:

    raw Phase 2 session
        -> Phase 3 preprocessing (validate + feature extraction)
        -> Phase 5 inference embeddings (Phase 7 checkpoint, read-only)
        -> Phase 8 enrollment centroid / L2 distance + calibrated threshold

Design rules
------------
* **Reusable abstractions only.** This module composes the existing Phase 3/5/8
  functions (``process_session``, ``FeatureScaler.load_state_dict``,
  ``enroll``, ``verify``, ``load_verifier``); it does NOT re-implement scaler
  fitting, embedding extraction, distance, enrollment or calibration.
* **No training; read-only model.** The checkpoint is loaded once into an
  evaluation-mode wrapper and never modified.
* **Train-only scalers.** The scalers come from the artifact
  ``models/behavioral_preprocessing.json`` (fitted on the deterministic
  training partition, seed 42). Incoming sessions are only *transformed*, never
  used to fit scalers.
* **Calibrated threshold.** The artifact ``models/verification_config.json``
  supplies the deployment threshold (development-partition calibration). The
  API never accepts a client-supplied threshold.
* **Lazy ML imports.** ``ml`` / ``torch`` module imports happen inside methods
  so that importing the app, the factory, or hitting ``/health`` never touches
  the ML stack. A missing/malformed artifact fails loudly on the ML endpoints
  with a structured error, while health stays green.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from backend.app.config import Settings
from backend.app.errors import AppError
from backend.app.services.profile_store import StoredProfile, utc_now

PREPROCESSING_ARTIFACT_NAME = "behavioral-preprocessing"
VERIFICATION_CONFIG_ARTIFACT_NAME = "verification-config"


class MLServiceError(AppError):
    """Base for ML-service failures (code + HTTP status carried by AppError)."""


class MLServiceUnavailable(MLServiceError):
    """The ML service cannot be initialised."""

    def __init__(self, message: str = "ML service is not available") -> None:
        super().__init__(code="ml_service_unavailable", message=message, status_code=503)


class ModelLoadError(MLServiceError):
    """The Phase 7 checkpoint could not be loaded."""

    def __init__(self, message: str = "the ML model could not be loaded") -> None:
        super().__init__(code="model_load_error", message=message, status_code=503)


class PreprocessingArtifactError(MLServiceError):
    """The train-only preprocessing artifact is missing or malformed."""

    def __init__(
        self, message: str = "the preprocessing artifact is missing or malformed"
    ) -> None:
        super().__init__(
            code="preprocessing_artifact_error", message=message, status_code=503
        )


class VerificationConfigError(MLServiceError):
    """The verification calibration artifact is missing or malformed."""

    def __init__(
        self, message: str = "the verification config artifact is missing or malformed"
    ) -> None:
        super().__init__(
            code="verification_config_error", message=message, status_code=503
        )


class InferenceError(MLServiceError):
    """An unexpected failure while running inference."""

    def __init__(self, message: str = "inference failed") -> None:
        super().__init__(code="inference_error", message=message, status_code=500)


class InvalidSessionError(MLServiceError):
    """The submitted session fails validation or has no usable behaviour."""

    def __init__(
        self, message: str = "session is invalid or has no usable behavioural content"
    ) -> None:
        super().__init__(code="invalid_session", message=message, status_code=422)


# ---------------------------------------------------------------------------
# Default seam implementations (lazy ml imports)
# ---------------------------------------------------------------------------


def default_session_processor(session: Mapping[str, Any]) -> Dict[str, Any]:
    """Phase 3: validate + preprocess one raw Phase 2 session."""
    from ml.preprocessing import process_session  # lazy

    return process_session(dict(session))


def default_verifier_factory(settings: Settings) -> Any:
    """Phase 8: load the Phase 7 checkpoint into a read-only wrapper."""
    from ml.evaluation import load_verifier  # lazy

    return load_verifier(settings.checkpoint_path)


def default_preprocessing_loader(settings: Settings) -> Tuple[Any, Any, Dict[str, Any]]:
    """Load train-only scalers + metadata from the preprocessing artifact."""
    from ml.preprocessing import FeatureScaler  # lazy

    path = settings.preprocessing_artifact_path
    if not os.path.isfile(path):
        raise PreprocessingArtifactError()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        raise PreprocessingArtifactError() from None
    if payload.get("artifact") != PREPROCESSING_ARTIFACT_NAME:
        raise PreprocessingArtifactError()
    keyboard_state = payload.get("keyboard_scaler")
    mouse_state = payload.get("mouse_scaler")
    if not isinstance(keyboard_state, dict) or not isinstance(mouse_state, dict):
        raise PreprocessingArtifactError()
    try:
        keyboard_scaler = FeatureScaler.load_state_dict(keyboard_state)
        mouse_scaler = FeatureScaler.load_state_dict(mouse_state)
    except Exception:  # noqa: BLE001 - malformed artifact -> structured error
        raise PreprocessingArtifactError() from None
    return keyboard_scaler, mouse_scaler, payload


def default_verification_config_loader(settings: Settings) -> Dict[str, Any]:
    """Load the calibrated threshold + metadata from the verification artifact."""
    path = settings.verification_config_path
    if not os.path.isfile(path):
        raise VerificationConfigError()
    try:
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, ValueError):
        raise VerificationConfigError() from None
    if payload.get("artifact") != VERIFICATION_CONFIG_ARTIFACT_NAME:
        raise VerificationConfigError()
    calibration = payload.get("calibration")
    if not isinstance(calibration, dict):
        raise VerificationConfigError()
    threshold = calibration.get("threshold")
    if not isinstance(threshold, (int, float)) or isinstance(threshold, bool):
        raise VerificationConfigError()
    if threshold < 0:
        raise VerificationConfigError()
    return payload


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------


class BehavioralMLService:
    """Inference-only bridge between the trained pipeline and the API."""

    def __init__(
        self,
        settings: Settings,
        *,
        verifier_factory: Callable[[Settings], Any] = default_verifier_factory,
        preprocessing_loader: Callable[[Settings], Tuple[Any, Any, Dict[str, Any]]] = (
            default_preprocessing_loader
        ),
        verification_config_loader: Callable[[Settings], Dict[str, Any]] = (
            default_verification_config_loader
        ),
        session_processor: Callable[[Mapping[str, Any]], Dict[str, Any]] = (
            default_session_processor
        ),
    ) -> None:
        if not isinstance(settings, Settings):
            raise TypeError("settings must be a Settings")
        self._settings = settings
        self._verifier_factory = verifier_factory
        self._preprocessing_loader = preprocessing_loader
        self._verification_config_loader = verification_config_loader
        self._session_processor = session_processor

        self._loaded = False
        self._verifier: Any = None
        self._keyboard_scaler: Any = None
        self._mouse_scaler: Any = None
        self._threshold: float = 0.0
        self._preprocessing_info: Dict[str, Any] = {}
        self._verification_info: Dict[str, Any] = {}

    @classmethod
    def from_settings(cls, settings: Settings) -> "BehavioralMLService":
        return cls(settings)

    # -- lifecycle ----------------------------------------------------------

    def load(self) -> "BehavioralMLService":
        """Lazily load checkpoint + artifacts once; idempotent."""
        if self._loaded:
            return self

        try:
            verifier = self._verifier_factory(self._settings)
        except MLServiceError:
            raise
        except Exception:  # noqa: BLE001 - any load failure is a model-load error
            raise ModelLoadError() from None

        try:
            keyboard_scaler, mouse_scaler, prep_info = self._preprocessing_loader(
                self._settings
            )
        except MLServiceError:
            raise
        except Exception:  # noqa: BLE001
            raise PreprocessingArtifactError() from None

        try:
            config = self._verification_config_loader(self._settings)
        except MLServiceError:
            raise
        except Exception:  # noqa: BLE001
            raise VerificationConfigError() from None

        self._verifier = verifier
        self._keyboard_scaler = keyboard_scaler
        self._mouse_scaler = mouse_scaler
        self._threshold = float(config["calibration"]["threshold"])
        self._preprocessing_info = dict(prep_info)
        self._verification_info = dict(config)
        self._loaded = True
        return self

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def threshold(self) -> float:
        return self.load()._threshold

    @property
    def embedding_dim(self) -> int:
        return int(self.load()._verifier.embedding_dim)

    @property
    def checkpoint_id(self) -> str:
        self.load()
        return str(self._verifier.checkpoint_id)

    # -- raw-session preprocessing ------------------------------------------

    def _preprocess(self, raw_session: Mapping[str, Any]) -> Dict[str, Any]:
        from ml.encoder.input import has_behavior  # lazy
        from ml.preprocessing import SessionValidationError  # lazy

        try:
            processed = self._session_processor(raw_session)
        except SessionValidationError:
            raise InvalidSessionError() from None
        if not has_behavior(processed):
            raise InvalidSessionError()
        return processed

    # -- enrollment ----------------------------------------------------------

    def enroll_sessions(
        self, user_ref: str, raw_sessions: Sequence[Mapping[str, Any]]
    ) -> Any:
        """Enroll a user from raw sessions; returns a Phase 8 ``EnrollmentProfile``."""
        from ml.evaluation import enroll  # lazy

        self.load()
        if not isinstance(raw_sessions, (list, tuple)) or not raw_sessions:
            raise InvalidSessionError("enrollment requires at least one session")
        processed = [self._preprocess(session) for session in raw_sessions]
        try:
            return enroll(
                processed,
                self._verifier,
                keyboard_scaler=self._keyboard_scaler,
                mouse_scaler=self._mouse_scaler,
                user_ref=user_ref,
            )
        except MLServiceError:
            raise
        except AppError:
            raise
        except Exception:  # noqa: BLE001
            raise InferenceError("enrollment inference failed") from None

    def enroll_to_store(
        self, user_ref: str, raw_sessions: Sequence[Mapping[str, Any]]
    ) -> StoredProfile:
        """Enroll and serialize the summary into a :class:`StoredProfile`."""
        profile = self.enroll_sessions(user_ref, raw_sessions)
        return _stored_from_enrollment(profile)

    # -- verification --------------------------------------------------------

    def verify_stored(
        self, user_ref: str, stored: StoredProfile, raw_session: Mapping[str, Any]
    ) -> Any:
        """Match a raw probe session against a stored profile."""
        from ml.evaluation import EnrollmentProfile, verify  # lazy

        self.load()
        probe = self._preprocess(raw_session)
        profile = EnrollmentProfile(
            user_id=stored.user_ref,
            centroid=_tensor_from_centroid(stored.centroid),
            n_sessions=stored.session_count,
            embedding_dim=stored.embedding_dim,
        )
        try:
            return verify(
                profile,
                probe,
                self._verifier,
                self._threshold,
                keyboard_scaler=self._keyboard_scaler,
                mouse_scaler=self._mouse_scaler,
            )
        except MLServiceError:
            raise
        except AppError:
            raise
        except Exception:  # noqa: BLE001
            raise InferenceError("verification inference failed") from None


def _stored_from_enrollment(profile: Any) -> StoredProfile:
    """Convert a Phase 8 ``EnrollmentProfile`` into a store-safe summary."""
    import torch  # lazy

    values = profile.centroid.detach().cpu().reshape(-1).tolist()
    created = utc_now()
    return StoredProfile(
        user_ref=profile.user_id,
        centroid=tuple(float(v) for v in values),
        embedding_dim=int(profile.embedding_dim),
        session_count=int(profile.n_sessions),
        created_at=created,
        updated_at=created,
    )


def _tensor_from_centroid(centroid: Sequence[float]) -> Any:
    """Reconstruct the centroid tensor from the stored float tuple."""
    import torch  # lazy

    return torch.tensor(list(centroid), dtype=torch.float32)


__all__ = [
    "BehavioralMLService",
    "InferenceError",
    "InvalidSessionError",
    "MLServiceError",
    "MLServiceUnavailable",
    "ModelLoadError",
    "PreprocessingArtifactError",
    "VerificationConfigError",
    "default_preprocessing_loader",
    "default_session_processor",
    "default_verification_config_loader",
    "default_verifier_factory",
]