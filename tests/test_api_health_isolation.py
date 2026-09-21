"""
Phase 9B — health isolation from the ML stack.

The health endpoints must stay lightweight: they never load the checkpoint,
artifacts or the ML service, and they keep returning 200 even when every ML
artifact is missing or the ML service is unavailable.
"""

import os
import sys

from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.dependencies import get_ml_service
from backend.app.main import create_app
from backend.app.services.ml_service import (
    BehavioralMLService,
    MLServiceUnavailable,
    ModelLoadError,
)
from tests.auth_testing import set_current_user

_MISSING = "/nonexistent/{}".format


def _raw_session(session_id, *, kb=3, mouse=3):
    t = 0.0
    keyboard_events = []
    for i in range(kb):
        keyboard_events.append({"event_type": "keyboard", "event": "keydown", "timestamp": t})
        t += 50.0
        keyboard_events.append({"event_type": "keyboard", "event": "keyup", "timestamp": t})
        t += 100.0
    x = y = mt = 0.0
    mouse_events = []
    for i in range(mouse):
        x += 4.0 + i
        y += 2.0
        mt += 16.0
        mouse_events.append(
            {"event_type": "mouse", "event": "mousemove", "x": x, "y": y, "timestamp": mt}
        )
    return {
        "session_id": session_id,
        "started_at": "2025-01-01T00:00:00Z",
        "ended_at": "2025-01-01T00:00:05Z",
        "timestamp_source": "test",
        "keyboard_events": keyboard_events,
        "mouse_events": mouse_events,
    }


def _broken_settings():
    return Settings(
        environment="test",
        debug=False,
        checkpoint_path=_MISSING("checkpoint.pt"),
        preprocessing_artifact_path=_MISSING("preprocessing.json"),
        verification_config_path=_MISSING("config.json"),
    )


class TestHealthIndependentOfArtifacts:
    def test_health_ok_with_missing_artifacts(self):
        app = create_app(_broken_settings())
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
            assert client.get("/health").json() == {"status": "ok"}
            assert client.get("/api/v1/health").json() == {"status": "ok"}

    def test_ml_endpoint_fails_clearly_while_health_is_ok(self):
        app = create_app(_broken_settings())
        set_current_user(app, username="u1")
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
            session = _raw_session("u_s1")
            response = client.post(
                "/api/v1/enrollment",
                json={"user_ref": "u1", "sessions": [session]},
            )
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "model_load_error"
            assert client.get("/health").status_code == 200

    def test_no_ml_modules_imported_when_serving_health(self):
        before = set(sys.modules)
        app = create_app(Settings(environment="test", debug=False))
        with TestClient(app) as client:
            client.get("/health")
            client.get("/api/v1/health")
        imported = set(sys.modules) - before
        assert not any(name == "ml" or name.startswith("ml.") for name in imported)
        assert not any(name.startswith("torch") for name in imported)


class TestHealthWithUnavailableService:
    def test_health_ok_when_service_override_raises(self):
        app = create_app(Settings(environment="test"))

        class _BrokenService:
            def load(self):
                raise MLServiceUnavailable()

        app.dependency_overrides[get_ml_service] = lambda: _BrokenService()
        with TestClient(app) as client:
            assert client.get("/health").status_code == 200
            assert client.get("/api/v1/health").status_code == 200

    def test_ml_endpoint_surfaces_service_unavailable(self):
        app = create_app(Settings(environment="test"))
        service = BehavioralMLService(
            _broken_settings(),
            verifier_factory=lambda s: (_ for _ in ()).throw(ModelLoadError()),
            preprocessing_loader=lambda s: (None, None, {}),
            verification_config_loader=lambda s: {},
        )
        # Force the unavailability path by overriding the store too.
        app.dependency_overrides[get_ml_service] = lambda: service
        set_current_user(app, username="u1")
        with TestClient(app) as client:
            response = client.post(
                "/api/v1/enrollment",
                json={"user_ref": "u1", "sessions": [_raw_session("u_s1")]},
            )
            assert response.status_code == 503
            assert response.json()["error"]["code"] == "model_load_error"