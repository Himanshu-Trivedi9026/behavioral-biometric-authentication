"""
Phase 9B — end-to-end integration through the REAL ML stack.

Drives the full journey over the actual artifacts and Phase 7 checkpoint:

    raw Phase 2 session -> schema validation -> Phase 3 preprocessing
    -> Phase 5 inference embedding -> Phase 8 centroid enrollment
    -> Phase 6 L2 distance -> calibrated threshold -> decision

The test is fully deterministic (identical raw session -> identical embedding,
so self-verification is exactly distance 0 -> VERIFIED) and also exercises the
persisted synthetic dataset by converting real entries back into raw sessions.

Each request runs as the authenticated user matching its ``user_ref`` (Phase
11); the token mechanics are covered in the auth test modules.

Skipped when the checkpoint or the Phase 9B inference artifacts are missing.
The tests never modify the checkpoint or the dataset.
"""

import json
import os

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.main import create_app
from dataset import load_dataset
from tests.auth_testing import set_current_user

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(_REPO, "models", "siamese_behavioral_encoder.pt")
PREPROCESSING_ARTIFACT = os.path.join(_REPO, "models", "behavioral_preprocessing.json")
VERIFICATION_CONFIG = os.path.join(_REPO, "models", "verification_config.json")

_READY = all(
    os.path.exists(path)
    for path in (CHECKPOINT_PATH, PREPROCESSING_ARTIFACT, VERIFICATION_CONFIG)
)

pytestmark = pytest.mark.skipif(
    not _READY,
    reason="Phase 9B inference artifacts/checkpoint missing; run scripts/export_inference_artifacts.py",
)

ENROLLMENT = "/api/v1/enrollment"
VERIFICATION = "/api/v1/verification"


@pytest.fixture(scope="function")
def app():
    return create_app(Settings(environment="test", debug=False))


@pytest.fixture(scope="function")
def client(app):
    # The ML journey always runs as the authenticated user matching the
    # request-body ``user_ref`` (identity -> authorization, Phase 11).
    with TestClient(app) as test_client:
        yield test_client, app


def _manual_raw(session_id, *, hold=55.0, flight=120.0, dx=6.0, dy=3.0, events=6):
    t = 0.0
    keyboard_events = []
    for i in range(events):
        keyboard_events.append({"event_type": "keyboard", "event": "keydown", "timestamp": t})
        t += hold
        keyboard_events.append({"event_type": "keyboard", "event": "keyup", "timestamp": t})
        t += flight
    x = y = mt = 0.0
    mouse_events = []
    for i in range(events):
        x += dx + i
        y += dy
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


def _entry_to_raw(entry):
    t = 0.0
    keyboard_events = []
    for sample in entry["keyboard_sequence"]:
        keyboard_events.append({"event_type": "keyboard", "event": "keydown", "timestamp": t})
        t += sample["hold_time"]
        keyboard_events.append({"event_type": "keyboard", "event": "keyup", "timestamp": t})
        t += max(sample["flight_time"], 0.0)
    x = y = mt = 0.0
    mouse_events = []
    for sample in entry["mouse_sequence"]:
        x += sample["dx"]
        y += sample["dy"]
        mt += sample["dt"]
        mouse_events.append(
            {"event_type": "mouse", "event": "mousemove", "x": x, "y": y, "timestamp": mt}
        )
    return {
        "session_id": entry["session_id"],
        "started_at": "2025-01-01T00:00:00Z",
        "ended_at": "2025-01-01T00:00:05Z",
        "timestamp_source": "test",
        "keyboard_events": keyboard_events,
        "mouse_events": mouse_events,
    }


class TestManualJourney:
    def test_full_pipeline_self_and_impostor(self, client):
        c, app = client
        set_current_user(app, username="e2e_user_a")
        enroll = c.post(
            ENROLLMENT,
            json={
                "user_ref": "e2e_user_a",
                "sessions": [_manual_raw("e2e_a_s1")],
            },
        )
        assert enroll.status_code == 201
        assert enroll.json()["embedding_dimension"] == 128
        assert enroll.json()["session_count"] == 1

        self_result = c.post(
            VERIFICATION,
            json={"user_ref": "e2e_user_a", "session": _manual_raw("e2e_a_s1")},
        )
        assert self_result.status_code == 200
        assert self_result.json()["decision"] == "VERIFIED"
        assert self_result.json()["distance"] < 1e-3

        impostor = c.post(
            VERIFICATION,
            json={
                "user_ref": "e2e_user_a",
                "session": _manual_raw("e2e_b_s1", hold=90.0, flight=220.0, dx=14.0, dy=9.0),
            },
        )
        assert impostor.status_code == 200
        body = impostor.json()
        assert body["distance"] > body["threshold"]
        assert body["decision"] == "SUSPICIOUS"

    def test_multi_session_enrollment(self, client):
        c, app = client
        set_current_user(app, username="e2e_two")
        response = c.post(
            ENROLLMENT,
            json={
                "user_ref": "e2e_two",
                "sessions": [_manual_raw("two_s1"), _manual_raw("two_s2")],
            },
        )
        assert response.status_code == 201
        assert response.json()["session_count"] == 2

    def test_duplicate_enrollment_conflict(self, client):
        c, app = client
        set_current_user(app, username="e2e_dup")
        payload = {"user_ref": "e2e_dup", "sessions": [_manual_raw("dup_s1")]}
        assert c.post(ENROLLMENT, json=payload).status_code == 201
        response = c.post(ENROLLMENT, json=payload)
        assert response.status_code == 409
        assert response.json()["error"]["code"] == "profile_exists"

    def test_unknown_profile_not_found(self, client):
        c, app = client
        set_current_user(app, username="e2e_ghost")
        response = c.post(
            VERIFICATION,
            json={"user_ref": "e2e_ghost", "session": _manual_raw("ghost_s1")},
        )
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "profile_not_found"

    def test_malformed_payload_validation_error(self, client):
        c, app = client
        set_current_user(app, username="e2e_bad")
        session = _manual_raw("bad_s1")
        session["keyboard_events"][0]["timestamp"] = float("inf")
        body = json.dumps({"user_ref": "e2e_bad", "sessions": [session]})
        response = c.post(
            ENROLLMENT, content=body, headers={"content-type": "application/json"}
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"


class TestDatasetJourney:
    def test_enroll_and_verify_from_persisted_dataset(self, client):
        c, app = client
        entries = load_dataset(os.path.join(_REPO, "data", "datasets", "dataset_synthetic.json"))
        user = "user_005"
        own = [e for e in entries if e["user_id"] == user]
        other = next(e for e in entries if e["user_id"] != user)
        set_current_user(app, username=user)

        enroll = c.post(
            ENROLLMENT,
            json={
                "user_ref": user,
                "sessions": [_entry_to_raw(own[0]), _entry_to_raw(own[1])],
            },
        )
        assert enroll.status_code == 201
        assert enroll.json()["embedding_dimension"] == 128
        assert enroll.json()["session_count"] == 2

        genuine = c.post(
            VERIFICATION, json={"user_ref": user, "session": _entry_to_raw(own[2])}
        ).json()
        impostor = c.post(
            VERIFICATION, json={"user_ref": user, "session": _entry_to_raw(other)}
        ).json()

        assert genuine["decision"] == (
            "VERIFIED" if genuine["distance"] <= genuine["threshold"] else "SUSPICIOUS"
        )
        assert impostor["decision"] == (
            "VERIFIED" if impostor["distance"] <= impostor["threshold"] else "SUSPICIOUS"
        )
        assert genuine["distance"] < impostor["distance"]
        assert impostor["distance"] > impostor["threshold"]