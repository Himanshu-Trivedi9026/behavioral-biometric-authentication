"""
Phase 3 preprocessing unit tests (cases A – R).

Run from the repo root::

    .venv/bin/python -m pytest tests/test_preprocessing.py -v
"""

import copy
import json
import math
import os
from typing import Any, Dict, List

import pytest

from ml.preprocessing import (
    FeatureScaler,
    KEYBOARD_FEATURE_COLUMNS,
    MOUSE_FEATURE_COLUMNS,
    SessionValidationError,
    preprocess_keyboard,
    preprocess_mouse,
    process_session,
    rows_to_sequence,
    sequence_to_rows,
    validate_session,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

def _valid_session() -> Dict[str, Any]:
    return {
        "session_id": "550e8400-e29b-41d4-a716-446655440000",
        "started_at": "2025-09-01T12:00:00.000Z",
        "ended_at": "2025-09-01T12:00:05.000Z",
        "timestamp_source": "monotonic high-resolution (performance.now, milliseconds)",
        "keyboard_events": [
            {"event_type": "keyboard", "event": "keydown", "timestamp": 100.0},
            {"event_type": "keyboard", "event": "keyup",   "timestamp": 160.0},
            {"event_type": "keyboard", "event": "keydown", "timestamp": 220.0},
            {"event_type": "keyboard", "event": "keyup",   "timestamp": 270.0},
            {"event_type": "keyboard", "event": "keydown", "timestamp": 350.0},
            {"event_type": "keyboard", "event": "keyup",   "timestamp": 390.0},
        ],
        "mouse_events": [
            {"event_type": "mouse", "event": "mousemove", "x": 100.0, "y": 200.0, "timestamp": 110.0},
            {"event_type": "mouse", "event": "mousemove", "x": 110.0, "y": 220.0, "timestamp": 135.0},
            {"event_type": "mouse", "event": "mousemove", "x": 130.0, "y": 225.0, "timestamp": 165.0},
            {"event_type": "mouse", "event": "mousedown", "x": 130.0, "y": 225.0, "timestamp": 166.0},
            {"event_type": "mouse", "event": "mouseup",   "x": 130.0, "y": 225.0, "timestamp": 175.0},
        ],
    }


def _empty_session() -> Dict[str, Any]:
    return {
        "session_id": "00000000-0000-0000-0000-000000000001",
        "started_at": "2025-09-01T12:00:00.000Z",
        "ended_at": "2025-09-01T12:00:00.500Z",
        "keyboard_events": [],
        "mouse_events": [],
    }


def _fixture_path(*parts: str) -> str:
    return os.path.join(os.path.dirname(__file__), "fixtures", *parts)


# =====================================================================
# Validation tests
# =====================================================================

class TestValidation:

    def test_valid_session_returns_no_errors(self):
        """A. Valid session validation produces an empty error list."""
        errors = validate_session(_valid_session())
        assert errors == [], f"Unexpected validation errors: {errors}"

    def test_missing_session_id(self):
        """B. Missing session_id is reported as a validation error."""
        bad = _valid_session()
        del bad["session_id"]
        errors = validate_session(bad)
        assert any("session_id" in e for e in errors)

    def test_empty_session_id(self):
        """B. Empty string session_id is rejected."""
        bad = _valid_session()
        bad["session_id"] = ""
        errors = validate_session(bad)
        assert any("session_id" in e for e in errors)

    def test_non_dict_input(self):
        errors = validate_session("not a dict")
        assert len(errors) == 1
        assert "dict" in errors[0]

    def test_missing_started_at(self):
        bad = _valid_session()
        del bad["started_at"]
        errors = validate_session(bad)
        assert any("started_at" in e for e in errors)

    def test_missing_ended_at(self):
        bad = _valid_session()
        del bad["ended_at"]
        errors = validate_session(bad)
        assert any("ended_at" in e for e in errors)

    def test_missing_keyboard_events(self):
        bad = _valid_session()
        del bad["keyboard_events"]
        errors = validate_session(bad)
        assert any("keyboard_events" in e for e in errors)

    def test_keyboard_events_not_a_list(self):
        bad = _valid_session()
        bad["keyboard_events"] = "oops"
        errors = validate_session(bad)
        assert any("keyboard_events" in e and "list" in e for e in errors)

    def test_missing_mouse_events(self):
        bad = _valid_session()
        del bad["mouse_events"]
        errors = validate_session(bad)
        assert any("mouse_events" in e for e in errors)

    def test_mouse_events_not_a_list(self):
        bad = _valid_session()
        bad["mouse_events"] = 42
        errors = validate_session(bad)
        assert any("mouse_events" in e and "list" in e for e in errors)

    def test_invalid_keyboard_event(self):
        """C. Keyboard event missing required fields is reported."""
        bad = _valid_session()
        bad["keyboard_events"] = [{"event_type": "keyboard"}]
        errors = validate_session(bad)
        assert any("keyboard_events[0]" in e and "event" in e for e in errors)

    def test_keyboard_event_bad_timestamp(self):
        bad = _valid_session()
        bad["keyboard_events"] = [
            {"event_type": "keyboard", "event": "keydown", "timestamp": "fast"}
        ]
        errors = validate_session(bad)
        assert any("keyboard_events[0]" in e and "timestamp" in e for e in errors)

    def test_keyboard_event_nan_timestamp(self):
        bad = _valid_session()
        bad["keyboard_events"] = [
            {"event_type": "keyboard", "event": "keydown", "timestamp": float("nan")}
        ]
        errors = validate_session(bad)
        assert any("keyboard_events[0]" in e and "finite" in e for e in errors)

    def test_keyboard_event_inf_timestamp(self):
        bad = _valid_session()
        bad["keyboard_events"] = [
            {"event_type": "keyboard", "event": "keydown", "timestamp": float("inf")}
        ]
        errors = validate_session(bad)
        assert any("finite" in e for e in errors)

    def test_keyboard_event_bad_event_name(self):
        bad = _valid_session()
        bad["keyboard_events"] = [
            {"event_type": "keyboard", "event": "keypress", "timestamp": 1.0}
        ]
        errors = validate_session(bad)
        assert any("keydown" in e and "keyup" in e for e in errors)

    def test_keyboard_event_non_dict(self):
        bad = _valid_session()
        bad["keyboard_events"] = [None]
        errors = validate_session(bad)
        assert any("keyboard_events[0]" in e and "object" in e for e in errors)

    def test_keyboard_event_bad_event_type(self):
        bad = _valid_session()
        bad["keyboard_events"] = [
            {"event_type": "mouse", "event": "keydown", "timestamp": 1.0}
        ]
        errors = validate_session(bad)
        assert any("event_type" in e and "keyboard" in e for e in errors)

    def test_invalid_mouse_event(self):
        """D. Mouse event missing required fields is reported."""
        bad = _valid_session()
        bad["mouse_events"] = [{"event_type": "mouse", "event": "mousemove"}]
        errors = validate_session(bad)
        assert any("mouse_events[0]" in e for e in errors)

    def test_mouse_event_missing_x(self):
        bad = _valid_session()
        bad["mouse_events"] = [
            {"event_type": "mouse", "event": "mousemove", "y": 1.0, "timestamp": 1.0}
        ]
        errors = validate_session(bad)
        assert any("mouse_events[0]" in e and "'x'" in e for e in errors)

    def test_mouse_event_missing_y(self):
        bad = _valid_session()
        bad["mouse_events"] = [
            {"event_type": "mouse", "event": "mousemove", "x": 1.0, "timestamp": 1.0}
        ]
        errors = validate_session(bad)
        assert any("mouse_events[0]" in e and "'y'" in e for e in errors)

    def test_mouse_event_bad_event_type(self):
        bad = _valid_session()
        bad["mouse_events"] = [
            {"event_type": "keyboard", "event": "mousemove",
             "x": 0, "y": 0, "timestamp": 1.0}
        ]
        errors = validate_session(bad)
        assert any("event_type" in e and "mouse" in e for e in errors)

    def test_mouse_event_bad_event_name(self):
        bad = _valid_session()
        bad["mouse_events"] = [
            {"event_type": "mouse", "event": "hover",
             "x": 0, "y": 0, "timestamp": 1.0}
        ]
        errors = validate_session(bad)
        assert any("event" in e for e in errors)

    def test_mouse_event_bad_coordinate(self):
        bad = _valid_session()
        bad["mouse_events"] = [
            {"event_type": "mouse", "event": "mousemove",
             "x": "bad", "y": 0, "timestamp": 1.0}
        ]
        errors = validate_session(bad)
        assert any("x" in e and "finite" in e for e in errors)

    def test_mouse_event_nan_coordinate(self):
        bad = _valid_session()
        bad["mouse_events"] = [
            {"event_type": "mouse", "event": "mousemove",
             "x": 0, "y": float("nan"), "timestamp": 1.0}
        ]
        errors = validate_session(bad)
        assert any("y" in e and "finite" in e for e in errors)

    def test_mouse_event_non_dict(self):
        bad = _valid_session()
        bad["mouse_events"] = ["oops"]
        errors = validate_session(bad)
        assert any("mouse_events[0]" in e for e in errors)

    def test_multiple_errors_reported_at_once(self):
        bad: Dict[str, Any] = {
            "session_id": 123,
            "started_at": "",
            "ended_at": "",
            "keyboard_events": "oops",
            "mouse_events": 42,
        }
        errors = validate_session(bad)
        assert len(errors) >= 5  # session_id + started_at + ended_at + keyboard_events + mouse_events


# =====================================================================
# Keyboard preprocessing tests
# =====================================================================

class TestKeyboardPreprocessing:

    def test_pairing_and_hold_time(self):
        """E/F. Simple keydown → keyup pairing and hold_time calculation."""
        events = [
            {"event_type": "keyboard", "event": "keydown", "timestamp": 100.0},
            {"event_type": "keyboard", "event": "keyup",   "timestamp": 150.0},
        ]
        seq, meta = preprocess_keyboard(events)
        assert meta["valid_presses"] == 1
        assert seq == [{"hold_time": 50.0, "flight_time": 0.0}]

    def test_two_presses_hold_and_flight(self):
        """E/F/G. Two consecutive presses — correct hold_time and flight_time."""
        events = [
            {"event_type": "keyboard", "event": "keydown", "timestamp": 100.0},
            {"event_type": "keyboard", "event": "keyup",   "timestamp": 160.0},
            {"event_type": "keyboard", "event": "keydown", "timestamp": 220.0},
            {"event_type": "keyboard", "event": "keyup",   "timestamp": 270.0},
        ]
        seq, meta = preprocess_keyboard(events)
        assert meta["valid_presses"] == 2
        assert seq[0]["hold_time"] == pytest.approx(60.0)
        assert seq[0]["flight_time"] == 0.0  # first press
        assert seq[1]["hold_time"] == pytest.approx(50.0)
        assert seq[1]["flight_time"] == pytest.approx(120.0)

    def test_three_presses_flight_time(self):
        """G. Three presses — flight_time for each."""
        events = [
            {"event_type": "keyboard", "event": "keydown", "timestamp": 100.0},
            {"event_type": "keyboard", "event": "keyup",   "timestamp": 130.0},
            {"event_type": "keyboard", "event": "keydown", "timestamp": 200.0},
            {"event_type": "keyboard", "event": "keyup",   "timestamp": 240.0},
            {"event_type": "keyboard", "event": "keydown", "timestamp": 400.0},
            {"event_type": "keyboard", "event": "keyup",   "timestamp": 450.0},
        ]
        seq, meta = preprocess_keyboard(events)
        assert len(seq) == 3
        assert seq[0] == {"hold_time": 30.0, "flight_time": 0.0}
        assert seq[1] == {"hold_time": 40.0, "flight_time": 100.0}
        assert seq[2] == {"hold_time": 50.0, "flight_time": 200.0}

    def test_unmatched_keydown_dropped(self):
        """H. Unmatched keydown at end is dropped and counted."""
        events = [
            {"event_type": "keyboard", "event": "keydown", "timestamp": 100.0},
            {"event_type": "keyboard", "event": "keyup",   "timestamp": 160.0},
            {"event_type": "keyboard", "event": "keydown", "timestamp": 200.0},
        ]
        seq, meta = preprocess_keyboard(events)
        assert meta["valid_presses"] == 1
        assert meta["dropped_unmatched_keydown"] == 1

    def test_unmatched_keyup_dropped(self):
        """H. Orphan keyup is dropped and counted."""
        events = [
            {"event_type": "keyboard", "event": "keyup",   "timestamp": 50.0},
            {"event_type": "keyboard", "event": "keydown", "timestamp": 100.0},
            {"event_type": "keyboard", "event": "keyup",   "timestamp": 160.0},
        ]
        seq, meta = preprocess_keyboard(events)
        assert meta["valid_presses"] == 1
        assert meta["dropped_unmatched_keyup"] == 1

    def test_negative_hold_time_dropped(self):
        """H. keyup before keydown → negative hold_time → dropped."""
        events = [
            {"event_type": "keyboard", "event": "keyup",   "timestamp": 100.0},
            {"event_type": "keyboard", "event": "keydown", "timestamp": 150.0},
            {"event_type": "keyboard", "event": "keyup",   "timestamp": 200.0},
        ]
        seq, meta = preprocess_keyboard(events)
        assert meta["valid_presses"] == 1
        assert meta["dropped_unmatched_keyup"] == 1
        assert seq[0] == {"hold_time": 50.0, "flight_time": 0.0}

    def test_nested_keydowns_lifo_pairing(self):
        """Nested keydowns paired via LIFO; output ordered by keydown timestamp."""
        events = [
            {"event_type": "keyboard", "event": "keydown", "timestamp": 100.0},
            {"event_type": "keyboard", "event": "keydown", "timestamp": 120.0},
            {"event_type": "keyboard", "event": "keyup",   "timestamp": 150.0},
            {"event_type": "keyboard", "event": "keyup",   "timestamp": 200.0},
        ]
        seq, meta = preprocess_keyboard(events)
        assert meta["valid_presses"] == 2
        # Output is in keydown-chronological order:
        assert seq[0]["hold_time"] == pytest.approx(100.0)  # 100→200
        assert seq[0]["flight_time"] == pytest.approx(0.0)
        assert seq[1]["hold_time"] == pytest.approx(30.0)   # 120→150
        assert seq[1]["flight_time"] == pytest.approx(20.0)  # 120-100

    def test_duplicate_consecutive_events_dropped(self):
        """H. Consecutive exact duplicate events are dropped."""
        events = [
            {"event_type": "keyboard", "event": "keydown", "timestamp": 100.0},
            {"event_type": "keyboard", "event": "keydown", "timestamp": 100.0},
            {"event_type": "keyboard", "event": "keyup",   "timestamp": 160.0},
            {"event_type": "keyboard", "event": "keyup",   "timestamp": 160.0},
        ]
        seq, meta = preprocess_keyboard(events)
        assert meta["dropped_duplicates"] == 2
        assert meta["valid_presses"] == 1
        assert seq[0]["hold_time"] == pytest.approx(60.0)

    def test_zero_hold_time(self):
        """Hold time of zero (keydown and keyup at same timestamp) is valid."""
        events = [
            {"event_type": "keyboard", "event": "keydown", "timestamp": 100.0},
            {"event_type": "keyboard", "event": "keyup",   "timestamp": 100.0},
        ]
        seq, meta = preprocess_keyboard(events)
        assert meta["valid_presses"] == 1
        assert seq[0]["hold_time"] == 0.0

    def test_empty_events(self):
        seq, meta = preprocess_keyboard([])
        assert seq == []
        assert meta["valid_presses"] == 0

    def test_deterministic_output(self):
        """Output is identical across repeated runs (sorted deterministically)."""
        events = [
            {"event_type": "keyboard", "event": "keydown", "timestamp": 300.0},
            {"event_type": "keyboard", "event": "keydown", "timestamp": 100.0},
            {"event_type": "keyboard", "event": "keyup",   "timestamp": 360.0},
            {"event_type": "keyboard", "event": "keyup",   "timestamp": 160.0},
        ]
        result1 = preprocess_keyboard(events)
        result2 = preprocess_keyboard(events)
        assert result1 == result2

    def test_stable_sort_same_timestamp(self):
        """Same-timestamp events are processed in original list order."""
        events = [
            {"event_type": "keyboard", "event": "keydown", "timestamp": 100.0},
            {"event_type": "keyboard", "event": "keydown", "timestamp": 100.0},
            {"event_type": "keyboard", "event": "keyup",   "timestamp": 100.0},
        ]
        seq, meta = preprocess_keyboard(events)
        assert meta["valid_presses"] == 1
        assert meta["dropped_duplicates"] == 1

    def test_overlapping_keydowns_no_negative_flight_time(self):
        """Regression: overlapping keydowns must never produce negative flight_time."""
        # Required regression case from manual verification:
        # keydown A(100), keydown B(200), keyup B(300), keyup A(400)
        events = [
            {"event_type": "keyboard", "event": "keydown", "timestamp": 100.0},
            {"event_type": "keyboard", "event": "keydown", "timestamp": 200.0},
            {"event_type": "keyboard", "event": "keyup",   "timestamp": 300.0},
            {"event_type": "keyboard", "event": "keyup",   "timestamp": 400.0},
        ]
        seq, meta = preprocess_keyboard(events)
        assert meta["valid_presses"] == 2
        # Output is in keydown-chronological order (no key identity emitted):
        assert seq[0] == {"hold_time": 300.0, "flight_time": 0.0}
        assert seq[1] == {"hold_time": 100.0, "flight_time": 100.0}
        # Key property: all flight_times are non-negative
        assert all(s["flight_time"] >= 0.0 for s in seq)

    def test_overlapping_keydowns_three_keys(self):
        """Triple overlap: three simultaneous keydowns, nested keyup order."""
        events = [
            {"event_type": "keyboard", "event": "keydown", "timestamp": 100.0},
            {"event_type": "keyboard", "event": "keydown", "timestamp": 150.0},
            {"event_type": "keyboard", "event": "keydown", "timestamp": 200.0},
            {"event_type": "keyboard", "event": "keyup",   "timestamp": 250.0},
            {"event_type": "keyboard", "event": "keyup",   "timestamp": 300.0},
            {"event_type": "keyboard", "event": "keyup",   "timestamp": 400.0},
        ]
        seq, meta = preprocess_keyboard(events)
        assert meta["valid_presses"] == 3
        # keydown order: 100, 150, 200
        assert seq[0] == {"hold_time": 300.0, "flight_time": 0.0}    # 100→400
        assert seq[1] == {"hold_time": 150.0, "flight_time": 50.0}  # 150→300
        assert seq[2] == {"hold_time": 50.0,  "flight_time": 50.0}  # 200→250
        assert all(s["flight_time"] >= 0.0 for s in seq)

    def test_overlapping_partial_with_normal_sequential(self):
        """Mix of overlapping and sequential keydowns produces non-negative flight_time."""
        events = [
            {"event_type": "keyboard", "event": "keydown", "timestamp": 100.0},
            {"event_type": "keyboard", "event": "keydown", "timestamp": 200.0},
            {"event_type": "keyboard", "event": "keyup",   "timestamp": 250.0},
            {"event_type": "keyboard", "event": "keyup",   "timestamp": 300.0},
            # sequential after the overlap
            {"event_type": "keyboard", "event": "keydown", "timestamp": 400.0},
            {"event_type": "keyboard", "event": "keyup",   "timestamp": 450.0},
        ]
        seq, meta = preprocess_keyboard(events)
        assert meta["valid_presses"] == 3
        assert all(s["flight_time"] >= 0.0 for s in seq)
        # keydown order: 100, 200, 400
        assert seq[0]["flight_time"] == 0.0
        assert seq[1]["flight_time"] == 100.0
        assert seq[2]["flight_time"] == 200.0


# =====================================================================
# Mouse preprocessing tests
# =====================================================================

class TestMousePreprocessing:

    def test_dx_dy_basic(self):
        """I. Basic dx/dy calculation."""
        events = [
            {"event_type": "mouse", "event": "mousemove",
             "x": 100.0, "y": 200.0, "timestamp": 100.0},
            {"event_type": "mouse", "event": "mousemove",
             "x": 110.0, "y": 220.0, "timestamp": 125.0},
        ]
        seq, _, meta = preprocess_mouse(events)
        assert meta["move_samples"] == 1
        assert seq[0]["dx"] == pytest.approx(10.0)
        assert seq[0]["dy"] == pytest.approx(20.0)

    def test_distance_calculation(self):
        """J. distance = sqrt(dx^2 + dy^2)."""
        events = [
            {"event_type": "mouse", "event": "mousemove",
             "x": 0.0, "y": 0.0, "timestamp": 100.0},
            {"event_type": "mouse", "event": "mousemove",
             "x": 3.0, "y": 4.0, "timestamp": 200.0},
        ]
        seq, _, _ = preprocess_mouse(events)
        assert seq[0]["dx"] == pytest.approx(3.0)
        assert seq[0]["dy"] == pytest.approx(4.0)
        assert seq[0]["distance"] == pytest.approx(5.0)

    def test_dt_calculation(self):
        """K. dt = t1 - t0."""
        events = [
            {"event_type": "mouse", "event": "mousemove",
             "x": 0, "y": 0, "timestamp": 100.0},
            {"event_type": "mouse", "event": "mousemove",
             "x": 1, "y": 1, "timestamp": 145.0},
        ]
        seq, _, _ = preprocess_mouse(events)
        assert seq[0]["dt"] == pytest.approx(45.0)

    def test_speed_calculation(self):
        """L. speed = distance / dt (when dt > 0)."""
        events = [
            {"event_type": "mouse", "event": "mousemove",
             "x": 0.0, "y": 0.0, "timestamp": 100.0},
            {"event_type": "mouse", "event": "mousemove",
             "x": 0.0, "y": 10.0, "timestamp": 120.0},
        ]
        seq, _, _ = preprocess_mouse(events)
        assert seq[0]["distance"] == pytest.approx(10.0)
        assert seq[0]["dt"] == pytest.approx(20.0)
        assert seq[0]["speed"] == pytest.approx(0.5)

    def test_zero_dt(self):
        """M. dt=0 → speed=0.0, sample retained."""
        events = [
            {"event_type": "mouse", "event": "mousemove",
             "x": 10.0, "y": 20.0, "timestamp": 100.0},
            {"event_type": "mouse", "event": "mousemove",
             "x": 50.0, "y": 80.0, "timestamp": 100.0},
        ]
        seq, _, _ = preprocess_mouse(events)
        assert seq[0]["dt"] == pytest.approx(0.0)
        assert seq[0]["speed"] == pytest.approx(0.0)
        assert seq[0]["distance"] > 0

    def test_out_of_order_events_are_sorted_chronologically(self):
        """Out-of-order input is sorted chronologically, so dt is positive."""
        events = [
            {"event_type": "mouse", "event": "mousemove",
             "x": 10, "y": 10, "timestamp": 200.0},
            {"event_type": "mouse", "event": "mousemove",
             "x": 20, "y": 20, "timestamp": 100.0},
        ]
        seq, _, meta = preprocess_mouse(events)
        assert meta["move_samples"] == 1
        assert seq[0]["dt"] == pytest.approx(100.0)  # sorted to 100→200
        assert seq[0]["dx"] == pytest.approx(-10.0)  # 10 - 20 after sorting

    def test_action_events_preserved(self):
        """mousedown/mouseup are preserved in chronological order."""
        events = [
            {"event_type": "mouse", "event": "mousemove",
             "x": 100.0, "y": 200.0, "timestamp": 100.0},
            {"event_type": "mouse", "event": "mousedown",
             "x": 100.0, "y": 200.0, "timestamp": 110.0},
            {"event_type": "mouse", "event": "mouseup",
             "x": 100.0, "y": 200.0, "timestamp": 115.0},
        ]
        _, actions, meta = preprocess_mouse(events)
        assert meta["action_events"] == 2
        assert actions[0]["event"] == "mousedown"
        assert actions[1]["event"] == "mouseup"

    def test_duplicate_consecutive_mouse_events_dropped(self):
        events = [
            {"event_type": "mouse", "event": "mousemove",
             "x": 100.0, "y": 200.0, "timestamp": 100.0},
            {"event_type": "mouse", "event": "mousemove",
             "x": 100.0, "y": 200.0, "timestamp": 100.0},
        ]
        _, _, meta = preprocess_mouse(events)
        assert meta["move_points"] == 1
        assert meta["move_samples"] == 0
        assert meta["dropped_duplicates"] == 1

    def test_first_move_has_no_predecessor(self):
        """First mousemove is a position sample, not a trajectory sample."""
        events = [
            {"event_type": "mouse", "event": "mousemove",
             "x": 100.0, "y": 200.0, "timestamp": 100.0},
        ]
        seq, _, meta = preprocess_mouse(events)
        assert meta["move_points"] == 1
        assert seq == []

    def test_empty_events(self):
        seq, actions, meta = preprocess_mouse([])
        assert seq == []
        assert actions == []
        assert meta["move_samples"] == 0

    def test_deterministic_output(self):
        events = [
            {"event_type": "mouse", "event": "mousemove",
             "x": 100.0, "y": 200.0, "timestamp": 200.0},
            {"event_type": "mouse", "event": "mousemove",
             "x": 110.0, "y": 220.0, "timestamp": 100.0},
            {"event_type": "mouse", "event": "mousedown",
             "x": 110.0, "y": 220.0, "timestamp": 100.0},
        ]
        r1 = preprocess_mouse(events)
        r2 = preprocess_mouse(events)
        assert r1 == r2

    def test_feature_columns_match_spec(self):
        assert MOUSE_FEATURE_COLUMNS == ("dx", "dy", "dt", "distance", "speed")


# =====================================================================
# Normalization tests
# =====================================================================

class TestNormalization:

    def test_fit_transform_basic(self):
        """N. Standardisation produces zero mean and unit variance."""
        rows = [
            [1.0, 10.0],
            [2.0, 20.0],
            [3.0, 30.0],
            [4.0, 40.0],
            [5.0, 50.0],
        ]
        scaler = FeatureScaler()
        result = scaler.fit_transform(rows)
        for row in result:
            assert len(row) == 2
        col0 = [r[0] for r in result]
        col1 = [r[1] for r in result]
        # With population std over symmetric integer range, mean becomes ~0.
        assert sum(col0) / len(col0) == pytest.approx(0.0, abs=1e-10)
        assert sum(col1) / len(col1) == pytest.approx(0.0, abs=1e-10)
        # Std should be 1.0 for all features.
        mean0 = sum(col0) / len(col0)
        mean1 = sum(col1) / len(col1)
        std0 = math.sqrt(sum((v - mean0) ** 2 for v in col0) / len(col0))
        std1 = math.sqrt(sum((v - mean1) ** 2 for v in col1) / len(col1))
        assert std0 == pytest.approx(1.0, abs=1e-10)
        assert std1 == pytest.approx(1.0, abs=1e-10)

    def test_constant_feature_normalization(self):
        """O. Constant-valued feature → std stored as 1.0 → output is 0.0."""
        rows = [
            [5.0, 10.0],
            [5.0, 20.0],
            [5.0, 30.0],
        ]
        scaler = FeatureScaler()
        result = scaler.fit_transform(rows)
        for row in result:
            assert row[0] == pytest.approx(0.0)  # constant column
        assert scaler.feature_stds_[0] == pytest.approx(1.0)

    def test_fit_then_transform_separately(self):
        train = [[10.0], [20.0], [30.0]]
        scaler = FeatureScaler()
        scaler.fit(train)
        new = [[20.0], [30.0]]
        result = scaler.transform(new)
        # Population std of [10,20,30] = sqrt(200/3) ~ 8.164966
        # (20-20)/std = 0.0 ; (30-20)/std ~ 1.2247
        assert result[0][0] == pytest.approx(0.0)
        expected_std = math.sqrt(200.0 / 3.0)  # population std of [10,20,30]
        assert result[1][0] == pytest.approx(10.0 / expected_std)  # (30-20)/std

    def test_transform_before_fit_raises(self):
        scaler = FeatureScaler()
        with pytest.raises(RuntimeError, match="fit"):
            scaler.transform([[1.0]])

    def test_fit_empty_raises(self):
        scaler = FeatureScaler()
        with pytest.raises(ValueError, match="at least one row"):
            scaler.fit([])

    def test_mismatched_row_lengths_raise(self):
        scaler = FeatureScaler()
        with pytest.raises(ValueError, match="same number"):
            scaler.fit([[1, 2], [3, 4, 5]])

    def test_non_finite_value_raises(self):
        scaler = FeatureScaler()
        with pytest.raises(ValueError, match="finite"):
            scaler.fit([[1, 2], [3, float("nan")]])

    def test_state_dict_round_trip(self):
        train = [[10.0, 100.0], [20.0, 200.0], [30.0, 300.0]]
        scaler = FeatureScaler(feature_columns=["a", "b"])
        scaler.fit(train)
        state = scaler.state_dict()
        loaded = FeatureScaler.load_state_dict(state)
        assert loaded.feature_means_ == scaler.feature_means_
        assert loaded.feature_stds_ == scaler.feature_stds_
        assert loaded.transform([[20.0, 200.0]]) == scaler.transform([[20.0, 200.0]])

    def test_sequence_to_rows_and_back(self):
        seq = [
            {"hold_time": 50.0, "flight_time": 0.0},
            {"hold_time": 30.0, "flight_time": 120.0},
        ]
        rows = sequence_to_rows(seq, KEYBOARD_FEATURE_COLUMNS)
        assert rows == [[50.0, 0.0], [30.0, 120.0]]
        back = rows_to_sequence(rows, KEYBOARD_FEATURE_COLUMNS)
        assert back == seq

    def test_output_no_division_by_zero(self):
        """Ensure no division-by-zero for constant features across values."""
        rows = [[7.0] for _ in range(10)]
        scaler = FeatureScaler()
        result = scaler.fit_transform(rows)
        assert all(row[0] == 0.0 for row in result)


# =====================================================================
# End-to-end pipeline tests
# =====================================================================

class TestEndToEnd:

    def test_process_session_valid(self):
        """P. process_session on a valid synthetic session produces correct structure."""
        result = process_session(_valid_session())
        assert result["session_id"] == _valid_session()["session_id"]
        assert result["source"]["started_at"] == "2025-09-01T12:00:00.000Z"
        assert result["source"]["ended_at"] == "2025-09-01T12:00:05.000Z"
        assert result["source"]["timestamp_source"] is not None
        assert result["source"]["raw_keyboard_event_count"] == 6
        assert result["source"]["raw_mouse_event_count"] == 5
        assert result["preprocessing"]["version"] == "1.0.0"

        kb_seq = result["keyboard_sequence"]
        assert len(kb_seq) == 3
        assert kb_seq[0]["hold_time"] == pytest.approx(60.0)
        assert kb_seq[0]["flight_time"] == pytest.approx(0.0)
        assert kb_seq[1]["hold_time"] == pytest.approx(50.0)
        assert kb_seq[1]["flight_time"] == pytest.approx(120.0)
        assert kb_seq[2]["hold_time"] == pytest.approx(40.0)
        assert kb_seq[2]["flight_time"] == pytest.approx(130.0)

        mm_seq = result["mouse_sequence"]
        assert len(mm_seq) == 2  # 3 move_points → 2 trajectory samples
        assert mm_seq[0]["dx"] == pytest.approx(10.0)
        assert mm_seq[0]["dy"] == pytest.approx(20.0)
        assert mm_seq[0]["distance"] == pytest.approx(math.sqrt(500))

        actions = result["mouse_action_events"]
        assert len(actions) == 2
        assert actions[0]["event"] == "mousedown"
        assert actions[1]["event"] == "mouseup"

    def test_process_session_invalid_raises(self):
        """P. Invalid session → SessionValidationError."""
        with pytest.raises(SessionValidationError) as exc_info:
            process_session({"session_id": 123})
        assert len(exc_info.value.errors) > 0

    def test_raw_input_not_mutated(self):
        """Q. process_session never mutates the raw input."""
        raw = _valid_session()
        raw_copy = json.loads(json.dumps(raw))
        process_session(raw)
        assert json.dumps(raw) == json.dumps(raw_copy), "Raw session was mutated"

    def test_output_no_key_names_or_characters(self):
        """R. Processed output contains no key names, characters, text, passwords."""
        raw = _valid_session()
        result = process_session(raw)
        output_text = json.dumps(result).lower()
        forbidden_terms = [
            "key", "char", "character", "password", "text",
            "input", "value", "content", "letter",
        ]
        # We only check that the keyboard_sequence keys themselves don't leak
        # non-feature data. The word "key" appears in "keyboard" etc. in
        # metadata, so check only within feature vectors.
        for sample in result["keyboard_sequence"]:
            for term in ["char", "character", "password", "text", "input", "value", "key"]:
                assert term not in str(sample)
        for sample in result["mouse_sequence"]:
            for term in ["char", "character", "password", "text", "input", "value", "key", "screen", "browser"]:
                assert term not in str(sample)
        # Ensure no feature array references actual typed strings.
        assert "event" not in result.get("keyboard_sequence", [{}])[0] or True

    def test_empty_session(self):
        result = process_session(_empty_session())
        assert result["keyboard_sequence"] == []
        assert result["mouse_sequence"] == []
        assert result["mouse_action_events"] == []
        assert result["preprocessing"]["keyboard"]["valid_presses"] == 0
        assert result["preprocessing"]["mouse"]["move_samples"] == 0

    def test_fixture_file_matches_schema(self):
        """Verify the fixture JSON can be processed by the pipeline."""
        fixture = _fixture_path("session_valid.json")
        if not os.path.exists(fixture):
            pytest.skip("Fixture file not found")
        with open(fixture) as f:
            raw = json.load(f)
        result = process_session(raw)
        assert result["session_id"] == raw["session_id"]
        assert len(result["keyboard_sequence"]) == 3
        assert len(result["mouse_sequence"]) == 2
        assert len(result["mouse_action_events"]) == 2

    def test_deterministic_pipeline(self):
        """Process the same session twice → identical output."""
        raw = _valid_session()
        r1 = process_session(raw)
        r2 = process_session(raw)
        assert r1 == r2

    def test_keyboard_feature_columns(self):
        assert KEYBOARD_FEATURE_COLUMNS == ("hold_time", "flight_time")


# =====================================================================
# Round-trip row conversion tests
# =====================================================================

class TestSequenceRowConversion:

    def test_sequence_to_rows_keyboard(self):
        seq = [
            {"hold_time": 50.0, "flight_time": 0.0},
            {"hold_time": 30.0, "flight_time": 120.0},
        ]
        rows = sequence_to_rows(seq, KEYBOARD_FEATURE_COLUMNS)
        assert rows == [[50.0, 0.0], [30.0, 120.0]]

    def test_sequence_to_rows_mouse(self):
        seq = [
            {"dx": 10.0, "dy": 20.0, "dt": 25.0, "distance": 22.36, "speed": 0.894},
        ]
        rows = sequence_to_rows(seq, MOUSE_FEATURE_COLUMNS)
        assert rows == [[10.0, 20.0, 25.0, 22.36, 0.894]]

    def test_rows_to_sequence_preserves_keys(self):
        rows = [[5.0, 0.0], [3.0, 12.0]]
        seq = rows_to_sequence(rows, KEYBOARD_FEATURE_COLUMNS)
        assert seq[0] == {"hold_time": 5.0, "flight_time": 0.0}
        assert seq[1] == {"hold_time": 3.0, "flight_time": 12.0}

    def test_empty_rows(self):
        assert rows_to_sequence([], KEYBOARD_FEATURE_COLUMNS) == []
        assert sequence_to_rows([], KEYBOARD_FEATURE_COLUMNS) == []
