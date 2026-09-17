"""
High-level preprocessing pipeline.

``process_session(raw_session)`` is the single entry point: it validates a
raw Phase 2 session, converts keyboard and mouse events into separate
numerical behavioral sequences, and returns a processed representation that
preserves traceability metadata. It is deterministic and never mutates the
raw input.

Processed output format (documented schema):

.. code-block:: python

    {
      "session_id": str,
      "source": {
        "started_at": str,
        "ended_at": str,
        "timestamp_source": str | None,
        "raw_keyboard_event_count": int,
        "raw_mouse_event_count": int,
      },
      "preprocessing": {
        "version": "1.0.0",
        "keyboard": {     # counts for traceability (see keyboard.py)
          "input_events": int, "valid_presses": int,
          "dropped_unmatched_keyup": int, "dropped_unmatched_keydown": int,
          "dropped_negative_hold": int, "dropped_duplicates": int,
        },
        "mouse": {        # counts for traceability (see mouse.py)
          "input_events": int, "move_points": int, "move_samples": int,
          "action_events": int, "dropped_duplicates": int,
        },
      },
      "keyboard_sequence": [
        {"hold_time": float, "flight_time": float}, ...
      ],
      "mouse_sequence": [
        {"dx": float, "dy": float, "dt": float,
         "distance": float, "speed": float}, ...
      ],
      "mouse_action_events": [
        {"event": "mousedown"|"mouseup", "x": float, "y": float,
         "timestamp": float}, ...
      ],
    }

Keyboard and mouse are intentionally kept as SEPARATE sequences; no fusion,
tensors, or neural-network inputs are created in this phase.
"""

from __future__ import annotations

from typing import Any, Dict

from .keyboard import preprocess_keyboard
from .mouse import preprocess_mouse
from .validation import SessionValidationError, validate_session

PREPROCESSING_VERSION = "1.0.0"


def process_session(raw_session: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and preprocess one raw Phase 2 session.

    Steps:
        1. validate the raw session (raises :class:`SessionValidationError`),
        2. preprocess keyboard events and extract hold/flight features,
        3. preprocess mouse events and extract trajectory/speed features
           (mousedown/mouseup preserved as action events),
        4. assemble and return the processed representation.

    The raw session is read-only; ``process_session`` never mutates it and
    never includes raw typed content in its output.
    """
    errors = validate_session(raw_session)
    if errors:
        raise SessionValidationError(errors)

    keyboard_events = raw_session["keyboard_events"]
    mouse_events = raw_session["mouse_events"]

    keyboard_sequence, keyboard_meta = preprocess_keyboard(keyboard_events)
    move_sequence, action_events, mouse_meta = preprocess_mouse(mouse_events)

    return {
        "session_id": raw_session["session_id"],
        "source": {
            "started_at": raw_session["started_at"],
            "ended_at": raw_session["ended_at"],
            "timestamp_source": raw_session.get("timestamp_source"),
            "raw_keyboard_event_count": len(keyboard_events),
            "raw_mouse_event_count": len(mouse_events),
        },
        "preprocessing": {
            "version": PREPROCESSING_VERSION,
            "keyboard": keyboard_meta,
            "mouse": mouse_meta,
        },
        "keyboard_sequence": keyboard_sequence,
        "mouse_sequence": move_sequence,
        "mouse_action_events": action_events,
    }