"""
Raw session validation.

Validates a raw Phase 2 session (a ``dict`` matching the browser collector's
JSON export) and reports clear, human-readable errors for malformed data.

Design:

- :func:`validate_session` NEVER mutates the input and returns a list of error
  strings (an empty list means the session is valid).
- :class:`SessionValidationError` wraps the same error list so the pipeline
  can reject an invalid session in one place. ``str(exc)`` is human-readable.

The expected raw schema (unchanged from Phase 2):

.. code-block:: python

    {
      "session_id": str,
      "started_at": str,          # ISO-8601 wall-clock reference
      "ended_at": str,            # ISO-8601 wall-clock reference
      "timestamp_source": str,    # informational (optional)
      "keyboard_events": [
        {"event_type": "keyboard", "event": "keydown"|"keyup",
         "timestamp": number}
      ],
      "mouse_events": [
        {"event_type": "mouse",
         "event": "mousemove"|"mousedown"|"mouseup",
         "x": number, "y": number, "timestamp": number}
      ]
    }

"Number" here means an ``int`` or ``float`` that is finite (never ``NaN`` or
``±inf``). Booleans are rejected because ``bool`` subclasses ``int``.
"""

from __future__ import annotations

import math
from typing import Any, List

# Canonical event names produced by the Phase 2 collector.
KEYBOARD_EVENT_NAMES = ("keydown", "keyup")
MOUSE_EVENT_NAMES = ("mousemove", "mousedown", "mouseup")


class SessionValidationError(ValueError):
    """Raised when a raw session fails validation.

    Attributes:
        errors: list of human-readable problems found in the raw session.
    """

    def __init__(self, errors: List[str]) -> None:
        self.errors = list(errors)
        summary = "; ".join(self.errors)
        super().__init__(
            "Raw session is not valid: " + summary if summary else "Raw session is not valid."
        )


def _is_finite_number(value: Any) -> bool:
    """True for int/float values that are finite; False for bool/None/NaN/Inf."""
    if isinstance(value, bool):
        return False
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return False


def _check_common_fields(session: dict, errs: List[str]) -> None:
    """Validate the session-level fields common to every raw session."""
    for key in ("session_id", "started_at", "ended_at"):
        if key not in session:
            errs.append("missing required field '{}'".format(key))
            continue
        value = session[key]
        if not isinstance(value, str) or len(value.strip()) == 0:
            errs.append("field '{}' must be a non-empty string".format(key))

    for key in ("keyboard_events", "mouse_events"):
        if key not in session:
            errs.append("missing required field '{}'".format(key))
        elif not isinstance(session[key], list):
            errs.append("field '{}' must be a list".format(key))


def _check_keyboard_event(index: int, event: Any, errs: List[str]) -> None:
    prefix = "keyboard_events[{}]".format(index)

    def problem(msg: str) -> None:
        errs.append("{}: {}".format(prefix, msg))

    if not isinstance(event, dict):
        problem("expected an object (got {})".format(type(event).__name__))
        return

    if "event_type" not in event:
        problem("missing field 'event_type'")
    elif event["event_type"] != "keyboard":
        problem("field 'event_type' must be 'keyboard'")

    if "event" not in event:
        problem("missing field 'event'")
    elif event["event"] not in KEYBOARD_EVENT_NAMES:
        problem(
            "field 'event' must be one of {} (got {!r})".format(
                KEYBOARD_EVENT_NAMES, event.get("event")
            )
        )

    if "timestamp" not in event:
        problem("missing field 'timestamp'")
    elif not _is_finite_number(event["timestamp"]):
        problem("field 'timestamp' must be a finite number")


def _check_mouse_event(index: int, event: Any, errs: List[str]) -> None:
    prefix = "mouse_events[{}]".format(index)

    def problem(msg: str) -> None:
        errs.append("{}: {}".format(prefix, msg))

    if not isinstance(event, dict):
        problem("expected an object (got {})".format(type(event).__name__))
        return

    if "event_type" not in event:
        problem("missing field 'event_type'")
    elif event["event_type"] != "mouse":
        problem("field 'event_type' must be 'mouse'")

    if "event" not in event:
        problem("missing field 'event'")
    elif event["event"] not in MOUSE_EVENT_NAMES:
        problem(
            "field 'event' must be one of {} (got {!r})".format(
                MOUSE_EVENT_NAMES, event.get("event")
            )
        )

    for key in ("x", "y", "timestamp"):
        if key not in event:
            problem("missing field '{}'".format(key))
        elif not _is_finite_number(event[key]):
            problem("field '{}' must be a finite number".format(key))


def validate_session(session: Any) -> List[str]:
    """Validate a raw session.

    Args:
        session: the raw Phase 2 session dict.

    Returns:
        A list of error strings. An empty list means the session is valid.

    Raises:
        SessionValidationError: when ``session`` is not a dict at all.
    """
    errors: List[str] = []
    if not isinstance(session, dict):
        return [
            "session must be a dict (got {})".format(type(session).__name__)
        ]

    _check_common_fields(session, errors)

    keyboard_events = session.get("keyboard_events")
    if isinstance(keyboard_events, list):
        for i, event in enumerate(keyboard_events):
            _check_keyboard_event(i, event, errors)

    mouse_events = session.get("mouse_events")
    if isinstance(mouse_events, list):
        for i, event in enumerate(mouse_events):
            _check_mouse_event(i, event, errors)

    return errors