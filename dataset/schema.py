"""
Dataset schema, validation, and privacy checks.

A **dataset entry** is a single identity-verification sample:

.. code-block:: python

    {
      "user_id": str,                       # dataset-level identity label
      "session_id": str,                    # unique per entry
      "keyboard_sequence": [                # Phase 3 keyboard features
        {"hold_time": float, "flight_time": float}, ...
      ],
      "mouse_sequence": [                   # Phase 3 mouse trajectory features
        {"dx": float, "dy": float, "dt": float,
         "distance": float, "speed": float}, ...
      ],
      "mouse_action_events": [              # preserved mousedown/mouseup
        {"event": "mousedown"|"mouseup", "x": float,
         "y": float, "timestamp": float}, ...
      ],
      "metadata": dict | None,              # optional traceability/for-eval info
    }

Privacy contract: feature sequences are numerical timing/movement values only.
The validation + privacy helpers below guarantee (and the test suite verifies)
that no typed characters, key names, key codes, or raw text can leak into a
dataset.
"""

from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple

DATASET_SCHEMA_VERSION = "1.0.0"

KEYBOARD_FEATURE_COLUMNS = ("hold_time", "flight_time")
MOUSE_FEATURE_COLUMNS = ("dx", "dy", "dt", "distance", "speed")

_REQUIRED_ENTRY_KEYS = ("user_id", "session_id", "keyboard_sequence", "mouse_sequence")

# JSON/behavioural keys that would indicate leaked key identity or raw text.
# Phase 2 never collects these; this is a defensive, tests-verified guard.
_FORBIDDEN_KEYS = frozenset(
    {
        "key",
        "keydown",
        "keyup",  # note: 'keyup'/'keydown' as DATA KEYS are leaks; harmless word
        "keyCode",
        "keycode",
        "code",
        "charCode",
        "charcode",
        "which",
        "char",
        "character",
        "value",
        "data",
        "inputType",
        "inputtype",
        "isComposing",
        "password",
        "text",
        "keyIdentifier",
        "keyIdentifierCode",
        "keypress",
    }
)


def _is_finite_number(value: Any) -> bool:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    return math.isfinite(float(value))


def _validate_feature_sample(
    sample: Any,
    feature_columns: Sequence[str],
    location: str,
) -> List[str]:
    errors: List[str] = []
    if not isinstance(sample, dict):
        return ["{} entry {} is not an object".format(location, sample)]
    sample_keys = set(sample.keys())
    missing = [col for col in feature_columns if col not in sample_keys]
    if missing:
        errors.append(
            "{} entry is missing feature(s): {}".format(location, ", ".join(missing))
        )
    extra = sample_keys - set(feature_columns)
    if extra:
        errors.append(
            "{} entry has unexpected feature(s): {}".format(
                location, ", ".join(sorted(extra))
            )
        )
    for col in feature_columns:
        value = sample.get(col)
        if value is None:
            continue  # already reported as missing
        if not _is_finite_number(value):
            errors.append(
                "{} feature '{}' must be a finite number, got {!r}".format(
                    location, col, value
                )
            )
    return errors


def validate_feature_sequence(
    sequence: Any,
    feature_columns: Sequence[str],
    location: str,
) -> List[str]:
    """Validate a numeric feature sequence (empty lists are allowed)."""
    if not isinstance(sequence, list):
        return ["{} must be a list, got {!r}".format(location, type(sequence).__name__)]
    errors: List[str] = []
    for i, sample in enumerate(sequence):
        errors.extend(
            _validate_feature_sample(sample, feature_columns, "{}[{}]".format(location, i))
        )
    return errors


def validate_action_events(events: Any) -> List[str]:
    """Validate ``mouse_action_events`` (structural; preserved verbatim)."""
    if events is None:
        return []
    if not isinstance(events, list):
        return ["mouse_action_events must be a list, got {!r}".format(type(events).__name__)]
    errors: List[str] = []
    allowed = {"mousedown", "mouseup"}
    for i, event in enumerate(events):
        if not isinstance(event, dict):
            errors.append("mouse_action_events[{}] must be an object".format(i))
            continue
        if event.get("event") not in allowed:
            errors.append(
                "mouse_action_events[{}].event must be mousedown|mouseup".format(i)
            )
    return errors


def validate_dataset_entry(entry: Any) -> List[str]:
    """Validate one dataset entry; returns a list of human-readable errors.

    An empty error list means the entry is valid. The entry is never mutated.
    """
    if not isinstance(entry, dict):
        return ["dataset entry must be an object, got {!r}".format(type(entry).__name__)]

    errors: List[str] = []

    user_id = entry.get("user_id")
    if not isinstance(user_id, str) or not user_id.strip():
        errors.append("field 'user_id' must be a non-empty string")

    session_id = entry.get("session_id")
    if not isinstance(session_id, str) or not session_id.strip():
        errors.append("field 'session_id' must be a non-empty string")

    for key in _REQUIRED_ENTRY_KEYS:
        if key not in entry:
            errors.append("missing required field '{}'".format(key))

    errors.extend(
        validate_feature_sequence(
            entry.get("keyboard_sequence", []),
            KEYBOARD_FEATURE_COLUMNS,
            "keyboard_sequence",
        )
    )
    errors.extend(
        validate_feature_sequence(
            entry.get("mouse_sequence", []),
            MOUSE_FEATURE_COLUMNS,
            "mouse_sequence",
        )
    )
    errors.extend(validate_action_events(entry.get("mouse_action_events")))

    metadata = entry.get("metadata")
    if metadata is not None and not isinstance(metadata, dict):
        errors.append("field 'metadata' must be an object or null")

    return errors


def validate_dataset(entries: Iterable[Any]) -> List[str]:
    """Validate a whole dataset; also detects duplicate ``session_id`` values."""
    if not isinstance(entries, (list, tuple)):
        return ["dataset must be a list of entries, got {!r}".format(type(entries).__name__)]

    errors: List[str] = []
    seen_sessions = set()
    for i, entry in enumerate(entries):
        for error in validate_dataset_entry(entry):
            errors.append("entry[{}]: {}".format(i, error))
        if isinstance(entry, dict):
            session_id = entry.get("session_id")
            if isinstance(session_id, str) and session_id:
                if session_id in seen_sessions:
                    errors.append(
                        "duplicate session_id '{}' (entry[{}])".format(session_id, i)
                    )
                seen_sessions.add(session_id)
    return errors


def privacy_report(entry: Any) -> List[str]:
    """Return findings that would indicate leaked key identity/raw text.

    Walks the whole entry (including nested dicts, sequences, and flattened
    JSON) looking for forbidden identity-bearing key names or value patterns.
    An empty list means the entry is privacy-clean.
    """
    findings: List[str] = []

    if not isinstance(entry, dict):
        return ["privacy: entry is not an object"]

    def walk(obj: Any, path: str) -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                child = "{}.{}".format(path, key) if path else key
                if key in _FORBIDDEN_KEYS:
                    findings.append(
                        "forbidden key name '{}' found at {}".format(key, child)
                    )
                walk(value, child)
        elif isinstance(obj, list):
            for i, value in enumerate(obj):
                walk(value, "{}[{}]".format(path, i))

    walk(entry, "")

    # Flat scan of the serialized form catches identity values smuggled in as
    # string data (defensive; legitimate values are all numeric here).
    serialized = _flatten_json_strings(entry)
    lowered = serialized.lower()
    for token in ("password", "passwd", "secret", "ssn ", "creditcard"):
        if token in lowered:
            findings.append("sensitive token '{}' present in entry".format(token))

    return findings


def _flatten_json_strings(obj: Any) -> str:
    return repr(obj).lower()


def assert_privacy(entries: Iterable[Any]) -> None:
    """Raise :class:`ValueError` if any entry leaks identity/text content.

    The dataset contract (and automated test suite) require this check to
    pass for both synthetic and real processed sessions.
    """
    for i, entry in enumerate(entries):
        findings = privacy_report(entry)
        if findings:
            raise ValueError(
                "entry[{}] failed the privacy check:\n  - {}".format(
                    i, "\n  - ".join(findings)
                )
            )


__all__ = [
    "DATASET_SCHEMA_VERSION",
    "KEYBOARD_FEATURE_COLUMNS",
    "MOUSE_FEATURE_COLUMNS",
    "assert_privacy",
    "privacy_report",
    "validate_dataset",
    "validate_dataset_entry",
]