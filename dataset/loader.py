"""
Loading real Phase 2/Phase 3 sessions into the Phase 4 dataset.

Pipeline support (required by Phase 4 spec):

    raw Phase 2 JSON
        |
        v  (Phase 3: ml.preprocessing.process_session — unchanged)
    processed session
        |
        v  (processed_to_entry)
    Phase 4 dataset entry

``loader`` provides:

- ``processed_to_entry`` — attach ``user_id`` and re-shape a Phase 3
  processed session into a dataset entry (numeric sequences are passed
  through unmodified; Phase 3 behaviour is never touched).
- ``raw_to_entry`` — run a raw Phase 2 session through Phase 3, then convert.
- ``load_processed_dir`` / ``load_processed_files`` — batch load real sessions.
- ``load_raw_files`` — batch load + process raw Phase 2 sessions.
- ``load_dataset`` / ``save_dataset`` — JSON persistence of a dataset.

User-id assignment
    Real collected sessions do not carry a dataset ``user_id``. Callers must
    provide a mapping or callable: ``user_id_map`` is a dict ``session_id ->
    user_id`` (optionally with filename-stem keys as a fallback), or a
    callable ``(session_id, filename) -> user_id``. Sessions missing an id
    raise :class:`DatasetLoadError` instead of silently inventing labels.
"""

from __future__ import annotations

import json
import os
from typing import Any, Callable, Dict, Iterable, List, Sequence, Union

from ml.preprocessing import SessionValidationError, process_session

from .schema import assert_privacy, validate_dataset_entry

_USER_ID_MAP = Union[Dict[str, str], Callable[[str, str], str]]


class DatasetLoadError(Exception):
    """Raised when a real session cannot be turned into a dataset entry."""


class DatasetSchemaError(Exception):
    """Raised when assembled dataset entries fail validation/privacy checks."""


def _resolve_user_id(
    session_id: str,
    filename: str,
    user_id_map: _USER_ID_MAP | None,
) -> str:
    if user_id_map is None:
        raise DatasetLoadError(
            "session '{}' (from {}) has no user_id; provide a user_id_map "
            "({} -> user_id) or a callable".format(session_id, filename, session_id)
        )
    if callable(user_id_map):
        user_id = user_id_map(session_id, filename)
    else:
        user_id = user_id_map.get(session_id) or user_id_map.get(filename)
    if not user_id or not isinstance(user_id, str):
        raise DatasetLoadError(
            "no user_id found for session '{}' (from {})".format(session_id, filename)
        )
    return user_id


def processed_to_entry(processed: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Convert one Phase 3 processed session into a dataset entry.

    Args:
        processed: output of ``ml.preprocessing.process_session``.
        user_id: dataset-level identity label for this session's owner.

    The two behavioral sequences are passed through byte-identical; only the
    envelope (``user_id`` + metadata) is added. Phase 3 code is not called here.
    """
    if not isinstance(user_id, str) or not user_id.strip():
        raise ValueError("user_id must be a non-empty string")

    entry = {
        "user_id": user_id,
        "session_id": processed["session_id"],
        "keyboard_sequence": processed.get("keyboard_sequence", []),
        "mouse_sequence": processed.get("mouse_sequence", []),
        "mouse_action_events": processed.get("mouse_action_events", []),
        "metadata": {
            "generated": False,
            "source": processed.get("source"),
            "preprocessing": processed.get("preprocessing"),
            "keyboard_samples": len(processed.get("keyboard_sequence", [])),
            "mouse_samples": len(processed.get("mouse_sequence", [])),
        },
    }
    _check_entry(entry)
    return entry


def raw_to_entry(raw: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    """Run a raw Phase 2 session through Phase 3, then convert to an entry.

    Delegates to :func:`ml.preprocessing.process_session` unchanged, so Phase 3
    preprocessing behaviour is preserved exactly. Raw input is never mutated.
    """
    try:
        processed = process_session(raw)
    except SessionValidationError as exc:
        raise DatasetLoadError(
            "raw session invalid: {}".format("; ".join(exc.errors))
        ) from exc
    return processed_to_entry(processed, user_id)


def _check_entry(entry: Dict[str, Any]) -> None:
    errors = validate_dataset_entry(entry)
    if errors:
        raise DatasetSchemaError(
            "assembled entry failed validation:\n  - {}".format("\n  - ".join(errors))
        )
    assert_privacy([entry])


def load_processed_file(path: str, user_id_map: _USER_ID_MAP | None) -> Dict[str, Any]:
    """Load one processed Phase 3 session JSON file as a dataset entry."""
    with open(path, "r") as f:
        processed = json.load(f)
    session_id = processed.get("session_id")
    if not session_id:
        raise DatasetLoadError("processed file '{}' has no session_id".format(path))
    user_id = _resolve_user_id(session_id, os.path.basename(path), user_id_map)
    return processed_to_entry(processed, user_id)


def load_raw_file(path: str, user_id_map: _USER_ID_MAP | None) -> Dict[str, Any]:
    """Load + preprocess one raw Phase 2 session JSON file as a dataset entry."""
    with open(path, "r") as f:
        raw = json.load(f)
    session_id = raw.get("session_id")
    if not session_id:
        raise DatasetLoadError("raw file '{}' has no session_id".format(path))
    user_id = _resolve_user_id(session_id, os.path.basename(path), user_id_map)
    return raw_to_entry(raw, user_id)


def load_processed_files(
    paths: Sequence[str], user_id_map: _USER_ID_MAP | None = None
) -> List[Dict[str, Any]]:
    """Load a batch of processed Phase 3 session files into dataset entries."""
    entries = [load_processed_file(path, user_id_map) for path in paths]
    _check_dataset(entries)
    return entries


def load_processed_dir(
    dirpath: str, user_id_map: _USER_ID_MAP | None = None, suffix: str = ".json"
) -> List[Dict[str, Any]]:
    """Load every ``*.json`` file in ``dirpath`` (sorted for determinism)."""
    candidates = [
        os.path.join(dirpath, name)
        for name in sorted(os.listdir(dirpath))
        if name.endswith(suffix)
    ]
    return load_processed_files(candidates, user_id_map)


def load_raw_files(
    paths: Sequence[str], user_id_map: _USER_ID_MAP | None = None
) -> List[Dict[str, Any]]:
    """Load + preprocess a batch of raw Phase 2 session files."""
    entries = [load_raw_file(path, user_id_map) for path in paths]
    _check_dataset(entries)
    return entries


def _check_dataset(entries: Sequence[Dict[str, Any]]) -> None:
    # duplicate detection + privacy are enforced at the dataset level here
    seen = set()
    for entry in entries:
        sid = entry["session_id"]
        if sid in seen:
            raise DatasetSchemaError("duplicate session_id '{}'".format(sid))
        seen.add(sid)
    assert_privacy(entries)


# ---------------------------------------------------------------------------
# Dataset persistence
# ---------------------------------------------------------------------------

DATASET_FILE_HEADER = "behavioral-biometric-dataset"


def save_dataset(entries: Sequence[Dict[str, Any]], path: str) -> str:
    """Write the dataset to a versioned JSON file; returns the written path."""
    from .schema import DATASET_SCHEMA_VERSION

    payload = {
        "format": DATASET_FILE_HEADER,
        "schema_version": DATASET_SCHEMA_VERSION,
        "n_entries": len(entries),
        "entries": list(entries),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    return path


def load_dataset(path: str) -> List[Dict[str, Any]]:
    """Load a dataset saved by :func:`save_dataset`; validates entries."""
    from .schema import DATASET_SCHEMA_VERSION, validate_dataset

    with open(path, "r") as f:
        payload = json.load(f)

    if payload.get("format") != DATASET_FILE_HEADER:
        raise DatasetSchemaError(
            "'{}' is not a Phase 4 dataset file (missing format header)".format(path)
        )
    if payload.get("schema_version") != DATASET_SCHEMA_VERSION:
        raise DatasetSchemaError(
            "dataset schema_version {} != supported {}".format(
                payload.get("schema_version"), DATASET_SCHEMA_VERSION
            )
        )

    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise DatasetSchemaError("dataset file has no 'entries' list")

    errors = validate_dataset(entries)
    if errors:
        raise DatasetSchemaError(
            "dataset failed validation:\n  - {}".format("\n  - ".join(errors))
        )
    assert_privacy(entries)
    return entries


__all__ = [
    "DATASET_FILE_HEADER",
    "DatasetLoadError",
    "DatasetSchemaError",
    "load_dataset",
    "load_processed_dir",
    "load_processed_files",
    "load_raw_files",
    "processed_to_entry",
    "raw_to_entry",
    "save_dataset",
]