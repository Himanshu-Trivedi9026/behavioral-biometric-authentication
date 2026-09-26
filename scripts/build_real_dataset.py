#!/usr/bin/env python3
"""
Phase 16A — assemble the local real-behavioral dataset.

Converts locally collected raw Phase 2 browser sessions into the existing
Phase 4 dataset schema file ``data/datasets/dataset_real.json``.

Input layout (expected, all private and gitignored)::

    data/raw/
        participant_manifest.json   {"participants": {user_001: ["a.json", ...], ...}}
        <session files>             raw Phase 2 session JSONs downloaded from the
                                    collector page (nothing else is read)

Only the sessions listed in the manifest are processed. User identity is
supplied **explicitly** through the manifest's pseudonymous participant ids
(``user_001``, ``user_002``, ...) — it is never inferred or invented. Real
names / emails / device details must not appear in the manifest or the files.

Privacy: the existing loader validates each entry and runs ``assert_privacy``
(no key identities, no typed text, no passwords, no clipboard/DOM/screenshot
content). Provenance ``metadata.generated=false`` is set by the loader.

The output dataset stays under ``data/datasets/`` which is gitignored — the
real data must never be committed or uploaded.

Usage:
    .venv/bin/python scripts/build_real_dataset.py \
        --raw-dir data/raw \
        --manifest data/raw/participant_manifest.json \
        --output data/datasets/dataset_real.json

Exit code 0 = success; 1 = execution error; 2 = usage/file error.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dataset import load_raw_files, save_dataset  # noqa: E402

DEFAULT_RAW_DIR = "data/raw"
DEFAULT_MANIFEST = os.path.join(DEFAULT_RAW_DIR, "participant_manifest.json")
DEFAULT_OUTPUT = "data/datasets/dataset_real.json"


def load_manifest(path: str) -> Dict[str, List[str]]:
    """Load ``{"participants": {user_id: [filenames, ...], ...}}``.

    Returns a dict ``user_id -> [filenames]``. Raises ``ValueError`` on a
    malformed manifest so collection mistakes fail loudly instead of silently
    inventing identities.
    """
    if not os.path.exists(path):
        raise ValueError(
            "participant manifest '{}' does not exist -- create it first "
            "(see docs/PHASE_16A_REAL_DATASET_EVALUATION.md)".format(path)
        )
    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("manifest must be a JSON object")
    participants = payload.get("participants")
    if not isinstance(participants, dict) or not participants:
        raise ValueError("manifest must contain a non-empty 'participants' object")

    result: Dict[str, List[str]] = {}
    for user_id, files in participants.items():
        if not isinstance(user_id, str) or not user_id.strip():
            raise ValueError("manifest participant ids must be non-empty strings")
        if not isinstance(files, list) or not files:
            raise ValueError("participant '{}' must list at least one session file".format(user_id))
        if any(not isinstance(name, str) or not name.strip() for name in files):
            raise ValueError("participant '{}' lists a non-string session filename".format(user_id))
        result[user_id.strip()] = files
    return result


def _resolve_paths(raw_dir: str, files: List[str]) -> List[str]:
    paths = [os.path.join(raw_dir, name) for name in files]
    missing = [p for p in paths if not os.path.isfile(p)]
    if missing:
        raise ValueError(
            "manifest references missing session file(s): {}".format(", ".join(missing))
        )
    return paths


def _build_user_id_map(participants: Dict[str, List[str]]) -> Dict[str, str]:
    """Map session filenames -> user_id (basename-keyed, like the loader)."""
    mapping: Dict[str, str] = {}
    for user_id, files in participants.items():
        for name in files:
            basename = os.path.basename(name)
            if basename in mapping and mapping[basename] != user_id:
                raise ValueError(
                    "session filename '{}' is listed for both '{}' and '{}'; a session "
                    "can belong to exactly one participant".format(
                        basename, mapping[basename], user_id
                    )
                )
            mapping[basename] = user_id
    return mapping


def _warn_unlisted_sessions(raw_dir: str, manifest: str, listed: List[str]) -> None:
    listed_basenames = {os.path.basename(name) for name in listed}
    manifest_basename = os.path.basename(manifest)
    unlisted = [
        name
        for name in sorted(os.listdir(raw_dir))
        if name.endswith(".json")
        and name != manifest_basename
        and name not in listed_basenames
    ]
    if unlisted:
        print(
            "warning: {} raw session file(s) in '{}' are NOT in the manifest and were "
            "skipped: {}".format(len(unlisted), raw_dir, ", ".join(unlisted)),
            file=sys.stderr,
        )


def build_real_dataset(
    raw_dir: str,
    manifest_path: str,
    output_path: str,
) -> Dict[str, Any]:
    """Assemble raw sessions -> validated/privacy-checked dataset file.

    Returns a JSON-serialisable summary and writes the dataset to
    ``output_path``. This function is importable so the CLI and tests share
    exactly one implementation.
    """
    if not os.path.isdir(raw_dir):
        raise ValueError("raw directory '{}' does not exist".format(raw_dir))

    participants = load_manifest(manifest_path)
    paths: List[str] = []
    for user_id, files in participants.items():
        paths.extend(_resolve_paths(raw_dir, files))

    user_id_map = _build_user_id_map(participants)
    _warn_unlisted_sessions(raw_dir, manifest_path, paths)

    # The existing loader validates the schema and runs assert_privacy on the
    # assembled entries (DataLoadError / DatasetSchemaError if anything leaks).
    entries = load_raw_files(paths, user_id_map)
    save_dataset(entries, output_path)

    counts: Dict[str, int] = {}
    for entry in entries:
        counts[entry["user_id"]] = counts.get(entry["user_id"], 0) + 1

    return {
        "output": output_path,
        "n_sessions": len(entries),
        "n_users": len(counts),
        "sessions_per_user": {
            user: counts[user] for user in sorted(counts.keys())
        },
        "generated": False,
        "note": "existing Phase 4 schema v1.0.0; privacy-validated; gitignored",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Assemble local real behavioral sessions into the Phase 4 dataset."
    )
    parser.add_argument("--raw-dir", default=DEFAULT_RAW_DIR, help="directory of raw session JSONs (default: %(default)s)")
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST, help="participant manifest file (default: %(default)s)")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="dataset output path (default: %(default)s)")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = build_real_dataset(args.raw_dir, args.manifest, args.output)
    except (ValueError, OSError) as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1
    print("sessions: {n_sessions} | users: {n_users}".format(**summary))
    print("sessions per user: {}".format(summary["sessions_per_user"]))
    print("dataset written: {} (generated=false, gitignored)".format(summary["output"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())