#!/usr/bin/env python3
"""
Phase 3 — preprocessing CLI demo.

Validates a raw Phase 2 session JSON file and prints (or writes) the processed
representation. For manual verification only; see docs/PHASE_3_PREPROCESSING.md.

Usage:
    .venv/bin/python scripts/preprocess_session.py <input.json> [--out <output.json>]

Examples:
    .venv/bin/python scripts/preprocess_session.py tests/fixtures/session_valid.json
    .venv/bin/python scripts/preprocess_session.py data/raw/session-x.json --out /tmp/processed.json

Exit code 0 = success; 1 = validation error; 2 = usage/file error.
"""

import argparse
import json
import os
import sys

# Allow importing the ml package when this script is invoked by path from the
# repository root (e.g. `python scripts/preprocess_session.py`).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ml.preprocessing import SessionValidationError, process_session  # noqa: E402


def main(argv=None):
    parser = argparse.ArgumentParser(description=_doc_synopsis())
    parser.add_argument("input", help="path to a raw Phase 2 session JSON file")
    parser.add_argument("--out", default=None, help="optional output JSON path")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.input):
        print("error: no such file: {}".format(args.input), file=sys.stderr)
        return 2

    with open(args.input, "r") as f:
        raw = json.load(f)

    try:
        processed = process_session(raw)
    except SessionValidationError as exc:
        print("raw session is invalid:", file=sys.stderr)
        for error in exc.errors:
            print("  - " + error, file=sys.stderr)
        return 1

    output = json.dumps(processed, indent=2)
    if args.out:
        with open(args.out, "w") as f:
            f.write(output + "\n")
        print("wrote processed session to: {}".format(args.out))
    else:
        print(output)
    return 0


def _doc_synopsis():
    return (
        "Validate and preprocess a raw Phase 2 session into behavioral "
        "feature sequences (keyboard + mouse kept separate)."
    )


if __name__ == "__main__":
    sys.exit(main())