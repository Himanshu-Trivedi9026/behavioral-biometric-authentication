#!/usr/bin/env python3
"""
Phase 4 — deterministic synthetic dataset generation CLI.

Usage:
    .venv/bin/python scripts/generate_dataset.py [--out PATH] [--seed N]
        [--n-users N] [--sessions-per-user N]

Default output:
    data/datasets/dataset_synthetic.json

Same seed + same scale always regenerates the exact same dataset (byte-for-byte).

Exit code 0 = success; 1 = error; 2 = usage/file error.
"""

import argparse
import json
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from dataset.generator import SyntheticDatasetGenerator  # noqa: E402
from dataset.loader import save_dataset  # noqa: E402
from dataset.schema import (  # noqa: E402
    DATASET_SCHEMA_VERSION,
    assert_privacy,
    validate_dataset,
)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Generate a deterministic synthetic Phase 4 dataset."
    )
    parser.add_argument(
        "--out",
        default=os.path.join("data", "datasets", "dataset_synthetic.json"),
        help="output JSON path (default: data/datasets/dataset_synthetic.json)",
    )
    parser.add_argument("--seed", type=int, default=0, help="deterministic seed (default 0)")
    parser.add_argument("--n-users", type=int, default=5, help="number of users (default 5)")
    parser.add_argument(
        "--sessions-per-user", type=int, default=10, help="sessions per user (default 10)"
    )
    args = parser.parse_args(argv)

    out_dir = os.path.dirname(os.path.abspath(args.out))
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as exc:
        print("error: cannot create output directory: {}".format(exc), file=sys.stderr)
        return 2

    try:
        generator = SyntheticDatasetGenerator(
            seed=args.seed,
            n_users=args.n_users,
            sessions_per_user=args.sessions_per_user,
        )
        entries = generator.generate()
    except ValueError as exc:
        print("error: {}".format(exc), file=sys.stderr)
        return 1

    errors = validate_dataset(entries)
    if errors:
        print("dataset validation failed:", file=sys.stderr)
        for error in errors:
            print("  - " + error, file=sys.stderr)
        return 1
    assert_privacy(entries)

    save_dataset(entries, args.out)

    kb_samples = sum(len(e["keyboard_sequence"]) for e in entries)
    mouse_samples = sum(len(e["mouse_sequence"]) for e in entries)
    print("wrote dataset to: {}".format(args.out))
    print("schema version: {}".format(DATASET_SCHEMA_VERSION))
    print(
        "entries: {} (users={}, sessions/user={})".format(
            len(entries), args.n_users, args.sessions_per_user
        )
    )
    print("keyboard samples total: {}".format(kb_samples))
    print("mouse samples total: {}".format(mouse_samples))
    print("privacy check: PASS")
    print("deterministic: same seed regenerates the identical dataset")
    return 0


if __name__ == "__main__":
    sys.exit(main())