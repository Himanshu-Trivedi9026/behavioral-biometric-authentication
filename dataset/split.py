"""
Deterministic, user/session-aware train/test split.

Guarantees
----------
- A session never appears in both splits (each ``session_id`` -> exactly one).
- **Per-user** split: every user keeps enrolment/training sessions AND separate
  evaluation sessions (subject to ``min_train`` / ``min_test``).
- Fully deterministic: sessions are grouped and stably sorted by ``session_id``,
  then shuffled with a per-user ``random.Random`` derived from the seed string
  (``f"split:{seed}:{user_id}"`` — deterministic across runs and processes).

Strategy
--------
For each user:
  1. collect that user's entries and sort by ``session_id``,
  2. shuffle the sorted list with the per-user RNG,
  3. take ``n_train`` for train, the rest for test, where ``n_train`` is
     derived from ``train_fraction`` clamped to ``[min_train, n - min_test]``.
"""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Sequence, Tuple

_Split = Tuple[List[Dict], List[Dict]]


def _group_by_user(entries: Sequence[Dict]) -> Dict[str, List[Dict]]:
    groups: Dict[str, List[Dict]] = {}
    for entry in entries:
        groups.setdefault(entry["user_id"], []).append(entry)
    return groups


def _n_train_for(n_sessions: int, fraction: float, min_train: int, min_test: int) -> int:
    if n_sessions < min_train + min_test:
        raise ValueError(
            "user has {} sessions but the split requires at least {} "
            "(min_train={} + min_test={})".format(
                n_sessions, min_train + min_test, min_train, min_test
            )
        )
    n_train = int(round(n_sessions * fraction))
    n_train = max(n_train, min_train)
    n_train = min(n_train, n_sessions - min_test)
    return n_train


def split_train_test(
    entries: Sequence[Dict],
    seed: int = 0,
    train_fraction: float = 0.7,
    min_train: int = 1,
    min_test: int = 1,
) -> _Split:
    """Split dataset entries into (train, test) with per-user isolation.

    Each user's sessions are split independently so that every user is
    represented in BOTH splits (enrollment + evaluation). Sessions are never
    shared between splits.

    Raises:
        ValueError: if any user has too few sessions to satisfy the minimums,
            or ``train_fraction`` is outside (0, 1).
    """
    if not (0.0 < train_fraction < 1.0):
        raise ValueError("train_fraction must be in (0, 1), got {!r}".format(train_fraction))
    if min_train < 1 or min_test < 1:
        raise ValueError("min_train and min_test must be >= 1")

    groups = _group_by_user(entries)
    train: List[Dict] = []
    test: List[Dict] = []

    for user_id in sorted(groups):
        user_entries = sorted(groups[user_id], key=lambda e: e["session_id"])
        n = len(user_entries)
        n_train = _n_train_for(n, train_fraction, min_train, min_test)

        rng = random.Random("split:{}:{}".format(seed, user_id))
        order = list(user_entries)
        rng.shuffle(order)

        train.extend(order[:n_train])
        test.extend(order[n_train:])

    return train, test


def split_report(
    entries: Sequence[Dict], train: Sequence[Dict], test: Sequence[Dict]
) -> Dict[str, Dict]:
    """Summarize counts per user for both splits (for reproducibility/docs)."""
    counts: Dict[str, Dict] = {}
    for user_id in sorted({e["user_id"] for e in entries}):
        n_train = sum(1 for e in train if e["user_id"] == user_id)
        n_test = sum(1 for e in test if e["user_id"] == user_id)
        counts[user_id] = {"train": n_train, "test": n_test, "total": n_train + n_test}
    return counts


def session_ids(entries: Sequence[Dict]) -> set[str]:
    """Return the set of ``session_id`` values (for leakage checks)."""
    return {e["session_id"] for e in entries}


def assert_no_leakage(train: Sequence[Dict], test: Sequence[Dict]) -> None:
    """Raise `ValueError` if any session appears in both train and test."""
    overlap = session_ids(train) & session_ids(test)
    if overlap:
        raise ValueError(
            "train/test leakage detected for session(s): {}".format(sorted(overlap))
        )


__all__ = [
    "assert_no_leakage",
    "session_ids",
    "split_report",
    "split_train_test",
]