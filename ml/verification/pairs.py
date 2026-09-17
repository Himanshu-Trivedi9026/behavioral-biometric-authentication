"""
Deterministic genuine/impostor session-pair generation (Phase 6).

Overview
--------
The Siamese verifier is trained / probed on **pairs** of behavioural sessions
together with a binary relationship label. This module turns the existing
Phase 4 dataset representation (list of entries) into those pairs without ever
letting identity metadata become a model feature.

Label convention (binary, explicit)
-----------------------------------
``label = 1``  -> **genuine**  : ``session_a`` and ``session_b`` belong to the
                                  **same** user.
``label = 0``  -> **impostor** : ``session_a`` and ``session_b`` belong to
                                  **different** users.

This follows the Phase 6 specification exactly.

Guarantees
----------
* Genuine pairs come from the same user; impostor pairs from two users.
* A session is **never** paired with itself (``session_a.session_id !=
  session_b.session_id`` enforced by :class:`VerificationPair`).
* Generation is deterministic: a fixed ``seed`` reproduces the exact pair set.
* Genuine / impostor counts are configurable independently.
* User ID only ever appears in the *pair label* construction — the emitted
  :class:`VerificationPair` carries metadata so callers may trace pairs, but
  the model input must be built from the behavioural sequences only (see
  ``ml.verification.siamese.SiameseVerifier``).
* Train/test leakage is prevented when a split is supplied: pairs are only
  formed *within* a partition, never across partitions.
* Invalid datasets / impossible pair requests raise :class:`ValueError`.

Identity separation
-------------------
Pairs keep full session dicts (which include ``user_id`` / ``session_id``) so
that **labels** and traceability can be recovered. The Siamese verifier
(intentionally) reads only ``keyboard_sequence`` / ``mouse_sequence`` from each
session, so identity metadata stays out of the model computation. This is
verified by the Phase 6 tests.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

GENUINE = 1
IMPOSTOR = 0

VALID_LABELS = (GENUINE, IMPOSTOR)


@dataclass(frozen=True)
class VerificationPair:
    """One (``session_a``, ``session_b``) pair with a binary relationship label.

    Attributes:
        session_a: first dataset entry (Phase 4 schema).
        session_b: second dataset entry (Phase 4 schema).
        label:     ``1`` genuine (same user) / ``0`` impostor (different users).

    The dataclass **validates at construction time** that the label matches the
    actual user relationship and that a session is never paired with itself.
    Session/user ids are metadata only — they are preserved for bookkeeping and
    label construction, never fed to the model.
    """

    session_a: Dict[str, Any]
    session_b: Dict[str, Any]
    label: int

    def __post_init__(self) -> None:
        for name in ("session_a", "session_b"):
            session = getattr(self, name)
            if not isinstance(session, Mapping):
                raise TypeError(
                    "{} must be a session/entry dict, got {!r}".format(
                        name, type(session).__name__
                    )
                )
            if not isinstance(session.get("user_id"), str) or not session["user_id"].strip():
                raise ValueError("{}['user_id'] must be a non-empty string".format(name))
            if not isinstance(session.get("session_id"), str) or not session["session_id"].strip():
                raise ValueError("{}['session_id'] must be a non-empty string".format(name))

        if self.label not in VALID_LABELS:
            raise ValueError(
                "pair label must be 1 (genuine) or 0 (impostor), got {!r}".format(
                    self.label
                )
            )

        if self.session_a["session_id"] == self.session_b["session_id"]:
            raise ValueError(
                "session '{}' cannot be paired with itself".format(
                    self.session_a["session_id"]
                )
            )

        same_user = self.session_a["user_id"] == self.session_b["user_id"]
        if (self.label == GENUINE) != same_user:
            raise ValueError(
                "label {} does not match user relationship "
                "(a={!r}, b={!r} -> same_user={})".format(
                    self.label,
                    self.session_a["user_id"],
                    self.session_b["user_id"],
                    same_user,
                )
            )

    # -- convenience metadata accessors --------------------------------------

    @property
    def user_a(self) -> str:
        """``user_id`` of ``session_a`` (metadata only)."""
        return self.session_a["user_id"]

    @property
    def user_b(self) -> str:
        """``user_id`` of ``session_b`` (metadata only)."""
        return self.session_b["user_id"]

    @property
    def session_id_a(self) -> str:
        """``session_id`` of ``session_a`` (bookkeeping only)."""
        return self.session_a["session_id"]

    @property
    def session_id_b(self) -> str:
        """``session_id`` of ``session_b`` (bookkeeping only)."""
        return self.session_b["session_id"]

    @property
    def is_genuine(self) -> bool:
        """``True`` iff this is a same-user pair."""
        return self.label == GENUINE

    def to_dict(self) -> Dict[str, Any]:
        """Metadata summary (never used as model input)."""
        return {
            "session_a_id": self.session_id_a,
            "session_b_id": self.session_id_b,
            "user_a": self.user_a,
            "user_b": self.user_b,
            "label": self.label,
        }


def group_by_user(entries: Sequence[Mapping[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Group dataset entries by ``user_id`` (order-preserving per user)."""
    _validate_pool(entries)
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for entry in entries:
        groups.setdefault(entry["user_id"], []).append(entry)
    return groups


def _validate_pool(entries: Sequence[Mapping[str, Any]]) -> None:
    if not isinstance(entries, (list, tuple)):
        raise TypeError(
            "entries must be a list/tuple of dataset entries, got {!r}".format(
                type(entries).__name__
            )
        )
    if not entries:
        raise ValueError("entries must contain at least one session")
    for i, entry in enumerate(entries):
        if not isinstance(entry, Mapping):
            raise TypeError(
                "entries[{}] must be a session/entry dict, got {!r}".format(
                    i, type(entry).__name__
                )
            )
        for field in ("user_id", "session_id"):
            value = entry.get(field)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(
                    "entries[{}]['{}'] must be a non-empty string".format(i, field)
                )


def genuine_candidates(
    entries: Sequence[Mapping[str, Any]],
) -> List[VerificationPair]:
    """All possible genuine pairs (distinct sessions of the same user).

    Ordered by: user (input order), then pair combination (input order).
    """
    candidates: List[VerificationPair] = []
    for user_id, sessions in group_by_user(entries).items():
        for a, b in itertools.combinations(sessions, 2):
            candidates.append(VerificationPair(session_a=a, session_b=b, label=GENUINE))
    return candidates


def impostor_candidates(
    entries: Sequence[Mapping[str, Any]],
) -> List[VerificationPair]:
    """All possible impostor pairs (sessions of two different users).

    Each unordered user pair is enumerated once; every cross-user session
    combination is included. Session ids are unique per dataset, so a session
    is never paired with itself.
    """
    groups = group_by_user(entries)
    users = list(groups.keys())
    candidates: List[VerificationPair] = []
    for user_a, user_b in itertools.combinations(users, 2):
        for a in groups[user_a]:
            for b in groups[user_b]:
                candidates.append(VerificationPair(session_a=a, session_b=b, label=IMPOSTOR))
    return candidates


def _sample(rng: random.Random, candidates: List[VerificationPair], n: int, kind: str) -> List[VerificationPair]:
    if n < 0:
        raise ValueError("requested {} pair count must be >= 0, got {}".format(kind, n))
    if n > len(candidates):
        raise ValueError(
            "requested {} {} pairs, but only {} unique candidates are available "
            "for this dataset; reduce n_{} or provide more sessions/users".format(
                n, kind, len(candidates), kind
            )
        )
    if n == 0:
        return []
    return rng.sample(candidates, n)


def generate_pairs(
    entries: Sequence[Mapping[str, Any]],
    n_genuine: int = 1,
    n_impostor: int = 1,
    seed: int = 0,
    partitions: Optional[Mapping[str, Sequence[Mapping[str, Any]]]] = None,
) -> List[VerificationPair]:
    """Generate deterministic genuine/impostor session pairs.

    Args:
        entries: list of Phase 4 dataset entries (the full pool).
        n_genuine: how many same-user pairs to produce (``>= 0``).
        n_impostor: how many different-user pairs to produce (``>= 0``).
        seed: RNG seed — identical seeds reproduce identical pair sets.
        partitions: optional mapping ``name -> entries`` (e.g.
            ``{"train": train, "test": test}``). When supplied, pairs are built
            **within each partition only**; a pair never mixes sessions from
            two different partitions, preventing train/test leakage.

    Returns:
        A list of :class:`VerificationPair` (length ``n_genuine + n_impostor``,
        genuine pairs first, then impostor pairs).

    Raises:
        ValueError: on invalid arguments, empty pools, or impossible requests
            (e.g. fewer than 2 distinct sessions of a user for genuine pairs,
            or fewer than 2 distinct users for impostor pairs).
    """
    if isinstance(n_genuine, bool) or isinstance(n_impostor, bool):
        raise ValueError("pair counts must be integers, not booleans")
    if not isinstance(n_genuine, int) or not isinstance(n_impostor, int):
        raise TypeError(
            "n_genuine and n_impostor must be integers, got {!r} and {!r}".format(
                type(n_genuine).__name__, type(n_impostor).__name__
            )
        )
    if n_genuine < 0 or n_impostor < 0:
        raise ValueError(
            "pair counts must be >= 0, got n_genuine={}, n_impostor={}".format(
                n_genuine, n_impostor
            )
        )
    if not isinstance(seed, int):
        raise TypeError("seed must be an integer, got {!r}".format(type(seed).__name__))

    if partitions is None:
        pools: List[Tuple[str, List[Dict[str, Any]]]] = [("all", list(entries))]
    else:
        if not isinstance(partitions, Mapping) or not partitions:
            raise ValueError("partitions must be a non-empty mapping of name -> entries")
        pools = [
            (name, list(pool))
            for name, pool in partitions.items()
        ]
        # Every partition pool is validated here; the union of partitions is
        # expected to cover the entries (but is not required to).
        for name, pool in pools:
            _validate_pool(pool)

    genuine_all: List[VerificationPair] = []
    impostor_all: List[VerificationPair] = []
    rng = random.Random(seed)

    for _name, pool in pools:
        genuine_all.extend(genuine_candidates(pool))
        impostor_all.extend(impostor_candidates(pool))

    rng.shuffle(genuine_all)
    rng.shuffle(impostor_all)

    genuine = _sample(rng, genuine_all, n_genuine, "genuine")
    impostor = _sample(rng, impostor_all, n_impostor, "impostor")

    return genuine + impostor


def assert_no_cross_split_pairs(
    pairs: Sequence[VerificationPair],
    train: Sequence[Mapping[str, Any]],
    test: Sequence[Mapping[str, Any]],
) -> None:
    """Raise :class:`ValueError` if any pair straddles the train/test boundary.

    A pair may contain two train sessions, or two test sessions, but never one
    train and one test session.
    """
    train_ids = {e["session_id"] for e in train}
    test_ids = {e["session_id"] for e in test}
    for i, pair in enumerate(pairs):
        in_train_a = pair.session_id_a in train_ids
        in_test_a = pair.session_id_a in test_ids
        in_train_b = pair.session_id_b in train_ids
        in_test_b = pair.session_id_b in test_ids
        if not (in_train_a or in_test_a) or not (in_train_b or in_test_b):
            raise ValueError(
                "pair[{}] references session(s) outside the supplied train/test "
                "splits".format(i)
            )
        if (in_train_a and in_test_b) or (in_test_a and in_train_b):
            raise ValueError(
                "pair[{}] mixes train/test sessions ({} / {})".format(
                    i, pair.session_id_a, pair.session_id_b
                )
            )


def pairs_to_dicts(pairs: Sequence[VerificationPair]) -> List[Dict[str, Any]]:
    """Convert a list of pairs to metadata dicts (for inspection/logging)."""
    return [pair.to_dict() for pair in pairs]


__all__ = [
    "GENUINE",
    "IMPOSTOR",
    "VALID_LABELS",
    "VerificationPair",
    "assert_no_cross_split_pairs",
    "generate_pairs",
    "genuine_candidates",
    "group_by_user",
    "impostor_candidates",
    "pairs_to_dicts",
]