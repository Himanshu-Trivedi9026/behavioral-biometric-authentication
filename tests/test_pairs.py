"""
Phase 6 pair-generation tests.

Run from the repo root::

    .venv/bin/python -m pytest tests/test_pairs.py -v
"""

import pytest

from dataset.generator import SyntheticDatasetGenerator
from ml.verification.pairs import (
    GENUINE,
    IMPOSTOR,
    VerificationPair,
    assert_no_cross_split_pairs,
    generate_pairs,
    genuine_candidates,
    group_by_user,
    impostor_candidates,
    pairs_to_dicts,
)
from dataset.split import split_train_test


def _dataset(seed=0, n_users=5, sessions=10):
    return SyntheticDatasetGenerator(seed=seed, n_users=n_users, sessions_per_user=sessions).generate()


def _signature(pairs):
    return [
        (p.session_id_a, p.session_id_b, p.label)
        for p in pairs
    ]


def _entry(session_id, user_id, kb=(1,), mo=(1,)):
    return {
        "user_id": user_id,
        "session_id": session_id,
        "keyboard_sequence": [{"hold_time": float(kb[0]), "flight_time": float(kb[0])}],
        "mouse_sequence": [
            {"dx": float(mo[0]), "dy": 0.0, "dt": 16.0, "distance": float(mo[0]), "speed": 0.1}
        ],
        "mouse_action_events": [],
    }


# ---------------------------------------------------------------------------
# Labelling
# ---------------------------------------------------------------------------

class TestLabelling:

    def test_same_user_pairs_are_genuine(self):
        pairs = generate_pairs(_dataset(seed=1), n_genuine=5, n_impostor=0)
        assert pairs
        assert all(p.label == GENUINE for p in pairs)
        assert all(p.is_genuine for p in pairs)
        assert all(p.user_a == p.user_b for p in pairs)

    def test_different_user_pairs_are_impostor(self):
        pairs = generate_pairs(_dataset(seed=1), n_genuine=0, n_impostor=5)
        assert pairs
        assert all(p.label == IMPOSTOR for p in pairs)
        assert all(not p.is_genuine for p in pairs)
        assert all(p.user_a != p.user_b for p in pairs)

    def test_pair_label_validated(self):
        entries = _dataset(seed=1)
        # entries[0] belongs to user_001, entries[10] to user_002
        a, b = entries[0], entries[10]
        assert a["user_id"] == "user_001"
        assert b["user_id"] == "user_002"
        with pytest.raises(ValueError, match="label"):
            VerificationPair(session_a=a, session_b=b, label=GENUINE)
        with pytest.raises(ValueError, match="label"):
            VerificationPair(session_a=entries[0], session_b=entries[1], label=IMPOSTOR)

    def test_non_binary_label_rejected(self):
        a, b = _dataset(seed=1)[:2]
        with pytest.raises(ValueError, match="1 \\(genuine\\) or 0"):
            VerificationPair(session_a=a, session_b=b, label=2)


class TestSelfPairing:

    def test_session_never_paired_with_itself(self):
        pairs = generate_pairs(_dataset(seed=2), n_genuine=20, n_impostor=20)
        for p in pairs:
            assert p.session_id_a != p.session_id_b

    def test_manual_self_pair_rejected(self):
        entry = _dataset(seed=2)[0]
        with pytest.raises(ValueError, match="itself"):
            VerificationPair(session_a=entry, session_b=entry, label=GENUINE)

    def test_pair_session_ids_respected_across_both_types(self):
        for p in generate_pairs(_dataset(seed=3), n_genuine=10, n_impostor=10):
            assert p.session_id_a != p.session_id_b


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_same_seed_identical_pairs(self):
        entries = _dataset(seed=4)
        a = generate_pairs(entries, n_genuine=7, n_impostor=9, seed=99)
        b = generate_pairs(entries, n_genuine=7, n_impostor=9, seed=99)
        assert _signature(a) == _signature(b)

    def test_same_seed_identical_object_identity(self):
        entries = _dataset(seed=4)
        a = generate_pairs(entries, n_genuine=7, n_impostor=9, seed=5)
        b = generate_pairs(entries, n_genuine=7, n_impostor=9, seed=5)
        # The dataset entries themselves are referenced, so pairs should match
        # field-for-field as well.
        assert [p.to_dict() for p in a] == [p.to_dict() for p in b]

    def test_different_seed_differs(self):
        entries = _dataset(seed=4)
        a = generate_pairs(entries, n_genuine=10, n_impostor=10, seed=1)
        b = generate_pairs(entries, n_genuine=10, n_impostor=10, seed=2)
        assert _signature(a) != _signature(b)


# ---------------------------------------------------------------------------
# Counts & validation
# ---------------------------------------------------------------------------

class TestCounts:

    def test_requested_counts_respected(self):
        pairs = generate_pairs(_dataset(seed=0), n_genuine=12, n_impostor=17, seed=3)
        assert len(pairs) == 12 + 17
        assert sum(p.is_genuine for p in pairs) == 12
        assert sum(not p.is_genuine for p in pairs) == 17

    def test_zero_counts_ok(self):
        pairs = generate_pairs(_dataset(seed=0), n_genuine=0, n_impostor=0)
        assert pairs == []

    def test_impossible_genuine_raises(self):
        entries = _dataset(seed=0)
        with pytest.raises(ValueError, match="genuine"):
            generate_pairs(entries, n_genuine=99999, n_impostor=0)

    def test_impossible_impostor_raises(self):
        entries = _dataset(seed=0)
        with pytest.raises(ValueError, match="impostor"):
            generate_pairs(entries, n_genuine=0, n_impostor=99999)

    def test_negative_count_raises(self):
        with pytest.raises(ValueError, match=">= 0"):
            generate_pairs(_dataset(seed=0), n_genuine=-1)

    def test_bool_count_rejected(self):
        with pytest.raises(ValueError, match="booleans"):
            generate_pairs(_dataset(seed=0), n_genuine=True)

    def test_non_int_count_rejected(self):
        with pytest.raises(TypeError, match="integers"):
            generate_pairs(_dataset(seed=0), n_genuine="3")

    def test_one_user_no_impostors_possible(self):
        entries = _dataset(seed=0, n_users=1, sessions=3)
        genuine = generate_pairs(entries, n_genuine=1, n_impostor=0)
        assert genuine and genuine[0].is_genuine
        with pytest.raises(ValueError, match="impostor"):
            generate_pairs(entries, n_genuine=0, n_impostor=1)

    def test_single_session_user_cannot_form_genuine(self):
        entries = [
            _entry("s1", "u1"),
        ]
        with pytest.raises(ValueError, match="genuine"):
            generate_pairs(entries, n_genuine=1)

    def test_empty_dataset_rejected(self):
        with pytest.raises(ValueError, match="at least one session"):
            generate_pairs([], n_genuine=1)

    def test_unsupported_entry_type_rejected(self):
        with pytest.raises(TypeError, match="entry dict"):
            generate_pairs("not-a-list", n_genuine=1)


# ---------------------------------------------------------------------------
# Identity separation
# ---------------------------------------------------------------------------

class TestIdentitySeparation:

    def test_identity_only_used_for_label_not_features(self):
        """Behavioural columns inside a pair contain no user identity."""
        pairs = generate_pairs(_dataset(seed=7), n_genuine=4, n_impostor=4)
        schema = {"hold_time", "flight_time"}
        for p in pairs:
            for sample in p.session_a["keyboard_sequence"]:
                assert set(sample.keys()) == schema
            for sample in p.session_b["keyboard_sequence"]:
                assert set(sample.keys()) == schema
            # mouse features carry only trajectory values
            for sample in p.session_a["mouse_sequence"]:
                assert "user_id" not in sample and "session_id" not in sample
                assert "dx" in sample and "dy" in sample

    def test_label_derived_from_user_relationship(self):
        pairs = generate_pairs(_dataset(seed=7), n_genuine=4, n_impostor=4)
        for p in pairs:
            expected = GENUINE if p.user_a == p.user_b else IMPOSTOR
            assert p.label == expected

    def test_pairs_carry_metadata_separately(self):
        pairs = generate_pairs(_dataset(seed=7), n_genuine=2, n_impostor=2)
        d = pairs_to_dicts(pairs)
        assert all(set(k) <= {"session_a_id", "session_b_id", "user_a", "user_b", "label"} for k in d)
        assert d[0]["label"] == GENUINE
        assert d[2]["label"] == IMPOSTOR

    def test_user_id_changes_pair_label_but_not_behaviour(self):
        """Two sessions with identical behaviour but different ids yield an
        impostor pair whose feature sequences are unchanged - identity is
        metadata, behaviour is the model signal."""
        s1 = _entry("s-1", "alice")
        s2 = _entry("s-2", "bob")
        pair = VerificationPair(session_a=s1, session_b=s2, label=IMPOSTOR)
        assert pair.label == IMPOSTOR
        assert pair.session_a["keyboard_sequence"] == s1["keyboard_sequence"]
        assert pair.session_b["keyboard_sequence"] == s2["keyboard_sequence"]


# ---------------------------------------------------------------------------
# Train/test split isolation
# ---------------------------------------------------------------------------

class TestSplitIsolation:

    def test_no_cross_split_pairs(self):
        entries = _dataset(seed=10)
        train, test = split_train_test(entries, seed=100)
        pairs = generate_pairs(
            entries,
            n_genuine=25,
            n_impostor=25,
            seed=0,
            partitions={"train": train, "test": test},
        )
        assert_no_cross_split_pairs(pairs, train, test)

    def test_generate_with_partitions_respects_boundaries(self):
        entries = _dataset(seed=11)
        train, test = split_train_test(entries, seed=200)
        pairs = generate_pairs(entries, n_genuine=10, n_impostor=10, seed=5,
                               partitions={"train": train, "test": test})
        train_ids = {e["session_id"] for e in train}
        test_ids = {e["session_id"] for e in test}
        for p in pairs:
            side_a = "train" if p.session_id_a in train_ids else "test"
            side_b = "train" if p.session_id_b in train_ids else "test"
            assert side_a == side_b, (p.session_id_a, p.session_id_b)

    def test_cross_split_pair_detector_raises(self):
        train, test = split_train_test(_dataset(seed=12), seed=3)
        train_session = train[0]
        test_session = next(e for e in test if e["user_id"] != train_session["user_id"])
        bad_pair = VerificationPair(session_a=train_session, session_b=test_session,
                                    label=IMPOSTOR)
        with pytest.raises(ValueError, match="mixes train/test"):
            assert_no_cross_split_pairs([bad_pair], train, test)

    def test_within_split_pair_passes_detector(self):
        train, test = split_train_test(_dataset(seed=12), seed=3)
        pairs = generate_pairs(train, n_genuine=3, n_impostor=3, seed=0)
        assert_no_cross_split_pairs(pairs, train, test)

    def test_invalid_partitions_argument(self):
        train, test = split_train_test(_dataset(seed=13), seed=3)
        with pytest.raises(ValueError, match="non-empty mapping"):
            generate_pairs(_dataset(seed=13), n_genuine=1, n_impostor=1, partitions={})

    def test_partition_pool_validated(self):
        with pytest.raises((ValueError, TypeError)):
            generate_pairs([], n_genuine=1, n_impostor=1,
                           partitions={"bad": []})


# ---------------------------------------------------------------------------
# Candidate enumeration helpers
# ---------------------------------------------------------------------------

class TestCandidates:

    def test_group_by_user(self):
        entries = _dataset(seed=0, n_users=2, sessions=3)
        groups = group_by_user(entries)
        assert set(groups) == {"user_001", "user_002"}
        assert len(groups["user_001"]) == 3

    def test_genuine_candidates_count(self):
        candidates = genuine_candidates(_dataset(seed=0, n_users=2, sessions=4))
        # per user C(4,2)=6 -> 12 total
        assert len(candidates) == 12
        assert all(c.is_genuine for c in candidates)

    def test_impostor_candidates_count(self):
        candidates = impostor_candidates(_dataset(seed=0, n_users=2, sessions=3))
        assert len(candidates) == 9  # 3x3 cross-user pairs
        assert all(not c.is_genuine for c in candidates)

    def test_candidates_unique_session_pairs(self):
        pairs = genuine_candidates(_dataset(seed=0, n_users=3, sessions=4))
        signatures = {(p.session_id_a, p.session_id_b) for p in pairs}
        assert len(signatures) == len(pairs)


# ---------------------------------------------------------------------------
# Small-dataset safety
# ---------------------------------------------------------------------------

class TestSmallDatasets:

    def test_two_users_one_session_each_impostor_ok_genuine_impossible(self):
        entries = [_entry("a1", "u1"), _entry("b1", "u2")]
        impostor = generate_pairs(entries, n_genuine=0, n_impostor=1)
        assert len(impostor) == 1 and not impostor[0].is_genuine
        with pytest.raises(ValueError, match="genuine"):
            generate_pairs(entries, n_genuine=1, n_impostor=0)

    def test_two_sessions_same_user_single_genuine(self):
        entries = [_entry("a1", "u1"), _entry("a2", "u1")]
        pairs = generate_pairs(entries, n_genuine=1, n_impostor=0)
        assert len(pairs) == 1 and pairs[0].is_genuine