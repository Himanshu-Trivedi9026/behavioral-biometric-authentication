"""
Phase 4 dataset tests.

Run from the repo root::

    .venv/bin/python -m pytest tests/test_dataset.py -v
"""

import copy
import json
import os

import pytest

from dataset.generator import SyntheticDatasetGenerator
from dataset.loader import (
    DatasetLoadError,
    DatasetSchemaError,
    load_dataset,
    load_processed_dir,
    load_raw_files,
    processed_to_entry,
    raw_to_entry,
    save_dataset,
)
from dataset.schema import (
    KEYBOARD_FEATURE_COLUMNS,
    MOUSE_FEATURE_COLUMNS,
    assert_privacy,
    privacy_report,
    validate_dataset,
    validate_dataset_entry,
)
from dataset.split import (
    assert_no_leakage,
    session_ids,
    split_report,
    split_train_test,
)
from ml.preprocessing import process_session

FIXTURES = os.path.join(os.path.dirname(__file__), "fixtures")
VALID_RAW = os.path.join(FIXTURES, "session_valid.json")


def _valid_entry(session_id="sess-1", user_id="user_1"):
    return {
        "user_id": user_id,
        "session_id": session_id,
        "keyboard_sequence": [
            {"hold_time": 60.0, "flight_time": 0.0},
            {"hold_time": 55.0, "flight_time": 150.0},
        ],
        "mouse_sequence": [
            {"dx": 10.0, "dy": 5.0, "dt": 20.0, "distance": 11.18, "speed": 0.559},
        ],
        "mouse_action_events": [
            {"event": "mousedown", "x": 10.0, "y": 10.0, "timestamp": 100.0},
            {"event": "mouseup", "x": 10.0, "y": 10.0, "timestamp": 120.0},
        ],
        "metadata": {"source": None, "preprocessing": {"version": "test"}},
    }


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------

class TestSchemaValidation:

    def test_valid_entry_passes(self):
        assert validate_dataset_entry(_valid_entry()) == []

    def test_missing_user_id(self):
        entry = _valid_entry()
        del entry["user_id"]
        errors = validate_dataset_entry(entry)
        assert any("user_id" in e for e in errors)

    def test_missing_session_id(self):
        entry = _valid_entry()
        entry["session_id"] = ""
        errors = validate_dataset_entry(entry)
        assert any("session_id" in e for e in errors)

    def test_missing_required_sequences(self):
        entry = _valid_entry()
        del entry["keyboard_sequence"]
        del entry["mouse_sequence"]
        errors = validate_dataset_entry(entry)
        assert any("keyboard_sequence" in e for e in errors)
        assert any("mouse_sequence" in e for e in errors)

    def test_keyboard_feature_must_be_numeric(self):
        entry = _valid_entry()
        entry["keyboard_sequence"][0]["hold_time"] = "abc"
        errors = validate_dataset_entry(entry)
        assert any("hold_time" in e and "finite" in e for e in errors)

    def test_keyboard_feature_unknown_column(self):
        entry = _valid_entry()
        entry["keyboard_sequence"][0]["extra"] = 1.0
        errors = validate_dataset_entry(entry)
        assert any("unexpected feature" in e for e in errors)

    def test_mouse_feature_validation(self):
        entry = _valid_entry()
        entry["mouse_sequence"][0] = {"dx": None}
        errors = validate_dataset_entry(entry)
        assert any("missing feature" in e for e in errors)

    def test_infinite_feature_rejected(self):
        entry = _valid_entry()
        entry["mouse_sequence"][0]["speed"] = float("inf")
        errors = validate_dataset_entry(entry)
        assert any("finite" in e for e in errors)

    def test_duplicate_session_detected(self):
        entries = [_valid_entry("s-1"), _valid_entry("s-1", "user_2")]
        errors = validate_dataset(entries)
        assert any("duplicate session_id" in e for e in errors)

    def test_dataset_not_list(self):
        errors = validate_dataset({})
        assert errors

    def test_entry_not_object(self):
        assert validate_dataset_entry([1, 2, 3])


# ---------------------------------------------------------------------------
# Privacy checks
# ---------------------------------------------------------------------------

class TestPrivacy:

    def test_valid_entry_privacy_clean(self):
        assert privacy_report(_valid_entry()) == []
        assert_privacy([_valid_entry()])

    def test_forbidden_key_name_detected(self):
        entry = copy.deepcopy(_valid_entry())
        entry["keyboard_sequence"][0]["key"] = "a"  # would violate the contract
        findings = privacy_report(entry)
        assert any("forbidden key name" in f for f in findings)
        with pytest.raises(ValueError):
            assert_privacy([entry])

    def test_generated_dataset_privacy_clean(self):
        entries = SyntheticDatasetGenerator(seed=1).generate()
        assert_privacy(entries)  # must not raise

    def test_serialized_text_scan(self):
        entry = copy.deepcopy(_valid_entry())
        entry["metadata"] = {"source": {"note": "password hint: hunter2"}}
        findings = privacy_report(entry)
        assert findings


# ---------------------------------------------------------------------------
# Synthetic generation
# ---------------------------------------------------------------------------

class TestSyntheticGeneration:

    def test_scale_and_user_ids(self):
        gen = SyntheticDatasetGenerator(seed=0, n_users=5, sessions_per_user=10)
        entries = gen.generate()
        assert len(entries) == 50
        users = sorted({e["user_id"] for e in entries})
        assert users == ["user_001", "user_002", "user_003", "user_004", "user_005"]
        per_user = {u: sum(1 for e in entries if e["user_id"] == u) for u in users}
        assert set(per_user.values()) == {10}

    def test_valid_schema(self):
        entries = SyntheticDatasetGenerator(seed=3).generate()
        assert validate_dataset(entries) == []
        assert_privacy(entries)

    def test_deterministic_same_seed(self):
        a = SyntheticDatasetGenerator(seed=42).generate()
        b = SyntheticDatasetGenerator(seed=42).generate()
        assert a == b

    def test_different_seed_different(self):
        a = SyntheticDatasetGenerator(seed=42).generate()
        b = SyntheticDatasetGenerator(seed=43).generate()
        assert a != b

    def test_no_duplicate_sessions(self):
        entries = SyntheticDatasetGenerator(seed=7).generate()
        ids = [e["session_id"] for e in entries]
        assert len(ids) == len(set(ids))

    def test_no_identical_duplicate_sessions(self):
        entries = SyntheticDatasetGenerator(seed=1).generate()
        # Sessions must not simply be repeated copies.
        from collections import Counter
        signatures = Counter(json.dumps(e, sort_keys=True) for e in entries)
        most_common = signatures.most_common(1)[0][1]
        assert most_common == 1

    def test_session_variation_within_user(self):
        entries = SyntheticDatasetGenerator(seed=2).generate()
        user_entries = [e for e in entries if e["user_id"] == "user_001"]
        lengths = [len(e["keyboard_sequence"]) for e in user_entries]
        assert len(set(lengths)) > 1  # sessions differ in length

    def test_samples_present(self):
        entries = SyntheticDatasetGenerator(seed=0).generate()
        for e in entries:
            assert len(e["keyboard_sequence"]) > 0
            assert len(e["mouse_sequence"]) > 0

    def test_keyboard_features_plausible(self):
        entries = SyntheticDatasetGenerator(seed=0).generate()
        for e in entries:
            for sample in e["keyboard_sequence"]:
                assert sample["hold_time"] >= 0.0
                assert sample["flight_time"] >= 0.0

    def test_mouse_features_plausible(self):
        entries = SyntheticDatasetGenerator(seed=0).generate()
        for e in entries:
            for sample in e["mouse_sequence"]:
                assert sample["dt"] > 0.0
                assert sample["distance"] >= 0.0
                assert sample["speed"] >= 0.0

    def test_action_events_schema(self):
        entries = SyntheticDatasetGenerator(seed=0).generate()
        for e in entries:
            for ev in e["mouse_action_events"]:
                assert ev["event"] in ("mousedown", "mouseup")


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------

class TestSplit:

    def _dataset(self, seed=0):
        return SyntheticDatasetGenerator(seed=seed).generate()

    def test_no_leakage(self):
        train, test = split_train_test(self._dataset())
        assert_no_leakage(train, test)
        assert session_ids(train).isdisjoint(session_ids(test))

    def test_per_user_both_splits(self):
        train, test = split_train_test(self._dataset())
        for u in {"user_001", "user_002", "user_003", "user_004", "user_005"}:
            assert any(e["user_id"] == u for e in train)
            assert any(e["user_id"] == u for e in test)

    def test_deterministic(self):
        a = split_train_test(self._dataset(), seed=11)
        b = split_train_test(self._dataset(), seed=11)
        assert a == b

    def test_split_counts(self):
        train, test = split_train_test(self._dataset(), seed=0, train_fraction=0.7)
        assert len(train) == 35
        assert len(test) == 15

    def test_report_counts_match(self):
        entries = self._dataset()
        train, test = split_train_test(entries, seed=3, train_fraction=0.6)
        report = split_report(entries, train, test)
        for user_id, counts in report.items():
            assert counts["total"] == 10
            assert counts["train"] == 6
            assert counts["test"] == 4
            assert counts["train"] + counts["test"] == 10

    def test_minimums_respected(self):
        train, test = split_train_test(self._dataset(), train_fraction=0.05)
        # fraction 0.05 -> round(0.5)=0 -> clamped up to min_train=1
        for user_id in {"user_001", "user_002", "user_003", "user_004", "user_005"}:
            n_train = sum(1 for e in train if e["user_id"] == user_id)
            n_test = sum(1 for e in test if e["user_id"] == user_id)
            assert n_train >= 1
            assert n_test >= 1

    def test_insufficient_sessions_raises(self):
        entries = SyntheticDatasetGenerator(seed=0, n_users=2, sessions_per_user=2).generate()
        with pytest.raises(ValueError):
            split_train_test(entries, min_train=2, min_test=2)

    def test_single_user_can_split(self):
        # A single-user dataset splits fine (1 user is still enrollment/eval);
        # the baseline *evaluation* requires 2+ users for impostors instead.
        entries = SyntheticDatasetGenerator(seed=0, n_users=1, sessions_per_user=3).generate()
        train, test = split_train_test(entries)
        assert len(train) == 2
        assert len(test) == 1

    def test_leakage_detection_raises(self):
        entries = self._dataset()
        train, _ = split_train_test(entries, seed=0)
        # Steal one session from test to simulate leakage.
        leaky_test = [e for e in entries if e["session_id"] not in session_ids(train)]
        assert_no_leakage(train, leaky_test)
        with pytest.raises(ValueError):
            assert_no_leakage(train, [train[0]] + leaky_test)


# ---------------------------------------------------------------------------
# Loading real Phase 3 sessions
# ---------------------------------------------------------------------------

class TestLoader:

    def test_processed_to_entry_roundtrip(self, tmp_path):
        raw = _load_raw_fixture()
        processed = process_session(raw)
        entry = processed_to_entry(processed, "real_user")
        assert entry["user_id"] == "real_user"
        assert entry["session_id"] == raw["session_id"]
        assert entry["keyboard_sequence"] == processed["keyboard_sequence"]
        assert entry["mouse_sequence"] == processed["mouse_sequence"]
        assert validate_dataset_entry(entry) == []
        assert_privacy([entry])

    def test_raw_to_entry_uses_phase_three(self):
        raw = _load_raw_fixture()
        entry = raw_to_entry(raw, "real_user")
        processed = process_session(raw)
        assert entry["keyboard_sequence"] == processed["keyboard_sequence"]
        assert entry["mouse_sequence"] == processed["mouse_sequence"]

    def test_raw_to_entry_invalid_raises(self):
        with pytest.raises(DatasetLoadError):
            raw_to_entry({"session_id": "x"}, "user")

    def test_load_processed_dir(self, tmp_path):
        raw = _load_raw_fixture()
        processed = process_session(raw)
        out_dir = tmp_path / "processed"
        out_dir.mkdir()
        with open(out_dir / "p1.json", "w") as f:
            json.dump(processed, f)
        entries = load_processed_dir(str(out_dir), user_id_map={"550e8400-e29b-41d4-a716-446655440000": "u_a"})
        assert len(entries) == 1
        assert entries[0]["user_id"] == "u_a"

    def test_load_processed_dir_missing_user_map(self, tmp_path):
        raw = _load_raw_fixture()
        processed = process_session(raw)
        out_dir = tmp_path / "processed2"
        out_dir.mkdir()
        with open(out_dir / "p2.json", "w") as f:
            json.dump(processed, f)
        with pytest.raises(DatasetLoadError):
            load_processed_dir(str(out_dir), user_id_map=None)

    def test_load_raw_files(self, tmp_path):
        dst = tmp_path / "raw.json"
        with open(dst, "w") as f:
            json.dump(_load_raw_fixture(), f)
        entries = load_raw_files([str(dst)], user_id_map={"550e8400-e29b-41d4-a716-446655440000": "u_b"})
        assert len(entries) == 1
        assert entries[0]["user_id"] == "u_b"
        assert len(entries[0]["keyboard_sequence"]) > 0

    def test_user_id_map_callable(self):
        entry = processed_to_entry(process_session(_load_raw_fixture()), "cb")
        assert entry["user_id"] == "cb"

    def test_save_load_roundtrip(self, tmp_path):
        entries = SyntheticDatasetGenerator(seed=9).generate()
        path = str(tmp_path / "ds.json")
        save_dataset(entries, path)
        loaded = load_dataset(path)
        assert loaded == entries

    def test_load_dataset_rejects_foreign_file(self, tmp_path):
        path = tmp_path / "other.json"
        with open(path, "w") as f:
            f.write('{"nope": true}')
        with pytest.raises(DatasetSchemaError):
            load_dataset(path)

    def test_load_dataset_rejects_bad_schema_version(self, tmp_path):
        entries = SyntheticDatasetGenerator(seed=9).generate()
        payload = {
            "format": "behavioral-biometric-dataset",
            "schema_version": "0.0.0",
            "n_entries": len(entries),
            "entries": entries,
        }
        path = tmp_path / "old.json"
        with open(path, "w") as f:
            json.dump(payload, f)
        with pytest.raises(DatasetSchemaError):
            load_dataset(path)

    def test_processed_sessions_duplicate_detected(self, tmp_path):
        raw = _load_raw_fixture()
        processed = process_session(raw)
        out_dir = tmp_path / "dup"
        out_dir.mkdir()
        for name in ("a.json", "b.json"):
            with open(out_dir / name, "w") as f:
                json.dump(processed, f)
        with pytest.raises(DatasetSchemaError):
            load_processed_dir(str(out_dir), user_id_map={"550e8400-e29b-41d4-a716-446655440000": "u"})


def _load_raw_fixture():
    with open(VALID_RAW, "r") as f:
        return json.load(f)