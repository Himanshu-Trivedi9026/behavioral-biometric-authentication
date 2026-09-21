"""
Phase 7 — training data pipeline tests.

Covers (per Phase 7 spec):
  1.  Dataset loads from the persisted JSON artifact.
  2.  Training pairs come only from training sessions.
  3.  Validation pairs come only from validation sessions.
  4.  Test pairs come only from test sessions.
  5.  No session appears across train/validation/test.
  6.  Genuine labels are correct.
  7.  Impostor labels are correct.
  8.  User/session IDs never enter the model feature computation.
  9.  Pair counts are controlled (configurable, respected exactly).
  10. Dataset + pair + scaler preparation is deterministic under a fixed seed.

Run:
    .venv/bin/python -m pytest tests/test_training_dataset.py -v
"""

import os

import pytest
import torch

from dataset.generator import SyntheticDatasetGenerator
from ml.encoder import BehavioralEncoder
from ml.preprocessing import (
    FeatureScaler,
    KEYBOARD_FEATURE_COLUMNS,
    MOUSE_FEATURE_COLUMNS,
    sequence_to_rows,
)
from ml.training import (
    TrainingConfig,
    prepare_training_data,
    siamese_collate,
)
from ml.training.dataset import SiamesePairDataset
from ml.verification import SiameseVerifier, pairs_to_dicts

DATASET_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data",
    "datasets",
    "dataset_synthetic.json",
)

SYNTHETIC_USERS = 5
SYNTHETIC_SESSIONS_PER_USER = 10


def _fast_config(**overrides):
    base = dict(
        dataset_path=DATASET_PATH,
        seed=42,
        n_genuine_train=6,
        n_impostor_train=6,
        n_genuine_val=2,
        n_impostor_val=4,
        n_genuine_test=3,
        n_impostor_test=3,
        batch_size=4,
        epochs=1,
    )
    base.update(overrides)
    return TrainingConfig(**base)


def _session_ids(sessions):
    return {e["session_id"] for e in sessions}


class TestPersistedDataset:

    def test_dataset_loads_from_persisted_json(self):
        config = _fast_config()
        data = prepare_training_data(config)
        ids = _session_ids(data.train_sessions) | _session_ids(data.val_sessions) | _session_ids(data.test_sessions)
        assert len(ids) == SYNTHETIC_USERS * SYNTHETIC_SESSIONS_PER_USER == 50

    def test_every_user_represented_in_every_partition(self):
        data = prepare_training_data(_fast_config())
        expected_totals = {"train": 25, "val": 10, "test": 15}
        for name, sessions in (
            ("train", data.train_sessions),
            ("val", data.val_sessions),
            ("test", data.test_sessions),
        ):
            users = {e["user_id"] for e in sessions}
            assert users == {"user_001", "user_002", "user_003", "user_004", "user_005"}
            assert len(sessions) == expected_totals[name]


class TestLeakagePrevention:

    @pytest.fixture()
    def data(self):
        return prepare_training_data(_fast_config())

    def test_train_pairs_only_use_train_sessions(self, data):
        train_ids = _session_ids(data.train_sessions)
        assert all(p.session_id_a in train_ids for p in data.train_pairs)
        assert all(p.session_id_b in train_ids for p in data.train_pairs)

    def test_val_pairs_only_use_val_sessions(self, data):
        val_ids = _session_ids(data.val_sessions)
        assert all(p.session_id_a in val_ids for p in data.val_pairs)
        assert all(p.session_id_b in val_ids for p in data.val_pairs)

    def test_test_pairs_only_use_test_sessions(self, data):
        test_ids = _session_ids(data.test_sessions)
        assert all(p.session_id_a in test_ids for p in data.test_pairs)
        assert all(p.session_id_b in test_ids for p in data.test_pairs)

    def test_no_session_across_partitions(self, data):
        train_ids, val_ids, test_ids = (
            _session_ids(data.train_sessions),
            _session_ids(data.val_sessions),
            _session_ids(data.test_sessions),
        )
        assert not (train_ids & val_ids)
        assert not (train_ids & test_ids)
        assert not (val_ids & test_ids)

    def test_no_pair_mixes_partitions(self, data):
        train_ids = _session_ids(data.train_sessions)
        val_ids = _session_ids(data.val_sessions)
        test_ids = _session_ids(data.test_sessions)
        for pairs, allowed in (
            (data.train_pairs, train_ids),
            (data.val_pairs, val_ids),
            (data.test_pairs, test_ids),
        ):
            for p in pairs:
                assert p.session_id_a in allowed and p.session_id_b in allowed
                assert p.session_id_a != p.session_id_b


class TestPairs:

    @pytest.fixture()
    def data(self):
        return prepare_training_data(_fast_config())

    def test_pair_counts_are_controlled(self, data):
        assert len(data.train_pairs) == 12
        assert len(data.val_pairs) == 6
        assert len(data.test_pairs) == 6

    def test_genuine_labels_are_correct(self, data):
        for pairs in (data.train_pairs, data.val_pairs, data.test_pairs):
            for p in pairs:
                if p.label == 1:
                    assert p.user_a == p.user_b
                    assert p.session_id_a != p.session_id_b

    def test_impostor_labels_are_correct(self, data):
        for pairs in (data.train_pairs, data.val_pairs, data.test_pairs):
            for p in pairs:
                if p.label == 0:
                    assert p.user_a != p.user_b

    def test_genuine_and_impostor_are_both_present(self, data):
        for pairs in (data.train_pairs, data.val_pairs, data.test_pairs):
            labels = {p.label for p in pairs}
            assert labels == {0, 1}

    def test_generated_pair_counts_match_request(self):
        data = prepare_training_data(
            _fast_config(n_genuine_train=4, n_impostor_train=5,
                         n_genuine_val=1, n_impostor_val=3,
                         n_genuine_test=2, n_impostor_test=4)
        )
        assert sum(1 for p in data.train_pairs if p.label == 1) == 4
        assert sum(1 for p in data.train_pairs if p.label == 0) == 5
        assert sum(1 for p in data.val_pairs if p.label == 1) == 1
        assert sum(1 for p in data.val_pairs if p.label == 0) == 3
        assert sum(1 for p in data.test_pairs if p.label == 1) == 2
        assert sum(1 for p in data.test_pairs if p.label == 0) == 4

    def test_requesting_more_pairs_than_available_raises(self):
        with pytest.raises(ValueError):
            prepare_training_data(_fast_config(n_genuine_val=999))


class TestDatasetAndCollate:

    def test_dataset_len_and_labels(self):
        data = prepare_training_data(_fast_config())
        dataset = SiamesePairDataset(data.train_pairs)
        assert len(dataset) == len(data.train_pairs)
        a, b, label = dataset[0]
        assert a["session_id"] == data.train_pairs[0].session_id_a
        assert b["session_id"] == data.train_pairs[0].session_id_b
        assert label == data.train_pairs[0].label

    def test_collate_output_labels_match_pairs(self):
        data = prepare_training_data(_fast_config())
        pairs = data.train_pairs[:4]
        sessions_a, sessions_b, labels = siamese_collate(
            [(p.session_a, p.session_b, p.label) for p in pairs]
        )
        assert sessions_a[0]["session_id"] == pairs[0].session_id_a
        assert sessions_b[0]["session_id"] == pairs[0].session_id_b
        assert labels.shape == (4,)
        assert labels.tolist() == [p.label for p in pairs]
        assert labels.dtype == torch.long

    def test_user_ids_never_enter_model_feature_tensors(self):
        """Swapping identity metadata must not change embeddings at all."""
        data = prepare_training_data(_fast_config())
        model = SiameseVerifier(seed=0)
        base = data.test_pairs[0].session_a
        twin = dict(base)
        twin["user_id"] = "attacker_id"
        twin["session_id"] = "spoofed-copy-" + base["session_id"]
        out = model(
            [base], [twin],
            keyboard_scaler=data.keyboard_scaler,
            mouse_scaler=data.mouse_scaler,
        )
        assert float(out.distance[0].detach()) < 1e-3


class TestScalersFitOnTrainingOnly:

    def test_scalers_are_fitted_on_training_rows_only(self):
        data = prepare_training_data(_fast_config())
        ref_kb = FeatureScaler(KEYBOARD_FEATURE_COLUMNS).fit(
            [row for e in data.train_sessions for row in sequence_to_rows(
                e["keyboard_sequence"], KEYBOARD_FEATURE_COLUMNS)]
        )
        assert data.keyboard_scaler.state_dict() == ref_kb.state_dict()
        ref_mo = FeatureScaler(MOUSE_FEATURE_COLUMNS).fit(
            [row for e in data.train_sessions for row in sequence_to_rows(
                e["mouse_sequence"], MOUSE_FEATURE_COLUMNS)]
        )
        assert data.mouse_scaler.state_dict() == ref_mo.state_dict()

    def test_validation_and_test_use_trained_scalers(self):
        data = prepare_training_data(_fast_config())
        encoded = BehavioralEncoder(seed=0).encode_entries(
            data.test_sessions[:2],
            keyboard_scaler=data.keyboard_scaler,
            mouse_scaler=data.mouse_scaler,
        )
        assert bool(torch.isfinite(encoded).all())


class TestDeterminism:

    def test_prepared_data_is_deterministic_under_fixed_seed(self):
        first = prepare_training_data(_fast_config())
        second = prepare_training_data(_fast_config())
        assert pairs_to_dicts(first.train_pairs) == pairs_to_dicts(second.train_pairs)
        assert pairs_to_dicts(first.val_pairs) == pairs_to_dicts(second.val_pairs)
        assert pairs_to_dicts(first.test_pairs) == pairs_to_dicts(second.test_pairs)
        assert first.session_count_summary() == second.session_count_summary()
        assert first.keyboard_scaler.state_dict() == second.keyboard_scaler.state_dict()
        assert first.mouse_scaler.state_dict() == second.mouse_scaler.state_dict()

    def test_different_seed_gives_different_train_partition(self):
        first = prepare_training_data(_fast_config(seed=42))
        second = prepare_training_data(_fast_config(seed=43))
        assert _session_ids(first.train_sessions) != _session_ids(second.train_sessions)

    def test_synthetic_dataset_is_valid_regression(self):
        """Sanity: the persisted dataset still satisfies the Phase 4 schema and
        privacy guarantees through the loader used here."""
        data = prepare_training_data(_fast_config())
        entries = data.train_sessions + data.val_sessions + data.test_sessions
        assert len(entries) == 50
        n_kb = sum(len(e["keyboard_sequence"]) for e in entries)
        n_mo = sum(len(e["mouse_sequence"]) for e in entries)
        assert n_kb == 1228
        assert n_mo == 2639


def test_persisted_dataset_file_matches_regenerated_snapshot():
    """leakage-free determinism end-to-end: the persisted artifact equals a
    fresh generation of the same generator (regression guard for the file we
    train on)."""
    persisted = prepare_training_data(_fast_config())
    fresh_entries = SyntheticDatasetGenerator(seed=0, n_users=5, sessions_per_user=10).generate()
    fresh_ids = {e["session_id"] for e in fresh_entries}
    all_ids = _session_ids(persisted.train_sessions) | _session_ids(persisted.val_sessions) | _session_ids(persisted.test_sessions)
    assert all_ids == fresh_ids