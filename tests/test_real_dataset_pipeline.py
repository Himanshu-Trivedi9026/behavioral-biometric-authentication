"""
Phase 16A — real-data pipeline tests (local fixtures only).

Covers the privacy-preserving offline workflow:
raw Phase 2 browser sessions -> explicit participant mapping -> existing Phase 4
dataset -> user-aware 70/20/10 split -> train-only scalers -> read-only
checkpoint embeddings -> pairs within partitions -> metrics at the FIXED deployed
threshold (no calibration) -> genuine/impostor distance distributions.

Only local, generated fixtures are used — no human-collected data is required,
and no model artifact is ever modified.

Run:
    .venv/bin/python -m pytest tests/test_real_dataset_pipeline.py -v
"""

import json
import os
from pathlib import Path

import numpy as np
import pytest

from dataset import (
    assert_privacy,
    privacy_report,
    validate_dataset_entry,
)
from dataset.loader import DatasetLoadError, load_raw_files, raw_to_entry
from ml.training import TrainingConfig, prepare_training_data
from ml.evaluation.metrics import auc, eer, evaluate_at_threshold, roc_points

from scripts.build_real_dataset import build_real_dataset
from scripts.evaluate_real_behavior import (
    DEPLOYED_THRESHOLD,
    RECALIBRATION_STATEMENT,
    _distribution_stats,
    _fixed_threshold_metrics,
    render_text,
    run_evaluation,
)

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(_REPO, "models", "siamese_behavioral_encoder.pt")
PREPROCESSING_ARTIFACT = os.path.join(_REPO, "models", "behavioral_preprocessing.json")
VERIFICATION_ARTIFACT = os.path.join(_REPO, "models", "verification_config.json")

has_checkpoint = pytest.mark.skipif(
    not os.path.exists(CHECKPOINT_PATH),
    reason="Phase 7 checkpoint not present on disk",
)
has_artifacts = pytest.mark.skipif(
    not os.path.exists(VERIFICATION_ARTIFACT) or not os.path.exists(PREPROCESSING_ARTIFACT),
    reason="deployed model artifacts not present on disk",
)


# ---------------------------------------------------------------------------
# Fixture helpers (generated, NOT human data)
# ---------------------------------------------------------------------------


def make_raw_session(session_id: str, t0: float = 1000.0, n_keys: int = 30, n_moves: int = 80) -> dict:
    t = t0
    keyboard_events = []
    for i in range(n_keys):
        t += 40 + (i % 5)
        keyboard_events.append({"event_type": "keyboard", "event": "keydown", "timestamp": t})
        t += 40 + (i % 7)
        keyboard_events.append({"event_type": "keyboard", "event": "keyup", "timestamp": t})
    mouse_events = []
    x = y = 100.0
    for i in range(n_moves):
        t += 5
        x += (i % 7) - 3
        y += (i % 5) - 2
        mouse_events.append({"event_type": "mouse", "event": "mousemove", "x": x, "y": y, "timestamp": t})
    mouse_events.append({"event_type": "mouse", "event": "mousedown", "x": x, "y": y, "timestamp": t + 1})
    mouse_events.append({"event_type": "mouse", "event": "mouseup", "x": x, "y": y, "timestamp": t + 2})
    return {
        "session_id": session_id,
        "started_at": "2025-09-01T12:00:00.000Z",
        "ended_at": "2025-09-01T12:00:10.000Z",
        "timestamp_source": "monotonic high-resolution (performance.now, milliseconds)",
        "keyboard_events": keyboard_events,
        "mouse_events": mouse_events,
    }


def write_raw_dataset(tmp_path, n_users: int = 5, sessions: int = 10) -> dict:
    """Write ``n_users`` x ``sessions`` raw session files + a manifest.

    The generated sessions use neutral synthetic timings; they exist purely to
    exercise the pipeline and MUST NOT be mistaken for human data.
    """
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    manifest = {"participants": {}}
    t0 = 1000.0
    files = []
    for u in range(1, n_users + 1):
        user_id = "user_{:03d}".format(u)
        manifest["participants"][user_id] = []
        for s in range(1, sessions + 1):
            session_id = "{}-s{:02d}".format(user_id, s)
            filename = "{}.json".format(session_id)
            t0 += 17.0
            (raw_dir / filename).write_text(
                json.dumps(make_raw_session(session_id, t0)), encoding="utf-8"
            )
            manifest["participants"][user_id].append(filename)
            files.append(filename)
    manifest_path = raw_dir / "participant_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return {
        "raw_dir": str(raw_dir),
        "manifest": str(manifest_path),
        "files": files,
        "dataset": str(tmp_path / "dataset_real.json"),
    }


def _config_for(dataset_path: str, seed: int = 42) -> TrainingConfig:
    return TrainingConfig(
        dataset_path=dataset_path,
        seed=seed,
        checkpoint_path=CHECKPOINT_PATH,
        n_genuine_train=1,
        n_impostor_train=1,
        n_genuine_val=1,
        n_impostor_val=1,
        n_genuine_test=1,
        n_impostor_test=1,
    )


class TestRawToDataset:
    def test_raw_session_to_processed_entry(self):
        entry = raw_to_entry(make_raw_session("s-0001"), "user_001")
        assert entry["user_id"] == "user_001"
        assert entry["session_id"] == "s-0001"
        assert entry["keyboard_sequence"]
        assert entry["mouse_sequence"]
        assert entry["metadata"]["generated"] is False
        assert entry["metadata"]["source"]["timestamp_source"] == (
            "monotonic high-resolution (performance.now, milliseconds)"
        )
        assert validate_dataset_entry(entry) == []
        assert_privacy([entry])

    def test_missing_user_id_rejected(self, tmp_path):
        raw = tmp_path / "orphan.json"
        raw.write_text(json.dumps(make_raw_session("orphan-1")), encoding="utf-8")
        with pytest.raises(DatasetLoadError):
            load_raw_files([str(raw)], user_id_map=None)

    def test_explicit_user_id_mapping_by_session_id(self, tmp_path):
        raw = tmp_path / "session.json"
        raw.write_text(json.dumps(make_raw_session("mapped-1")), encoding="utf-8")
        entry = load_raw_files([str(raw)], {"mapped-1": "user_007"})[0]
        assert entry["user_id"] == "user_007"

    def test_explicit_user_id_mapping_by_filename(self, tmp_path):
        raw = tmp_path / "session_file.json"
        raw.write_text(json.dumps(make_raw_session("mapped-2")), encoding="utf-8")
        entry = load_raw_files([str(raw)], {"session_file.json": "user_008"})[0]
        assert entry["user_id"] == "user_008"

    def test_forbidden_key_detected_by_privacy_report(self):
        leaky = {
            "user_id": "u",
            "session_id": "s",
            "keyboard_sequence": [],
            "mouse_sequence": [],
            "metadata": {"generated": False, "key": "a"},
        }
        assert privacy_report(leaky)
        with pytest.raises(ValueError):
            assert_privacy([leaky])

    def test_clean_entry_passes_privacy_report(self):
        entry = raw_to_entry(make_raw_session("s-0001"), "user_001")
        assert privacy_report(entry) == []


class TestBuildDataset:
    def test_build_end_to_end_and_deterministic(self, tmp_path):
        loc = write_raw_dataset(tmp_path, n_users=5, sessions=10)
        summary = build_real_dataset(loc["raw_dir"], loc["manifest"], loc["dataset"])
        assert summary["n_users"] == 5
        assert summary["n_sessions"] == 50
        assert summary["generated"] is False

        second = str(tmp_path / "second.json")
        build_real_dataset(loc["raw_dir"], loc["manifest"], second)
        assert Path(loc["dataset"]).read_bytes() == Path(second).read_bytes()
        assert len(raw_to_entry(make_raw_session("x"), "y")["keyboard_sequence"]) > 0

    def test_build_skips_unlisted_files(self, tmp_path, capsys):
        loc = write_raw_dataset(tmp_path, n_users=2, sessions=3)
        (Path(loc["raw_dir"]) / "unlisted.json").write_text(
            json.dumps(make_raw_session("unlisted-1")), encoding="utf-8"
        )
        summary = build_real_dataset(loc["raw_dir"], loc["manifest"], loc["dataset"])
        assert summary["n_sessions"] == 6
        assert "NOT in the manifest" in capsys.readouterr().err

    def test_build_raises_on_missing_file(self, tmp_path):
        loc = write_raw_dataset(tmp_path, n_users=2, sessions=3)
        manifest = json.loads(Path(loc["manifest"]).read_text(encoding="utf-8"))
        manifest["participants"]["user_001"].append("ghost.json")
        Path(loc["manifest"]).write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="missing session file"):
            build_real_dataset(loc["raw_dir"], loc["manifest"], loc["dataset"])

    def test_build_raises_on_session_in_two_participants(self, tmp_path):
        loc = write_raw_dataset(tmp_path, n_users=2, sessions=3)
        manifest = json.loads(Path(loc["manifest"]).read_text(encoding="utf-8"))
        shared = manifest["participants"]["user_001"][0]
        manifest["participants"]["user_002"].append(shared)
        Path(loc["manifest"]).write_text(json.dumps(manifest), encoding="utf-8")
        with pytest.raises(ValueError, match="exactly one participant"):
            build_real_dataset(loc["raw_dir"], loc["manifest"], loc["dataset"])


@pytest.fixture(scope="module")
def split_data(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("split")
    loc = write_raw_dataset(tmp, n_users=5, sessions=8)
    build_real_dataset(loc["raw_dir"], loc["manifest"], loc["dataset"])
    return prepare_training_data(_config_for(loc["dataset"]))


@pytest.fixture(scope="module")
def evaluation_report(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("eval")
    loc = write_raw_dataset(tmp, n_users=5, sessions=10)
    build_real_dataset(loc["raw_dir"], loc["manifest"], loc["dataset"])
    return run_evaluation(loc["dataset"], output_dir=None)


class TestSplitLeakage:
    def test_every_user_in_every_partition(self, split_data):
        data = split_data
        users = [sorted({e["user_id"] for e in partition}) for partition in (
            data.train_sessions, data.val_sessions, data.test_sessions
        )]
        assert users[0] == users[1] == users[2] == ["user_001", "user_002", "user_003", "user_004", "user_005"]

    def test_no_session_leakage_across_partitions(self, split_data):
        data = split_data
        ids = [set(e["session_id"] for e in partition) for partition in (
            data.train_sessions, data.val_sessions, data.test_sessions
        )]
        assert ids[0].isdisjoint(ids[1])
        assert ids[0].isdisjoint(ids[2])
        assert ids[1].isdisjoint(ids[2])

    def test_no_cross_partition_pairs(self, split_data):
        data = split_data
        val_ids = {e["session_id"] for e in data.val_sessions}
        test_ids = {e["session_id"] for e in data.test_sessions}
        for pair in data.val_pairs:
            assert pair.session_id_a in val_ids and pair.session_id_b in val_ids
        for pair in data.test_pairs:
            assert pair.session_id_a in test_ids and pair.session_id_b in test_ids

    def test_deterministic_split_for_same_seed(self, split_data, tmp_path):
        loc = write_raw_dataset(tmp_path, n_users=5, sessions=8)
        build_real_dataset(loc["raw_dir"], loc["manifest"], loc["dataset"])
        again = prepare_training_data(_config_for(loc["dataset"]))
        assert [e["session_id"] for e in again.train_sessions] == [
            e["session_id"] for e in split_data.train_sessions
        ]


class TestFixedThresholdMetrics:
    def test_fixed_threshold_metrics_derive_precision_recall_f1(self):
        distances = [0.1, 0.5, 0.9, 0.4, 0.8, 1.1]
        labels = [1, 1, 1, 0, 0, 0]
        m = _fixed_threshold_metrics(distances, labels, threshold=0.5)
        assert m["tp"] == 2.0
        assert m["fp"] == 1.0
        assert m["fn"] == 1.0
        assert m["tn"] == 2.0
        assert m["tar"] == pytest.approx(2 / 3)
        assert m["far"] == pytest.approx(1 / 3)
        assert m["frr"] == pytest.approx(1 / 3)
        assert m["accuracy"] == pytest.approx(2 / 3)
        assert m["precision"] == pytest.approx(2 / 3)
        assert m["recall"] == pytest.approx(2 / 3)
        assert m["f1"] == pytest.approx(2 / 3)
        assert m["confusion_matrix"]["matrix"] == [[2.0, 1.0], [1.0, 2.0]]

    def test_distance_distribution_stats(self):
        dist = [1.0, 2.0, 3.0, 4.0, 100.0]
        stats = _distribution_stats(dist)
        assert stats["count"] == 5
        assert stats["mean"] == pytest.approx(22.0)
        assert stats["min"] == pytest.approx(1.0)
        assert stats["max"] == pytest.approx(100.0)
        assert stats["median"] == pytest.approx(3.0)

    def test_genuine_impostor_separation_metrics(self):
        genuine = [0.1, 0.1, 0.1, 0.1]
        impostor = [0.9, 0.9, 0.9, 0.9]
        roc = roc_points(genuine, impostor)
        assert auc(roc) == pytest.approx(1.0)
        assert eer(genuine, impostor)["eer"] == pytest.approx(0.0)
        m = _fixed_threshold_metrics(
            genuine + impostor, [1] * 4 + [0] * 4, threshold=0.5
        )
        assert m["tar"] == 1.0
        assert m["far"] == 0.0
        assert m["confusion_matrix"]["matrix"] == [[4.0, 0.0], [0.0, 4.0]]


@has_checkpoint
class TestRealEvaluation:
    def test_uses_fixed_deployed_threshold(self, evaluation_report):
        report = evaluation_report
        assert report["threshold"]["value"] == DEPLOYED_THRESHOLD
        assert report["measurement"]["recalibration_performed"] is False
        assert report["measurement"]["statement"] == RECALIBRATION_STATEMENT

    def test_split_counts(self, evaluation_report):
        assert evaluation_report["split"]["n_train_sessions"] == 25
        assert evaluation_report["split"]["n_dev_sessions"] == 10
        assert evaluation_report["split"]["n_test_sessions"] == 15

    def test_pairs_and_metrics_present(self, evaluation_report):
        report = evaluation_report
        for partition in ("dev", "test"):
            block = report[partition]
            assert block["pairs"]["genuine"] >= 1
            assert block["pairs"]["impostor"] >= 1
            m = block["metrics_at_fixed_threshold"]
            for key in ("tar", "far", "frr", "accuracy", "precision", "recall", "f1",
                        "tp", "tn", "fp", "fn", "confusion_matrix"):
                assert key in m
            assert block["auc"] == pytest.approx(auc(block["roc"]))
            assert "eer" in block and "threshold" in block["eer"]
            for kind in ("genuine", "impostor"):
                d = block["distributions"][kind]
                for key in ("count", "mean", "std", "min", "max", "median", "p25", "p75"):
                    assert key in d

    def test_leakage_checks_all_pass(self, evaluation_report):
        assert all(evaluation_report["leakage_checks"][key] is True for key in (
            "train_dev_disjoint", "train_test_disjoint", "dev_test_disjoint",
            "dev_pairs_within_dev_partition", "test_pairs_within_test_partition",
        ))

    def test_deterministic_across_runs(self, evaluation_report, tmp_path_factory):
        tmp = tmp_path_factory.mktemp("eval2")
        loc = write_raw_dataset(tmp, n_users=5, sessions=10)
        build_real_dataset(loc["raw_dir"], loc["manifest"], loc["dataset"])
        second = run_evaluation(loc["dataset"], output_dir=None)
        first_dump = {}
        second_dump = {}
        # the dataset path reflects where the fixtures happen to live; every
        # measured field must be identical across identical runs
        first_dump = {
            k: v for k, v in evaluation_report.items() if k != "output_files"
        }
        second_dump = {
            k: v for k, v in second.items() if k != "output_files"
        }
        first_dump["dataset"] = {k: v for k, v in evaluation_report["dataset"].items() if k != "path"}
        second_dump["dataset"] = {k: v for k, v in second["dataset"].items() if k != "path"}
        assert json.dumps(first_dump, sort_keys=True) == json.dumps(second_dump, sort_keys=True)

    def test_no_wallclock_activity_timestamps_in_report(self, evaluation_report):
        serialized = json.dumps(evaluation_report)
        assert "started_at" not in serialized
        assert "ended_at" not in serialized

    def test_report_text_states_no_recalibration(self, evaluation_report):
        assert RECALIBRATION_STATEMENT in render_text(evaluation_report)

    def test_writes_only_output_dir(self, tmp_path, tmp_path_factory):
        models_before = _list_recursive("models")
        ml_before = _list_recursive("ml")
        tmp = tmp_path_factory.mktemp("eval3")
        loc = write_raw_dataset(tmp, n_users=5, sessions=10)
        build_real_dataset(loc["raw_dir"], loc["manifest"], loc["dataset"])
        out = str(tmp / "res")
        run_evaluation(loc["dataset"], output_dir=out)
        assert os.path.exists(os.path.join(out, "real_behavior_report.json"))
        assert os.path.exists(os.path.join(out, "real_behavior_report.txt"))
        assert _list_recursive("models") == models_before
        assert _list_recursive("ml") == ml_before


@has_checkpoint
@has_artifacts
class TestArtifactsUntouched:
    def test_model_artifacts_unchanged(self, tmp_path_factory):
        targets = [CHECKPOINT_PATH, PREPROCESSING_ARTIFACT, VERIFICATION_ARTIFACT]
        hashes = {path: _sha256(path) for path in targets}
        listing = _list_recursive("models")

        tmp = tmp_path_factory.mktemp("art")
        loc = write_raw_dataset(tmp, n_users=5, sessions=10)
        build_real_dataset(loc["raw_dir"], loc["manifest"], loc["dataset"])
        run_evaluation(loc["dataset"], output_dir=None)

        for path, digest in hashes.items():
            assert _sha256(path) == digest
        assert _list_recursive("models") == listing

    def test_deployed_threshold_matches_artifact(self):
        with open(VERIFICATION_ARTIFACT, "r") as f:
            payload = json.load(f)
        assert float(payload["calibration"]["threshold"]) == DEPLOYED_THRESHOLD


def test_script_never_imports_calibration():
    source = Path(_REPO, "scripts", "evaluate_real_behavior.py").read_text(encoding="utf-8")
    assert "calibrate_threshold" not in source
    assert "DEPLOYED_THRESHOLD = 0.4635127782821655" in source


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _list_recursive(rel: str):
    base = os.path.join(_REPO, rel)
    out = []
    for root, _dirs, files in os.walk(base):
        if "__pycache__" in root:
            continue
        for name in sorted(files):
            full = os.path.join(root, name)
            out.append((os.path.relpath(full, _REPO), os.path.getsize(full)))
    return sorted(out)


def _sha256(path: str) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            digest.update(chunk)
    return digest.hexdigest()