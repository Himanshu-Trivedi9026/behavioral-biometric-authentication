"""
Phase 8 — tests for loading the trained checkpoint into the inference-only
:class:`~ml.evaluation.model.LoadedVerifier` wrapper.

Covers: correct embedding shape/dimension, deterministic inference, parameter
invariance under inference (no_grad/eval), Phase 6 pair-distance convention,
and loud failures on missing / corrupt / behaviourless input.

Skipped automatically when the Phase 7 checkpoint is not present on disk.

Run:
    .venv/bin/python -m pytest tests/test_evaluation_model.py -v
"""

import json
import os

import pytest
import torch

from ml.evaluation.model import LoadedVerifier, load_verifier
from ml.training import CheckpointError

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_PATH = os.path.join(_REPO, "models", "siamese_behavioral_encoder.pt")
DATASET_PATH = os.path.join(_REPO, "data", "datasets", "dataset_synthetic.json")

pytestmark = pytest.mark.skipif(
    not os.path.exists(CHECKPOINT_PATH) or not os.path.exists(DATASET_PATH),
    reason="Phase 7 checkpoint and/or synthetic dataset not present on disk",
)


@pytest.fixture(scope="function")
def verifier():
    return load_verifier(CHECKPOINT_PATH)


@pytest.fixture(scope="function")
def sessions():
    with open(DATASET_PATH, "r", encoding="utf-8") as fh:
        wrapped = json.load(fh)
    entries = wrapped["entries"] if isinstance(wrapped, dict) else wrapped
    behavioral = [e for e in entries if e.get("keyboard_sequence") or e.get("mouse_sequence")]
    assert len(behavioral) >= 2
    return behavioral[0], behavioral[1]


def test_load_reports_checkpoint_identity(verifier):
    assert isinstance(verifier, LoadedVerifier)
    assert verifier.embedding_dim == 128
    assert verifier.epoch >= 1
    assert verifier.checkpoint_id == os.path.basename(CHECKPOINT_PATH)
    assert verifier.model.training is False
    assert verifier.model.shared_encoder.config.embedding_dim == 64


def test_embed_session_shape_and_finiteness(verifier, sessions):
    embedding = verifier.embed_session(sessions[0])
    assert embedding.shape == (128,)
    assert bool(torch.isfinite(embedding).all())
    assert embedding.dtype.is_floating_point


def test_embed_sessions_stacks_and_matches_single(verifier, sessions):
    stack = verifier.embed_sessions(list(sessions), batch_size=1)
    assert stack.shape == (2, 128)
    single_a = verifier.embed_session(sessions[0])
    single_b = verifier.embed_session(sessions[1])
    assert torch.equal(stack[0], single_a)
    assert torch.equal(stack[1], single_b)


def test_inference_is_deterministic(verifier, sessions):
    first = verifier.embed_session(sessions[0])
    second = verifier.embed_session(sessions[0])
    assert torch.equal(first, second)


def test_inference_does_not_change_parameters(verifier, sessions):
    before = {k: v.clone() for k, v in verifier.model.state_dict().items()}
    verifier.embed_sessions(list(sessions), batch_size=2)
    verifier.embed_session(sessions[1])
    after = verifier.model.state_dict()
    for name, tensor in before.items():
        assert name in after
        assert torch.equal(tensor, after[name]), "parameter {} changed during inference".format(name)


def test_pair_distance_matches_embedding_geometry(verifier, sessions):
    a, b = sessions
    ea = verifier.embed_session(a)
    eb = verifier.embed_session(b)
    manual = float(torch.sqrt(((ea - eb) ** 2).sum() + 1e-8))
    distance = float(verifier.pair_distance(ea.unsqueeze(0), eb.unsqueeze(0))[0])
    assert distance == pytest.approx(manual, rel=1e-5)
    assert distance >= 0.0


def test_pair_distance_is_symmetric(verifier, sessions):
    ea = verifier.embed_session(sessions[0])
    eb = verifier.embed_session(sessions[1])
    dab = float(verifier.pair_distance(ea.unsqueeze(0), eb.unsqueeze(0))[0])
    dba = float(verifier.pair_distance(eb.unsqueeze(0), ea.unsqueeze(0))[0])
    assert dab == pytest.approx(dba, abs=1e-6)


def test_verify_pair_returns_float(verifier, sessions):
    a, b = sessions
    distance = verifier.verify_pair(a, b)
    assert isinstance(distance, float)
    assert distance >= 0.0


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(CheckpointError):
        load_verifier(os.path.join(str(tmp_path), "does_not_exist.pt"))


def test_corrupt_checkpoint_raises(tmp_path):
    bad = tmp_path / "bad.pt"
    bad.write_text("this is not a phase 7 checkpoint at all", encoding="utf-8")
    with pytest.raises((CheckpointError, RuntimeError, Exception)):
        try:
            load_verifier(str(bad))
        except CheckpointError:
            raise
        except Exception as exc:  # any loud failure is acceptable
            if "pickle" in str(exc).lower() or "unsafe" in str(exc).lower() or "corrupt" in str(exc).lower():
                raise


def test_behaviorless_session_rejected(verifier):
    empty = {"user_id": "u", "session_id": "s", "keyboard_sequence": [], "mouse_sequence": []}
    with pytest.raises(ValueError):
        verifier.embed_session(empty)


def test_non_mapping_session_rejected(verifier):
    with pytest.raises(TypeError):
        verifier.embed_session("not-a-session")  # type: ignore[arg-type]


def test_embed_sessions_empty_is_ok(verifier):
    stack = verifier.embed_sessions([], batch_size=4)
    assert tuple(stack.shape) == (0, 128)