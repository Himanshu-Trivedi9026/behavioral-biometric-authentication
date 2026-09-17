"""
Phase 6 end-to-end integration smoke test.

Pipeline under test (a single forward/backward pass, NOT a training loop):

    dataset sessions
        |  (Phase 4 entries)
        v
    pairs (genuine + impostor, labels 1/0)
        |  Phase 3 FeatureScaler fit on the dataset
        v
    Phase 5 BehavioralEncoder (shared by both Siamese branches)
        |  embedding_a / embedding_b [B, 128]
        v
    Siamese L2 distance [B]
        |
        v
    contrastive loss (scalar) -> backward()

Run from the repo root::

    .venv/bin/python -m pytest tests/test_verification_integration.py -v
"""

import pytest
import torch

from dataset.generator import SyntheticDatasetGenerator
from dataset.split import split_train_test
from ml.encoder import BehavioralEncoder
from ml.encoder.input import _validated_rows
from ml.preprocessing import (
    FeatureScaler,
    KEYBOARD_FEATURE_COLUMNS,
    MOUSE_FEATURE_COLUMNS,
)
from ml.verification import (
    ContrastiveLoss,
    SiameseVerifier,
    generate_pairs,
)


def _dataset(seed=0, n_users=3, sessions=4):
    return SyntheticDatasetGenerator(seed=seed, n_users=n_users, sessions_per_user=sessions).generate()


def _scalers(entries):
    kb_scaler = FeatureScaler(KEYBOARD_FEATURE_COLUMNS).fit(
        [row for e in entries for row in _validated_rows(e["keyboard_sequence"],
                                                         KEYBOARD_FEATURE_COLUMNS)]
    )
    mo_scaler = FeatureScaler(MOUSE_FEATURE_COLUMNS).fit(
        [row for e in entries for row in _validated_rows(e["mouse_sequence"],
                                                         MOUSE_FEATURE_COLUMNS)]
    )
    return kb_scaler, mo_scaler


class TestIntegration:

    def test_full_pipeline_shapes_and_finite(self):
        entries = _dataset(seed=0)
        kb_scaler, mo_scaler = _scalers(entries)
        pairs = generate_pairs(entries, n_genuine=6, n_impostor=6, seed=1)
        assert len(pairs) == 12

        verifier = SiameseVerifier(seed=0)
        out = verifier(
            [p.session_a for p in pairs],
            [p.session_b for p in pairs],
            keyboard_scaler=kb_scaler,
            mouse_scaler=mo_scaler,
        )
        assert out.embedding_a.shape == (12, 128)
        assert out.embedding_b.shape == (12, 128)
        assert out.distance.shape == (12,)
        assert bool(torch.isfinite(out.embedding_a).all())
        assert bool(torch.isfinite(out.embedding_b).all())
        assert bool(torch.isfinite(out.distance).all())

        labels = torch.tensor([p.label for p in pairs])
        criterion = ContrastiveLoss(margin=1.0)
        loss = criterion(out.distance, labels)
        assert loss.ndim == 0
        assert bool(torch.isfinite(loss).all())

    def test_loss_and_gradients_propagate(self):
        entries = _dataset(seed=1)
        kb_scaler, mo_scaler = _scalers(entries)
        pairs = generate_pairs(entries, n_genuine=8, n_impostor=8, seed=2)

        verifier = SiameseVerifier(seed=0)
        out = verifier(
            [p.session_a for p in pairs],
            [p.session_b for p in pairs],
            keyboard_scaler=kb_scaler,
            mouse_scaler=mo_scaler,
        )
        labels = torch.tensor([p.label for p in pairs])
        loss = ContrastiveLoss(margin=1.0)(out.distance, labels)
        loss.backward()

        missing = [
            name for name, p in verifier.shared_encoder.named_parameters()
            if p.requires_grad and (p.grad is None or not bool(torch.isfinite(p.grad).all()))
        ]
        assert missing == []

    def test_partition_aware_pairs_integration(self):
        entries = _dataset(seed=2)
        train, test = split_train_test(entries, seed=100)
        # non-mixing pairs are generated from train and test separately
        pairs = generate_pairs(
            train,
            n_genuine=3,
            n_impostor=3,
            seed=0,
            partitions={"train": train},
        )
        verifier = SiameseVerifier(seed=0)
        out = verifier(
            [p.session_a for p in pairs],
            [p.session_b for p in pairs],
        )
        assert bool(torch.isfinite(out.distance).all())
        assert (out.distance >= 0).all()

    def test_identity_metadata_does_not_affect_pipeline(self):
        """Two sessions with identical behaviour but different identities give
        ~zero Siamese distance through the full real pipeline."""
        entries = _dataset(seed=3)
        kb_scaler, mo_scaler = _scalers(entries)
        verifier = SiameseVerifier(seed=0)
        base = entries[0]
        twin = dict(base)
        twin["user_id"] = "someone_else"
        twin["session_id"] = base["session_id"] + "-copy"
        out = verifier.verify(
            base, twin,
            keyboard_scaler=kb_scaler,
            mouse_scaler=mo_scaler,
        )
        assert float(out.distance[0].detach()) < 1e-3

    def test_contrastive_loss_directionality_raw_pipeline(self):
        """Untrained-encoder smoke check: a session drawn compared against
        itself is a genuine pair and should receive smaller distance than a
        session compared with a different user's session is *representable*."""
        entries = _dataset(seed=4)
        verifier = SiameseVerifier(seed=0)
        s = entries[0]
        other = entries[10]  # a different user's session
        self_dist = float(verifier.verify(s, s).distance[0].detach())
        cross_dist = float(verifier.verify(s, other).distance[0].detach())
        # identical input -> zero-ish distance; different input -> bigger
        assert self_dist < cross_dist

    def test_encoder_input_contract_holds_with_scalers(self):
        """Phase 5 encoder API used by the verifier must keep its exact
        contracts ([N,128] / [128])."""
        entries = _dataset(seed=5)
        kb_scaler, mo_scaler = _scalers(entries)
        enc = BehavioralEncoder(seed=0)
        batch = enc.encode_entries(entries, keyboard_scaler=kb_scaler, mouse_scaler=mo_scaler)
        single = enc.encode_session(entries[0], keyboard_scaler=kb_scaler, mouse_scaler=mo_scaler)
        assert batch.shape == (len(entries), 128)
        assert single.shape == (128,)