"""
Behavioral Biometric Authentication — Phase 6 Siamese verification package.

Provides identity-verification machinery on top of the Phase 5 128-dim
behavioral embeddings:

* :class:`~ml.verification.pairs.VerificationPair` + :func:`~ml.verification.pairs.generate_pairs`
  — deterministic genuine/impostor session-pair generation.
* :class:`~ml.verification.siamese.SiameseVerifier` — shared-weight Siamese
  verifier around the Phase 5 :class:`~ml.encoder.BehavioralEncoder` producing
  ``embedding_a [B,128]``, ``embedding_b [B,128]`` and L2 ``distance [B]``.
* :class:`~ml.verification.loss.ContrastiveLoss` / :func:`~ml.verification.loss.contrastive_loss`
  — contrastive loss (label ``1`` genuine / ``0`` impostor).
* :func:`~ml.verification.scoring.decide` / :func:`~ml.verification.scoring.decision`
  — simple deterministic distance→decision scoring (NOT production-ready).

Label convention: ``1`` = genuine (same user), ``0`` = impostor (different
users). See ``docs/PHASE_6_SIAMESE_VERIFICATION.md`` for the full design,
equations, and scope boundaries.
"""

from .pairs import (
    GENUINE,
    IMPOSTOR,
    VALID_LABELS,
    VerificationPair,
    assert_no_cross_split_pairs,
    generate_pairs,
    genuine_candidates,
    group_by_user,
    impostor_candidates,
    pairs_to_dicts,
)
from .siamese import SiameseOutput, SiameseVerifier
from .loss import ContrastiveLoss, contrastive_loss
from .scoring import (
    VerificationDecision,
    decide,
    decide_batch,
    decision,
    default_threshold,
)

__version__ = "6.0.0"

__all__ = [
    "GENUINE",
    "IMPOSTOR",
    "VALID_LABELS",
    "ContrastiveLoss",
    "SiameseOutput",
    "SiameseVerifier",
    "VerificationDecision",
    "VerificationPair",
    "assert_no_cross_split_pairs",
    "contrastive_loss",
    "decide",
    "decide_batch",
    "decision",
    "default_threshold",
    "generate_pairs",
    "genuine_candidates",
    "group_by_user",
    "impostor_candidates",
    "pairs_to_dicts",
]