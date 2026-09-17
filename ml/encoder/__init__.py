"""
Behavioral Biometric Authentication — Phase 5 CNN + GRU behavioral encoder.

Converts Phase 3/4 feature sequences (keyboard + mouse) into a fixed-size
behavioral embedding via per-modality ``Conv1d -> GRU -> Linear`` encoders
fused by concatenation (default 128 dims). See
``docs/PHASE_5_CNN_GRU_ENCODER.md`` for the full design.

Public API (re-exported from ``ml.encoder``):

- :class:`BehavioralEncoder` — fused keyboard+mouse encoder producing
  ``[B, embedding_dim]`` embeddings. Use ``encode_session`` /
  ``encode_entries`` on Phase 3/4 sessions, or ``forward`` directly with
  :class:`~ml.encoder.input.PaddedBatch` objects.
- :class:`ConvGruEncoder` — single-modality CNN+GRU encoder.
- :class:`EncoderConfig` — width/hidden-size/feature-count configuration.
- :func:`set_seed` — seed ``torch`` for reproducible construction/inference.
- :class:`~ml.encoder.input.PaddedBatch` — zero-padded variable-length batch.
- :func:`~ml.encoder.input.collate_sequences` /
  :func:`~ml.encoder.input.collate_sessions` — build encoder inputs from Phase
  3/4 feature sequences or processed sessions.
- :func:`~ml.encoder.input.has_behavior` — filter sessions with no behaviour.

Normalization: reuse the Phase 3 ``FeatureScaler`` (fit on training data only)
and pass the fitted scaler to ``collate_sessions`` / ``encode_session`` /
``encode_entries``.

Privacy invariant: the encoder only ever sees NUMERICAL timing/movement
features. Key names, characters, passwords and raw text never enter this
package.
"""

from .input import (
    ModalityBatch,
    PaddedBatch,
    collate_rows,
    collate_sequences,
    collate_sessions,
    has_behavior,
    sequence_tensor,
)
from .cnn_gru import (
    BehavioralEncoder,
    ConvGruEncoder,
    EncoderConfig,
    set_seed,
)

__version__ = "5.0.0"

__all__ = [
    "BehavioralEncoder",
    "ConvGruEncoder",
    "EncoderConfig",
    "ModalityBatch",
    "PaddedBatch",
    "collate_rows",
    "collate_sequences",
    "collate_sessions",
    "has_behavior",
    "sequence_tensor",
    "set_seed",
]