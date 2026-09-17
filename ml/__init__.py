"""
Behavioral Biometric Authentication — ML package.

Phase 3 provides ``ml.preprocessing``: validation and feature extraction that
turn a raw Phase 2 session (browser-collected keyboard/mouse events) into
machine-learning-ready behavioral sequences.

Only preprocessing and feature extraction live here. Model training
(CNN/GRU, Siamese, ...) is intentionally out of scope for this phase.
"""