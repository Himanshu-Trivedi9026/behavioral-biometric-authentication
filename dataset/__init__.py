"""
Behavioral Biometric Authentication — Phase 4 dataset package.

Turns processed behavioral sessions (Phase 3 output) into a deterministic,
evaluation-ready dataset. A dataset entry keeps keyboard and mouse as
separate modalities and carries an explicit dataset-level ``user_id`` for
identity verification experiments.

Public API (re-exported from ``dataset``):

- :func:`validate_dataset_entry` — validate one dataset entry.
- :func:`validate_dataset` — validate a whole dataset (entries list).
- :func:`assert_privacy` — raise if any entry leaks key identity/text/characters.
- :class:`SyntheticDatasetGenerator` — deterministic multi-user synthetic data.
- :func:`processed_to_entry` — Phase 3 processed session -> dataset entry.
- :func:`load_processed_dir` / :func:`load_processed_files` — real sessions.
- :func:`raw_to_entry` / :func:`load_raw_files` — raw Phase 2 JSON -> entry.
- :func:`load_dataset` / :func:`save_dataset` — JSON persistence.
- :func:`split_train_test` — deterministic per-user train/test split.

Privacy invariant: entries contain numerical behavioral features only. No
typed characters, key names, key codes, or raw text input ever.
"""

from .schema import (
    DATASET_SCHEMA_VERSION,
    KEYBOARD_FEATURE_COLUMNS,
    MOUSE_FEATURE_COLUMNS,
    assert_privacy,
    privacy_report,
    validate_dataset,
    validate_dataset_entry,
)
from .generator import SyntheticDatasetGenerator
from .loader import (
    load_dataset,
    load_processed_dir,
    load_processed_files,
    load_raw_files,
    processed_to_entry,
    raw_to_entry,
    save_dataset,
)
from .split import split_train_test

__version__ = "1.0.0"

__all__ = [
    "DATASET_SCHEMA_VERSION",
    "KEYBOARD_FEATURE_COLUMNS",
    "MOUSE_FEATURE_COLUMNS",
    "SyntheticDatasetGenerator",
    "assert_privacy",
    "load_dataset",
    "load_processed_dir",
    "load_processed_files",
    "load_raw_files",
    "privacy_report",
    "processed_to_entry",
    "raw_to_entry",
    "save_dataset",
    "split_train_test",
    "validate_dataset",
    "validate_dataset_entry",
]