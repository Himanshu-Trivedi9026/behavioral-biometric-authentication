#!/usr/bin/env python3
"""
Phase 5 — manual verification harness for the CNN + GRU behavioral encoder.

Runs two checks against a real Phase 2 session:

  1. real-session smoke : raw Phase 2 JSON -> Phase 3 processing -> (fitted
     Phase 3 FeatureScaler) -> 128-dim behavioral embedding.
  2. determinism        : seeded construction + seeded inference are
     byte-identical.

CORRECTED OUTPUT CONTRACT (the reason this harness exists)
----------------------------------------------------------
The encoder returns plain ``torch.Tensor`` objects — NOT a dict, NOT a
dataclass, NOT a 3-D tensor:

    BehavioralEncoder(...)(keyboard=..., mouse=...)  -> [B, embedding_dim] (2-D)
    encoder.encode_session(session)                  -> [embedding_dim]    (1-D)
    encoder.encode_entries(sessions)                 -> [N, embedding_dim] (2-D)

Therefore the 2-D results must be indexed with AT MOST two axes, e.g.
``embeddings[0]`` (a row). Indexing ``embeddings[0, :, :]`` (three axes) raises
``IndexError: too many indices for tensor of dimension 2`` — that was the
earlier manual-verification failure, NOT an encoder bug.

Usage:
    .venv/bin/python scripts/verify_encoder.py --session PATH [--seed N]

Exit code 0 = both checks passed; 1 = a check failed; 2 = usage/file error.
"""

import argparse
import json
import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ml.encoder import BehavioralEncoder, set_seed  # noqa: E402
from ml.encoder.input import collate_sessions  # noqa: E402
from ml.preprocessing import (  # noqa: E402
    FeatureScaler,
    KEYBOARD_FEATURE_COLUMNS,
    MOUSE_FEATURE_COLUMNS,
    process_session,
)


def _fit_scalers(processed):
    keyboard_scaler = FeatureScaler(KEYBOARD_FEATURE_COLUMNS).fit(
        [[row["hold_time"], row["flight_time"]] for row in processed["keyboard_sequence"]]
    )
    mouse_scaler = FeatureScaler(MOUSE_FEATURE_COLUMNS).fit(
        [[row[column] for column in MOUSE_FEATURE_COLUMNS]
         for row in processed["mouse_sequence"]]
    )
    return keyboard_scaler, mouse_scaler


def check_smoke(processed, seed):
    """Real-session smoke: session must encode to a finite 128-dim embedding."""
    keyboard_scaler, mouse_scaler = _fit_scalers(processed)
    enc = BehavioralEncoder(seed=seed)

    # 1) full-pipeline convenience wrapper -> [embedding_dim]
    single = enc.encode_session(
        processed,
        keyboard_scaler=keyboard_scaler,
        mouse_scaler=mouse_scaler,
    )
    assert isinstance(single, torch.Tensor) and type(single).__name__ == "Tensor", \
        "encode_session must return a torch.Tensor"
    assert single.ndim == 1 and tuple(single.shape) == (128,), \
        "encode_session must return a (128,) vector, got shape {}".format(tuple(single.shape))
    assert bool(torch.isfinite(single).all()), "embedding contains non-finite values"

    # 2) batch wrapper -> [N, embedding_dim]
    batch = enc.encode_entries(
        [processed],
        keyboard_scaler=keyboard_scaler,
        mouse_scaler=mouse_scaler,
    )
    assert batch.ndim == 2 and tuple(batch.shape) == (1, 128), \
        "encode_entries must return a (1, 128) tensor, got shape {}".format(tuple(batch.shape))

    # 3) low-level forward over PaddedBatch inputs -> [B, embedding_dim]
    mods = collate_sessions(
        [processed], keyboard_scaler=keyboard_scaler, mouse_scaler=mouse_scaler
    )
    direct = enc(keyboard=mods.keyboard, mouse=mods.mouse)
    assert direct.ndim == 2 and tuple(direct.shape) == (1, 128), \
        "forward must return a (1, 128) tensor, got shape {}".format(tuple(direct.shape))
    assert torch.equal(batch, direct), "encode_entries vs forward disagree"

    # documented indexing: a row of a 2-D result is reached with ONE index
    row = direct[0]
    assert row.ndim == 1 and tuple(row.shape) == (128,)

    return single


def check_determinism(processed, seed):
    """Same seed => byte-identical embeddings (construction AND inference)."""
    keyboard_scaler, mouse_scaler = _fit_scalers(processed)

    first = BehavioralEncoder(seed=seed).encode_entries(
        [processed], keyboard_scaler=keyboard_scaler, mouse_scaler=mouse_scaler
    )
    second = BehavioralEncoder(seed=seed).encode_entries(
        [processed], keyboard_scaler=keyboard_scaler, mouse_scaler=mouse_scaler
    )
    assert torch.equal(first, second), "same seed did not reproduce construction"

    enc = BehavioralEncoder(seed=seed)
    set_seed(seed)
    a = enc.encode_entries(
        [processed], keyboard_scaler=keyboard_scaler, mouse_scaler=mouse_scaler
    )
    set_seed(seed)
    b = enc.encode_entries(
        [processed], keyboard_scaler=keyboard_scaler, mouse_scaler=mouse_scaler
    )
    assert torch.equal(a, b), "reseeding before inference did not reproduce output"


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Phase 5 encoder manual verification (smoke + determinism)."
    )
    parser.add_argument(
        "--session",
        required=True,
        help="path to a raw Phase 2 session JSON (behavioral-session-*.json)",
    )
    parser.add_argument("--seed", type=int, default=0, help="deterministic seed (default 0)")
    args = parser.parse_args(argv)

    if not os.path.isfile(args.session):
        print("error: session file not found: {}".format(args.session), file=sys.stderr)
        return 2

    try:
        with open(args.session) as f:
            raw = json.load(f)
        processed = process_session(raw)
    except Exception as exc:
        print("error: could not load/process session: {}".format(exc), file=sys.stderr)
        return 1

    try:
        emb = check_smoke(processed, seed=args.seed)
        check_determinism(processed, seed=args.seed)
    except AssertionError as exc:
        print("FAIL: {}".format(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print("FAIL: unexpected error: {!r}".format(exc), file=sys.stderr)
        return 1

    print("PASS  real-session smoke : session `{}` -> embedding {}".format(
        processed["session_id"], tuple(emb.shape)
    ))
    print("        keyboard={} samples, mouse={} samples".format(
        len(processed["keyboard_sequence"]), len(processed["mouse_sequence"])
    ))
    print("PASS  output contract     : forward/encode_entries -> [B, 128] (2-D); "
          "encode_session -> [128] (1-D)")
    print("PASS  determinism         : seed {} reproduces byte-identically".format(args.seed))
    return 0


if __name__ == "__main__":
    sys.exit(main())