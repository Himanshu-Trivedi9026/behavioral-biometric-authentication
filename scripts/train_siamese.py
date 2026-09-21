#!/usr/bin/env python3
"""
Phase 7 — Siamese CNN+GRU training CLI.

Usage:
    .venv/bin/python scripts/train_siamese.py \
        --dataset data/datasets/dataset_synthetic.json \
        --epochs 10 --batch-size 8 --learning-rate 0.001 \
        --margin 1.0 --seed 42

Pipeline (in order):
    persisted dataset -> user-aware train/val/test split -> per-partition
    genuine/impostor pairs -> shared Siamese encoder -> L2 distance ->
    contrastive loss -> Adam -> per-epoch train/validation -> checkpoint.

The dataset is LOADED from the persisted artifact and never regenerated here.

Interpretation guard-rail: metrics below come from a SYNTHETIC dataset of
generated identities. They validate the training pipeline only and are NOT
real-world biometric performance.

Exit code 0 = success; 1 = execution error; 2 = usage/file error.
"""

import argparse
import math
import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from ml.training import (  # noqa: E402
    DEFAULT_CHECKPOINT,
    DEFAULT_DATASET,
    TrainingConfig,
    create_criterion,
    create_optimizer,
    create_verifier,
    load_checkpoint,
    prepare_loaders,
    prepare_training_data,
    run_validation,
    save_checkpoint,
    verify_reload,
)
from ml.training.trainer import SiameseTrainer  # noqa: E402


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the shared Siamese CNN+GRU behavioral encoder "
        "with the Phase 6 contrastive objective on the persisted dataset."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="persisted dataset path (default: %(default)s)")
    parser.add_argument("--epochs", type=int, default=10, help="epochs (default: %(default)s)")
    parser.add_argument("--batch-size", type=int, default=8, help="batch size (default: %(default)s)")
    parser.add_argument("--learning-rate", type=_positive_float, default=0.001, help="Adam lr (default: %(default)s)")
    parser.add_argument("--margin", type=_positive_float, default=1.0, help="contrastive margin (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=42, help="deterministic seed (default: %(default)s)")
    parser.add_argument("--val-fraction", type=_non_negative_float, default=2 / 7,
                        help="fraction of train sessions used as validation (default: %(default)s)")
    parser.add_argument("--n-genuine-train", type=int, default=40, help="genuine training pairs (default: %(default)s)")
    parser.add_argument("--n-impostor-train", type=int, default=40, help="impostor training pairs (default: %(default)s)")
    parser.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="checkpoint path (default: %(default)s)")
    parser.add_argument("--overwrite", action="store_true",
                        help="overwrite an existing checkpoint instead of failing")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.epochs < 1:
        print("error: --epochs must be >= 1", file=sys.stderr)
        return 2
    if args.batch_size < 1:
        print("error: --batch-size must be >= 1", file=sys.stderr)
        return 2
    if args.val_fraction <= 0 or args.val_fraction >= 1:
        print("error: --val-fraction must be in (0, 1)", file=sys.stderr)
        return 2

    if os.path.exists(args.checkpoint) and not args.overwrite:
        print(
            "error: checkpoint '{}' already exists; pass --overwrite to replace it "
            "(refusing to overwrite without explicit intent)".format(args.checkpoint),
            file=sys.stderr,
        )
        return 2

    config = TrainingConfig(
        dataset_path=args.dataset,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        margin=args.margin,
        val_fraction=args.val_fraction,
        n_genuine_train=args.n_genuine_train,
        n_impostor_train=args.n_impostor_train,
        checkpoint_path=args.checkpoint,
    )

    try:
        data = prepare_training_data(config)
    except Exception as exc:  # noqa: BLE001 - report at CLI level
        print("error: failed to prepare training data: {!r}".format(exc), file=sys.stderr)
        return 1

    users = sorted({e["user_id"] for e in data.train_sessions + data.val_sessions + data.test_sessions})
    print("dataset loaded from: {}".format(args.dataset))
    print("synthetic users: {}".format(len(users)))
    print("sessions -> train/val/test: {train_sessions}/{val_sessions}/{test_sessions}".format(
        **data.session_count_summary()))
    counts = data.pair_count_summary()
    print("pairs -> train: {train[genuine]} genuine + {train[impostor]} impostor; "
          "val: {val[genuine]} genuine + {val[impostor]} impostor; "
          "test: {test[genuine]} genuine + {test[impostor]} impostor".format(**counts))
    print("keyboard/mouse scalers fitted on TRAINING sessions only")

    torch.manual_seed(config.seed)
    model = create_verifier(config)
    criterion = create_criterion(config)
    optimizer = create_optimizer(model, config)
    train_loader, val_loader, test_loader, train_generator = prepare_loaders(config, data)
    trainer = SiameseTrainer(
        model,
        criterion,
        optimizer,
        config,
        keyboard_scaler=data.keyboard_scaler,
        mouse_scaler=data.mouse_scaler,
        train_generator=train_generator,
    )

    print("training epochs: {epochs}, batch size: {batch}, lr: {lr}, margin: {margin}, seed: {seed}"
          .format(epochs=config.epochs, batch=config.batch_size, lr=config.learning_rate,
                  margin=config.margin, seed=config.seed))
    try:
        history = trainer.fit(train_loader, val_loader)
    except Exception as exc:  # noqa: BLE001 - report at CLI level
        print("error: training failed: {!r}".format(exc), file=sys.stderr)
        return 1

    finite = all(math.isfinite(v) for r in history.records for v in r.values())
    print("{:<6} {:<14} {:<14}".format("epoch", "train_loss", "val_loss"))
    for record in history.records:
        print("{:<6} {:<14.6f} {:<14.6f}".format(int(record["epoch"]), record["train_loss"], record["val_loss"]))
    print("all losses finite: {}".format("PASS" if finite else "FAIL"))

    test_loss = run_validation(
        model, criterion, test_loader,
        keyboard_scaler=data.keyboard_scaler, mouse_scaler=data.mouse_scaler,
    )
    print("SYNTHETIC-DATA test loss (after training, one pass, isolated test set): {:.6f}".format(test_loss))

    try:
        if os.path.exists(args.checkpoint) and not args.overwrite:
            raise RuntimeError("checkpoint exists without --overwrite")
        save_checkpoint(
            args.checkpoint,
            model=model,
            optimizer=optimizer,
            epoch=config.epochs,
            history=history,
            config=config,
        )
    except Exception as exc:  # noqa: BLE001
        print("error: checkpoint save failed: {!r}".format(exc), file=sys.stderr)
        return 1
    print("checkpoint saved: {}".format(args.checkpoint))

    try:
        payload = load_checkpoint(args.checkpoint)
        rebuild = create_verifier(config)
        rebuild.load_state_dict(payload["model_state_dict"])
        probe_a = data.test_pairs[0].session_a
        probe_b = data.test_pairs[0].session_b
        deviations = verify_reload(
            payload,
            model,
            [probe_a],
            [probe_b],
            keyboard_scaler=data.keyboard_scaler,
            mouse_scaler=data.mouse_scaler,
        )
    except Exception as exc:  # noqa: BLE001
        print("error: checkpoint reload verification failed: {!r}".format(exc), file=sys.stderr)
        return 1
    print("checkpoint reload: PASS {} (max abs diff over probe pair)".format(deviations))
    return 0


if __name__ == "__main__":
    sys.exit(main())