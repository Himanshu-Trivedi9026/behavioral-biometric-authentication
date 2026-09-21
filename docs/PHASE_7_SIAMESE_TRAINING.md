# Phase 7 — Siamese CNN+GRU Training Pipeline

## 1. Objective

Train the Phase 5 CNN+GRU `BehavioralEncoder` through the Phase 6 Siamese
verification architecture and `ContrastiveLoss` so that behavioural sessions
of the **same user** map closer together in embedding space while sessions of
**different users** map farther apart.

This phase is an **engineering / training-pipeline validation**. It proves the
whole loop works: persisted dataset -> leakage-free split -> pairs ->
DataLoader -> shared encoder -> L2 distance -> contrastive loss ->
backpropagation -> optimizer -> checkpoint -> reload.

## 2. Dataset Used

`data/datasets/dataset_synthetic.json` (Phase 4 artifact, schema v1.0.0),
loaded with `dataset.loader.load_dataset`. The CLI never regenerates the
dataset; training always operates on the persisted artifact.

## 3. Why Synthetic Data

Real multi-user behavioural data has not been collected yet. The synthetic
generator that produced `dataset_synthetic.json` creates **generated
identities** — users are labeled `user_001..user_005` — precisely so the
training pipeline, split logic, pair construction, determinism, checkpointing,
and reload semantics can be validated deterministically before real data is
collected. Synthetic sessions are **not** real participants.

## 4. Dataset Size

- Users: 5 (`user_001`–`user_005`)
- Sessions: 50 (10 per user)
- Keyboard samples: 1,228; Mouse samples: 2,639

## 5. Train/Validation/Test Strategy

Deterministic, user-aware, three-way split (documented in
`ml/training/config.py`):

| Step | Operation | Per user | Global |
|------|-----------|----------|--------|
| outer split (`train_fraction=0.7`) | train / test | 7 / 3 | 35 / 15 |
| inner split (`val_fraction=2/7`) | train / val out of the 7 | 5 / 2 | 25 / 10 |
| final | train / val / test | 5 / 2 / 3 | **25 / 10 / 15** |

Every user appears in **every** partition (each user has ≥ 1 session per
partition); this is asserted in `prepare_training_data`
(`_verify_user_partition_coverage`). The test partition is only ever used for
the post-training synthetic test-loss and reload verification — never during an
epoch.

## 6. Pair-Generation Strategy

Pairs are generated with the Phase 6 `generate_pairs` **after** the split,
separately per partition and only from that partition's sessions:

- train: 40 genuine + 40 impostor (capacities: 50 genuine / 250 impostor)
- val: 5 genuine + 10 impostor (capacities: 5 genuine / 40 impostor)
- test: 15 genuine + 15 impostor (capacities: 15 genuine / 90 impostor)

Counts are configurable (`n_genuine_train` … `n_impostor_test`). Requests
beyond the available unique candidates raise `ValueError`. The defaults are
chosen to keep CPU training fast (~10 batches/epoch at batch size 8) while
remaining balanced enough to avoid extreme class imbalance.

## 7. Data Leakage Prevention

- Split first, then pair: `sessions -> user-aware split -> per-partition pairs`.
  Pairs are never split randomly after generation.
- `assert_no_leakage` on train/val and train/test (session-id disjointness).
- Per-partition pair generation means a pair can never straddle two partitions.
- The `FeatureScaler` is fitted **on training rows only** and reused for
  validation/test (`ml/training/dataset._fit_scalers`).
- Regression tests assert: train/val/test ids are disjoint; every pair's
  sessions belong to its own partition; test sessions never reach the trainer.

## 8. Siamese Architecture

Single `ml.verification.SiameseVerifier` wrapping **one shared**
`BehavioralEncoder` (default 64+64 -> 128-dim). Both branches use the same
parameter tensor set; the verifier collates both sides in one `encode_entries`
pass and splits the result, then computes Euclidean (L2) distance with
`eps=1e-8`. No second encoder is introduced.

## 9. Contrastive-Loss Configuration

Phase 6 `ContrastiveLoss` (`ml/verification/loss.py`):

    L = mean( y·D² + (1−y)·max(0, margin − D)² )

- labels: `1` genuine, `0` impostor
- margin: `1.0` (configurable)
- reduction: `mean`

## 10. Optimizer

Adam (`torch.optim.Adam`) with `lr=1e-3` (default) and `weight_decay=0.0`
(default), both configurable. Gradient clipping is enabled by default
(`gradient_clip=1.0`) via `torch.nn.utils.clip_grad_norm_`. No hyperparameter
search is performed in this phase.

## 11. Training Configuration

Defaults (all configurable through `TrainingConfig` / the CLI):

| Setting | Default |
|---------|---------|
| dataset | `data/datasets/dataset_synthetic.json` |
| epochs | 10 |
| batch size | 8 |
| learning rate | 1e-3 |
| margin | 1.0 |
| seed | 42 |
| num_workers | 0 (CPU, single-process) |
| checkpoint | `models/siamese_behavioral_encoder.pt` |

## 12. Reproducibility

- `random`/pair RNG is seed-isolated (`random.Random(seed)`, derived seeds for
  val/test pairs).
- Split RNG is per-user (`"split:{seed}:{user_id}"`).
- Encoder initialization is seeded (`ml.encoder.set_seed`) so two runs with the
  same config & seed construct identical weights.
- The training DataLoader's `torch.Generator` is reseeded to `seed + epoch` at
  the start of every epoch: deterministic across runs, different each epoch.
- Tests assert that two runs with the same seed reproduce identical loss
  history **and bit-identical model state dicts** (CPU; the project does not
  claim cross-hardware bit-identical training).

CLI usage example:

```bash
.venv/bin/python scripts/train_siamese.py \
    --dataset data/datasets/dataset_synthetic.json \
    --epochs 10 --batch-size 8 --learning-rate 0.001 \
    --margin 1.0 --seed 42
```

The CLI refuses to overwrite an existing checkpoint unless `--overwrite` is
passed, and exits 0 only if loading, training, checkpointing, and reload
verification all succeed.

## 13. Checkpoint Format

`ml/training/checkpoint.py` writes `siamese-behavioral-checkpoint-v1`:

- `model_state_dict` (SiameseVerifier / shared encoder)
- `optimizer_state_dict` (Adam moments)
- `epoch`, `seed`
- `history` (train/val loss per epoch)
- `training_config` (JSON-serializable)
- `encoder_config` (EncoderConfig fields)

Saving is atomic (`<path>.tmp` + `os.replace`). Loading uses
`torch.load(weights_only=True)` and rejects wrong-format or unsafe files via
`CheckpointError`. `verify_reload` reconstructs a fresh verifier from the
checkpoint and proves output parity on a probe pair.

## 14. Training Results (synthetic data, seed 42)

10 epochs, batch 8, lr 1e-3, margin 1.0:

| epoch | train_loss | val_loss |
|------:|-----------:|---------:|
| 1 | 0.124812 | 0.051647 |
| 2 | 0.070945 | 0.043462 |
| 3 | 0.055153 | 0.037324 |
| 4 | 0.044322 | 0.029376 |
| 5 | 0.032526 | 0.024930 |
| 6 | 0.025951 | 0.024123 |
| 7 | 0.021081 | 0.025314 |
| 8 | 0.017442 | 0.024570 |
| 9 | 0.014446 | 0.029076 |
| 10 | 0.011864 | 0.029225 |

- All train/val losses finite.
- Synthetic-data test loss (one post-training pass over the isolated test set,
  labelled as synthetic): **0.168076**.
- Checkpoint: `models/siamese_behavioral_encoder.pt`
- Reload verification: PASS — embedding/distance max abs diff = 0.0 on the
  probe pair.

## 15. Limitations

- The dataset is small (5 users x 10 sessions) and **synthetic**. Training
  curves/val/test losses indicate the pipeline runs and gradients flow; they do
  NOT establish authentication quality.
- Only 15 test sessions / 30 test pairs — far too small for any accuracy claim.
- CPU-only, `num_workers=0`; no multi-GPU, no mixed precision.
- Adaptive thresholding, ROC/AUC/EER evaluation, and hyperparameter search are
  intentionally **out of scope** for Phase 7.

## 16. Interpretation Guard-Rail

> Training results in Phase 7 are **not** real-world biometric performance.
> `dataset_synthetic.json` contains **generated identities**, not human
> participants. No claim of "accuracy", "production readiness", or real-user
> authentication quality is made from these results. Real multi-user data will
> be collected and evaluated in later phases.

## Verification checklist

- Phase 7 tests: 49/49 passed
- Full python suite: 383/383 passed (334 previous + 49 new)
- Frontend: 19/19 passed
- `compileall` and import checks: PASS
- Training smoke test (10 epochs on the persisted dataset): completed
- Losses & gradients finite; parameters update; same-seed determinism PASS
- Checkpoint saved and reload reproduces saved outputs exactly
- No Phase 8 or later functionality implemented