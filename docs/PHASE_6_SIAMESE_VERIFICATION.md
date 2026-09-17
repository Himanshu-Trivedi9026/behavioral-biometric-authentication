# Phase 6 — Siamese Verification + Contrastive Loss

## 1. Objective

Phase 6 adds the **identity-verification layer** on top of the Phase 5
128-dimensional behavioral embeddings. Given two behavioral sessions, the layer
answers the question *"are these the same user?"* using a **Siamese network**
(two branches that share one Phase 5 encoder), an **L2 distance**, and a
**contrastive loss** that learns to pull same-user embeddings together and push
different-user embeddings apart.

> **Phase 6 establishes the verification architecture.**
> It does **not** establish production authentication performance.

## 2. Why Siamese verification?

The task is a **verification** (1:1) rather than identification (1:N): an
enrolment session and a fresh probe session are compared directly. A Siamese
architecture is the natural fit:

* It compares two sessions with *exactly the same feature extractor* — the
  learned behavioural representation is shared, so no per-user model is needed.
* It produces a single scalar *distance* that is directly optimisable with a
  contrastive objective.
* It generalizes to enrolling new users without re-training (only the
  enrollee's sessions go through the shared encoder to produce reference
  embeddings).

## 3. Pair-generation strategy

`ml/verification/pairs.py` turns the Phase 4 dataset representation into
training/analysis pairs. It operates on dataset **entries** (the same dict
schema as Phase 4) and yields `VerificationPair` objects:

* **Genuine pairs** — two distinct sessions of the **same** user
  (all `combinations(sessions_of_user, 2)` per user).
* **Impostor pairs** — two sessions belonging to **two different** users
  (all unordered user pairs × every cross-user session combination).

Generation is **deterministic** (`random.Random(seed)`): the same seed
reproduces the exact same pair set; a different seed produces a different set.

Counts are independently configurable (`n_genuine`, `n_impostor`). Impossible
requests fail loudly (`ValueError`) — e.g. asking for an impostor pair from a
single-user dataset, or a genuine pair when a user has only one session. A
session is never paired with itself (enforced by `VerificationPair`
construction).

### Train/test leakage control

`generate_pairs(..., partitions={"train": ..., "test": ...})` builds pairs
**within each partition only**; a pair never mixes a train session with a test
session. `assert_no_cross_split_pairs()` independently verifies this invariant.
Purposely, even when the full pool is supplied, pair boundaries respect the
split.

## 4. Genuine/impostor label convention

| label | meaning | relationship |
|-------|---------|--------------|
| `1`   | genuine | `session_a.user_id == session_b.user_id` |
| `0`   | impostor | `session_a.user_id != session_b.user_id` |

The convention is explicit and shared by `ml/verification/pairs.py` and
`ml/verification/loss.py` (`GENUINE = 1`, `IMPOSTOR = 0`).

## 5. Shared-encoder (Siamese) architecture

```
      session A                     session B
         |                              |
         v                              v
   shared BehavioralEncoder        shared BehavioralEncoder
   (Phase 5, [T]->128)             (Phase 5, [T]->128)
         |                              |
         v                              v
   embedding_a [B,128]             embedding_b [B,128]
              \                        /
               -----------------------
                            |
                            v
                     distance [B]
```

Critically, **both branches are the same module instance** — there is exactly
one `ml.encoder.BehavioralEncoder` inside `SiameseVerifier.shared_encoder`. No
second, independently-initialized encoder exists (verified by tests comparing
parameter object identities). `embedding_dim` is the encoder's own (default
**128**).

Batching is implemented by collating the B "A-sessions" and B "B-sessions"
into a single stack, running the shared encoder once, and splitting the
output — this keeps the two branches trivially on identical parameters.

## 6. Distance function

Euclidean (L2) distance between the two behavioral embeddings:

```
diff[i]   = embedding_a[i] - embedding_b[i]
distance[i] = sqrt( sum_j diff[i,j]^2  +  eps ),   eps = 1e-8
```

The tiny `eps` (added under the sqrt) keeps the derivative finite when two
embeddings are identical. The distance is differentiable, always non-negative,
and symmetric.

## 7. Contrastive-loss equation

For a batch of pair distances `D` with binary labels `y`
(`1` = genuine, `0` = impostor) and a positive `margin`:

```
L_genuine   = y        * D^2
L_impostor  = (1 - y)  * max(0, margin - D)^2
L           = mean( L_genuine + L_impostor )     (reduction = "mean", default)
```

* Genuine pairs are penalized quadratically the farther apart they are.
* Impostor pairs are penalized **only while** `D < margin`; at/above the margin
  they contribute exactly `0`.
* The computation stays entirely in PyTorch tensors (fully differentiable,
  no NumPy in the differentiable path). `reduction="sum"` is also supported.

## 8. Margin meaning

`margin` is the target separation for impostor pairs: the loss pushes impostor
distances to be **at least** `margin`, and genuine distances toward `0`. Larger
margins demand more aggressive separation. `margin` must be positive; the
module/functional both validate this.

## 9. Tensor shapes

| quantity | shape |
|----------|-------|
| `embedding_a` | `[B, 128]` |
| `embedding_b` | `[B, 128]` |
| `distance` | `[B]` |
| `labels` | `[B]` (int64/float/bool, values 0/1) |
| contrastive loss | scalar |

## 10. Identity / privacy separation

* `user_id` / `session_id` exist **only** in dataset metadata and are used
  **only** to build genuine/impostor labels.
* The Siamese verifier reads only `keyboard_sequence` / `mouse_sequence` from
  each session; identity fields never enter the model computation
  (tested: identical behavior with different ids → zero distance).
* The Phase 4 privacy contract (numeric features only, no typed characters,
  key names, or passwords) is inherited and unchanged.

## 11. What Phase 6 does NOT implement

* No FastAPI/Flask/REST, no PostgreSQL/MongoDB/Redis, no JWT, no
  login/registration, no authentication endpoints.
* No frontend dashboard, no continuous/adaptive authentication.
* No full ROC/AUC/EER evaluation pipeline, no threshold calibration, no
  adaptive thresholding (these belong to later phases).
* No model training loop: no optimizer, no epochs, no checkpoints, no CLI, no
  production training pipeline, no hyperparameter search. Only a forward/
  backward gradient smoke test is exercised.
* `ml/verification/scoring.py` provides only a *fixed-threshold* decision
  helper (`distance <= threshold -> accept`) and explicitly declares that its
  default threshold is **not production-ready**.

## 12. Testing performed

* `tests/test_pairs.py` — labelling, self-pairing prohibition, determinism,
  count handling, invalid requests, identity separation, split isolation,
  candidate enumeration, small-dataset safety.
* `tests/test_siamese.py` — shapes, distance properties (non-negative, zero for
  identical input, symmetry), shared-parameter identity, gradient flow,
  batching, determinism, Phase 5 contract preservation, identity independence.
* `tests/test_contrastive_loss.py` — label/margin validation, finiteness,
  analytic loss values, penalty behaviour (genuine/impostor), gradients.
* `tests/test_verification_scoring.py` — threshold rule, validation, no
  adaptive/calibration machinery.
* `tests/test_verification_integration.py` — end-to-end pipeline from Phase 4
  entries → pairs → Phase 5 shared encoder → distance → contrastive loss →
  `backward()`, with shapes/finiteness/gradient verification.
* Real-session smoke test: the existing real Phase 2 session is run through
  Phase 3 → Phase 5 → Siamese self-compare; distance ≈ 0, finite, backprops.

## 13. Known limitations

* The encoder used here is the Phase 5 encoder **as released** — it has not
  been trained with the contrastive objective yet; genuine/impostor separation
  is the *learned objective* of Phase 7+, not a claim made here.
* The distance/L2 metric is fixed; no learned metric, temperature, or margins
  beyond a constant are tuned in Phase 6.
* `scoring.default_threshold()` is an arbitrary placeholder (`1.0`) with **no
  accuracy guarantee**.
* Pair generation is in-memory over the existing dataset representation; very
  large datasets would need a streaming equivalent (out of scope).
* No augmentation of sessions (perturbed copies) is performed.