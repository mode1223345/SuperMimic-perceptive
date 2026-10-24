# Shared elevation AE implementation plan

> Execute inline using executing-plans, with a final independent code review. User has already approved the design and execution.

**Goal:** Pretrain and freeze a shared elevation encoder without interrupting the existing experiment.

**Architecture:** Keep HeightEncoder and network dimensions. Add AE pretraining and frozen-encoder ownership/checkpoint helpers. Inject one frozen encoder into all three networks; preserve input gradients and optimizer isolation.

**Tech Stack:** PyTorch, existing SuperMimic simulation environment, pytest, existing server environment.

**Spec:** [Spec.md](Spec.md)

## Global constraints
- Baseline and running training remain unchanged. New files begin with capitals.
- Same input semantics, 32-D bottleneck, 4096 env/pi/wm, 8-step dynamics, 32-step imagination.
- No AE parameter updates after pretraining, including after unfreezing dynamics/critic or resume.

## Review focus
- Re-enabling network gradients must not unfreeze the encoder.
- Shared state dictionaries cannot silently overwrite inconsistent encoder weights.
- A frozen encoder must preserve nonzero gradients to queried pose.
- Evaluation must load self-contained frozen checkpoints without requiring the pretrain file.
- Synthetic/pretrain split and old-vs-new training budgets must be disclosed honestly.

## Task 1: AE and data
Files: `supermimic/Ae.py`, `supermimic/Pretrain.py`, `Tests/TestAe.py`.
Interfaces: `Autoencoder`, `sample_maps`, `reconstruction_loss`, `save_encoder`, `load_encoder`, `encoder_hash`.
- [x] Write failing behavior tests for reconstruction, invalid masks, data splits and artifact validation.
- [x] Implement deterministic geometry sampling, AE training, validation metrics/plots and portable checkpoints.
- [x] Run CPU tests and small reconstruction fit.

## Task 2: Frozen shared integration
Files: `networks.py`, `Rollout.py`, `train.py`, `evaluate.py`, `Checkpoint.py`, `Tests/TestShared.py`.
Interfaces: optional injected `map_encoder`; optional precomputed `map_latent`; freeze-safe `set_trainable`; encoder metadata in map configuration.
- [x] Test single object ownership, optimizer exclusion, immutable weights after updates, retained pose gradients, and checkpoint guards.
- [x] Implement training/evaluation integration with legacy behavior unchanged when no AE is supplied.
- [x] Run full suite and round-trip tests.

## Task 3: Server experiment
Files: `Server/Env.sh`, `Server/Run.sh`, `Server/Monitor.sh`, `Server/Report.py`, `Guide.md`.
- [x] Deploy independent source and reuse existing dependencies/data read-only.
- [x] Pretrain AE on an idle GPU; inspect held-out plots and errors.
- [x] Eager/compiled full-size smoke, resume and full-motion evaluation; inspect encoder hashes.
- [x] Independent review, then start fresh Flat/Steps training and evaluation/metrics export.
- [x] Record exact commands, version, resource allocation and limits of the comparison.
