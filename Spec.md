# Frozen shared elevation AE

User-approved experiment: pretrain a local elevation-map autoencoder, freeze its encoder, and share that same encoder between actor, critic and dynamics. Preserve the original Heightmap checkout and all running jobs. New files use short names starting with a capital letter.

- Branch `Ae`, independent checkout `Code/SuperMimicAE`, baseline `2049839`.
- Same map semantics `[2,15,23]`, relative height and validity, same 32-dimensional HeightEncoder architecture. Train a decoder only during pretraining.
- Pretraining covers Flat, Steps, varied rectangular terrain, robot height, heading and map boundaries. Hold out procedural geometry and poses; record separate reconstruction/edge/mask metrics and representative images. No simulator rewards enter pretraining.
- One shared encoder object, always frozen and excluded from all three optimizers. Input derivatives remain enabled in imagination. Reuse one embedding for actor and dynamics at the same imagined pose.
- AE checkpoint and downstream checkpoints record architecture, map semantics and encoder hash; downstream snapshots are self-contained. Reject incompatible resumes. Do not replace existing trained representations in-place.
- Train fresh Flat and Steps models with the current full-4096 configuration; same motions, reward, horizon, seed and evaluation protocol. AE pretraining cost is reported separately. Existing original curves are informative but not a strict equal-configuration baseline because they switched configuration at iteration 5000.
- Use idle server GPUs, separate `/home/wzy/SuperMimicAE` logs/tmux/cache. Read-only reuse of installed dependencies and motion data. No changes to original sessions or training files.

Acceptance: CPU behavioral suite, GPU pretrain/reconstruction validation, eager and compiled actor-update smoke tests, checkpoint/evaluation/resume checks, then independent long-running comparison jobs and reproducible launch instructions.
