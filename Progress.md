# Execution ledger — AePlan.md

- User approved design and execution; implementation proceeds inline.
- Native worktree tool could not act on the projectless parent directory (`Not a git repository`). Created a git worktree in `Code/SuperMimicAE` on branch `Ae` instead. Original checkout unchanged.
- Baseline source hashes match the running server. Initial CPU suite: 37 passed, one temporary-directory setup error caused by sandbox temp permissions. Re-run using a workspace-local basetemp.
- Interfaces: AE load returns a frozen HeightEncoder and metadata; networks consume that object; train/evaluate/checkpoint validation consume the same metadata and hash.
- Experimental decision: keep latent dimension and CNN identical to baseline; decoder only exists for pretraining. This tests frozen shared AE as a combined intervention, not separate attribution of sharing versus pretraining.
- Task 1 and Task 2: complete in bdad622; eight new tests failed before implementation, full suite then 46 passed using workspace-local temp directory. Original tests passed too.
- First local pretrain CLI reached checkpoint/metrics then lacked matplotlib in the small CPU venv. Server environment already has matplotlib; no dependency changes to the original environment.
- Deployment shell quoting error was corrected by uploading standalone shell scripts. Local git commit uses command-scoped Codex identity; global git settings unchanged.
- First server pretrain completed: canonical Steps RMSE 1.04 cm, edge RMSE 1.78 cm. A second run uses 60,000 steps at lr=0.0003 (first run 12,000 at 0.001) to reduce reconstruction error; all artifacts retained.
- Ruling: launch two fresh original-structure controls on idle GPUs 5/6, in addition to AE runs 0/1. Original historical runs changed batch/compile at 5000, so these new controls provide a cleaner comparison. Costs two idle GPUs during the experiment.
- Formal AE: 60,000 steps, selected step 59,400, 111.78 s. Held-out random geometry height RMSE 12.16 mm, edge RMSE 29.29 mm; canonical Steps 8.07/14.78 mm; Flat 3.20 mm. Encoder SHA256 `6fd7a16209283863820c40d0903e60a00afe3149d67e4ad82fb3c045874fdb54`. Reconstruction images inspected: geometry retained with visible edge smoothing; this remains a testable baseline, not lossless encoding.
- Independent whole-branch review (`ae_review`) found no actionable correctness issue. GPU compilation/runtime and eventual performance remain empirical checks; single-seed combined-intervention limitations are documented in Ae.md.
- Server CPU suite: 46 passed in 2.74 s. Eager/resume/evaluation/compiled GPU checks in progress.
- GPU verification complete: eager 6 iterations, self-contained resume to 8 without `--ae`, evaluation across all 40 full motions, compiled 4096 env/pi/wm for 6 iterations including actor/critic updates. Metrics finite, actor gradients nonzero, shared encoder hash unchanged in all three saved networks. Evidence: `results/Validation.json`.
- All four long runs launched 2026-10-24 18:33:24 China time under separate `SuperMimicAE` tmux socket. GPU 0/1 AE, 5/6 fresh controls, 7 evaluator. Original GPU 3/4 jobs untouched. Planned 16,000 iterations, checkpoints/evaluation every 1,000.
- New TensorBoard 6007 and local SSH forward verified HTTP 200; original 6006 unchanged. Report exports scalar curves and `results/Status.json` every 30 seconds.
- Independent reviewer deferred empirical adequacy/performance: GPU functional checks now passed; training quality remains pending the scheduled evaluations. Combined intervention and single seed remain explicit experimental limits.
- Branch is intentionally retained, not merged into Heightmap, per user request.
