set -euo pipefail
source /home/wzy/SuperMimicAE/Server/Env.sh
export CUDA_VISIBLE_DEVICES=1
python -m pytest -q > Server/Tests.log 2>&1
python supermimic/train.py --num_envs 128 --pi_batch 64 --wm_batch 64 \
  --buffer_steps 32 --warmup_steps 16 --wm_batches 2 --horizon 8 \
  --wm_only_iters 2 --iterations 6 --ckpt_every 6 --log_every 1 --amp bf16 \
  --terrain steps --ae logs/AeFinal/Encoder.pt --log_dir logs/Smoke > Server/Smoke.log 2>&1
# Resume without --ae: reconstruct shared weights entirely from the checkpoint.
python supermimic/train.py --num_envs 128 --pi_batch 64 --wm_batch 64 \
  --buffer_steps 32 --warmup_steps 16 --wm_batches 2 --horizon 8 \
  --wm_only_iters 2 --iterations 8 --ckpt_every 8 --log_every 1 --amp bf16 \
  --terrain steps --resume logs/Smoke/last.pt --log_dir logs/Smoke > Server/Resume.log 2>&1
python supermimic/evaluate.py --run_dir logs/Smoke > Server/Evaluate.log 2>&1
python supermimic/train.py --num_envs 4096 --pi_batch 4096 --wm_batch 4096 \
  --compile --buffer_steps 512 --warmup_steps 64 --wm_window 8 --wm_batches 5 \
  --horizon 32 --value_batch 4096 --amp bf16 --terrain steps --ae logs/AeFinal/Encoder.pt \
  --wm_only_iters 2 --iterations 6 --ckpt_every 6 --log_every 1 --log_dir logs/Compiled > Server/Compiled.log 2>&1
python Server/Check.py > Server/Check.log 2>&1
printf 'AE_GPU_SMOKE_COMPLETE\n'
