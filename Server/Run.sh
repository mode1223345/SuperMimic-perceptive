set -euo pipefail
source /home/wzy/SuperMimicAE/Server/Env.sh
export CUDA_VISIBLE_DEVICES="$1"
RUN="$2"
TERRAIN="$3"
MODE="$4"
test ! -e "logs/$RUN"
EXTRA=()
if [ "$MODE" = ae ]; then
  test -f logs/AeFinal/Encoder.pt
  EXTRA=(--ae logs/AeFinal/Encoder.pt)
elif [ "$MODE" != baseline ]; then
  echo 'Mode must be ae or baseline' >&2
  exit 2
fi
exec python supermimic/train.py --num_envs 4096 --pi_batch 4096 --wm_batch 4096 \
  --compile --buffer_steps 512 --warmup_steps 64 --wm_window 8 --wm_batches 5 \
  --horizon 32 --value_batch 4096 --amp bf16 --terrain "$TERRAIN" \
  --wm_only_iters 500 --iterations 16000 --ckpt_every 1000 --log_every 50 \
  --seed 0 --log_dir "logs/$RUN" "${EXTRA[@]}"
