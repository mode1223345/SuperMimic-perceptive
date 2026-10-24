set -euo pipefail
source /home/wzy/SuperMimicAE/Server/Env.sh
export CUDA_VISIBLE_DEVICES=7
for step in $(seq 1000 1000 16000); do
  printf -v name 'ckpt_%06d.pt' "$step"
  for run in FlatAe StepsAe FlatBase StepsBase; do
    while [ ! -f "logs/$run/$name" ]; do
      if ! tmux -L SuperMimicAE has-session -t "$run" 2>/dev/null; then
        echo "$run ended before $name; inspect Server/$run.log" >&2
        exit 1
      fi
      sleep 30
    done
    python supermimic/evaluate.py --run_dir "logs/$run"
  done
done
printf 'AE_COMPARISON_EVALUATION_COMPLETE\n'
