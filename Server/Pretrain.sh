set -euo pipefail
source /home/wzy/SuperMimicAE/Server/Env.sh
export CUDA_VISIBLE_DEVICES=0
exec python supermimic/Pretrain.py --out logs/AeFinal --steps 60000 --lr .0003
