set -euo pipefail
cd /home/wzy/SuperMimicAE
export PATH="$PWD/.venv/bin:$PATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1
export WARP_CACHE_PATH="$PWD/.cache/warp"
export TORCHINDUCTOR_CACHE_DIR="$PWD/.cache/inductor"
export TORCHINDUCTOR_COMPILE_THREADS=4
export MPLCONFIGDIR="$PWD/.cache/mpl"
export PYTHONUNBUFFERED=1 MUJOCO_GL=egl
