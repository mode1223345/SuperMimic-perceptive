#!/bin/bash
# Start the isolated 'supermimic-heightmap' container with this repository mounted.
# Usage: docker/run_container.sh [GPU_INDEX] [MEMORY_LIMIT]
set -euo pipefail
GPU=${1:-0}
MEM=${2:-128g}
REPO="$(cd "$(dirname "$0")/.." && pwd)"
docker run -d --name supermimic-heightmap --gpus "\"device=${GPU}\"" --memory="$MEM" --shm-size=16g --user root \
  -e HOME=/root -e WARP_CACHE_PATH=/workspace/supermimic/.cache/warp -e MPLCONFIGDIR=/tmp/mpl \
  -e TORCHINDUCTOR_CACHE_DIR=/workspace/supermimic/.cache/inductor \
  -v "$REPO":/workspace/supermimic -w /workspace/supermimic supermimic-heightmap:latest sleep infinity
