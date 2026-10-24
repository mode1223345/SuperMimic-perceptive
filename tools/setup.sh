#!/bin/bash
# Clone Isaac Lab at the pinned commit and build the kit-less image + the SuperMimic image.
set -euo pipefail
cd "$(dirname "$0")/.."
ISAACLAB_COMMIT=46d6cc3d7b115906fad12efd372fbf69f4e3d3ed
if [ ! -d IsaacLab ]; then
  git clone https://github.com/isaac-sim/IsaacLab.git
fi
git -C IsaacLab checkout "$ISAACLAB_COMMIT"
(cd IsaacLab && docker build -f docker/Dockerfile.kitless -t isaac-lab-kitless:main-46d6cc3 .)
docker build -t supermimic-heightmap:latest docker/
