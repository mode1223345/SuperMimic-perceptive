set -euo pipefail
BASE=/home/wzy/SuperMimicHeightmap
DEST=/home/wzy/SuperMimicAE
test -d "$BASE/.venv"
test -d "$BASE/data"
test ! -e "$DEST"
mkdir "$DEST"
tar -xf /home/wzy/SuperMimicAE.tar -C "$DEST"
mkdir -p "$DEST/.cache" "$DEST/logs"
git -C "$DEST" init -b Ae
git -C "$DEST" config user.name Codex
git -C "$DEST" config user.email codex@local
git -C "$DEST" add .
git -C "$DEST" commit -m 'Isolated frozen shared AE experiment snapshot'
ln -s "$BASE/.venv" "$DEST/.venv"
ln -s "$BASE/data" "$DEST/data"
ln -s "$BASE/IsaacLab" "$DEST/IsaacLab"
