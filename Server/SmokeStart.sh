set -euo pipefail
tmux -L SuperMimicAE new-session -d -s Smoke 'bash /home/wzy/SuperMimicAE/Server/Smoke.sh > /home/wzy/SuperMimicAE/Server/SmokeSession.log 2>&1'
