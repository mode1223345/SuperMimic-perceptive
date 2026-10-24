set -euo pipefail
tmux -L SuperMimicAE new-session -d -s Pretrain 'bash /home/wzy/SuperMimicAE/Server/Pretrain.sh > /home/wzy/SuperMimicAE/Server/Pretrain.log 2>&1'
