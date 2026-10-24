set -euo pipefail
source /home/wzy/SuperMimicAE/Server/Env.sh
exec tensorboard --logdir logs/Tensorboard --host 127.0.0.1 --port 6007
