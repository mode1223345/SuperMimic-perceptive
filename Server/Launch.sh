set -euo pipefail
cd /home/wzy/SuperMimicAE
test -f logs/AeFinal/Encoder.pt
grep -q AE_GPU_SMOKE_COMPLETE Server/SmokeSession.log
.venv/bin/python Server/Check.py > Server/Check.log 2>&1
tmux -L SuperMimicAE new-session -d -s FlatAe 'bash /home/wzy/SuperMimicAE/Server/Run.sh 0 FlatAe flat ae > /home/wzy/SuperMimicAE/Server/FlatAe.log 2>&1'
tmux -L SuperMimicAE new-session -d -s StepsAe 'bash /home/wzy/SuperMimicAE/Server/Run.sh 1 StepsAe steps ae > /home/wzy/SuperMimicAE/Server/StepsAe.log 2>&1'
tmux -L SuperMimicAE new-session -d -s FlatBase 'bash /home/wzy/SuperMimicAE/Server/Run.sh 5 FlatBase flat baseline > /home/wzy/SuperMimicAE/Server/FlatBase.log 2>&1'
tmux -L SuperMimicAE new-session -d -s StepsBase 'bash /home/wzy/SuperMimicAE/Server/Run.sh 6 StepsBase steps baseline > /home/wzy/SuperMimicAE/Server/StepsBase.log 2>&1'
tmux -L SuperMimicAE new-session -d -s Monitor 'bash /home/wzy/SuperMimicAE/Server/Monitor.sh > /home/wzy/SuperMimicAE/Server/Monitor.log 2>&1'
tmux -L SuperMimicAE new-session -d -s Report 'bash /home/wzy/SuperMimicAE/Server/Report.sh > /home/wzy/SuperMimicAE/Server/Report.log 2>&1'
tmux -L SuperMimicAE new-session -d -s Tensorboard 'bash /home/wzy/SuperMimicAE/Server/Tensorboard.sh > /home/wzy/SuperMimicAE/Server/Tensorboard.log 2>&1'
