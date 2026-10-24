# 冻结共享高程图 AE 实验

这是独立的 `Ae` 分支，基于 `Heightmap` 的 `2049839`。原目录 `../SuperMimicHeightmap`、原服务器目录和原训练会话均保留。

## 网络与更新

1. 预训练：`局部图 → HeightEncoder → 32 维 z → Decoder → 重建图`。
2. 下游：冻结同一个 HeightEncoder 实例，分别交给 actor、world model、critic。
3. Actor 更新时冻结整个 world model 和 critic；恢复它们的可训练参数时仍跳过 AE。
4. AE 参数不加入任何下游优化器；每次保存 checkpoint 都核对编码器哈希。冻结参数不切断对输入地图和位姿的梯度。

输入 `[B,2,15,23]`，两个通道为 `terrain_z-base_z` 与有效性；采样范围、坐标系、CNN 结构和 32 维维度与原版一致。编码器为 `Conv(2→8) → Conv(8→16) → Flatten(384) → Linear(64) → Linear(32)`，激活为 ELU。AE 解码器为 `32→128→256→690`，输出高度和有效性 logit。

`L_AE = 有效格子上的高度 MSE + 0.01 × 有效性 BCE`。没有额外边缘损失，单独记录边缘误差，避免平均误差掩盖台阶问题。控制阶段没有重建损失，解码器不参与控制。

同一想象步里，actor 和 world model 复用同一次编码；critic 对它评价的状态查询相应地图。不同预测位姿需要重新查询和编码。

## 预训练数据

- 65,536 个训练局部图，8,192 个验证局部图，使用生产环境的 Heightmap 查询实现。
- 平地、现有 4/8/4 cm 台阶和随机矩形台地；随机台地高度 2–24 cm。当前数据不包含任意斜坡、负高度坑洞或真实深度噪声，不能由此声称任意地形泛化。
- 随机朝向、身体高度和地图边界。大多数根部高度 0.25–1.25 m，少量 0.02–1.6 m。
- 验证集的随机地形布局与训练集不同；固定 Flat/Steps 几何两边均出现，但采样位姿不同。使用验证重建损失选择 checkpoint，不使用控制评估奖励选 AE。
- `logs/Ae` 是首轮预训练，`logs/AeFinal` 是较低学习率的正式候选；保留两轮报告和耗时。

## 对照设计

| 运行 | GPU | 地形 | 地图编码器 |
|---|---:|---|---|
| FlatAe | 0 | 平地 | 预训练、冻结、共享 |
| StepsAe | 1 | 台阶 | 预训练、冻结、共享 |
| FlatBase | 5 | 平地 | 三个独立、可训练 CNN |
| StepsBase | 6 | 台阶 | 三个独立、可训练 CNN |

四个新运行均从零开始，seed=0，4096 env / pi batch / wm batch，torch.compile，buffer=512，warmup=64，8 步世界模型拟合，每轮 5 次 WM 更新，32 步想象，前 500 次仅训练 WM，总计 16,000 次迭代。奖励、机器人、动作数据、reset 和全 40 段动作评估协议保持一致。GPU 7 做评估。

原始 FlatFull / StepsFull 仍在 `/home/wzy/SuperMimicHeightmap` 的 GPU 3/4 运行。它们第 5000 次改变过训练配置，因此主要对比新增的同配置 Base/Ae；原曲线作为背景参考。

这是单个种子下，**AE 预训练＋共享＋冻结**的组合实验，不能分别归因于其中一项。相同种子也不保证架构改变后下游参数逐元素相同。样本效率看 sim_steps，训练速度看耗时，并另计 AE 预训练成本。跨 GPU 并行负载会影响墙钟时间，不能据此直接声称算法加速。

## 文件与命令

服务器根目录 `/home/wzy/SuperMimicAE`，复用原 `.venv`、`data` 和 IsaacLab 依赖；代码、缓存、日志和 tmux socket 独立。

```bash
source /home/wzy/SuperMimicAE/Server/Env.sh
# 已执行的预训练；输出目录必须是新的，防止覆盖
CUDA_VISIBLE_DEVICES=0 python supermimic/Pretrain.py --out logs/AeFinal --steps 60000 --lr .0003
# 完整启动参数见 Server/Run.sh；新运行必须使用新名称
bash Server/Run.sh 0 FlatAe flat ae
# 自包含恢复：无需原始 AE 文件，仍需兼容的训练超参数
python supermimic/train.py ... --resume logs/FlatAe/last.pt --log_dir logs/FlatAe
tmux -L SuperMimicAE list-sessions
```

预训练报告：`logs/AeFinal/Report.json`、`Metrics.csv`、`Reconstruction.png`；模型：`Encoder.pt`（仅编码器）和 `Best.pt`（完整 AE）。

本地已取回正式编码器与验证报告到 `results/Encoder.pt`、`results/Report.json` 和 `results/Reconstruction.png`。正式预训练约 112 秒，前一轮探索约 24 秒，均为该机器上观测到的耗时。正式候选的独立随机地形高度 RMSE 12.16 mm、边缘 29.29 mm；当前台阶几何的独立位姿采样为 8.07 mm / 14.78 mm；平地 3.20 mm。重建边缘仍有平滑，不能当作无损几何表示。

控制训练：`logs/<运行>/train_log.csv`、`last.pt`、`ckpt_*.pt`。全动作评估：`logs/<运行>/Eval_full/eval.csv` 和逐 checkpoint JSON。对比图：`results/Curves.png`；TensorBoard：服务器 `127.0.0.1:6007`，需 SSH 转发。

所有下游 checkpoint 内嵌编码器权重和哈希，评估不依赖 AE 预训练文件路径。原独立编码器 checkpoint 可以在无 AE 模式下使用，但禁止将它直接恢复成共享 AE 训练。

## 验证记录

实现前已观察到新增测试失败；首轮完整 CPU suite 为 46 passed。Windows 本地 `.venv` 缺少 matplotlib，绘图在已有 matplotlib 的服务器环境完成。最终 GPU smoke、恢复、全动作评估和运行状态见 `Progress.md` 与 `results/Status.json`。

2026-10-24 18:33（中国时间）四组正式训练启动。GPU eager、恢复、全 40 动作评估流程和全 4096 编译配置均通过，详见 `results/Validation.json`。此处的评估只是验证未训练策略可完成评估流程，不能据其分数判断效果。正式对照结果随训练逐步写入 TensorBoard。

本机入口：[新实验 TensorBoard](http://127.0.0.1:6007/#scalars)。原实验仍使用 6006。SSH 转发随当前本机 SSH 进程保持；连接断开时重新转发服务器 6007 即可。

本地 `results/Curves.png` 和 `results/Status.json` 是启动阶段抓取的快照；持续更新的数据在服务器同名路径及 TensorBoard。前 500 次迭代仅拟合世界模型，actor/critic 的损失暂为 0 属于预定预热流程。
