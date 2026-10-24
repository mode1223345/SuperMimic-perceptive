# SuperMimic Heightmap — 首版实施计划

**目标：** 在独立副本与 `Heightmap` 分支中接入高程图感知，保留原始 SuperMimic 复现不变。先验证真值地形上的可微闭环，不宣称已完成深度相机部署。

**依据：** `论文精读/GMT/SuperMimic/CodeReview.md` 第 11 节及用户 2026-10-24 的实施授权。基线提交 `a1508d42c9d1699a52e8233ab16d08c488441ab9`。

**架构：** 同一份不可变地形配置生成仿真静态方块与栅格真值图；在当前/预测的机身位置与 yaw 下查询局部图。Actor、WorldModel、Critic 各有独立地图编码器。世界模型训练使用实测动作与实测下一状态；策略训练冻结世界模型/critic 参数，但保留它们对输入状态、动作及地图查询的梯度。

**技术栈：** 现有 PyTorch + Isaac Lab 3 / Newton / MuJoCo Warp；沿用上游固定提交，不在此步迁移 mjlab。纯 PyTorch 测试可在 CPU 执行；真实仿真需要上游 Linux/CUDA 环境、G1 资产及动作数据。

## 边界与约束

- 新建业务文件首字母大写；已有上游文件名保留，便于 diff。
- 独立本地 Git clone，不改原仓库文件或 Git 元数据。当前工作区根目录不是 Git 仓库，原生 worktree 工具无法指定嵌套仓库，因此使用独立 clone。
- 首版地形全环境共享且在整个训练/恢复期间保持不变；保存配置指纹。禁止训练中变更地形而继续复用旧 replay。
- 局部图为重力对齐、跟随 yaw 的网格；高度符号统一为 `terrain_z - base_z`，附有效性通道。不使用机身 roll/pitch 倾斜采样格。
- 默认平地验证回归；浅台阶是实验选项。参考动作不自动适配任意台阶，不宣称“任意地形 mimic”。
- 静态 cuboid 碰撞面与栅格采样来自同一配置。台阶边缘的双线性采样仅是连续梯度代理，有栅格近似误差。
- 部署端需要深度→点云→里程计融合→同规格局部高程图；本步不实现/声称已有这个映射器。

## 实施顺序

1. `supermimic/Heightmap.py`：地形配置、栅格化、局部可微查询、两通道 CNN；测试 xy/yaw/高度符号、缺失和输入梯度。
2. `supermimic/networks.py`：三个网络接地图；各自编码器随所属优化器训练。`WorldModel.step` 每步接重新查询的图。
3. `supermimic/env.py`：同源静态台阶配置；`train.py`/`evaluate.py` 接通真实采样和想象展开的地图，检查 checkpoint 配置一致性。
4. `supermimic/Training.py`：恢复训练重新预填 replay；重复 motion bin 回报先聚合再更新 EMA，配行为测试。
5. 端到端 CPU 小网络冒烟：多步模型训练、冻结模型下的 actor BPTT、critic 更新及参数隔离；真实 CUDA 仿真 smoke 命令与前置条件单独记录。
6. 代码审查、运行测试、记录未验证项；检查原仓库干净且仍在原提交。

## 审查重点

- 想象动作变化后重新查询地图，不复用真实轨迹旧图。
- 图像编码器的参数冻结和输入梯度不是一回事；不在 actor rollout 内用 `no_grad` 或 detach 截断链路。
- Replay/reset 不改变地形语义；恢复时拒绝不兼容地图或上游无地图 checkpoint。
- Critic 的地图编码在 critic 训练时有梯度，不能在特征预计算时误冻结。
- 无仿真环境时只报告 CPU 验证，不能把通过单元测试等同于可用的机器人训练结果。
