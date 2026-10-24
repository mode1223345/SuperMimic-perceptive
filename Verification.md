# 验证记录

日期：2026-10-24。基线：`a1508d42c9d1699a52e8233ab16d08c488441ab9`。

## 已执行

- Python 3.12.14、PyTorch 2.14.1+cpu、NumPy 2.5.4、pytest 9.1.1，项目内 `.venv`。
- `OMP_NUM_THREADS=1`，`python -m pytest -q --tb=line -p no:cacheprovider`：**38 passed in 5.47s**。
- `supermimic/` 与 `tools/` 共 **23 个 Python 文件** AST 解析通过。
- Git Bash `bash -n` 检查 `run_train.sh`、`docker/run_container.sh`、`tools/setup.sh`、`tools/download_data.sh` 通过。
- `git diff --check` 通过。
- 独立只读审查后修正：失败恢复覆写元数据、恢复新目录日志表头、完整/安全起点评测混用、旧 benchmark 漏传地图、绘图读取路径失配。
- 图像查询梯度的突变验证：临时将输入位姿 detach，新坡面测试明确失败；生产实现保持可微。
- 原 `Code/SuperMimicVibed` 工作树干净，HEAD 仍为上面的基线提交。

## 主要测试行为

1. 高程图 xy/yaw/roll-pitch 约定、高度符号、缺失与越界、几何参数合法性和指纹。
2. 相同身体状态在不同地形下产生不同 actor/value 输出；小动力学模型可学习地形导致的不同状态变化。
3. 冻结 WM/critic 参数时，actor 能经多步预测更新，WM/critic 参数不变。
4. 第二步坡面查询对第一步动作的梯度，与解析高度 `.2*x-z` 的梯度一致。
5. 真实多步 `predict_sequence` 的状态误差反传更新 WM 地图编码器；critic 的 MSE 更新自己的地图编码器。
6. 恢复 replay 补满、重复 motion bin 回报先平均再 EMA、障碍 reset 候选与无候选保护。
7. 不兼容 checkpoint 拒绝、轻量快照不可当完整恢复、失败恢复不覆写旧运行配置。

## 尚未验证

本地没有该复现的 `IsaacLab/` 和 `data/`，本次未建立大型仿真环境。以下仍需在其固定 Linux/CUDA 环境实测：

- Isaac Lab 的静态盒体加载、Newton/MuJoCo 接触与采样地图之间的一致性。
- 训练/评估完整入口、真实 reward/FK、恢复后的真实采样及 checkpoint 周期。
- CUDA `grid_sample` 反传、`torch.compile`、默认 batch/horizon 的峰值显存和吞吐。
- 渲染视频和完整/安全起点两种评测曲线的实际输出。
- 深度相机部署端建图、实际台阶运动能力与实验收益。

CPU 验证证明了模块与梯度链路，不构成 G1 仿真训练成功或策略性能提升的证据。GPU smoke 命令见 `Guide.md`。
