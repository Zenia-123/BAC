# 无仿真缓存分析：当前实验与下一步

本入口对应分享对话中“分析 Threshold，再进入 ACS”的阶段。它不创建
Environment Runner，不需要 gym、TF-Agents 或 PyBullet。

## 运行

```bash
conda activate bac
cd /home/zenia/Projects/BAC/BAC
python -m BACInfer.analysis.analyze_cache_no_env \
  -c 'checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_0/checkpoints/epoch=7550-test_mean_score=1.000.ckpt' \
  -o results/block_pushing_no_env \
  -d cuda:0 --num_caches 5,20 --trials 3
```

默认使用原始尺度的零观测 `[1, 3, 16]`，由策略的 normalizer 归一化。
默认加载 `workspace.model`，与已有速度 Benchmark 保持一致。
官方 `run_policy.py` 会在启用 EMA 时选择 `workspace.ema_model`；需要切换时
明确传入 `--policy_weights ema`，并在同一组对比中使用同一套权重。

零观测只用于验证流程。当前是 FP32 推理，没有切换到 FP16，也没有测任务成功率。

## 数据流与保存的文件

1. 加载 Checkpoint，构造同一套权重的策略和观测。
2. 运行一次原始策略，在每层 `dropout1/2/3` 收集 SA/CA/FFN 的残差分支输出。
3. 将每个输出展平，计算 100 × 100 的余弦相似度矩阵。
4. 调用已有 `OptimalCacheScheduler` 动态规划，为每个 Block 分别选更新点。
5. 对 Original、Threshold 1、Threshold 5、ACS 5、ACS 20，使用相同观测与
   随机种子 11、12、13，检查动作误差、实际 Block 计算次数和缓存重置。
6. 移除采集 Hook，预热后单独测量延迟。延迟包括统一的随机种子设置，
   是本机短测量结果。

```text
results/block_pushing_no_env/
├── analysis_summary.json
├── original/
│   ├── activations.pkl
│   └── similarity_matrices.pkl
└── optimal_steps/cosine/
    └── decoder.layers.<层号>.dropout<1|2|3>/
        └── optimal_steps_<模块名>_<更新预算>_cosine.{pkl,json,txt}
```

`activations.pkl` 是模块名到张量列表的字典。当前共 24 个模块，每个列表长度
100，单个张量形状 `[1, 5, 256]`，类型 `torch.float32`。

列表和调度文件中的索引是执行顺序：索引 0 对应 DDPM timestep 99，
索引 99 对应 timestep 0。完整对应表记录在 summary 的 `denoising_timesteps`。

## 2026-10-14 实测结果

以下是新入口同一次实验的结果，BU 未启用。动作 MAE 是三个随机种子的均值，
单位为策略返回的动作尺度，不能直接解释为任务成功率。

| 模式 | 平均推理延迟 | 加速比 | 每个 Block 实际计算次数 | 动作 MAE |
|---|---:|---:|---:|---:|
| Original | 605.0 ms | 1.00× | 100 | 0 |
| Threshold 1 对照 | 641.1 ms | 0.94× | 100 | 0 |
| Threshold 5 | 195.3 ms | 3.10× | 20 | 0.000647 |
| ACS 5 | 119.1 ms | 5.08× | 6 | 0.002564 |
| ACS 20 | 204.9 ms | 2.95× | 21 | 0.001951 |

Threshold 1 在三个种子下的 `action` 和完整 `action_pred` 均与原始策略一致。
Threshold 5 的更新点为 `0, 5, 10, ..., 95`，其余 80 步复用缓存。

`cache_threshold=5` 是更新间隔；`num_caches=5` 是动态规划选择的更新点数量。
当前 `_update_cache_flags` 还会向 Optimal 调度添加索引 99，因此本次 ACS 5/20
实际计算 6/21 次。不能把 Threshold 5 与 ACS 5 当作同预算比较。

本次零观测实验中 ACS 20 的动作 MAE 高于 Threshold 5；这些结果不能支持
“ACS 必然更准”或“加速无损”的结论。调度只在 seed 11 的一次推理上拟合，
种子变化时的误差也说明需要更有代表性的校准数据。

## 已有 Benchmark 的 FLOPs 应怎样理解

旧 `eval_fast_diffusion_policy.py` 调用的是
`profile(fast_policy.model, inputs=(sample, timestep, cond))`，只运行一次
Transformer 前向，不包含完整 100 步去噪。

预热后缓存计数器为 99，该 profiling 前向将其增加到 100，Threshold 5
恰好重新计算所有 Block，因此两份旧 JSON 都报告 `46,428,160`。
此外，本地 THOP 没有为 `MultiheadAttention` 注册计数器，注意力计算统计不完整。
这两个旧数字不能用来比较完整 Original/Threshold 推理的计算量。

实测的 Block 次数可证明缓存跳过了计算，但仍不等于完整 FLOPs、显存流量或能耗。

## 为什么另建入口

官方链路 `get_optimal_cache_update_steps -> collect_activations -> run_policy`
仍会实例化环境，本地导入 Runner 时缺少 `gym`。此外：

- `run_policy` 给当前不接受 `output_dir` 参数的 lowdim Workspace 传入了该参数。
- 集成入口检查 `<output>/optimal_steps`，底层计算函数实际保存到
  `<output>/original/optimal_steps`，路径不一致。
- 官方 ACS 默认可能选择 EMA，与已有 Benchmark 使用的 model 权重不同。

本入口直接采集特征，并显式保存调度路径，以完成当前无仿真实验。

## 下一步

先打开第 0 层的三个调度 JSON，对照 summary 中的特征相似度和实际计算次数，
理解 ACS 如何给 SA、CA、FFN 分配不同更新时间。

本次第 0 层热力图保存为 `results/block_pushing_no_env/layer0_cosine.png`。
横纵轴是执行索引，颜色表示跨步余弦相似度，红圈是动态规划选出的 5 个更新点。

随后用真实任务观测重新校准。观测文件格式是由 `torch.save` 保存的
`{'obs': tensor[B, 3, 16]}`，使用原始尺度、有限数值；传入 `--obs_file <路径>`，
并指定新的输出目录。一个观测窗口仍只代表一个校准输入，需要进一步扩展
到多个任务状态与独立验证样本。

在这些数据上建立 ACS 基线后，再采集缓存误差、选择 BU Block 并验证 BU
实际增加的更新点。当前结果明确记录 `num_bu_blocks=0`，尚未完成 ACS + BU
或论文成功率复现。
