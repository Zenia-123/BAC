# Lowdim Transformer FP16：运行、结果与源码管理

本分支直接管理全部 `diffusion_policy/` 源码。以后的流程是：修改 `.py` → 运行实验 → 查看结果 → Git 提交和推送。无需每次应用 patch，不需要第二个子模块私有仓库。

## 一条命令运行完整实验

沿用现有 `bac` 环境和本机 checkpoint：

```bash
cd /home/zenia/Projects/BAC/BAC
conda activate bac
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B \
  -m BACInfer.analysis.lowdim_fp16_experiment \
  --output-dir results/lowdim_fp16_my_run \
  --budgets 5,20 --trials 10 --warmup 3 \
  --rollout-episodes 3 --rollout-budget 20 \
  --report-dir reports
```

输出目录必须不存在，每次使用新名称，避免覆盖已有结果。实验使用真实 Block Pushing 观测，并运行真实 rollout；不启用视频。设备默认为 `cuda:0`，权重默认 `workspace.model`，与 eval 保持一致。可用 `--weights ema` 明确切换，但不要混用不同权重的结果。

checkpoint 默认位置：

```text
checkpoint/low_dim/block_pushing/diffusion_policy_transformer/train_0/checkpoints/epoch=7550-test_mean_score=1.000.ckpt
```

checkpoint 和环境依赖不随 Git 上传；新电脑需具备相同模型文件和依赖。无需重新安装现有电脑的环境。

只检查流程、不跑 rollout 时增加 `--rollout-episodes 0`；只有明确选择 `--observation-source zeros` 才使用零观测。这类运行不能报告任务成功率。

## 实验做什么

1. 加载一次同一 checkpoint，打印并保存真实模型配置；使用 FP32 权重。
2. 对同一输入做 FP32 / AMP 单次 Transformer forward，记录模块 dtype。
3. 用环境 seed 0 的初始观测、噪声 seed 11，分别采集 FP32 和 AMP 的完整 100 步分支特征。原始特征保留各自 dtype，相似度及 L1 统计用 FP32。
4. 调用项目已有 ACS 动态规划，为每层 SA/CA/FFN 生成独立调度。FP32 和 AMP 保存到不同目录。
5. 用项目已有的跨步分支 L1 误差定义选择 BUA blocks，并保留原来的下游 FFN 调度并集传播规则。
6. 使用独立的环境观测 seed 10000–10002 和噪声 seeds 101/102/103，检查完整采样精度、缓存读写和动作误差。
7. 移除所有观察 hook 后，每个模式预热 3 次、同步计时 10 次，记录 mean/std/p50、峰值已分配显存和两种加速比。
8. 对 Original 与 ACS 20（含 BUA 消融及两种调度来源）运行相同环境 seeds 的 rollout。每次预测重新设置相同的对应噪声 seed；状态因动作不同而分化是实际闭环行为。

模式包括 FP32 / AMP Original、Threshold 1 正确性对照、Threshold 5，以及预算 5/20 下：FP32 ACS、AMP 加 FP32 调度、AMP 加 AMP 调度，各自开启/关闭 BUA。

若只想重新计时已完成的实验，可使用：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B \
  -m BACInfer.analysis.lowdim_fp16_experiment \
  --output-dir results/lowdim_fp16_20261017_full \
  --rebenchmark --trials 10 --warmup 3 --report-dir reports
```

这会保留误差、调度和 rollout，先保存旧 summary 的副本，再更新计时与报告。本轮初次计时发现缓存方法闭包的循环引用让前序候选暂留显存，已经在各候选之间增加 `gc.collect()` 并按此命令重新测量；最终报告使用复测值，旧值保留在运行目录。

## 文件和函数调用关系

```text
lowdim_fp16_experiment.main
  -> load_policy                 [checkpoint / model 或 EMA]
  -> observations                [真实初始环境观测]
  -> observe_forward             [单步模块 dtype]
  -> calibrate
       -> audit -> capture -> policy.predict_action
       -> FP32 相似度 -> OptimalCacheScheduler
       -> compute_block_l1_errors -> BUA 排序
  -> candidate                   [深拷贝；精度、调度、BUA]
       -> FastDiffusionPolicy.apply_cache
  -> audit                       [完整采样和缓存精度检查]
  -> benchmark                   [无 hook 的同步计时]
  -> rollout                     [闭环环境评估]
  -> summary.json / 报告

policy.predict_action
  -> conditional_sample
       -> autocast 内 model.forward
       -> AMP 输出 .float()
       -> autocast 外 scheduler.step
```

传统离线入口 `run_policy.py`、`collect_activations.py` 和 `get_optimal_cache_update_steps.py` 也已贯通 `--precision fp32|fp16_amp` 与 `--policy_weights model|ema`。采集路径现在包含精度目录，避免互相覆盖。

## 哪些源码改了，为什么

下面的路径均相对于 BAC 根目录；前七项原来保存在子模块 patch 中，本次按原内容直接纳入主仓库，未重新改变模型计算逻辑。

| 文件 / 函数 | 作用 |
|---|---|
| `diffusion_policy/diffusion_policy/policy/diffusion_transformer_lowdim_policy.py`：`__init__`、`conditional_sample` | 保存 `inference_precision`，默认 FP32；仅 `self.model(...)` 进入 FP16 autocast，输出 `.float()` 后交给 FP32 Scheduler。 |
| `diffusion_policy/diffusion_policy/model/diffusion/transformer_for_diffusion.py` | 保留已有学习注释，Transformer 算法不变。 |
| `diffusion_policy/diffusion_policy/env_runner/blockpush_lowdim_runner.py` | 低维向量环境关闭共享内存，适配本机 Gym。 |
| `diffusion_policy/diffusion_policy/gym_util/async_vector_env.py`、`sync_vector_env.py` | 适配现有 Gym 的 reset 和 concatenate 接口。 |
| `diffusion_policy/diffusion_policy/workspace/train_diffusion_transformer_lowdim_workspace.py` | 接受并转发 `output_dir`，与评估加载方式兼容。 |
| `diffusion_policy/setup.py` | 保留之前 namespace package 发现方式；本轮不执行安装。 |
| `BACInfer/scripts/eval_fast_diffusion_policy.py`：`main` | 已有 `--precision` 传到 policy；本次让独立 THOP forward 也使用同一精度。 |
| `BACInfer/analysis/check_lowdim_fp16.py`：`observe_forward` | hook 接收位置参数和关键字参数，支持 Transformer 的 `tgt=...` 调用；记录真实 dtype，结束后移除 hook。 |
| `BACInfer/analysis/run_policy.py`：`run_policy` | 增加精度和权重选择，禁止把当前 wrapper 不支持的参数继续传入；不再重开 stdout。 |
| `BACInfer/analysis/collect_activations.py`：`collect_activations` | 精度传给 policy，输出路径隔离；异常时也清理 hook 和临时 forward。 |
| `BACInfer/analysis/get_optimal_cache_update_steps.py` | 传递精度和权重；修正实际调度保存路径，设置可用默认参数。 |
| `BACInfer/analysis/activation_analysis.py`、`bu_block_selection.py` | FP16 特征在相似度、误差统计前转 FP32，避免低精度统计影响比较。 |
| `BACInfer/analysis/optimal_cache_scheduler.py`：`compute_optimal_steps` | 检查矩阵和预算，修正 `l2` 指标拼写，保留原 DP 目标。 |
| `BACInfer/core/diffusion_cache_wrapper.py`：`apply_cache`、`_load_block_optimal_steps`、`_update_cache_flags` | 校验完整调度；BUA 显式选择或找到校准数据；末步索引适配实际步数，保留原并集规则。 |
| `BACInfer/analysis/lowdim_fp16_experiment.py` | 新增统一实验入口：采集、ACS/BUA、dtype 审计、误差、计时、闭环评估、报告。 |
| `BACInfer/__init__.py` | 优先导入当前 checkout 自带的 diffusion_policy，避免误用其他目录的 editable 安装。 |
| `.gitmodules`、`diffusion_policy/VENDORED.md`、恢复脚本和旧 patch 说明 | 普通源码替代 gitlink；记录来源和许可证，旧恢复脚本安全提示无需补丁。 |

FP16 的边界可以具体理解为：轨迹 `sample` 和条件 `cond` 以 FP32 进入 `model.forward`；autocast 自动让符合规则的 Linear/Attention 等运算使用 FP16，并保留部分 FP32 运算。模块输出的实测 dtype 见 P2 报告。没有执行 `model.half()`，也没有把整个推理流程改为 FP16。

## THOP 是什么

THOP 是 PyTorch 模型的参数量和 MACs（乘加次数）估算工具。`profile(model, inputs=...)` 会执行模型一次，并通过计数规则估算运算量；它不是 GPU 计时器。FP16 不改变同一结构的乘加次数，不能期待 FLOPs 直接减半。

本项目 THOP 对 MultiheadAttention 的统计覆盖不完整，而且一次 Transformer forward 不是完整 100 步采样。正式实验以同步计时、峰值显存和实际 SA/CA/FFN 执行次数为证据，不用 THOP 估算数字推导完整加速比。eval 的 THOP 路径已补齐独立 autocast，使其执行精度跟随 `--precision`。

本轮真实 checkpoint 的 THOP 检查中，FP32 和 AMP 都得到 23,214,080 个 MACs；模型实际参数为 8,961,538，THOP 只统计到 4,748,546。这也直接说明应区分实际参数量与工具覆盖到的参数量。eval 的日志与 JSON 已标明 THOP 的单次、部分统计范围。

## 看哪些结果

- `results/<运行目录>/summary.json`：完整实验配置、模块 dtype、误差、计时、调度和 rollout。
- `calibration/fp32/` 与 `calibration/fp16_amp/`：各自原始特征、相似度与调度文件。
- `reports/P2_dtype_audit.md`：完整采样精度和缓存审计。
- `reports/P3_original_precision_comparison.md`：同输入、同噪声的 Original 对比。
- `reports/P4_acs_bua.md`：ACS / BUA 与调度来源消融。
- `reports/P5_lowdim_fp16_results.md`：最终结果、局限及交付。
- `reports/lowdim_fp16/summary.json`：随 Git 保存的结构化小型结果。

当前校准只使用一个初始观测窗口；三个 rollout episodes 的统计不代表论文规模的成功率复现。要扩展实验，可增加 `--rollout-episodes`、验证噪声 seeds 和独立校准数据。正式扩展须保存新的输出目录，并明确数据和协议变化。

## 测试和 Git

```bash
python -B -m unittest discover -s tests -p 'test_lowdim_fp16.py' -v
git status --short
git diff --check
git diff --stat
```

代码已经直接归属于 BAC。以后明确选择修改过的文件，例如：

```bash
git add BACInfer/analysis/lowdim_fp16_experiment.py \
        BACInfer/core/diffusion_cache_wrapper.py \
        diffusion_policy/diffusion_policy/policy/diffusion_transformer_lowdim_policy.py
git diff --cached
git commit -m "Update lowdim FP16 experiment"
git push origin feature/lowdim_fp16
```

这只是以后继续开发的示例；只加入本次实际修改的文件。数据、checkpoint、原始激活和大型日志不上传。原始第三方 LICENSE 与源码资产完整保留，旧 patch 仅作历史记录。
