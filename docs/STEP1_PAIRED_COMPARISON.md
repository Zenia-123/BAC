# 第一步：同条件任务对照

本步只建立可解释的任务对照，不调整权重、严格 FP16 内核、ACS 目标或 BUA 算法。
四组依次为 FP32 Original、严格 FP16 Original、FP32 ACS20、严格 FP16 ACS20，全部关闭 BUA。
官方入口仍是 `python -m BACInfer.scripts.eval_fast_diffusion_policy`。

## 已补充的源码

- `BACInfer/scripts/eval_fast_diffusion_policy.py`：新增 `--test_start_seed`、`--seed`、`--rollout_only`。
  保存 checkpoint 指纹、实际任务配置、p1/p2、逐 seed 得分，以及独立的 `rollout_protocol.json`。
- `BACInfer/core/reproducibility.py`：指定 seed 时固定初始化随机数，关闭 TF32、cuDNN benchmark，启用确定性算法。
  cuBLAS 工作区配置在首次矩阵运算前设置。
- 源码目录 `diffusion_policy/.../env_runner/blockpush_lowdim_runner.py`：新增可选 `policy_seed`。
  每次 `predict_action()` 前重新设置 PyTorch seed，并记录每批初始观测指纹、噪声 seeds 和首个动作。
  没有指定 seed 时仍沿用原来的随机数流。
- `tests/test_paired_rollout.py`：用真实 runner 循环和轻量环境替身验证跨预测、跨批次的种子规则。

只在脚本开始设置一次 seed 不够：模型初始化、预热、THOP、计时都会消耗随机数。
本修改在每次真实动作预测前重新设置 seed，使初始 trajectory 和 DDPM scheduler 步进使用对应的随机数流。
这没有更改 policy 的去噪公式和精度路径。

## 四组共用的条件

| 条件 | 数值 |
|---|---|
| checkpoint | `checkpoints/epoch=7550-test_mean_score=1.000.ckpt` |
| 权重 | `workspace.model`，不切换 EMA |
| 测试 seeds | `4300000 / 4300001 / 4300002`，延续之前失败的同三个设置 |
| 训练任务数 / 测试任务数 / 并行环境 | `0 / 3 / 3` |
| 环境步上限 | 350 |
| 去噪步骤 | checkpoint 的 100 步；ACS20 仍然执行 100 步 |
| 动作预测长度 / 每次执行长度 | checkpoint 的 5 / 1 |
| 推理噪声 | 第 k 次批量预测使用 `11+k`，k 从 0 开始 |
| BUA | 关闭，`--num_bu_blocks 0` |
| 视频 | 关闭 |
| 性能测试 | 跳过，本步只测任务表现 |

一般多批次规则为：`seed + chunk_index * (max_steps + 1) + prediction_index`。
每次 seed 对整个 batch 生效，三个环境按固定 batch 顺序取随机数，不是每个环境获得相同噪声。
因此四组须保持相同 `n_envs`、任务顺序和噪声规则。
不同策略执行后环境状态会逐渐不同；同条件指相同初始状态及随机数安排，不要求后续观测相同。

这是新增的受控协议。之前没有固定推理噪声的结果不能直接与本轮组合成四组对照。
三例用于定位问题，不能作为稳定成功率估计。

## ACS20 准备

本机已从现有特征、相似度矩阵生成两种精度的 ACS20 调度，已有的 ACS5 文件保留。
没有重新运行模型采集激活。两组 ACS 使用各自精度的特征生成调度，
所以两组 ACS 之间的差异也包含调度差异；A/B 两组无缓存才直接隔离精度切换。

若需要重新生成调度，可在项目目录执行以下命令；已有文件会复用，无需加 `--force_recompute`：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B \
  -m BACInfer.analysis.get_optimal_cache_update_steps \
  --checkpoint checkpoints/epoch=7550-test_mean_score=1.000.ckpt \
  -o assets/blockpush_fp32 --device cuda:0 \
  --precision fp32 --metrics cosine --num_caches 20

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B \
  -m BACInfer.analysis.get_optimal_cache_update_steps \
  --checkpoint checkpoints/epoch=7550-test_mean_score=1.000.ckpt \
  -o assets/blockpush_strict --device cuda:0 \
  --precision decoder_fp16_strict --metrics cosine --num_caches 20
```

## 正式运行命令

在同一个 Bash 终端依次运行。数组仅用于复用完全相同的公共参数；
`"${STEP1_COMMON[@]}"` 会逐个展开参数，不会把整个数组拼成一个字符串。

```bash
cd /home/zenia/Projects/BAC/BAC-original-fp32
conda activate bac
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

STEP1_COMMON=(
  --checkpoint checkpoints/epoch=7550-test_mean_score=1.000.ckpt
  --device cuda:0
  --n_train 0 --n_test 3 --n_envs 3
  --test_start_seed 4300000 --max_steps 350
  --seed 11 --num_bu_blocks 0 --skip_video --rollout_only
)
```

先执行 A，确认基线结果：

```bash
python -B -m BACInfer.scripts.eval_fast_diffusion_policy "${STEP1_COMMON[@]}" \
  --precision fp32 --cache_mode original \
  -o results/step1_paired/fp32_original
```

B：只切换严格 decoder 精度，仍无缓存。

```bash
python -B -m BACInfer.scripts.eval_fast_diffusion_policy "${STEP1_COMMON[@]}" \
  --precision decoder_fp16_strict --cache_mode original \
  -o results/step1_paired/strict_original
```

C：FP32 + ACS20。

```bash
python -B -m BACInfer.scripts.eval_fast_diffusion_policy "${STEP1_COMMON[@]}" \
  --precision fp32 --cache_mode optimal --num_caches 20 --metric cosine \
  --optimal_steps_dir assets/blockpush_fp32/original/optimal_steps/cosine \
  -o results/step1_paired/fp32_acs20
```

D：严格 FP16 + ACS20。

```bash
python -B -m BACInfer.scripts.eval_fast_diffusion_policy "${STEP1_COMMON[@]}" \
  --precision decoder_fp16_strict --cache_mode optimal --num_caches 20 --metric cosine \
  --optimal_steps_dir assets/blockpush_strict/decoder_fp16_strict/original/optimal_steps/cosine \
  -o results/step1_paired/strict_acs20
```

每条执行结束后再执行下一条。输出目录用于保存本轮结果，重复评估应换新名称。
如果 A 也全部失败，应先检查基线和环境适配；不要将结果直接解释为严格 FP16 的问题。

## 查看与核对

各目录的 `eval_results.json` 直接提供：

- `scores`：各环境 seed 的分数。
- `p1`：第一阶段完成比例，项目定义为 score > 0.4。
- `p2`：两阶段完成比例，项目定义为 score > 0.9；本步主要比较它。
- `mean_score`：分数均值，不一定等于 p2。
- `precision`、`cache_mode`、`seed`、`checkpoint_sha256` 和 `env_runner_config`。

`rollout_protocol.json` 额外保存：

- `num_inference_steps`，应为 100。
- 确定性设置与 TF32 状态。
- `rollout_trace[0].environment_seeds`，应为相同三个 seeds。
- `initial_observation_sha256`，四组应相同。
- `prediction_seeds`，应从 `[11,12,13,...]` 开始；长度可因策略提前完成而不同。
- `first_action`，用于同配置重复运行的一致性检查，不要求不同策略相同。

`--rollout_only` 不生成性能统计，`speedup` 和 `flops` 保存为 `null`。
严格模式仍在任务结束后导出 `kernel_audit`，对实际执行的内核检查精度。

可用下列命令汇总本轮四组结果，并检查公共条件：

```bash
python -B - <<'PY'
import json
from pathlib import Path

root = Path('results/step1_paired')
records = []
for name in ('fp32_original','strict_original','fp32_acs20','strict_acs20'):
    folder = root/name
    if not (folder/'eval_results.json').exists():
        print(name, '尚未运行完毕')
        continue
    result = json.loads((folder/'eval_results.json').read_text())
    protocol = json.loads((folder/'rollout_protocol.json').read_text())
    signature = (result['checkpoint_sha256'], protocol['num_inference_steps'],
                 result['env_runner_config'], protocol['rollout_trace'][0]['initial_observation_sha256'])
    records.append((name, signature))
    assert protocol['num_inference_steps'] == 100
    assert protocol['rollout_trace'][0]['prediction_seeds'][0] == 11
    assert result['num_bu_blocks'] == 0
    print(name, 'scores=',result['scores'], 'p2=',result['p2'], 'mean_score=',result['mean_score'])
if records:
    assert all(signature == records[0][1] for _,signature in records), '公共条件不一致'
    print('已完成组的 checkpoint、环境配置、初始观测一致。')
PY
```

解释对照时优先检查：A 是否完成任务；A/B 判断精度影响；A/C 和 B/D 分别判断缓存影响。
若 A/B 都很好、C/D 退化，下一步先调整缓存。
若 A 很好、B 退化，下一步优先检查严格半精度数值路径。
