# 严格 FP16 + ACS20 使用与提交

当前结论：已测任务中，严格 FP16 对任务完成情况的影响小于缓存更新预算的影响。
FP32 / 严格 FP16 Original 均通过同条件三任务；ACS5 在两种精度下均失败；
严格 FP16 ACS10、ACS15 各通过 2/3，ACS20 通过 3/3。
后续 6 个新环境、各 2 组新噪声的配对验证中，严格 FP16 Original 与 ACS20 均为 12/12。
这些 ACS20 结果关闭 BUA，不能当作 ACS20+BUA3 已验证的完成率，也不代表所有输入下精度无损。

## 运行

沿用官方入口，关键参数为 `--precision decoder_fp16_strict`、`--num_caches 20`、
`--num_bu_blocks 3`。本机已有严格 FP16 的 ACS20 校准和特征，可直接运行：

```bash
cd /home/zenia/Projects/BAC/BAC-original-fp32
conda activate bac

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B \
  -m BACInfer.scripts.eval_fast_diffusion_policy \
  --checkpoint checkpoints/epoch=7550-test_mean_score=1.000.ckpt \
  -o results/strict_fp16_acs20_bua3_run01 \
  --device cuda:0 --precision decoder_fp16_strict \
  --cache_mode optimal \
  --optimal_steps_dir assets/blockpush_strict/decoder_fp16_strict/original/optimal_steps/cosine \
  --metric cosine --num_caches 20 --num_bu_blocks 3 \
  --n_train 0 --n_test 3 --n_envs 3 \
  --test_start_seed 4300000 --seed 11 --max_steps 350 \
  --skip_video --rollout_only
```

这是 3 个任务的闭环评估，`--rollout_only` 跳过计时。去掉此参数会额外运行原入口的计时流程。
再次运行时更换输出目录名称。`eval_results.json` 中的 `p2` 为 score > 0.9 的任务比例，
`bu_blocks` 记录实际启用 BUA 的分支。设置 `--num_bu_blocks 0` 可关闭 BUA。

权重和激活文件不随 Git 上传。在新机器放好权重、安装
`requirements-lowdim-fp32.txt` 后，首次生成严格 FP16 校准：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B \
  -m BACInfer.analysis.get_optimal_cache_update_steps \
  --checkpoint checkpoints/epoch=7550-test_mean_score=1.000.ckpt \
  -o assets/blockpush_strict --device cuda:0 \
  --precision decoder_fp16_strict --metrics cosine --num_caches 20
```

BUA 需要同一校准目录中的 `original/activations.pkl`，不能只保留调度文件。
预算 20 是各分支的 ACS 更新预算，仍执行完整 100 个去噪步；末步刷新和 BUA 可增加实际更新数。

## 整理后的源码

| 文件 | 用途 |
|---|---|
| `diffusion_policy/.../model/diffusion/strict_fp16.py` | 严格 FP16 Decoder 算子、残差、归约及指令审计 |
| `diffusion_policy/.../model/diffusion/transformer_for_diffusion.py` | 设置 Decoder 入口 / 出口精度，默认仍为 FP32 |
| `BACInfer/core/precision.py` | 统一精度选择与校准来源检查 |
| `BACInfer/core/diffusion_cache_wrapper.py` | ACS/BUA 接入，缓存命中和重算均使用严格残差加法 |
| `BACInfer/analysis/` 中的校准入口 | 采集对应精度的特征，离线 FP32 统计，生成调度 |
| `BACInfer/scripts/eval_fast_diffusion_policy.py` | 官方终端入口的精度与任务控制参数 |
| `BACInfer/analysis/check_strict_fp16.py` | 默认检查 ACS20+BUA3，可用参数更改预算和 BUA 数量 |
| `BACInfer/core/reproducibility.py`、环境 runner | 固定评估随机性、记录实际任务协议 |
| `tests/` | 精度、缓存、BUA、Gym 与配对随机性回归检查 |

全部 Decoder 层及 `ln_f` 为严格 FP16，包括 LayerNorm 均值 / 方差、Attention 归约、
矩阵乘法累加、残差和分支缓存。embedding、条件 encoder、head、Scheduler、动作保持 FP32。
ACS/BUA 离线统计保持 FP32，在线更新时间和开关是整数 / 布尔量。
精度开关与缓存算法没有在本次文档整理中改变。

整理后验证：13 项回归测试通过；实际 checkpoint 的 ACS20+BUA3 固定输入检查在噪声
seeds 101/102/103 下均输出有限值，Scheduler 各执行 100 次，缓存为 FP16，
实际选中 3 个 BUA 分支并新增 23 个分支更新时间；执行指令审计通过。
这是接线及数值有限性检查，不是 ACS20+BUA3 的闭环完成率测试。

`diffusion_policy/` 已由子模块改为普通受控目录，保留上游全部 369 个受控文件及现有修改，
并加入严格内核；保留原许可证与上游版本说明
[VENDORED.md](../diffusion_policy/VENDORED.md)。因此一个主仓库分支即可保存完整代码，
无需另找子模块远程仓库，也不再需要运行 `git submodule update`。

## 创建分支并提交到自己的远程仓库

代码已整理好，尚未自动创建分支、提交或推送。下面命令创建独立功能分支，
推送到本机 `origin`（`git@github.com:Zenia-123/BAC.git`）：

```bash
cd /home/zenia/Projects/BAC/BAC-original-fp32
git switch -c feature/strict-fp16-acs20
git add -A
git diff --cached --check
git diff --cached --stat
git commit -m "feat: add strict FP16 decoder with ACS20 and BUA support"
git push -u origin feature/strict-fp16-acs20
```

子模块转换的源码已经加入索引，`git add -A` 会一并补上其余代码和文档。
checkpoint、激活、结果目录和生成的内核审计产物仍被忽略，不会因以上命令上传。
后续在该功能分支继续修改时，使用 `git add -A`、`git commit`、`git push` 即可，
不必重复创建分支。完整实现与语法说明见 [STRICT_FP16_DECODER.md](STRICT_FP16_DECODER.md)。
