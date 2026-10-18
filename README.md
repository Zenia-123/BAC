# Block-wise Adaptive Caching for Diffusion Policy

<div align="center">

### 🚀 An Efficient Acceleration Technique for Transformer-based Diffusion Policies

[![arXiv](https://img.shields.io/badge/arXiv-2506.13456-b31b1b.svg)](https://arxiv.org/pdf/2506.13456)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://block-wise-adaptive-caching.github.io/)
[![GitHub](https://img.shields.io/badge/GitHub-Code-black.svg)](https://github.com/ky-ji/BAC.git)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-purple.svg)](https://iclr.cc/)

**ICLR 2026**

[Paper](https://arxiv.org/pdf/2506.13456) | [Project Page](https://block-wise-adaptive-caching.github.io/) | [Code](https://github.com/ky-ji/BAC.git)

</div>

---

## 📋 Overview

<div align="center">
  <img src="figure/method.png" width="90%" alt="BAC Method Overview">
</div>

**BAC (Block-wise Adaptive Caching)** is a novel acceleration technique designed specifically for Transformer-based Diffusion Policies in robotic manipulation tasks. Our method achieves significant speedup while maintaining high task success rates.

### ✨ Key Features

- **🔧 Adaptive Caching Scheduler (ACS)**: Dynamically determines optimal cache update steps for each transformer block based on temporal activation similarity analysis
- **🫧 Bubbling Union (BU) Algorithm**: Propagates cache schedules from downsteam rFFN blocks to upstream blocks to mitigate the error surge phenomenon
- **⚡ Significant Speedup**: Achieves up to **3x** lossless acceleration
- **🔌 Plug-and-Play**: Easy to integrate with existing Diffusion Policy implementations

---

## 🛠️ Installation

### Prerequisites

- Python 3.9
- CUDA 11.6 or higher
- Conda package manager

### Setup Environment

```bash
git clone https://github.com/ky-ji/BAC.git
cd BAC

conda env create -f conda_environment.yaml
conda activate robodiff

pip install -e .
```

---

## 🚀 Quick Start

### 严格 FP16 decoder + ACS20 + BUA

本分支保留官方评估入口，通过 `--precision fp32|decoder_fp16_strict` 选择精度，默认 FP32。
严格模式覆盖全部 decoder 层及 `ln_f`，包括 LayerNorm、Attention 归约、矩阵乘法累加、残差与缓存；
embedding、条件 encoder、head、Scheduler 和动作接口仍为 FP32。ACS/BUA 离线统计保持 FP32。
需要 CUDA、Triton 3.6.0，限推理使用。完整实现说明见 [STRICT_FP16_DECODER.md](docs/STRICT_FP16_DECODER.md)。

当前推荐从 **ACS20** 开始。已测对照中，FP32 与严格 FP16 的 Original 均通过；
ACS5 在两种精度下都失败，严格 FP16 ACS10/15 各通过 2/3，ACS20 通过 3/3。
随后新增的 6 个环境 × 2 组噪声中，严格 FP16 Original 和 ACS20 均为 12/12。
这些 ACS20 成绩均关闭 BUA；下方按需开启 BUA3，其闭环表现需单独查看结果。

```bash
cd /home/zenia/Projects/BAC/BAC-original-fp32
conda activate bac
python -m pip install -r requirements-lowdim-fp32.txt

# 仅在首次使用或缺少调度时运行；本机已有严格 FP16 的 ACS20 调度。
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B \
  -m BACInfer.analysis.get_optimal_cache_update_steps \
  --checkpoint checkpoints/epoch=7550-test_mean_score=1.000.ckpt \
  -o assets/blockpush_strict --device cuda:0 \
  --precision decoder_fp16_strict --metrics cosine --num_caches 20

# 官方入口：严格 FP16 + ACS20 + BUA3，评估三个任务。
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

`--num_caches 20` 是每分支的 ACS 预算，仍做完整 100 个去噪步；末步刷新及 BUA 可能增加实际更新次数。
`--num_bu_blocks 3` 开启 BUA 并选择 3 个分支补点；设为 `0` 可关闭。
`--rollout_only` 仅评估任务；去掉它才会额外运行预热、profile 和延迟计时。
输出目录重复运行时请改名；`eval_results.json` 中查看 `scores`、`p2`、`bu_blocks`。

请保留完整校准目录，BUA 需要其中的 `original/activations.pkl`；严格模式不能混用 FP32/AMP 调度。
`OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` 只限制 CPU 数值库线程，不决定精度或 GPU 线程。
`diffusion_policy/` 现为主仓库直接管理的源码，普通克隆即可获取精度修改，无需单独提交子模块。

[改动摘要及 Git 分支提交命令](docs/STRICT_FP16_RELEASE.md) ·
[历史同条件对照协议](docs/STEP1_PAIRED_COMPARISON.md)

### 本机 BlockPush FP32 运行方法

此 worktree 默认使用原版 FP32 推理，也支持上方新增的严格 FP16 decoder 精度选项；
已补充当前 `bac` 环境所需的入口和 Gym 兼容修复。
以下命令在 `/home/zenia/Projects/BAC/BAC-original-fp32` 中运行。
README 其他示例里的 `<checkpoint_path>`、`<output_dir>`、`<optimal_steps_dir>` 是占位符，
不能原样粘贴到终端；尖括号会被 Bash 当作重定向符号。

```bash
cd /home/zenia/Projects/BAC/BAC-original-fp32
conda activate bac
python -m pip install -r requirements-lowdim-fp32.txt
```

清单针对 Python 3.10、CUDA 12.8 版 PyTorch 和低维 BlockPush 的校准/评估。
本机所列依赖已全部满足，不需要安装原始 Conda 文件中其他机器人任务的依赖。
入口会优先导入当前 worktree 自带的 `diffusion_policy`，无需重装另一个目录的 editable package。

`optimal` 模式必须先采集原版 FP32 特征并生成调度。已有调度时会复用；重新生成可加 `--force_recompute`。
校准与评估均使用 checkpoint 的 `workspace.model` 权重。

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B \
  -m BACInfer.analysis.get_optimal_cache_update_steps \
  --checkpoint /home/zenia/Projects/BAC/BAC-original-fp32/checkpoints/epoch=7550-test_mean_score=1.000.ckpt \
  -o assets/blockpush_fp32 \
  --device cuda:0 --metrics cosine --num_caches 5
```

生成的调度位于 `assets/blockpush_fp32/original/optimal_steps/cosine/`。
同一校准的 `original/activations.pkl` 用于选择 BUA 分支，不能只复制调度文件而漏掉它。
下面是可直接执行的小规模评估命令；输出文件会写入 `-o` 指定的目录：

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 python -B \
  -m BACInfer.scripts.eval_fast_diffusion_policy \
  --checkpoint /home/zenia/Projects/BAC/BAC-original-fp32/checkpoints/epoch=7550-test_mean_score=1.000.ckpt \
  -o results/fp32_acs5_bu3 \
  --device cuda:0 \
  --cache_mode optimal \
  --optimal_steps_dir /home/zenia/Projects/BAC/BAC-original-fp32/assets/blockpush_fp32/original/optimal_steps/cosine \
  --metric cosine --num_caches 5 --num_bu_blocks 3 --skip_video \
  --n_train 0 --n_test 3 --n_envs 3
```

去掉最后一行的三个环境参数，会使用 checkpoint 中的评估数量：6 个训练环境初始设置和 50 个测试初始设置。
只去掉 `--n_train 0 --n_test 3`、保留 `--n_envs 3`，可分批运行这 56 个任务。
`--skip_video` 只关闭视频，不跳过任务评估；每个 episode 最多 350 个环境步。
ACS5 是强近似缓存配置，命令执行完成不等于每个 episode 都成功。

结果文件包括 `benchmark_results.json`、`eval_results.json`、`fast_eval_log.json`。
评估结果中的 `bu_blocks` 记录实际选中的分支；若调度或 BUA 特征缺失，入口会报错，避免静默跳过。
THOP 的 FLOPs 是单次 forward 的部分算子估算，不能当作完整 100 步采样的计算量。

本次已用上述三环境命令完成端到端评估：seeds 为 `4300000 / 4300001 / 4300002`，
ACS5 + BUA 的 scores 为 `[0, 0, 0]`，`p2=0`；程序正常结束并保存了结果，但三个任务均未完成。
BUA 实际选择 layer 7 FFN、layer 6 FFN、layer 7 SA，新增 8 次更新。
这是较早的 ACS5 结果。后续同条件对照已完成，当前推荐使用上方的 ACS20 配置。

```bash
python -B -m unittest discover -s tests -p 'test_fp32_entrypoints.py' -v
```

FP32 兼容修复、Transformer 精度开关和新增 `strict_fp16.py` 均随主仓库保存；
policy 的采样算法仍沿用原实现。`diffusion_policy/` 已改为普通受控目录，无需单独提交或推送子模块。

### 1. Prepare Data and Checkpoints

This implementation works with checkpoints from the [Diffusion Policy](https://diffusion-policy.cs.columbia.edu/) project. Please refer to the original project for dataset downloads and model training.

### 2. Compute Optimal Cache Schedules

Before running accelerated inference, compute the optimal caching schedules using our **Adaptive Caching Scheduler (ACS)**:

```bash
# For a single task
python -m BACInfer.analysis.get_optimal_cache_update_steps \
  -c <checkpoint_path> \
  -o <output_dir> \
  -d cuda:0 \
  --num_caches 5,8,10,20 \
  --force_recompute

# Batch process all tasks
bash BACInfer/shell/get_steps/get_optimal_steps.sh --device cuda:0 --num_caches 5
```

### 3. Run Evaluation

Evaluate the policy with BAC acceleration:

```bash
# With BAC (optimal caching + BU algorithm)
python -m BACInfer.scripts.eval_fast_diffusion_policy \
  --checkpoint <checkpoint_path> \
  -o <output_dir> \
  --device cuda:0 \
  --cache_mode optimal \
  --optimal_steps_dir <optimal_steps_dir> \
  --metric cosine \
  --num_caches 5 \
  --num_bu_blocks 3 \
  --skip_video

# No caching
python -m BACInfer.scripts.eval_fast_diffusion_policy \
  --checkpoint <checkpoint_path> \
  -o <output_dir> \
  --device cuda:0 \
  --cache_mode original \
  --skip_video

# Uniform caching 
python -m BACInfer.scripts.eval_fast_diffusion_policy \
  --checkpoint <checkpoint_path> \
  -o <output_dir> \
  --device cuda:0 \
  --cache_mode threshold \
  --cache_threshold 5 \
  --skip_video
```

### 4. Run Full Benchmarks

Run comprehensive benchmarks across all tasks:

```bash
# Run BAC benchmarks
bash BACInfer/shell/main/run_bac_all.sh --device cuda:0 --steps 5 --metric cosine
```

---


## 🤖 Real Robot Deployment

BAC can be deployed for real-time robot inference. The `realworld/` directory contains inference servers optimized for real robot control.

### Quick Start

```bash
cd realworld

# Run original inference server
python inference_server.py

# Run BAC accelerated inference server
python inference_server_bac.py
```

### Configuration

Edit `realworld/server_config.py` to configure BAC settings:

```python
ENABLE_BAC = True
CACHE_MODE = 'optimal'  # or 'threshold'
NUM_CACHES = 5
NUM_BU_BLOCKS = 3
OPTIMAL_STEPS_DIR = None  # auto-find or specify path
```

### Collect Optimal Steps for Real Robot

Before deploying BAC on the real robot, collect optimal cache schedules:

```bash
cd realworld

# Collect activations at iteration 10
python inference_server_analyze.py --iteration 10 --num_bu_blocks 3

```

### Real Robot Inference Files

- `inference_server.py` - Original inference server
- `inference_server_bac.py` - BAC accelerated inference server
- `inference_server_analyze.py` - Activation collection server
- `bac_policy_wrapper.py` - BAC policy wrapper
- `bac_realworld_analyzer.py` - Optimal steps computation
- `server_config.py` - Server configuration


## 📊 Performance

<div align="center">
  <img src="figure/performance.png" width="85%" alt="Performance Results">
</div>

BAC achieves significant speedup while maintaining competitive task success rates across various robotic manipulation benchmarks.

---

## 📁 Project Structure

```
BAC/
├── BACInfer/                      # BAC acceleration module
│   ├── core/                      # Core caching wrapper (FastDiffusionPolicy)
│   ├── analysis/                  # Activation analysis and optimal schedule computation
│   ├── scripts/                   # Evaluation scripts
│   └── shell/                     # Shell scripts for batch experiments
├── diffusion_policy/              # Base diffusion policy implementation
│   ├── model/                     # Transformer and diffusion model architectures
│   ├── env/                       # Simulation environments
│   ├── env_runner/                # Environment runners
│   └── workspace/                 # Training workspace
├── checkpoint/                    # Model checkpoints (not included)
├── data/                          # Training datasets (not included)
├── assets/                        # Cached activations and optimal schedules
│   └── {task}/
│       └── optimal_steps/         # Per-module optimal cache schedules
└── results/                       # Evaluation results
```

---

## 📝 Citation

If you find BAC useful for your research, please consider citing our paper:

```bibtex
@article{ji2025block,
  title={Block-wise Adaptive Caching for Accelerating Diffusion Policy},
  author={Ji, Kangye and Meng, Yuan and Cui, Hanyun and Li, Ye and Hua, Shengjia and Chen, Lei and Wang, Zhi},
  journal={arXiv preprint arXiv:2506.13456},
  year={2025}
}
```

---

## ❤️ Acknowledgments

This work builds upon:
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) - Original diffusion policy implementation
- [robomimic](https://github.com/ARISE-Initiative/robomimic) - Robot learning framework and benchmarks

We sincerely thank the authors of these projects for their open-source contributions!

---

## 🌟 Star History

<div align="center">

[![Star History Chart](https://api.star-history.com/svg?repos=ky-ji/BAC&type=Date)](https://star-history.com/#ky-ji/BAC&Date)

</div>

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**If you have any questions, please feel free to open an issue!**

</div>
