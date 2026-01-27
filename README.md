# BAC: Block-wise Adaptive Caching for Diffusion Policy

Official implementation of **BAC (Block-wise Adaptive Caching)**, a novel acceleration technique for Transformer-based Diffusion Policies in robotic manipulation tasks.


### Key Features

- **Adaptive Caching Scheduler (ACS)**: Dynamically determines optimal cache update steps for each transformer block based on temporal activation similarity
- **Bubbling Union (BU) Algorithm**: Propagates cache schedules from FFN blocks to upstream blocks to mitigate error surge phenomenon.
## Installation

### Prerequisites

- Python 3.9
- CUDA 11.6 or higher
- Conda package manager

### Setup Environment

```bash
# Clone the repository with submodules
git clone --recursive <REPO_URL>
cd BAC

# Initialize Diffusion Policy
git submodule update --init --recursive

# Create and activate conda environment
conda env create -f conda_environment.yaml
conda activate robodiff

# Install the package
pip install -e .
```

## Quick Start

### 1. Prepare Data and Checkpoints

This implementation can be directly pluged in the Diffusion Policy codebase. Please refer to the base implementation for dataset downloads and model training.

### 2. Compute Optimal Cache Schedules

Before running accelerated inference, compute the optimal caching schedules:

```bash
# For a single task
python -m BACInfer.analysis.get_optimal_cache_update_steps \
  -c checkpoint/can_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt \
  -o assets/can_ph \
  -d cuda:0 \
  --num_caches 5,8,10,20 \
  --force_recompute

# Batch process all tasks
bash BACInfer/shell/get_steps/get_optimal_steps.sh --device cuda:0 --num_caches 5
```

### 3. Run Evaluation

Evaluate the policy with BAC acceleration:

```bash
# BAC 
python -m BACInfer.scripts.eval_fast_diffusion_policy \
  --checkpoint checkpoint/can_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt \
  -o results/can_ph/bac \
  --device cuda:0 \
  --cache_mode optimal \
  --optimal_steps_dir assets/can_ph/optimal_steps/cosine \
  --metric cosine \
  --num_caches 5 \
  --num_bu_blocks 3 \
  --skip_video

# Baseline 
python -m BACInfer.scripts.eval_fast_diffusion_policy \
  --checkpoint checkpoint/can_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt \
  -o results/can_ph/baseline \
  --device cuda:0 \
  --cache_mode original \
  --skip_video

# Uniform Caching
python -m BACInfer.scripts.eval_fast_diffusion_policy \
  --checkpoint checkpoint/can_ph/diffusion_policy_transformer/train_0/checkpoints/latest.ckpt \
  -o results/can_ph/threshold \
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

## Real Robot Deployment

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


## Project Structure

```
BAC/
├── BACInfer/           # BAC acceleration module
├── diffusion_policy/   # Base diffusion policy (submodule)
├── realworld/          # Real robot inference servers
├── checkpoint/         # Model checkpoints (not included)
├── data/              # Training datasets (not included)
├── assets/            # Cached activations and optimal schedules
└── results/           # Evaluation results
```


## License

This project is licensed under the MIT License.

