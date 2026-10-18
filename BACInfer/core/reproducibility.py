"""Optional reproducibility controls for paired BlockPush evaluations."""
import os
import random

import numpy as np
import torch

PREDICTION_SEED_RULE = 'seed + chunk_index * (max_steps + 1) + prediction_index'


def configure_reproducibility(seed):
    if seed is None:
        return
    # Must be set before the first cuBLAS operation; supported CUDA workspace size.
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
