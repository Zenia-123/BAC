from pathlib import Path
import sys

# Resolve the source shipped with this checkout before any editable installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'diffusion_policy'))

__version__ = "1.0.0"

from BACInfer.core.diffusion_cache_wrapper import FastDiffusionPolicy
from BACInfer.analysis import (
    collect_activations,
    compute_similarity_matrix,
    load_activations,
    get_optimal_cache_update_steps,
    analyze_block_errors,
)

__all__ = [
    'FastDiffusionPolicy',
    'collect_activations',
    'compute_similarity_matrix',
    'load_activations',
    'get_optimal_cache_update_steps',
    'analyze_block_errors',
]

