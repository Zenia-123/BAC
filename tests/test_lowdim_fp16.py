"""Small algorithm/caching regressions; the GPU experiment is a separate entry."""
from itertools import combinations
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from BACInfer.analysis.optimal_cache_scheduler import OptimalCacheScheduler
from BACInfer.core.diffusion_cache_wrapper import FastDiffusionPolicy
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.policy.diffusion_transformer_lowdim_policy import DiffusionTransformerLowdimPolicy


def policy():
    model = TransformerForDiffusion(input_dim=2, output_dim=2, horizon=3,
        n_obs_steps=2, cond_dim=16, n_emb=8, n_head=2, n_layer=2,
        p_drop_emb=0, p_drop_attn=0)
    return DiffusionTransformerLowdimPolicy(model=model,
        noise_scheduler=DDPMScheduler(num_train_timesteps=4), horizon=3,
        obs_dim=16, action_dim=2, n_action_steps=1, n_obs_steps=2,
        num_inference_steps=4, obs_as_cond=True).eval()


class SchedulerTests(unittest.TestCase):
    def test_half_features_use_float_statistics(self):
        from BACInfer.analysis.activation_analysis import compute_temporal_metrics
        from BACInfer.analysis.bu_block_selection import compute_block_l1_errors
        a, b = torch.tensor([40000.], dtype=torch.float16), torch.tensor([-40000.], dtype=torch.float16)
        self.assertEqual(compute_temporal_metrics(a, b)['l1'], 80000.)
        values = compute_block_l1_errors({'decoder.layers.0.dropout1': [a, b]})
        self.assertEqual(values['decoder.layers.0_sa_block'], 80000.)

    def test_dynamic_program_matches_exhaustive_partitions(self):
        matrix = np.random.default_rng(11).normal(size=(5, 5))
        for metric in ('cosine', 'l2'):
            steps = OptimalCacheScheduler(matrix, metric=metric).compute_optimal_steps(3)
            def cost(starts):
                stops = list(starts[1:]) + [5]
                return sum(matrix[a, a:b].sum() for a, b in zip(starts, stops))
            scores = [cost((0, *rest)) for rest in combinations(range(1, 5), 2)]
            expected = max(scores) if metric == 'cosine' else min(scores)
            self.assertAlmostEqual(cost(steps), expected)

    def test_invalid_budget(self):
        for budget in (0, 5):
            with self.assertRaises(ValueError):
                OptimalCacheScheduler(np.eye(4)).compute_optimal_steps(budget)

    def test_terminal_execution_index_uses_step_count(self):
        cache = {'mode': 'optimal', 'num_steps': 4, 'block_steps': {'block': [0, 1]}}
        FastDiffusionPolicy._update_cache_flags(cache, 3)
        self.assertEqual(cache['steps_lookup']['block'], {0, 1, 3})
        self.assertTrue(cache['should_cache']['block'])

    def test_partial_schedules_rejected_and_bua_union(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for layer in range(2):
                for dropout in range(1, 4):
                    name = f'decoder.layers.{layer}.dropout{dropout}'
                    path = root / name / f'optimal_steps_{name}_2_cosine.pkl'
                    path.parent.mkdir()
                    steps = [0, 2 if layer == 0 else 3] if dropout == 3 else [0, 1]
                    path.write_bytes(pickle.dumps(steps))
            selected = ['decoder.layers.0_sa_block', 'decoder.layers.0_mha_block']
            cached = FastDiffusionPolicy.apply_cache(policy(), cache_mode='optimal',
                optimal_steps_dir=root, num_caches=2, num_bu_blocks=2, bu_blocks=selected)
            self.assertEqual(cached._cache['block_steps'][selected[0]], [0, 1, 2, 3])
            name = 'decoder.layers.1.dropout3'
            (root / name / f'optimal_steps_{name}_2_cosine.pkl').unlink()
            with self.assertRaisesRegex(ValueError, 'Missing ACS schedules'):
                FastDiffusionPolicy.apply_cache(policy(), cache_mode='optimal',
                    optimal_steps_dir=root, num_caches=2)

    def test_threshold_one_preserves_sampling(self):
        original = policy()
        from copy import deepcopy
        cached = FastDiffusionPolicy.apply_cache(deepcopy(original), cache_mode='threshold', cache_threshold=1)
        condition = torch.zeros((1, 3, 2))
        mask = torch.zeros_like(condition, dtype=torch.bool)
        cond = torch.zeros((1, 2, 16))
        with torch.inference_mode():
            torch.manual_seed(11)
            expected = original.conditional_sample(condition, mask, cond=cond)
            torch.manual_seed(11)
            actual = cached.conditional_sample(condition, mask, cond=cond)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
