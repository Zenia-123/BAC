"""Regressions for Gym compatibility and complete ACS/BUA setup."""
from pathlib import Path
import pickle
import tempfile
import unittest

import BACInfer  # Resolve this worktree before an editable diffusion_policy.
import gym
import numpy as np
import torch
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from BACInfer.core.diffusion_cache_wrapper import FastDiffusionPolicy
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.model.diffusion.transformer_for_diffusion import TransformerForDiffusion
from diffusion_policy.policy.diffusion_transformer_lowdim_policy import DiffusionTransformerLowdimPolicy


class ToyEnv(gym.Env):
    observation_space = gym.spaces.Box(-10, 10, shape=(2,), dtype=np.float32)
    action_space = gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32)
    metadata = {}

    def seed(self, seed=None):
        return [seed]

    def reset(self):
        return np.zeros(2, dtype=np.float32)

    def step(self, action):
        return np.ones(2, dtype=np.float32), 1.0, False, {}


def policy():
    model = TransformerForDiffusion(input_dim=2, output_dim=2, horizon=3,
        n_obs_steps=2, cond_dim=16, n_emb=8, n_head=2, n_layer=2,
        p_drop_emb=0, p_drop_attn=0)
    return DiffusionTransformerLowdimPolicy(model=model,
        noise_scheduler=DDPMScheduler(num_train_timesteps=4), horizon=3,
        obs_dim=16, action_dim=2, n_action_steps=1, n_obs_steps=2,
        num_inference_steps=4, obs_as_cond=True).eval()


def schedules(root):
    features = {}
    for layer in range(2):
        for dropout in range(1, 4):
            name = f'decoder.layers.{layer}.dropout{dropout}'
            folder = root / 'optimal_steps/cosine' / name
            folder.mkdir(parents=True)
            steps = [0, 2 + layer] if dropout == 3 else [0, 1]
            (folder / f'optimal_steps_{name}_2_cosine.pkl').write_bytes(pickle.dumps(steps))
            scale = 10 - layer if dropout == 1 else 1
            features[name] = [torch.zeros(1), torch.ones(1) * scale]
    (root / 'activations.pkl').write_bytes(pickle.dumps(features))
    return root / 'optimal_steps/cosine'


class EntrypointTests(unittest.TestCase):
    def test_vector_reset_and_step_in_all_transport_modes(self):
        for mode in ('sync', 'shared', 'pipe'):
            with self.subTest(mode=mode):
                env = SyncVectorEnv([ToyEnv, ToyEnv]) if mode == 'sync' else AsyncVectorEnv(
                    [ToyEnv, ToyEnv], shared_memory=(mode == 'shared'))
                try:
                    np.testing.assert_array_equal(env.reset(), np.zeros((2, 2)))
                    obs, reward, done, _ = env.step(np.zeros((2, 1), dtype=np.float32))
                    np.testing.assert_array_equal(obs, np.ones((2, 2)))
                    np.testing.assert_array_equal(reward, [1., 1.])
                    np.testing.assert_array_equal(done, [False, False])
                finally:
                    env.close()

    def test_partial_acs_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = schedules(Path(directory))
            next(path.rglob('*.pkl')).unlink()
            with self.assertRaisesRegex(ValueError, 'Missing ACS schedules'):
                FastDiffusionPolicy.apply_cache(policy(), cache_mode='optimal',
                    optimal_steps_dir=path, num_caches=2)

    def test_bua_uses_matching_calibration_and_adds_ffn_union(self):
        with tempfile.TemporaryDirectory() as directory:
            path = schedules(Path(directory))
            cached = FastDiffusionPolicy.apply_cache(policy(), cache_mode='optimal',
                optimal_steps_dir=path, num_caches=2, num_bu_blocks=2)
            self.assertEqual(cached._cache['bu_blocks'],
                             ['decoder.layers.0_sa_block', 'decoder.layers.1_sa_block'])
            self.assertEqual(cached._cache['block_steps']['decoder.layers.0_sa_block'], [0, 1, 2, 3])

    def test_missing_bua_features_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = schedules(root)
            (root / 'activations.pkl').unlink()
            with self.assertRaisesRegex(FileNotFoundError, 'BUA requires activations.pkl'):
                FastDiffusionPolicy.apply_cache(policy(), cache_mode='optimal',
                    optimal_steps_dir=path, num_caches=2, num_bu_blocks=2)


if __name__ == '__main__':
    unittest.main()
