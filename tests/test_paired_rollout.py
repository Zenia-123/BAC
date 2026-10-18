"""Check paired rollout RNG independence from prior and per-policy RNG use."""
import unittest

import BACInfer
import numpy as np
import torch
from diffusion_policy.env_runner.blockpush_lowdim_runner import BlockPushLowdimRunner


class RecordingVectorEnv:
    def __init__(self):
        self.actions = []

    def call_each(self, *args, **kwargs):
        return None

    def reset(self):
        self.actions = []
        return np.zeros((1,2,16), dtype=np.float32)

    def step(self, action):
        self.actions.append(action.copy())
        return (np.zeros((1,2,16),dtype=np.float32), np.array([0.]),
                np.array([len(self.actions) == 3]), [{'REACH_0': np.array([1])}])

    def render(self):
        return [None]

    def call(self, *args):
        return [[0.,0.,0.]]


class RandomPolicy:
    device = torch.device('cpu')
    dtype = torch.float32

    def __init__(self, extra_draws):
        self.extra_draws = extra_draws

    def reset(self):
        pass

    def predict_action(self, obs):
        action = torch.randn(1,1,2)
        torch.randn(self.extra_draws)
        return {'action':action}


def run(extra_draws, chunk_count=1):
    # Exercise the actual runner loop without spawning PyBullet workers.
    runner = BlockPushLowdimRunner.__new__(BlockPushLowdimRunner)
    runner.env = RecordingVectorEnv()
    runner.env_fns = [None]
    runner.env_init_fn_dills = [b'init'] * chunk_count
    runner.env_seeds = list(range(4300000,4300000+chunk_count))
    runner.env_prefixs = ['test/'] * chunk_count
    runner.max_steps = 3
    runner.obs_eef_target = True
    runner.past_action = False
    runner.tqdm_interval_sec = 100
    runner.policy_seed = 11
    log = runner.run(RandomPolicy(extra_draws))
    return runner, log


class PairedRolloutTests(unittest.TestCase):
    def test_prior_and_internal_rng_consumption_do_not_shift_next_prediction(self):
        torch.manual_seed(900)
        first, log = run(7)
        torch.randn(300)
        second, _ = run(101)
        for a,b in zip(first.env.actions,second.env.actions):
            np.testing.assert_array_equal(a,b)
        self.assertEqual(first.rollout_trace[0]['prediction_seeds'],[11,12,13])
        self.assertEqual(first.rollout_trace[0]['initial_observation_sha256'],
                         second.rollout_trace[0]['initial_observation_sha256'])
        self.assertEqual(log['test/p2'],0.)

    def test_chunk_seed_ranges_do_not_repeat(self):
        runner, _ = run(1,chunk_count=2)
        self.assertEqual([trace['prediction_seeds'] for trace in runner.rollout_trace],
                         [[11,12,13],[15,16,17]])


if __name__ == '__main__':
    unittest.main()
