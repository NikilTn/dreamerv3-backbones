import unittest

import numpy as np
from omegaconf import OmegaConf

from envs import make_env
from envs.popgym import POPGym


class ProposalEnvSmokeTest(unittest.TestCase):
    def test_popgym_hydra_task_runs_through_make_env(self):
        config = OmegaConf.create(
            {
                "task": "popgym_RepeatPreviousEasy-v0",
                "action_repeat": 1,
                "time_limit": 1024,
                "seed": 0,
            }
        )
        env = make_env(config, 0)
        obs = env.reset()
        self.assertIn("vector", obs)
        sample = np.zeros(env.action_space.shape, dtype=np.float32)
        sample[0] = 1.0
        obs, reward, done, info = env.step(sample)
        self.assertIn("vector", obs)
        self.assertIsInstance(float(reward), float)
        self.assertIsInstance(bool(done), bool)
        self.assertIsInstance(info, dict)
        env.close()

    def test_popgym_wrapper_runs(self):
        env = POPGym("popgym-RepeatPreviousEasy-v0", seed=0)
        obs = env.reset()
        self.assertIn("vector", obs)
        self.assertTrue(obs["is_first"])
        sample = env.action_space.sample()
        obs, reward, done, info = env.step(sample)
        self.assertIn("vector", obs)
        self.assertIsInstance(float(reward), float)
        self.assertIsInstance(bool(done), bool)
        self.assertIsInstance(info, dict)
        env.close()

    def test_bsuite_wrapper_imports_when_available(self):
        try:
            from envs.bsuite import BSuite
            env = BSuite("memory_len/0", seed=0)
        except ImportError:
            self.skipTest("bsuite is unavailable on this local interpreter")
        obs = env.reset()
        self.assertIn("vector", obs)
        sample = env.action_space.sample()
        obs, reward, done, info = env.step(sample)
        self.assertIn("vector", obs)
        self.assertIsInstance(float(reward), float)
        self.assertIsInstance(bool(done), bool)
        self.assertIsInstance(info, dict)
        env.close()


if __name__ == "__main__":
    unittest.main()
