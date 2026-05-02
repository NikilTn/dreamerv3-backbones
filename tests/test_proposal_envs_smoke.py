import unittest

from envs.popgym import POPGym


class ProposalEnvSmokeTest(unittest.TestCase):
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
