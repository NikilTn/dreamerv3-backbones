import unittest

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from buffer import Buffer


def make_buffer_config():
    return OmegaConf.create(
        {
            "batch_size": 2,
            "batch_length": 3,
            "max_size": 64,
            "device": "cpu",
            "storage_device": "cpu",
            "sampling": {
                "strategy": "dfs",
                "dfs": {
                    "alpha": 1.0,
                    "beta": 0.0,
                    "eps": 1e-8,
                    "world_temperature": 1.0,
                    "policy_temperature": 1.0,
                    "priority_clip": 50.0,
                },
            },
        }
    )


def make_transition(step):
    return TensorDict(
        {
            "image": torch.randint(0, 256, (2, 8, 8, 3), dtype=torch.uint8),
            "reward": torch.randn(2, 1),
            "is_first": torch.tensor([[step == 0], [step == 0]], dtype=torch.bool),
            "is_last": torch.zeros(2, 1, dtype=torch.bool),
            "is_terminal": torch.zeros(2, 1, dtype=torch.bool),
            "action": torch.randn(2, 3),
            "stoch": torch.randn(2, 4, 4),
            "deter": torch.randn(2, 8),
            "episode": torch.tensor([0, 1], dtype=torch.int64),
        },
        batch_size=(2,),
    )


class BufferDFSSmokeTest(unittest.TestCase):
    def test_dfs_sampling_and_policy_starts_work(self):
        buffer = Buffer(make_buffer_config())
        for step in range(8):
            buffer.add_transition(make_transition(step))

        data, index, initial, metrics = buffer.sample()
        self.assertEqual(tuple(data.batch_size), (2, 3))
        self.assertEqual(tuple(initial[0].shape), (2, 4, 4))
        self.assertIn("buffer/world_count_mean", metrics)
        self.assertEqual(tuple(index[0].shape), (2, 3))

        starts, start_metrics = buffer.sample_policy_starts(6)
        self.assertEqual(tuple(starts[0].shape), (6, 4, 4))
        self.assertEqual(tuple(starts[1].shape), (6, 8))
        self.assertIn("buffer/policy_start_count_mean", start_metrics)

    def test_latent_update_writes_sampled_storage_coordinates(self):
        buffer = Buffer(make_buffer_config())
        for step in range(8):
            buffer.add_transition(make_transition(step))

        data, index, initial, metrics = buffer.sample()
        new_stoch = torch.full((2, 3, 4, 4), 123.0)
        new_deter = torch.full((2, 3, 8), 456.0)
        buffer.update(index, new_stoch, new_deter)

        sampled = buffer._buffer.storage[index[0].reshape(-1), index[1].reshape(-1)]
        self.assertTrue(torch.all(sampled["stoch"] == 123.0))
        self.assertTrue(torch.all(sampled["deter"] == 456.0))


if __name__ == "__main__":
    unittest.main()
