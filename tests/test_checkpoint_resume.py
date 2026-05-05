import tempfile
import unittest
from pathlib import Path

import torch
from omegaconf import OmegaConf

from trainer import OnlineTrainer
from train import load_resume_checkpoint


class TinyAgent(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(2, 2)
        self.opt = torch.optim.Adam(self.parameters(), lr=1e-3)


def make_trainer_config(**overrides):
    cfg = {
        "steps": 1000,
        "pretrain": 0,
        "eval_every": 1000,
        "eval_episode_num": 0,
        "video_pred_log": False,
        "params_hist_log": False,
        "batch_length": 4,
        "batch_size": 2,
        "train_ratio": 1,
        "update_log_every": 1000,
        "checkpoint_every": 100,
        "checkpoint_keep_all": True,
        "action_repeat": 1,
    }
    cfg.update(overrides)
    return OmegaConf.create(cfg)


class CheckpointResumeTest(unittest.TestCase):
    def test_checkpoint_contains_resume_state_and_loads_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            logdir = Path(tmp)
            agent = TinyAgent()
            trainer = OnlineTrainer(make_trainer_config(), None, None, logdir, None, None)

            loss = agent.net(torch.ones(1, 2)).sum()
            loss.backward()
            agent.opt.step()
            trainer.save_checkpoint(agent, 100)

            latest = logdir / "latest.pt"
            numbered = logdir / "checkpoint_000000100.pt"
            self.assertTrue(latest.exists())
            self.assertTrue(numbered.exists())

            checkpoint = torch.load(latest, map_location="cpu", weights_only=False)
            self.assertEqual(checkpoint["trainer_step"], 100)
            self.assertFalse(checkpoint["final"])
            self.assertFalse(checkpoint["replay_buffer_included"])
            self.assertIn("opt", checkpoint["optims_state_dict"])

            restored = TinyAgent()
            step = load_resume_checkpoint(restored, latest, "cpu")
            self.assertEqual(step, 100)
            self.assertTrue(restored.opt.state_dict()["state"])

    def test_resume_aligns_next_periodic_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            trainer = OnlineTrainer(make_trainer_config(), None, None, Path(tmp), None, None)
            trainer._align_checkpoint_schedule(250)
            self.assertEqual(trainer._next_checkpoint_step, 300)

            trainer._align_checkpoint_schedule(300)
            self.assertEqual(trainer._next_checkpoint_step, 400)


if __name__ == "__main__":
    unittest.main()
