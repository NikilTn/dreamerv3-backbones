import unittest

import torch
from omegaconf import OmegaConf
from tensordict import TensorDict

from dreamer import Dreamer


class FakeSpace:
    def __init__(self, shape):
        self.shape = tuple(shape)


class FakeDictSpace:
    def __init__(self, spaces):
        self.spaces = spaces


class FakeActSpace:
    def __init__(self, shape):
        self.shape = tuple(shape)


class FakeReplayBuffer:
    def __init__(self, data, initial):
        self.data = data
        self.initial = initial
        self.updated = None
        self.strategy = "uniform"

    def sample(self):
        return self.data.clone(), [torch.zeros(1), torch.zeros(1)], self.initial, {}

    def sample_policy_starts(self, batch_size):
        return None, {}

    def update(self, index, stoch, deter):
        self.updated = (tuple(stoch.shape), tuple(deter.shape))


def make_test_config(backbone):
    return OmegaConf.create(
        {
            "device": "cpu",
            "act_entropy": 3e-4,
            "kl_free": 1.0,
            "imag_horizon": 3,
            "horizon": 15,
            "lamb": 0.95,
            "compile": False,
            "log_grads": False,
            "rep_loss": "r2dreamer",
            "backbone": backbone,
            "deter": 32,
            "hidden": 32,
            "discrete": 4,
            "depth": 8,
            "units": 32,
            "act": "SiLU",
            "norm": True,
            "lr": 1e-4,
            "agc": 0.3,
            "pmin": 1e-3,
            "eps": 1e-20,
            "beta1": 0.9,
            "beta2": 0.999,
            "warmup": 0,
            "slow_target_update": 1,
            "slow_target_fraction": 0.02,
            "loss_scales": {
                "barlow": 0.05,
                "cpc": 1.0,
                "dyn": 1.0,
                "rep": 0.1,
                "rew": 1.0,
                "con": 1.0,
                "policy": 1.0,
                "value": 1.0,
                "repval": 0.3,
            },
            "r2dreamer": {"lambd": 5e-4},
            "cpc": {
                "enabled": False,
                "horizon": 2,
                "proj_dim": 16,
                "hidden_dim": 32,
                "temperature": 0.1,
                "aug": {
                    "max_delta": 1.0,
                    "same_across_time": True,
                    "bilinear": False,
                },
            },
            "dreamer_pro": {
                "warm_up": 1,
                "num_prototypes": 8,
                "proto_dim": 8,
                "temperature": 0.1,
                "sinkhorn_eps": 0.05,
                "sinkhorn_iters": 3,
                "ema_update_every": 1,
                "ema_update_fraction": 0.05,
                "freeze_prototypes_iters": 10,
                "aug": {
                    "max_delta": 1.0,
                    "same_across_time": True,
                    "bilinear": False,
                },
            },
            "rssm": {
                "stoch": 4,
                "deter": 32,
                "hidden": 32,
                "discrete": 4,
                "img_layers": 2,
                "obs_layers": 1,
                "dyn_layers": 1,
                "blocks": 4,
                "act": "SiLU",
                "norm": True,
                "unimix_ratio": 0.01,
                "initial": "learned",
                "device": "cpu",
            },
            "transformer": {"tokens": 4, "layers": 1, "heads": 2, "ff_mult": 2},
            "mamba": {"tokens": 4, "layers": 1, "expand": 2, "conv_kernel": 3},
            "s4": {"tokens": 4, "layers": 1, "state_size": 8},
            "s5": {"tokens": 4, "layers": 1, "heads": 2, "state_size": 8},
            "encoder": {
                "mlp_keys": "^$",
                "cnn_keys": "image",
                "mlp": {
                    "shape": None,
                    "layers": 1,
                    "units": 32,
                    "act": "SiLU",
                    "norm": True,
                    "device": "cpu",
                    "outscale": None,
                    "symlog_inputs": True,
                    "name": "mlp_encoder",
                },
                "cnn": {
                    "act": "SiLU",
                    "norm": True,
                    "kernel_size": 3,
                    "minres": 4,
                    "depth": 8,
                    "mults": [1, 2],
                },
            },
            "reward": {
                "shape": [31],
                "layers": 1,
                "units": 32,
                "act": "SiLU",
                "norm": True,
                "dist": {"name": "symexp_twohot", "bin_num": 31},
                "outscale": 0.0,
                "device": "cpu",
                "symlog_inputs": False,
                "name": "reward",
            },
            "cont": {
                "shape": [1],
                "layers": 1,
                "units": 32,
                "act": "SiLU",
                "norm": True,
                "dist": {"name": "binary"},
                "outscale": 1.0,
                "device": "cpu",
                "symlog_inputs": False,
                "name": "cont",
            },
            "actor": {
                "shape": None,
                "layers": 2,
                "units": 32,
                "act": "SiLU",
                "norm": True,
                "device": "cpu",
                "dist": {
                    "cont": {"name": "bounded_normal", "min_std": 0.1, "max_std": 1.0},
                    "disc": {"name": "onehot", "unimix_ratio": 0.01},
                    "multi_disc": {"name": "multi_onehot", "unimix_ratio": 0.01},
                },
                "outscale": 0.01,
                "symlog_inputs": False,
                "name": "actor",
            },
            "critic": {
                "shape": [31],
                "layers": 2,
                "units": 32,
                "act": "SiLU",
                "norm": True,
                "device": "cpu",
                "dist": {"name": "symexp_twohot", "bin_num": 31},
                "outscale": 0.0,
                "symlog_inputs": False,
                "name": "value",
            },
        }
    )


def make_spaces():
    obs_space = FakeDictSpace(
        {
            "image": FakeSpace((32, 32, 3)),
            "reward": FakeSpace((1,)),
            "is_first": FakeSpace((1,)),
            "is_last": FakeSpace((1,)),
            "is_terminal": FakeSpace((1,)),
        }
    )
    act_space = FakeActSpace((3,))
    return obs_space, act_space


def make_step_obs(batch_size):
    return TensorDict(
        {
            "image": torch.randint(0, 256, (batch_size, 32, 32, 3), dtype=torch.uint8),
            "reward": torch.zeros(batch_size, 1, dtype=torch.float32),
            "is_first": torch.zeros(batch_size, 1, dtype=torch.bool),
            "is_last": torch.zeros(batch_size, 1, dtype=torch.bool),
            "is_terminal": torch.zeros(batch_size, 1, dtype=torch.bool),
        },
        batch_size=(batch_size,),
    )


def make_batch(batch_size, length, act_dim):
    data = TensorDict(
        {
            "image": torch.randint(0, 256, (batch_size, length, 32, 32, 3), dtype=torch.uint8),
            "action": torch.randn(batch_size, length, act_dim, dtype=torch.float32),
            "reward": torch.randn(batch_size, length, 1, dtype=torch.float32),
            "is_first": torch.zeros(batch_size, length, 1, dtype=torch.bool),
            "is_last": torch.zeros(batch_size, length, 1, dtype=torch.bool),
            "is_terminal": torch.zeros(batch_size, length, 1, dtype=torch.bool),
        },
        batch_size=(batch_size, length),
    )
    data["is_first"][:, 0] = True
    return data


class BackboneSmokeTest(unittest.TestCase):
    def test_backbones_execute_on_cpu(self):
        obs_space, act_space = make_spaces()
        for backbone in ("gru", "transformer", "mamba", "mamba2", "s4", "s3m", "s5"):
            with self.subTest(backbone=backbone):
                config = make_test_config(backbone)
                agent = Dreamer(config, obs_space, act_space).to("cpu")

                state = agent.get_initial_state(2)
                step_obs = make_step_obs(2)
                action, next_state = agent.act(step_obs, state, eval=False)
                self.assertEqual(tuple(action.shape), (2, 3))
                self.assertEqual(tuple(next_state["deter"].shape), (2, config.rssm.deter))

                initial = agent.rssm.initial(2)
                replay = FakeReplayBuffer(make_batch(2, 4, 3), initial)
                metrics = agent.update(replay)
                self.assertIn("opt/loss", metrics)
                self.assertEqual(replay.updated[0], (2, 4, config.rssm.stoch, config.rssm.discrete))
                self.assertEqual(replay.updated[1], (2, 4, config.rssm.deter))

    def test_cpc_option_executes_on_cpu(self):
        obs_space, act_space = make_spaces()
        config = make_test_config("transformer")
        config.cpc.enabled = True
        agent = Dreamer(config, obs_space, act_space).to("cpu")
        initial = agent.rssm.initial(2)
        replay = FakeReplayBuffer(make_batch(2, 4, 3), initial)
        metrics = agent.update(replay)
        self.assertIn("loss/cpc", metrics)

    def test_burn_in_changes_world_model_loss(self):
        """A non-zero ``burn_in`` mask should change the world-model loss
        (dyn/rep/rew/con) for a fixed seed and batch, without crashing."""
        obs_space, act_space = make_spaces()
        torch.manual_seed(0)
        cfg_no_burn = make_test_config("gru")
        cfg_no_burn.burn_in = 0
        agent = Dreamer(cfg_no_burn, obs_space, act_space).to("cpu")
        initial = agent.rssm.initial(2)
        batch = make_batch(2, 4, 3)
        replay = FakeReplayBuffer(batch.clone(), initial)
        metrics_no_burn = agent.update(replay)

        torch.manual_seed(0)
        cfg_burn = make_test_config("gru")
        cfg_burn.burn_in = 2
        agent2 = Dreamer(cfg_burn, obs_space, act_space).to("cpu")
        initial2 = agent2.rssm.initial(2)
        replay2 = FakeReplayBuffer(batch.clone(), initial2)
        metrics_burn = agent2.update(replay2)

        # Different burn_in values should produce different per-step losses
        # on the same data (otherwise the mask had no effect). We use cont
        # rather than dyn (which hits the kl_free clamp for a fresh model and
        # is constant in time) and rather than rew (whose head has
        # outscale=0 in the default config, so predictions are also flat).
        self.assertNotAlmostEqual(
            float(metrics_no_burn["loss/con"].detach()),
            float(metrics_burn["loss/con"].detach()),
            places=5,
            msg="burn_in mask did not change the cont loss; mask is not being applied",
        )


if __name__ == "__main__":
    unittest.main()
