import unittest

import torch
from omegaconf import OmegaConf

from rssm_mamba2 import Mamba2Deter


def make_backbone_config():
    return OmegaConf.create(
        {
            "layers": 2,
            "d_state": 4,
            "d_conv": 3,
            "headdim": 8,
            "expand": 2,
            "chunk_size": 4,
        }
    )


class Mamba2ParallelEquivalenceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.d_model = 16
        self.stoch = 8  # flat stoch dim (S * K)
        self.act_dim = 3
        self.B = 2
        self.T = 7
        self.cfg = make_backbone_config()
        self.net = Mamba2Deter(self.d_model, self.stoch, self.act_dim, self.cfg)
        self.net.eval()
        self.state_size = self.net._block_state_size

    def _run_sequential(self, stochs, actions, reset, init_memory):
        B, T = stochs.shape[:2]
        deter = torch.zeros(B, self.d_model)
        memory = init_memory
        deters, memories = [], []
        for t in range(T):
            r_t = reset[:, t]
            memory = self.net.reset_memory(memory, r_t)
            stoch_t = torch.where(
                r_t.reshape(B, *((1,) * (stochs.dim() - 2))).expand_as(stochs[:, t]),
                torch.zeros_like(stochs[:, t]),
                stochs[:, t],
            )
            action_t = torch.where(
                r_t.reshape(B, *((1,) * (actions.dim() - 2))).expand_as(actions[:, t]),
                torch.zeros_like(actions[:, t]),
                actions[:, t],
            )
            deter, memory = self.net(stoch_t, deter, memory, action_t)
            deters.append(deter)
            memories.append(memory)
        return torch.stack(deters, dim=1), torch.stack(memories, dim=1)

    def test_no_reset_matches(self):
        stochs = torch.randn(self.B, self.T, self.stoch)
        actions = torch.randn(self.B, self.T, self.act_dim)
        reset = torch.zeros(self.B, self.T, dtype=torch.bool)
        init_memory = torch.zeros(self.B, self.cfg.layers, self.state_size)

        with torch.no_grad():
            seq_deter, seq_mem = self._run_sequential(stochs, actions, reset, init_memory)
            par_deter, par_mem = self.net.forward_parallel(stochs, actions, init_memory, reset)

        diff_deter = (seq_deter - par_deter).abs().max().item()
        diff_mem = (seq_mem - par_mem).abs().max().item()
        self.assertLess(diff_deter, 1e-4, f"deter mismatch {diff_deter}")
        self.assertLess(diff_mem, 1e-4, f"memory mismatch {diff_mem}")

    def test_with_reset_matches(self):
        stochs = torch.randn(self.B, self.T, self.stoch)
        actions = torch.randn(self.B, self.T, self.act_dim)
        reset = torch.zeros(self.B, self.T, dtype=torch.bool)
        # Mid-segment reset for one batch element; reset at position 0 for the other.
        reset[0, 3] = True
        reset[1, 0] = True
        init_memory = torch.randn(self.B, self.cfg.layers, self.state_size)

        with torch.no_grad():
            seq_deter, seq_mem = self._run_sequential(stochs, actions, reset, init_memory)
            par_deter, par_mem = self.net.forward_parallel(stochs, actions, init_memory, reset)

        diff_deter = (seq_deter - par_deter).abs().max().item()
        diff_mem = (seq_mem - par_mem).abs().max().item()
        self.assertLess(diff_deter, 1e-4, f"deter mismatch {diff_deter}")
        self.assertLess(diff_mem, 1e-4, f"memory mismatch {diff_mem}")

    def test_reset_spanning_chunk_boundary(self):
        # T=7 with chunk_size=4 splits into chunks of 4 and 3 (padded). Place
        # resets on either side of the boundary to exercise cross-chunk carry.
        stochs = torch.randn(self.B, self.T, self.stoch)
        actions = torch.randn(self.B, self.T, self.act_dim)
        reset = torch.zeros(self.B, self.T, dtype=torch.bool)
        reset[0, 4] = True  # first position of chunk 1
        reset[1, 2] = True  # last-but-one of chunk 0
        init_memory = torch.randn(self.B, self.cfg.layers, self.state_size)

        with torch.no_grad():
            seq_deter, seq_mem = self._run_sequential(stochs, actions, reset, init_memory)
            par_deter, par_mem = self.net.forward_parallel(stochs, actions, init_memory, reset)

        diff_deter = (seq_deter - par_deter).abs().max().item()
        diff_mem = (seq_mem - par_mem).abs().max().item()
        self.assertLess(diff_deter, 1e-4, f"deter mismatch {diff_deter}")
        self.assertLess(diff_mem, 1e-4, f"memory mismatch {diff_mem}")

    def test_gradients_flow(self):
        stochs = torch.randn(self.B, self.T, self.stoch, requires_grad=True)
        actions = torch.randn(self.B, self.T, self.act_dim, requires_grad=True)
        reset = torch.zeros(self.B, self.T, dtype=torch.bool)
        init_memory = torch.randn(self.B, self.cfg.layers, self.state_size, requires_grad=True)

        deter_seq, _ = self.net.forward_parallel(stochs, actions, init_memory, reset)
        deter_seq.sum().backward()
        self.assertIsNotNone(stochs.grad)
        self.assertIsNotNone(actions.grad)
        self.assertIsNotNone(init_memory.grad)
        self.assertGreater(stochs.grad.abs().mean().item(), 0)
        self.assertGreater(init_memory.grad.abs().mean().item(), 0)


if __name__ == "__main__":
    unittest.main()
