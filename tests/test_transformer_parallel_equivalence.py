import unittest

import torch
from omegaconf import OmegaConf

from rssm_transformer import TransformerDeter


def make_backbone_config():
    return OmegaConf.create({"tokens": 4, "layers": 2, "heads": 2, "ff_mult": 2, "detach_memory": False})


class TransformerParallelEquivalenceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.token_dim = 16
        self.stoch = 8  # flat stoch dim (S * K)
        self.act_dim = 3
        self.B = 2
        self.T = 7
        self.cfg = make_backbone_config()
        self.net = TransformerDeter(self.token_dim, self.stoch, self.act_dim, self.cfg)
        self.net.eval()

    def _run_sequential(self, stochs, actions, reset, init_memory):
        B, T = stochs.shape[:2]
        deter = torch.zeros(B, self.token_dim)
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
        init_memory = torch.zeros(self.B, self.cfg.tokens, self.token_dim)

        with torch.no_grad():
            seq_deter, seq_mem = self._run_sequential(stochs, actions, reset, init_memory)
            par_deter, par_mem = self.net.forward_parallel(stochs, actions, init_memory, reset)

        self.assertTrue(torch.allclose(seq_deter, par_deter, atol=1e-5), (seq_deter - par_deter).abs().max())
        self.assertTrue(torch.allclose(seq_mem, par_mem, atol=1e-5), (seq_mem - par_mem).abs().max())

    def test_with_reset_matches(self):
        stochs = torch.randn(self.B, self.T, self.stoch)
        actions = torch.randn(self.B, self.T, self.act_dim)
        reset = torch.zeros(self.B, self.T, dtype=torch.bool)
        # Insert a reset mid-segment for one batch element.
        reset[0, 3] = True
        reset[1, 0] = True
        init_memory = torch.randn(self.B, self.cfg.tokens, self.token_dim)

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
        init_memory = torch.randn(self.B, self.cfg.tokens, self.token_dim, requires_grad=True)

        deter_seq, _ = self.net.forward_parallel(stochs, actions, init_memory, reset)
        deter_seq.sum().backward()
        self.assertIsNotNone(stochs.grad)
        self.assertIsNotNone(actions.grad)
        self.assertIsNotNone(init_memory.grad)
        self.assertGreater(stochs.grad.abs().mean().item(), 0)
        self.assertGreater(init_memory.grad.abs().mean().item(), 0)


if __name__ == "__main__":
    unittest.main()
