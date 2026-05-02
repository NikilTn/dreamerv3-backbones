import torch
from torch import nn

from rssm_base import CategoricalRSSM


class S5Block(nn.Module):
    """Grouped state-space block with explicit recurrent state."""

    def __init__(self, dim, state_size, heads):
        super().__init__()
        self.state_size = int(state_size)
        self.heads = int(heads)
        assert dim % self.heads == 0, (dim, self.heads)
        self.head_dim = dim // self.heads
        self.norm = nn.RMSNorm(dim, eps=1e-04, dtype=torch.float32)
        self.a = nn.Parameter(torch.eye(self.state_size).unsqueeze(0).repeat(self.heads, 1, 1))
        self.b = nn.Parameter(torch.randn(self.heads, self.head_dim, self.state_size) * 0.02)
        self.c = nn.Parameter(torch.randn(self.heads, self.state_size, self.head_dim) * 0.02)
        self.skip = nn.Parameter(torch.ones(self.heads, self.head_dim))
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, tokens):
        x = self.norm(tokens)
        batch_size, steps, dim = x.shape
        x = x.reshape(batch_size, steps, self.heads, self.head_dim)
        state = x.new_zeros(batch_size, self.heads, self.state_size)
        a = torch.tanh(self.a)
        outs = []
        for index in range(steps):
            state = torch.tanh(torch.einsum("bhs,hst->bht", state, a) + torch.einsum("bhd,hds->bhs", x[:, index], self.b))
            mixed = torch.einsum("bhs,hsd->bhd", state, self.c) + self.skip.unsqueeze(0) * x[:, index]
            outs.append(mixed.reshape(batch_size, dim))
        return tokens + self.out_proj(torch.stack(outs, dim=1))


class S5Deter(nn.Module):
    def __init__(self, deter, stoch, act_dim, config, act="SiLU"):
        super().__init__()
        self.tokens = int(config.tokens)
        assert deter % self.tokens == 0, (deter, self.tokens)
        self.token_dim = deter // self.tokens
        act_cls = getattr(torch.nn, act)
        self.input_proj = nn.Sequential(
            nn.Linear(stoch + act_dim, self.token_dim, bias=True),
            nn.RMSNorm(self.token_dim, eps=1e-04, dtype=torch.float32),
            act_cls(),
        )
        self.blocks = nn.ModuleList(
            [S5Block(self.token_dim, int(config.state_size), int(config.heads)) for _ in range(int(config.layers))]
        )
        self.output_norm = nn.RMSNorm(self.token_dim, eps=1e-04, dtype=torch.float32)

    def forward(self, stoch, deter, action):
        batch_size = action.shape[0]
        action = action / torch.clip(torch.abs(action), min=1.0).detach()
        new_token = self.input_proj(torch.cat([stoch.reshape(batch_size, -1), action], dim=-1)).unsqueeze(1)
        tokens = deter.reshape(batch_size, self.tokens, self.token_dim)
        tokens = torch.cat([tokens[:, 1:], new_token], dim=1)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.output_norm(tokens)
        return tokens.reshape(batch_size, -1)


class RSSM(CategoricalRSSM):
    def __init__(self, config, embed_size, act_dim, backbone_config):
        deter_net = S5Deter(
            int(config.deter),
            int(config.stoch) * int(config.discrete),
            act_dim,
            backbone_config,
            act=config.act,
        )
        super().__init__(config, embed_size, act_dim, deter_net)
