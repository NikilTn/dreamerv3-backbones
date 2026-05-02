import torch
from torch import nn

from rssm_base import CategoricalRSSM


class S4Block(nn.Module):
    """Diagonal state-space style mixer for smoke-testable S4-like experiments."""

    def __init__(self, dim, state_size):
        super().__init__()
        self.state_size = int(state_size)
        self.norm = nn.RMSNorm(dim, eps=1e-04, dtype=torch.float32)
        self.in_proj = nn.Linear(dim, self.state_size, bias=True)
        self.out_proj = nn.Linear(self.state_size, dim, bias=True)
        self.skip = nn.Linear(dim, dim, bias=True)
        self.log_decay = nn.Parameter(torch.zeros(self.state_size))

    def forward(self, tokens):
        x = self.norm(tokens)
        batch_size, steps, _ = x.shape
        state = x.new_zeros(batch_size, self.state_size)
        decay = torch.sigmoid(self.log_decay).unsqueeze(0)
        outs = []
        for index in range(steps):
            state = decay * state + self.in_proj(x[:, index])
            outs.append(self.out_proj(state) + self.skip(x[:, index]))
        return tokens + torch.stack(outs, dim=1)


class S4Deter(nn.Module):
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
        self.blocks = nn.ModuleList([S4Block(self.token_dim, int(config.state_size)) for _ in range(int(config.layers))])
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
        deter_net = S4Deter(
            int(config.deter),
            int(config.stoch) * int(config.discrete),
            act_dim,
            backbone_config,
            act=config.act,
        )
        super().__init__(config, embed_size, act_dim, deter_net)
