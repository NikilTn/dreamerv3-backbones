import torch
import torch.nn.functional as F
from torch import nn

from rssm_base import CategoricalRSSM


class MambaMixerBlock(nn.Module):
    """CPU-friendly selective state-space inspired mixer."""

    def __init__(self, dim, expand, conv_kernel):
        super().__init__()
        hidden = int(expand) * dim
        self.norm = nn.RMSNorm(dim, eps=1e-04, dtype=torch.float32)
        self.in_proj = nn.Linear(dim, hidden * 2, bias=True)
        self.conv = nn.Conv1d(
            hidden,
            hidden,
            kernel_size=int(conv_kernel),
            groups=hidden,
            padding=int(conv_kernel) - 1,
            bias=True,
        )
        self.out_proj = nn.Linear(hidden, dim, bias=True)

    def forward(self, tokens):
        x = self.norm(tokens)
        value, gate = self.in_proj(x).chunk(2, dim=-1)
        value = self.conv(value.transpose(1, 2))[..., : tokens.shape[1]].transpose(1, 2)
        value = F.silu(value)
        gate = torch.sigmoid(gate)
        return tokens + self.out_proj(value * gate)


class MambaDeter(nn.Module):
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
            [MambaMixerBlock(self.token_dim, int(config.expand), int(config.conv_kernel)) for _ in range(int(config.layers))]
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
        deter_net = MambaDeter(
            int(config.deter),
            int(config.stoch) * int(config.discrete),
            act_dim,
            backbone_config,
            act=config.act,
        )
        super().__init__(config, embed_size, act_dim, deter_net)
