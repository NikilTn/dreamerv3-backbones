import torch
from torch import nn

from rssm_base import CategoricalRSSM


class TransformerDeter(nn.Module):
    """Transformer-style deterministic state update over a rolling token memory."""

    def __init__(self, deter, stoch, act_dim, config, act="SiLU"):
        super().__init__()
        self.tokens = int(config.tokens)
        assert deter % self.tokens == 0, (deter, self.tokens)
        self.token_dim = deter // self.tokens
        self.heads = int(config.heads)
        assert self.token_dim % self.heads == 0, (self.token_dim, self.heads)

        act_cls = getattr(torch.nn, act)
        self.input_proj = nn.Sequential(
            nn.Linear(stoch + act_dim, self.token_dim, bias=True),
            nn.RMSNorm(self.token_dim, eps=1e-04, dtype=torch.float32),
            act_cls(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=self.token_dim,
            nhead=self.heads,
            dim_feedforward=int(config.ff_mult) * self.token_dim,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=int(config.layers))
        self.output_norm = nn.RMSNorm(self.token_dim, eps=1e-04, dtype=torch.float32)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.tokens, self.token_dim))

    def forward(self, stoch, deter, action):
        batch_size = action.shape[0]
        action = action / torch.clip(torch.abs(action), min=1.0).detach()
        new_token = self.input_proj(torch.cat([stoch.reshape(batch_size, -1), action], dim=-1)).unsqueeze(1)
        tokens = deter.reshape(batch_size, self.tokens, self.token_dim)
        tokens = torch.cat([tokens[:, 1:], new_token], dim=1)
        tokens = self.encoder(tokens + self.pos_emb)
        tokens = self.output_norm(tokens)
        return tokens.reshape(batch_size, -1)


class RSSM(CategoricalRSSM):
    def __init__(self, config, embed_size, act_dim, backbone_config):
        deter_net = TransformerDeter(
            int(config.deter),
            int(config.stoch) * int(config.discrete),
            act_dim,
            backbone_config,
            act=config.act,
        )
        super().__init__(config, embed_size, act_dim, deter_net)
