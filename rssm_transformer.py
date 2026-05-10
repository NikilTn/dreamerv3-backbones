import torch
import torch.nn.functional as F
from torch import nn

from rssm_base import CategoricalRSSM


class RelativePositionalEncoding(nn.Module):
    """Sinusoidal relative positional encoding from Transformer-XL (Dai et al., 2019).

    Produces a (length, dim) tensor of sinusoidal embeddings for offsets
    [length-1, length-2, ..., 1, 0], to be projected into per-head key space
    inside the relative attention.
    """

    def __init__(self, dim):
        super().__init__()
        assert dim % 2 == 0, dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2.0) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.dim = dim

    def forward(self, length, device, dtype):
        pos_seq = torch.arange(length - 1, -1, -1.0, device=device, dtype=dtype)
        sinusoid = pos_seq.unsqueeze(-1) * self.inv_freq.to(device=device, dtype=dtype)
        return torch.cat([sinusoid.sin(), sinusoid.cos()], dim=-1)


def _rel_shift(x):
    """Transformer-XL relative-shift trick to align (q_pos, k_pos) -> (q_pos, q_pos - k_pos)."""
    B, H, Tq, Tk = x.shape
    zero_pad = x.new_zeros(B, H, Tq, 1)
    x_padded = torch.cat([zero_pad, x], dim=-1)
    x_padded = x_padded.reshape(B, H, Tk + 1, Tq)
    return x_padded[:, :, 1:].reshape(B, H, Tq, Tk)


class RelativeMultiheadAttention(nn.Module):
    """Multi-head attention with Transformer-XL relative positional encoding.

    Implements the four-term decomposition (AC + BD) of Dai et al. (2019)
    with learnable content bias u and positional bias v, replacing the
    absolute positional embedding that was added at the input layer.
    """

    def __init__(self, dim, heads):
        super().__init__()
        assert dim % heads == 0, (dim, heads)
        self.dim = dim
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.qkv_proj = nn.Linear(dim, dim * 3, bias=False)
        self.r_proj = nn.Linear(dim, dim, bias=False)
        self.u = nn.Parameter(torch.zeros(heads, self.head_dim))
        self.v = nn.Parameter(torch.zeros(heads, self.head_dim))
        self.out_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x, r):
        B, T, _ = x.shape
        H, D_h = self.heads, self.head_dim
        qkv = self.qkv_proj(x).reshape(B, T, 3, H, D_h)
        q, k, val = qkv.unbind(dim=2)
        r_h = self.r_proj(r).reshape(T, H, D_h)

        AC = torch.einsum("bthd,bshd->bhts", q + self.u, k)
        BD = torch.einsum("bthd,shd->bhts", q + self.v, r_h)
        BD = _rel_shift(BD)

        scores = (AC + BD) * self.scale
        causal_mask = torch.ones(T, T, device=scores.device, dtype=torch.bool).triu(1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn = F.softmax(scores, dim=-1)
        out = torch.einsum("bhts,bshd->bthd", attn, val).reshape(B, T, self.dim)
        return self.out_proj(out)


class TransformerXLLayer(nn.Module):
    def __init__(self, dim, heads, ff_mult, act="GELU"):
        super().__init__()
        act_cls = getattr(torch.nn, act)
        self.norm1 = nn.RMSNorm(dim, eps=1e-04, dtype=torch.float32)
        self.attn = RelativeMultiheadAttention(dim, heads)
        self.norm2 = nn.RMSNorm(dim, eps=1e-04, dtype=torch.float32)
        self.ff = nn.Sequential(
            nn.Linear(dim, ff_mult * dim),
            act_cls(),
            nn.Linear(ff_mult * dim, dim),
        )

    def forward(self, x, r):
        x = x + self.attn(self.norm1(x), r)
        x = x + self.ff(self.norm2(x))
        return x


class TransformerDeter(nn.Module):
    """Transformer-XL deterministic state update over a rolling token memory.

    The rolling token window plays the role of the segment memory: at every
    RSSM step the oldest token is dropped, the new (stoch, action) token is
    appended, and self-attention runs over the window with relative
    positional encoding rather than learned absolute embeddings.
    """

    def __init__(self, deter, stoch, act_dim, config, act="SiLU"):
        super().__init__()
        self.tokens = int(config.tokens)
        assert deter % self.tokens == 0, (deter, self.tokens)
        self.token_dim = deter // self.tokens
        self.heads = int(config.heads)
        assert self.token_dim % self.heads == 0, (self.token_dim, self.heads)
        self.detach_memory = bool(getattr(config, "detach_memory", False))

        act_cls = getattr(torch.nn, act)
        self.input_proj = nn.Sequential(
            nn.Linear(stoch + act_dim, self.token_dim, bias=True),
            nn.RMSNorm(self.token_dim, eps=1e-04, dtype=torch.float32),
            act_cls(),
        )
        self.layers = nn.ModuleList(
            [
                TransformerXLLayer(self.token_dim, self.heads, int(config.ff_mult))
                for _ in range(int(config.layers))
            ]
        )
        self.output_norm = nn.RMSNorm(self.token_dim, eps=1e-04, dtype=torch.float32)
        self.rel_pos = RelativePositionalEncoding(self.token_dim)

    def forward(self, stoch, deter, action):
        batch_size = action.shape[0]
        action = action / torch.clip(torch.abs(action), min=1.0).detach()
        new_token = self.input_proj(torch.cat([stoch.reshape(batch_size, -1), action], dim=-1)).unsqueeze(1)
        tokens = deter.reshape(batch_size, self.tokens, self.token_dim)
        memory = tokens[:, 1:]
        if self.detach_memory:
            memory = memory.detach()
        tokens = torch.cat([memory, new_token], dim=1)
        r = self.rel_pos(self.tokens, device=tokens.device, dtype=tokens.dtype)
        for layer in self.layers:
            tokens = layer(tokens, r)
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
