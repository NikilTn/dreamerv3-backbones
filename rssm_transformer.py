import torch
import torch.nn.functional as F
from torch import nn

from rssm_base import CategoricalRSSM
from tools import rpad


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

    def forward(self, x, r, key_mask=None):
        """Args:
            x: (B, T, dim)
            r: (T, dim) sinusoidal relative positional encoding
            key_mask: (B, T) bool, optional — True positions are masked out (invalid keys).
        """
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
        if key_mask is not None:
            scores = scores.masked_fill(key_mask[:, None, None, :], float("-inf"))
        attn = F.softmax(scores, dim=-1)
        # Rows where every key was masked produce NaN from 0/0 in softmax. Clamp to 0
        # so values don't propagate; we only consume the last query position downstream
        # and the mask logic guarantees its row always has at least one valid key.
        attn = torch.nan_to_num(attn, nan=0.0)
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

    def forward(self, x, r, key_mask=None):
        x = x + self.attn(self.norm1(x), r, key_mask=key_mask)
        x = x + self.ff(self.norm2(x))
        return x


class TransformerDeter(nn.Module):
    """Transformer-XL deterministic state update with explicit memory cache.

    `deter` is the layer-L output of the current step (shape `(B, token_dim)`,
    with `token_dim = config.rssm.deter`). `memory` is the rolling window of
    the last M *new_token* projections (shape `(B, M, token_dim)`), serving as
    the attention cache. Each step appends the new input projection to the
    window, drops the oldest, and runs causal self-attention over the M-sized
    window with TXL relative positional encoding to produce the new layer-L
    output.

    This split — `deter` = current state, `memory` = input cache — lets
    sequential and parallel paths share the same attention contents, since
    both attend over rolling windows of new_tokens rather than feeding layer-L
    outputs back as inputs.
    """

    def __init__(self, deter, stoch, act_dim, config, act="SiLU"):
        super().__init__()
        self.token_dim = int(deter)
        self.tokens = int(config.tokens)
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

    def initial_memory(self, batch_size):
        device = next(self.parameters()).device
        return torch.zeros(batch_size, self.tokens, self.token_dim, dtype=torch.float32, device=device)

    def reset_memory(self, memory, reset):
        mask = rpad(reset, memory.dim() - int(reset.dim()))
        return torch.where(mask, torch.zeros_like(memory), memory)

    def _new_token(self, stoch, action):
        batch_size = action.shape[0]
        action = action / torch.clip(torch.abs(action), min=1.0).detach()
        return self.input_proj(torch.cat([stoch.reshape(batch_size, -1), action], dim=-1))

    def forward(self, stoch, deter, memory, action):
        new_token = self._new_token(stoch, action).unsqueeze(1)  # (B, 1, token_dim)
        prev_window = memory[:, 1:]
        if self.detach_memory:
            prev_window = prev_window.detach()
        new_memory = torch.cat([prev_window, new_token], dim=1)  # (B, M, token_dim)
        r = self.rel_pos(self.tokens, device=new_memory.device, dtype=new_memory.dtype)
        x = new_memory
        for layer in self.layers:
            x = layer(x, r)
        x = self.output_norm(x)
        new_deter = x[:, -1]  # (B, token_dim) — last position is the current state
        return new_deter, new_memory

    def forward_parallel(self, stochs, actions, init_memory, reset):
        """Process a whole segment of T timesteps in parallel.

        Equivalent to iterating `forward` T times, but batches the M-sized
        attention windows across the segment. Uses `unfold` to gather sliding
        windows from `cat(init_memory, new_tokens)`; each window is the input
        cache for one segment position. Causal attention with TXL relative
        positional encoding runs over each window independently.

        Args:
            stochs:      (B, T, S, K) — sampled posterior stochs.
            actions:     (B, T, A) — actions; new_tokens use prev-step pairing
                upstream, so caller must supply the already-shifted sequence.
            init_memory: (B, M, token_dim) — initial cache, zeros at episode start.
            reset:       (B, T) bool — episode reset flags.

        Returns:
            deter_seq:  (B, T, token_dim) — per-position layer-L outputs.
            memory_seq: (B, T, M, token_dim) — per-position rolling-window cache
                snapshot (the cache state right after consuming position t).
        """
        B, T = stochs.shape[:2]
        # Normalize reset to (B, T) bool. Callers pass (B, T) or (B, T, 1).
        reset = reset.reshape(B, T).to(torch.bool)
        actions = actions / torch.clip(torch.abs(actions), min=1.0).detach()
        # Mirror sequential reset semantics: zero stoch/action at reset positions
        # before computing new_tokens, so input_proj gets the same inputs as the
        # per-step path.
        reset_for_stoch = rpad(reset, stochs.dim() - reset.dim())
        reset_for_action = rpad(reset, actions.dim() - reset.dim())
        stochs = torch.where(reset_for_stoch, torch.zeros_like(stochs), stochs)
        actions = torch.where(reset_for_action, torch.zeros_like(actions), actions)
        flat_stoch = stochs.reshape(B, T, -1)
        new_tokens = self.input_proj(torch.cat([flat_stoch, actions], dim=-1))  # (B, T, token_dim)

        if self.detach_memory:
            init_memory = init_memory.detach()
        extended = torch.cat([init_memory, new_tokens], dim=1)  # (B, M+T, token_dim)

        # Sliding windows: window t covers extended positions [t+1, t+M].
        # unfold(dim=1, size=M, step=1) -> (B, M+T-M+1, token_dim, M) = (B, T+1, token_dim, M)
        windows = extended.unfold(1, self.tokens, 1)[:, 1 : T + 1]  # (B, T, token_dim, M)
        windows = windows.transpose(-2, -1).contiguous()  # (B, T, M, token_dim)

        # Per-window validity mask for reset boundaries.
        # For window at segment position t, position w in [0, M-1] maps to
        # extended index t+1+w. Collapse init_memory positions to seg_w = -1 so
        # the no-reset sentinel (last_reset = -1) keeps them all valid.
        device = extended.device
        positions_t = torch.arange(T, device=device)
        ws = torch.arange(self.tokens, device=device)
        extended_idx = positions_t[:, None] + 1 + ws[None, :]  # (T, M)
        seg_w = (extended_idx - self.tokens).clamp(min=-1)  # init -> -1, segment -> [0, t]
        reset_pos = torch.where(reset, positions_t[None, :], torch.full_like(positions_t[None, :], -1))
        last_reset = torch.cummax(reset_pos, dim=1).values  # (B, T) — -1 if no reset, else last reset segment-idx
        valid = seg_w[None, :, :] >= last_reset[:, :, None]  # (B, T, M)

        # Zero out invalid window positions instead of attention-masking. This
        # matches sequential mode's "reset_memory zeros the rolling window"
        # semantics: invalid tokens still take softmax weight (via the BD
        # positional-bias term) but contribute zero values to the output.
        windows = windows * valid[..., None].to(windows.dtype)
        memory_seq = windows  # cache snapshot at each segment position

        windows_flat = windows.reshape(B * T, self.tokens, self.token_dim)

        r = self.rel_pos(self.tokens, device=windows_flat.device, dtype=windows_flat.dtype)
        x = windows_flat
        for layer in self.layers:
            x = layer(x, r)
        # Take the last (query) position of each window.
        x_last = x[:, -1]  # (B*T, token_dim)
        x_last = self.output_norm(x_last)
        deter_seq = x_last.reshape(B, T, self.token_dim)
        return deter_seq, memory_seq


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
