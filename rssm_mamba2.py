import math

import torch
import torch.nn.functional as F
from torch import nn

from rssm_base import CategoricalRSSM
from tools import rpad


try:
    from mamba_ssm.ops.triton.ssd_combined import (
        mamba_chunk_scan_combined as _mamba_ssd_fast,  # noqa: F401
    )

    _HAS_FAST_SSD = True
except Exception:
    _HAS_FAST_SSD = False


class Mamba2Block(nn.Module):
    """Single Mamba-2 (selective scan / SSD) block.

    Scalar `A` per head, `B`/`C` shared across heads. Pre-norm + residual.
    Provides two equivalent code paths over the same parameters:

    - `forward(u, state)`: per-step recurrent update used during imagination
      and `obs_step`.
    - `forward_parallel(u_seq, init_state, reset)`: chunked SSD that processes
      a whole segment in one call. Mathematically equal to iterating `forward`
      over the segment, modulo float roundoff.

    `state` packs the SSM hidden state and the depthwise conv1d state into a
    single flat tensor so the outer RSSM can treat memory as one tensor across
    backbones.
    """

    def __init__(self, d_model, d_state, d_conv, headdim, expand, chunk_size):
        super().__init__()
        self.d_model = int(d_model)
        self.d_state = int(d_state)        # N
        self.d_conv = int(d_conv)          # K
        self.headdim = int(headdim)        # P
        self.expand = int(expand)          # E
        self.d_inner = self.d_model * self.expand
        assert self.d_inner % self.headdim == 0, (self.d_inner, self.headdim)
        self.nheads = self.d_inner // self.headdim  # H
        self.chunk_size = int(chunk_size)           # L
        assert self.d_conv >= 1, self.d_conv

        self.in_norm = nn.RMSNorm(self.d_model, eps=1e-4, dtype=torch.float32)
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=0,
            bias=True,
        )
        self.dt_proj = nn.Linear(self.d_inner, self.nheads, bias=True)
        self.B_proj = nn.Linear(self.d_inner, self.d_state, bias=False)
        self.C_proj = nn.Linear(self.d_inner, self.d_state, bias=False)
        # A = -exp(A_log); A_log ~ Uniform(log 1, log 16) per Mamba-2 init.
        self.A_log = nn.Parameter(torch.empty(self.nheads).uniform_(0.0, math.log(16.0)))
        self.D = nn.Parameter(torch.ones(self.nheads))
        self.out_norm = nn.RMSNorm(self.d_inner, eps=1e-4, dtype=torch.float32)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    @property
    def ssm_state_size(self):
        return self.nheads * self.d_state * self.headdim

    @property
    def conv_state_size(self):
        return (self.d_conv - 1) * self.d_inner

    @property
    def state_size(self):
        return self.ssm_state_size + self.conv_state_size

    def _unpack_state(self, state):
        B = state.shape[0]
        ssm_flat = state[:, : self.ssm_state_size]
        conv_flat = state[:, self.ssm_state_size :]
        ssm_state = ssm_flat.reshape(B, self.nheads, self.d_state, self.headdim)
        if self.d_conv > 1:
            conv_state = conv_flat.reshape(B, self.d_conv - 1, self.d_inner)
        else:
            conv_state = conv_flat.reshape(B, 0, self.d_inner)
        return ssm_state, conv_state

    def _pack_state(self, ssm_state, conv_state):
        B = ssm_state.shape[0]
        return torch.cat([ssm_state.reshape(B, -1), conv_state.reshape(B, -1)], dim=-1)

    def forward(self, u, state):
        """One-step recurrent path.

        u: (B, d_model). state: (B, state_size).
        Returns (out: (B, d_model), new_state: (B, state_size)).
        """
        B = u.shape[0]
        ssm_state, conv_state = self._unpack_state(state)

        x_z = self.in_proj(self.in_norm(u))
        x, z = x_z.chunk(2, dim=-1)  # (B, d_inner)

        conv_window = torch.cat([conv_state, x.unsqueeze(1)], dim=1)  # (B, K, d_inner)
        w = self.conv1d.weight.squeeze(1)  # (d_inner, K)
        bias = self.conv1d.bias
        x_conv = (conv_window * w.transpose(0, 1).unsqueeze(0)).sum(dim=1) + bias
        x_conv = F.silu(x_conv)
        new_conv_state = conv_window[:, 1:]

        dt = F.softplus(self.dt_proj(x_conv))                    # (B, H)
        B_t = self.B_proj(x_conv)                                # (B, N)
        C_t = self.C_proj(x_conv)                                # (B, N)
        A = -torch.exp(self.A_log.float()).to(x_conv.dtype)      # (H,)
        alpha = torch.exp(dt * A)                                # (B, H), in (0, 1)

        x_h = x_conv.reshape(B, self.nheads, self.headdim)       # (B, H, P)
        beta = torch.einsum("bh,bn,bhp->bhnp", dt, B_t, x_h)     # (B, H, N, P)
        new_ssm_state = alpha[..., None, None] * ssm_state + beta

        y = torch.einsum("bn,bhnp->bhp", C_t, new_ssm_state) + self.D[None, :, None] * x_h
        y = y.reshape(B, self.d_inner)
        y = F.silu(z) * self.out_norm(y)
        y = self.out_proj(y)
        return u + y, self._pack_state(new_ssm_state, new_conv_state)

    def forward_parallel(self, u_seq, init_state, reset):
        """Chunked SSD parallel path over a segment of length T.

        u_seq:      (B, T, d_model).
        init_state: (B, state_size). Zeros at the start of an episode.
        reset:      (B, T) bool.

        Returns:
            out_seq:   (B, T, d_model)
            state_seq: (B, T, state_size) — packed state right after step t
        """
        B, T, _ = u_seq.shape
        device = u_seq.device
        dtype = u_seq.dtype
        ssm_state0, conv_state0 = self._unpack_state(init_state)

        x_z = self.in_proj(self.in_norm(u_seq))
        x, z = x_z.chunk(2, dim=-1)  # (B, T, d_inner)

        # ----- Conv with reset-aware unfold -----
        K = self.d_conv
        x_extended = torch.cat([conv_state0, x], dim=1)  # (B, K-1+T, d_inner)
        windows = x_extended.unfold(1, K, 1)             # (B, T, d_inner, K)

        ws = torch.arange(K, device=device)
        ts = torch.arange(T, device=device)
        x_pos = ts[:, None] + ws[None, :] - (K - 1)      # (T, K) — source x_seq index per (t, k)
        reset_pos = torch.where(
            reset,
            ts.unsqueeze(0).expand(B, T),
            torch.full((B, T), -1, device=device, dtype=torch.long),
        )
        last_reset = torch.cummax(reset_pos, dim=1).values  # (B, T)
        # Validity:
        # - No reset has occurred yet (last_reset == -1): lookback is valid in
        #   both the init-conv-state region (x_pos < 0) and the in-segment
        #   region (x_pos >= 0).
        # - A reset has occurred at position r: init lookback is wiped (the
        #   per-step path's reset_memory zeros conv_state) and in-segment
        #   lookback is valid only at-or-after r.
        x_pos_b = x_pos[None].expand(B, T, K)
        last_reset_b = last_reset[:, :, None]
        valid = (last_reset_b == -1) | ((x_pos_b >= last_reset_b) & (x_pos_b >= 0))
        windows = windows * valid.unsqueeze(2).to(dtype)

        w = self.conv1d.weight.squeeze(1)  # (d_inner, K)
        bias = self.conv1d.bias
        x_conv = (windows * w[None, None, :, :]).sum(dim=-1) + bias
        x_conv = F.silu(x_conv)

        # End-of-step conv state per position, shape (B, T, K-1, d_inner).
        if K > 1:
            new_conv_state_seq = x_extended.unfold(1, K - 1, 1)[:, 1 : T + 1]  # (B, T, d_inner, K-1)
            new_conv_state_seq = new_conv_state_seq.transpose(-2, -1).contiguous()
            # Apply reset masking: source x_seq position for lookback-k after step t is t + k - (K-2).
            ws_state = torch.arange(K - 1, device=device)
            x_pos_state = ts[:, None] + ws_state[None, :] - (K - 2)  # (T, K-1)
            x_pos_state_b = x_pos_state[None].expand(B, T, K - 1)
            valid_state = (last_reset_b == -1) | (
                (x_pos_state_b >= last_reset_b) & (x_pos_state_b >= 0)
            )
            new_conv_state_seq = new_conv_state_seq * valid_state.unsqueeze(-1).to(dtype)
        else:
            new_conv_state_seq = x_conv.new_zeros(B, T, 0, self.d_inner)

        # ----- SSM projections over the segment -----
        dt = F.softplus(self.dt_proj(x_conv))             # (B, T, H)
        B_t = self.B_proj(x_conv)                         # (B, T, N)
        C_t = self.C_proj(x_conv)                         # (B, T, N)
        A = -torch.exp(self.A_log.float()).to(dtype)      # (H,)
        log_alpha = dt * A                                 # (B, T, H)
        x_h_seq = x_conv.reshape(B, T, self.nheads, self.headdim)

        # ----- Pad to chunk multiple -----
        L = self.chunk_size
        if T % L != 0:
            pad = L - (T % L)
            x_h_seq = F.pad(x_h_seq, (0, 0, 0, 0, 0, pad))
            B_t = F.pad(B_t, (0, 0, 0, pad))
            C_t = F.pad(C_t, (0, 0, 0, pad))
            dt = F.pad(dt, (0, 0, 0, pad))
            log_alpha = F.pad(log_alpha, (0, 0, 0, pad))
            reset_pad = F.pad(reset, (0, pad), value=False)
        else:
            reset_pad = reset
        T_pad = log_alpha.shape[1]
        nchunks = T_pad // L

        x_h_c = x_h_seq.reshape(B, nchunks, L, self.nheads, self.headdim)
        B_c = B_t.reshape(B, nchunks, L, self.d_state)
        C_c = C_t.reshape(B, nchunks, L, self.d_state)
        dt_c = dt.reshape(B, nchunks, L, self.nheads)
        la_c = log_alpha.reshape(B, nchunks, L, self.nheads)
        reset_c = reset_pad.reshape(B, nchunks, L)

        cum_la = torch.cumsum(la_c, dim=2)  # (B, nchunks, L, H)

        q_idx = torch.arange(L, device=device)
        lower_tri = (q_idx[:, None] >= q_idx[None, :])  # (L, L) bool

        H_state = ssm_state0  # (B, H, N, P)
        out_chunks = []
        ssm_state_chunks = []
        D_per_head = self.D[None, None, :, None]  # (1, 1, H, 1)

        for c in range(nchunks):
            X = x_h_c[:, c]              # (B, L, H, P)
            BB = B_c[:, c]               # (B, L, N)
            CC = C_c[:, c]               # (B, L, N)
            dT = dt_c[:, c]              # (B, L, H)
            cla = cum_la[:, c]           # (B, L, H)
            r_c = reset_c[:, c]          # (B, L) bool

            # Within-chunk last reset (chunk-local index, -1 if none).
            reset_pos_c = torch.where(
                r_c,
                q_idx.unsqueeze(0).expand(B, L),
                torch.full((B, L), -1, device=device, dtype=torch.long),
            )
            chunk_last_reset = torch.cummax(reset_pos_c, dim=1).values  # (B, L)

            mask_within = q_idx[None, None, :] >= chunk_last_reset[:, :, None]  # (B, L, L)
            mask_within = mask_within & lower_tri[None, :, :]
            mask_carry = chunk_last_reset == -1  # (B, L)

            diff = cla.unsqueeze(2) - cla.unsqueeze(1)              # (B, L, L, H)
            M = torch.exp(diff) * mask_within.unsqueeze(-1).to(dtype)  # (B, L, L, H)

            CB = torch.einsum("bln,bkn->blk", CC, BB)               # (B, L, L)
            W = M * CB.unsqueeze(-1) * dT.unsqueeze(1)              # (B, L, L, H)
            Y_attn = torch.einsum("blkh,bkhp->blhp", W, X)          # (B, L, H, P)

            CH = torch.einsum("bln,bhnp->blhp", CC, H_state)        # (B, L, H, P)
            Y_carry = (
                CH
                * mask_carry.unsqueeze(-1).unsqueeze(-1).to(dtype)
                * torch.exp(cla).unsqueeze(-1)
            )

            Y_skip = D_per_head * X
            Y = Y_attn + Y_carry + Y_skip
            out_chunks.append(Y)

            # Per-position SSM state: explicit so the parallel state_seq matches per-step exactly.
            state_contrib = torch.einsum("blkh,bkh,bkn,bkhp->blhnp", M, dT, BB, X)  # (B, L, H, N, P)
            carry_q = (
                mask_carry[:, :, None, None, None].to(dtype)
                * torch.exp(cla)[:, :, :, None, None]
                * H_state.unsqueeze(1)
            )
            H_step_seq = carry_q + state_contrib                    # (B, L, H, N, P)
            ssm_state_chunks.append(H_step_seq)
            H_state = H_step_seq[:, -1]

        Y_full = torch.cat(out_chunks, dim=1)[:, :T]                # (B, T, H, P)
        ssm_state_seq = torch.cat(ssm_state_chunks, dim=1)[:, :T]   # (B, T, H, N, P)

        y = Y_full.reshape(B, T, self.d_inner)
        y = F.silu(z[:, :T]) * self.out_norm(y)
        y = self.out_proj(y)
        out_seq = u_seq + y

        ssm_flat = ssm_state_seq.reshape(B, T, -1)
        conv_flat = new_conv_state_seq.reshape(B, T, -1)
        state_seq = torch.cat([ssm_flat, conv_flat], dim=-1)
        return out_seq, state_seq


class Mamba2Deter(nn.Module):
    """Stack of Mamba-2 blocks with a shared input projection."""

    def __init__(self, deter, stoch, act_dim, config, act="SiLU"):
        super().__init__()
        self.d_model = int(deter)
        self.layers = int(config.layers)
        self.d_state = int(config.d_state)
        self.d_conv = int(config.d_conv)
        self.headdim = int(config.headdim)
        self.expand = int(config.expand)
        self.chunk_size = int(config.chunk_size)

        act_cls = getattr(torch.nn, act)
        self.input_proj = nn.Sequential(
            nn.Linear(stoch + act_dim, self.d_model, bias=True),
            nn.RMSNorm(self.d_model, eps=1e-4, dtype=torch.float32),
            act_cls(),
        )
        self.blocks = nn.ModuleList(
            [
                Mamba2Block(
                    self.d_model,
                    self.d_state,
                    self.d_conv,
                    self.headdim,
                    self.expand,
                    self.chunk_size,
                )
                for _ in range(self.layers)
            ]
        )
        self.output_norm = nn.RMSNorm(self.d_model, eps=1e-4, dtype=torch.float32)
        self._block_state_size = self.blocks[0].state_size

    def initial_memory(self, batch_size):
        device = next(self.parameters()).device
        return torch.zeros(
            batch_size, self.layers, self._block_state_size, dtype=torch.float32, device=device
        )

    def reset_memory(self, memory, reset):
        mask = rpad(reset, memory.dim() - reset.dim())
        return torch.where(mask, torch.zeros_like(memory), memory)

    def forward(self, stoch, deter, memory, action):
        action = action / torch.clip(torch.abs(action), min=1.0).detach()
        flat_stoch = stoch.reshape(stoch.shape[0], -1)
        u = self.input_proj(torch.cat([flat_stoch, action], dim=-1))
        new_states = []
        for i, block in enumerate(self.blocks):
            u, new_state = block(u, memory[:, i])
            new_states.append(new_state)
        u = self.output_norm(u)
        return u, torch.stack(new_states, dim=1)

    def forward_parallel(self, stochs, actions, init_memory, reset):
        B, T = actions.shape[:2]
        reset = reset.reshape(B, T).to(torch.bool)

        # Mirror obs_step's at-reset zeroing of the previous (stoch, action).
        reset_stoch = rpad(reset, stochs.dim() - reset.dim())
        reset_action = rpad(reset, actions.dim() - reset.dim())
        stochs = torch.where(reset_stoch, torch.zeros_like(stochs), stochs)
        actions = torch.where(reset_action, torch.zeros_like(actions), actions)

        actions = actions / torch.clip(torch.abs(actions), min=1.0).detach()
        flat_stoch = stochs.reshape(B, T, -1)
        u_seq = self.input_proj(torch.cat([flat_stoch, actions], dim=-1))

        block_state_seqs = []
        for i, block in enumerate(self.blocks):
            u_seq, state_seq = block.forward_parallel(u_seq, init_memory[:, i], reset)
            block_state_seqs.append(state_seq)
        u_seq = self.output_norm(u_seq)
        memory_seq = torch.stack(block_state_seqs, dim=2)  # (B, T, layers, S)
        return u_seq, memory_seq


class RSSM(CategoricalRSSM):
    def __init__(self, config, embed_size, act_dim, backbone_config):
        deter_net = Mamba2Deter(
            int(config.deter),
            int(config.stoch) * int(config.discrete),
            act_dim,
            backbone_config,
            act=config.act,
        )
        super().__init__(config, embed_size, act_dim, deter_net)
