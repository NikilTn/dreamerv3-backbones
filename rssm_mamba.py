"""Mamba-2 SSD-style selective state-space backbone for the Dreamer RSSM
(Phase 2b — block layout matches official ``state-spaces/mamba``'s
``Mamba2.step()``).

Architectural reference: Dao & Gu, ICML 2024 ("Transformers are SSMs:
Generalized Models and Efficient Algorithms Through Structured State Space
Duality") — reference [5] in the project proposal. Proposal §3.2: drop-in
replacement of the GRU with a Mamba-2 SSSM, without the architectural
differences posed by Drama.

Block layout (matches the official ``Mamba2`` block's per-step path):

    x         ∈ R^{B×dim}                (block input, after pre-norm)
    [z, xBC, dt] = InProj(x)             (in_proj is bias-free, like official)
        z   ∈ R^{B×hidden}                  (gate path)
        xBC ∈ R^{B×(hidden + 2·d_state)}   (will pass through the conv)
        dt  ∈ R^{B×nheads}                 (raw dt, before bias + softplus)
    xBC = SiLU(causal_depthwise_conv(xBC))    ← the conv mixes x, B, C jointly
    x_post, B_post, C_post = split(xBC, [hidden, d_state, d_state])
    x_h = x_post.view(B, nheads, headdim)
    Δ   = softplus(dt + dt_bias)         ∈ R^{B×nheads}
    dA  = exp(Δ · A)                     ∈ R^{B×nheads}    where A = -exp(A_log) ∈ R^{nheads}
    dBx = einsum("bh,bn,bhp -> bhpn", Δ, B_post, x_h)        ∈ R^{B×nheads×headdim×d_state}
    state[t+1] = state[t] · dA[..., None, None] + dBx
    y_h = einsum("bhpn,bn -> bhp", state[t+1], C_post) + D · x_h
    y   = flatten(y_h)                   ∈ R^{B×hidden}
    y   = RMSNorm(SiLU(z) · y)           (gated post-norm before out_proj)
    out = OutProj(y) + residual

Critical correction relative to an earlier Mamba-1-style draft of this file:
**B and C also pass through the depthwise conv** (as part of the xBC concat),
matching the official block. The earlier draft applied the conv only to the
``x`` branch and recomputed B/C/dt from the conv output, which is a real
architectural difference (different short-time mixing of B/C with x).

Persistence: ``extra["conv_state_*"]`` of shape ``(B, hidden + 2·d_state, kernel-1)``
holds the recent conv inputs; ``extra["ssm_state_*"]`` of shape
``(B, nheads, headdim, d_state)`` holds the SSD hidden state. Both persist
across env steps via the unified extra interface from Phase 1.

Pure PyTorch — no dependency on the ``mamba_ssm`` CUDA package. Trade-off vs.
the official: this is **slower** than the fused-kernel + chunked-SSD-scan
implementation in ``state-spaces/mamba``. We avoid CUDA-build risk on
PyTorch 2.8 + CUDA 13.1 academic clusters at the cost of throughput. For
Dreamer's per-env-step ``observe()`` loop (which loops one step at a time
anyway), parallel scan would only speed up training, not change correctness.

Honest gaps relative to ``mamba_ssm.Mamba2`` (do not block production for the
proposal's per-step Dreamer use):

- **No SSD chunked parallel scan during training.** Per-step Python loop.
  Identical output, slower throughput on long replay chunks.
- **No fused CUDA/Triton kernels.** Pure PyTorch eager execution.
- **``ngroups`` fixed at 1.** B and C shared across all heads (single group).
  The official allows ``ngroups > 1`` for parameter scaling at large model
  sizes; not measurably relevant at world-model scale.
- **Gated norm uses ``RMSNorm(SiLU(z) · y)``** rather than the fused
  ``RMSNormGated`` kernel. Mathematically equivalent.
- **Persistence test still uses private ``_call_deter_net`` path** with
  ``stoch_mode="onehot"``; useful but not the same as a full training
  ``observe()``-path validation.

Cross-checked against:
- ``state-spaces/mamba/mamba_ssm/modules/mamba2.py::Mamba2.step``
- ``tommyip/mamba2-minimal/mamba2.py``
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from rssm_base import CategoricalRSSM


class MambaSSD(nn.Module):
    """Mamba-2 SSD recurrence parameters and per-step update.

    Owns only the SSD recurrence parameters: ``A_log`` (scalar per head),
    ``D`` (per-head skip), and ``dt_bias`` (per-head). The ``x``, ``B``, ``C``,
    and raw ``dt`` are computed in :py:class:`Mamba2Block` (post-conv split)
    and passed in to :py:meth:`step`.

    Initialization matches ``state-spaces/mamba``:
    - ``A`` drawn uniform ``[A_init_min, A_init_max]`` per head (default
      ``[1, 16]``); stored as ``A_log = log(A)``; runtime ``A = -exp(A_log)``.
    - ``dt_bias`` such that ``softplus(dt_bias)`` recovers Δ ∈ log-uniform
      ``[dt_min, dt_max]`` per head (Mamba paper §3.6 convention).
    """

    def __init__(
        self,
        nheads: int,
        headdim: int,
        d_state: int,
        A_init_range: tuple[float, float] = (1.0, 16.0),
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
    ):
        super().__init__()
        self.nheads = int(nheads)
        self.headdim = int(headdim)
        self.d_state = int(d_state)

        # Stash for re-init in :py:meth:`special_init` (called after the
        # Dreamer global ``weight_init_`` would otherwise wipe ``dt_bias``).
        self._dt_min = float(dt_min)
        self._dt_max = float(dt_max)
        self._A_init_range = (float(A_init_range[0]), float(A_init_range[1]))

        # A: scalar per head (the SSD restriction). Init uniform [1, 16] then log.
        with torch.no_grad():
            a_init = torch.empty(self.nheads).uniform_(*A_init_range)
        self.A_log = nn.Parameter(torch.log(a_init))

        # D: per-head skip gain.
        self.D = nn.Parameter(torch.ones(self.nheads))

        # dt_bias: per-head, log-uniform Δ via softplus^{-1}.
        self.dt_bias = nn.Parameter(torch.zeros(self.nheads))
        self.special_init()

    def special_init(self):
        """Restore Mamba-2 dt_bias init.

        ``A_log`` and ``D`` are ``nn.Parameter`` directly on this module
        (no ``weight`` attribute), so ``tools.weight_init_`` skips them.
        ``dt_bias`` is also a ``nn.Parameter`` (not a Linear's bias), so it's
        also untouched by ``weight_init_`` — but since the Dreamer global
        init runs *after* ``__init__``, we keep this method explicit so the
        intent is obvious and so the RSSM wrapper can re-trigger it if any
        upstream pass overrides it.
        """
        with torch.no_grad():
            dt = torch.exp(
                torch.rand(self.nheads, device=self.dt_bias.device)
                * (math.log(self._dt_max) - math.log(self._dt_min))
                + math.log(self._dt_min)
            ).clamp(min=1e-4)
            inv_dt = dt + torch.log(-torch.expm1(-dt))  # softplus^{-1}(dt)
            self.dt_bias.copy_(inv_dt)

    def step(
        self,
        x_heads: torch.Tensor,
        B: torch.Tensor,
        C: torch.Tensor,
        dt_raw: torch.Tensor,
        state: torch.Tensor,
    ):
        """One env-step Mamba-2 SSD recurrent update.

        Args:
            x_heads: ``(B, nheads, headdim)`` post-conv reshaped input.
            B: ``(B, d_state)`` post-conv B projection.
            C: ``(B, d_state)`` post-conv C projection.
            dt_raw: ``(B, nheads)`` raw dt slice from in_proj (pre-bias, pre-softplus).
            state: ``(B, nheads, headdim, d_state)`` SSD hidden state.

        Returns:
            ``(B, nheads, headdim)`` per-head output, new state.
        """
        # Δ: softplus(raw_dt + dt_bias) per head.
        dt = F.softplus(dt_raw + self.dt_bias.unsqueeze(0))   # (B, H)
        # Continuous-time A: scalar per head, strictly negative.
        A = -torch.exp(self.A_log)                            # (H,)
        # Discretize: dA = exp(Δ * A) per head.
        dA = torch.exp(dt * A.unsqueeze(0))                   # (B, H)

        # dBx[b, h, p, n] = Δ[b, h] * B[b, n] * x_heads[b, h, p].
        dBx = torch.einsum("bh,bn,bhp->bhpn", dt, B, x_heads)  # (B, H, P, N)

        # Recurrent update.
        new_state = state * dA[..., None, None] + dBx

        # Output: y[b, h, p] = sum_n new_state[b, h, p, n] * C[b, n].
        y_heads = torch.einsum("bhpn,bn->bhp", new_state, C)
        # Per-head D skip.
        y_heads = y_heads + self.D[None, :, None] * x_heads
        return y_heads, new_state


class Mamba2Block(nn.Module):
    """Mamba-2 block: in_proj → split [z, xBC, dt] → conv on xBC → split → SSD → gated norm → out_proj.

    Matches the official ``state-spaces/mamba`` per-step block layout:
    - ``in_proj`` is bias-free and emits ``[z, xBC, dt]`` of widths
      ``[hidden, hidden + 2·d_state, nheads]``.
    - The causal depthwise conv operates on ``xBC`` (so B and C also receive
      short-time mixing), not just on ``x``.
    - Post-conv ``xBC`` is split into ``x``, ``B``, ``C`` and passed into
      :py:class:`MambaSSD` for the recurrence.
    - Output is gated by ``SiLU(z)``, normalized, and projected back to
      ``dim`` with a residual add.

    Persistent extras:
    - ``conv_state``: ``(B, hidden + 2·d_state, kernel-1)`` (NOT just hidden).
    - ``ssm_state``: ``(B, nheads, headdim, d_state)``.
    """

    def __init__(
        self,
        dim: int,
        expand: int,
        conv_kernel: int,
        nheads: int,
        d_state: int,
    ):
        super().__init__()
        self.dim = int(dim)
        self.hidden = int(expand) * int(dim)
        assert self.hidden % int(nheads) == 0, (
            f"hidden ({self.hidden}) must be divisible by nheads ({nheads})"
        )
        self.nheads = int(nheads)
        self.headdim = self.hidden // self.nheads
        self.d_state = int(d_state)
        self.conv_kernel = int(conv_kernel)
        # xBC width: hidden (x) + d_state (B) + d_state (C). ngroups=1.
        self.conv_dim = self.hidden + 2 * self.d_state

        self.norm = nn.RMSNorm(dim, eps=1e-04, dtype=torch.float32)

        # in_proj: emits [z, xBC, dt]. Bias-free per official Mamba-2 default.
        in_proj_dim = self.hidden + self.conv_dim + self.nheads
        self.in_proj = nn.Linear(dim, in_proj_dim, bias=False)

        # Causal depthwise conv on xBC (the channel-mixing path that gives
        # B and C their short-time temporal context, matching the official).
        self.conv = nn.Conv1d(
            self.conv_dim, self.conv_dim,
            kernel_size=self.conv_kernel, groups=self.conv_dim,
            padding=self.conv_kernel - 1, bias=True,
        )

        # SSD recurrence parameters.
        self.ssm = MambaSSD(
            nheads=self.nheads, headdim=self.headdim, d_state=self.d_state,
        )

        # Mamba-2 gated RMSNorm layout: y *= SiLU(z); y = norm(y); out_proj.
        self.post_norm = nn.RMSNorm(self.hidden, eps=1e-04, dtype=torch.float32)
        self.out_proj = nn.Linear(self.hidden, dim, bias=True)

    def step(
        self,
        x_t: torch.Tensor,
        conv_state: torch.Tensor,
        ssm_state: torch.Tensor,
    ):
        """One env-step forward with persistent conv + SSD state.

        Args:
            x_t: ``(B, dim)`` block input.
            conv_state: ``(B, conv_dim, kernel-1)`` past conv inputs (covers
                xBC, not just x).
            ssm_state: ``(B, nheads, headdim, d_state)`` SSD hidden state.

        Returns:
            ``(B, dim)`` block output, new conv_state, new ssm_state.
        """
        x = self.norm(x_t)
        # in_proj produces concat[z, xBC, dt].
        proj = self.in_proj(x)                                # (B, hidden + conv_dim + nheads)
        z, xBC, dt_raw = torch.split(
            proj, [self.hidden, self.conv_dim, self.nheads], dim=-1
        )

        # Causal conv with persistent state on xBC. The conv mixes x, B, and
        # C together over the past `kernel` env steps (so B and C see the
        # marker through the same short-time receptive field as x). Manual
        # elementwise multiply-and-sum is used instead of F.conv1d because
        # for kernel=3 / single-timestep / small conv_dim the F.conv1d
        # per-call overhead (~700ms vs 37ms on CPU) far exceeds the math.
        # On GPU the trade-off may differ — re-measure on cluster before
        # switching.
        xBC_seq = torch.cat([conv_state, xBC.unsqueeze(-1)], dim=-1)  # (B, conv_dim, kernel)
        weight = self.conv.weight.squeeze(1)                          # (conv_dim, kernel)
        bias = self.conv.bias                                          # (conv_dim,)
        xBC_conv = (weight.unsqueeze(0) * xBC_seq).sum(-1) + bias.unsqueeze(0)
        xBC_conv = F.silu(xBC_conv)
        new_conv_state = xBC_seq[..., 1:].contiguous()

        # Split post-conv xBC into x, B, C.
        x_post, b_proj, c_proj = torch.split(
            xBC_conv, [self.hidden, self.d_state, self.d_state], dim=-1
        )
        # Reshape x into heads for the SSD recurrence.
        batch = x_t.shape[0]
        x_heads = x_post.view(batch, self.nheads, self.headdim)

        # SSD recurrent step.
        y_heads, new_ssm_state = self.ssm.step(x_heads, b_proj, c_proj, dt_raw, ssm_state)
        y = y_heads.reshape(batch, self.hidden)

        # Mamba-2 gated norm (SiLU(z) · y, then RMSNorm).
        gated = y * F.silu(z)
        gated = self.post_norm(gated)

        out = self.out_proj(gated)
        return x_t + out, new_conv_state, new_ssm_state


class MambaDeter(nn.Module):
    """Deter network with stacked Mamba-2 blocks.

    Each block carries its own ``conv_state`` (covers xBC, not just x) and
    SSD ``ssm_state`` in ``extra``. The deter representation is a rolling
    tape of the last ``tokens`` final-block outputs (kept for backward
    compatibility with the obs/img heads — the genuine recurrent memory is
    the SSD state).
    """

    def __init__(self, deter, stoch, act_dim, config, act="SiLU"):
        super().__init__()
        self.tokens = int(config.tokens)
        assert deter % self.tokens == 0, (deter, self.tokens)
        self.token_dim = deter // self.tokens
        self.expand = int(config.expand)
        self.conv_kernel = int(config.conv_kernel)
        # Mamba-2 knobs. Defaults match the proposal-cited Mamba-2 paper:
        # d_state ~ 64 (paper-typical), nheads ~ 8 (so headdim = hidden/8).
        self.d_state = int(getattr(config, "state_size", 64))
        self.nheads = int(getattr(config, "nheads", 8))

        hidden = self.expand * self.token_dim
        if hidden % self.nheads != 0:
            raise ValueError(
                f"hidden ({hidden}) = expand ({self.expand}) * token_dim "
                f"({self.token_dim}) must be divisible by nheads ({self.nheads})"
            )

        act_cls = getattr(torch.nn, act)
        self.input_proj = nn.Sequential(
            nn.Linear(stoch + act_dim, self.token_dim, bias=True),
            nn.RMSNorm(self.token_dim, eps=1e-04, dtype=torch.float32),
            act_cls(),
        )
        self.blocks = nn.ModuleList(
            [
                Mamba2Block(
                    self.token_dim,
                    self.expand,
                    self.conv_kernel,
                    nheads=self.nheads,
                    d_state=self.d_state,
                )
                for _ in range(int(config.layers))
            ]
        )
        self.output_norm = nn.RMSNorm(self.token_dim, eps=1e-04, dtype=torch.float32)

    def initial_extra(self, batch_size, device):
        """Per-block conv state (over xBC) + SSD hidden state + per-step x_t.

        - ``conv_state_*`` shape: ``(B, hidden + 2·d_state, kernel-1)`` —
          covers x, B, C jointly (matches official Mamba-2 conv on xBC).
        - ``ssm_state_*`` shape: ``(B, nheads, headdim, d_state)``.
        - ``x_t`` shape: ``(B, token_dim)`` — the post-final-block per-step
          output, used by the SSSM hidden-state policy ``π(a | z_t, x_t)``
          (proposal §3.1). Initialized to zeros; overwritten each step in
          :py:meth:`forward`.
        """
        extras: dict = {}
        for i, block in enumerate(self.blocks):
            extras[f"conv_state_{i}"] = torch.zeros(
                batch_size, block.conv_dim, block.conv_kernel - 1,
                dtype=torch.float32, device=device,
            )
            extras[f"ssm_state_{i}"] = torch.zeros(
                batch_size, block.nheads, block.headdim, block.d_state,
                dtype=torch.float32, device=device,
            )
        extras["x_t"] = torch.zeros(
            batch_size, self.token_dim, dtype=torch.float32, device=device,
        )
        return extras

    def forward(self, stoch, deter, action, extra=None):
        if extra is None:
            extra = self.initial_extra(action.shape[0], action.device)
        batch_size = action.shape[0]
        action = action / torch.clip(torch.abs(action), min=1.0).detach()
        new_token = self.input_proj(torch.cat([stoch.reshape(batch_size, -1), action], dim=-1))

        new_extra: dict = {}
        x = new_token
        for i, block in enumerate(self.blocks):
            conv_state = extra.get(f"conv_state_{i}")
            ssm_state = extra.get(f"ssm_state_{i}")
            if conv_state is None:
                conv_state = x.new_zeros(
                    batch_size, block.conv_dim, block.conv_kernel - 1
                )
            if ssm_state is None:
                ssm_state = x.new_zeros(
                    batch_size, block.nheads, block.headdim, block.d_state
                )
            x, conv_state, ssm_state = block.step(x, conv_state, ssm_state)
            new_extra[f"conv_state_{i}"] = conv_state
            new_extra[f"ssm_state_{i}"] = ssm_state

        # Expose the post-final-block per-step output for the SSSM hidden-
        # state policy π(a | z_t, x_t) (proposal §3.1). Shape (B, token_dim).
        new_extra["x_t"] = x

        # Maintain the deter rolling tape (last `tokens` final-block outputs).
        tokens = deter.reshape(batch_size, self.tokens, self.token_dim)
        tokens = torch.cat([tokens[:, 1:], x.unsqueeze(1)], dim=1)
        tokens = self.output_norm(tokens)
        return tokens.reshape(batch_size, -1), new_extra


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
        # Restore Mamba-2 dt_bias init that may have been overridden by the
        # Dreamer global ``weight_init_`` pass. (dt_bias is a Parameter not a
        # Linear's bias, so ``weight_init_`` does NOT actually touch it; this
        # is defensive in case of any additional global init we add later.)
        for block in deter_net.blocks:
            block.ssm.special_init()

    @property
    def policy_feat_size(self):
        # Hidden-state policy: actor sees cat([stoch_flat, x_t]) with
        # x_t of shape (B, token_dim). Per proposal §3.1.
        return self.flat_stoch + self._deter_net.token_dim

    def policy_feat(self, stoch, deter, extra=None):
        """SSSM hidden-state policy ``π(a | z_t, x_t)`` (proposal §3.1).

        ``x_t`` is the post-final-block per-step output, threaded via
        ``extra["x_t"]`` by :py:class:`MambaDeter.forward`. Falls back to
        :py:meth:`get_feat` if ``extra`` lacks ``x_t`` (e.g., construction-
        time sizing checks before any forward pass) — in that case the
        returned tensor will be ``feat_size``-wide and the caller is
        responsible for re-checking once a real ``extra`` is available.
        """
        if extra is None or "x_t" not in extra:
            return self.get_feat(stoch, deter)
        stoch_flat = stoch.reshape(*stoch.shape[:-2], self.flat_stoch)
        return torch.cat([stoch_flat, extra["x_t"]], dim=-1)
