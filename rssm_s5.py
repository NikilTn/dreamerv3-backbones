"""S5 (Simplified State Space) backbone for the Dreamer RSSM (Phase 2d).

Architectural reference: Smith, Warrington & Linderman, ICLR 2023
("Simplified State Space Layers for Sequence Modeling") — reference [7] in
the project proposal. Hieros (Mattes et al., 2024; reference [6] in the
proposal) uses S5 as its world-model backbone.

Distinctive S5 design points (vs. S4D and Mamba-2):

- **Complex-diagonal A**, not real-diagonal. Each pole of A is a complex
  eigenvalue with negative real part (stable decay) and non-zero imaginary
  part (rotational/oscillatory dynamics). Init: HiPPO-N approximation —
  ``A_n = -0.5 + i · π · n`` for ``n ∈ [1, d_state]``.
- **MIMO structure** (multi-input multi-output). Each "head" runs ONE shared
  SSM cell with ``d_state`` complex poles, with B mapping
  ``R^{head_dim} → C^{d_state}`` and C mapping ``C^{d_state} → R^{head_dim}``.
  All ``head_dim`` channels within a head share the same poles. This trades
  capacity (one shared state per head) for parallelism (single scan over
  time, no per-channel parallelism inside a head).
- **Per-pole-per-head learned timestep Δ**, shape ``(heads, d_state)``,
  init log-uniform ``[1e-3, 0.1]``.
- **ZOH discretization for diagonal A** (closed form via element-wise
  ``A_bar = exp(Δ · A)``, ``B_bar = (1/A) · (A_bar - 1) · B``).
- **Output**: ``y = 2 · Re(C · h) + D · x``. The factor 2 accounts for the
  conjugate-paired structure that ensures the output is real-valued.

State persistence: ``extra["ssm_state_*"]`` holds the complex SSM hidden
state of shape ``(B, heads, d_state)``, dtype ``complex64``. Unlike S4 or
Mamba (real-state), S5's state is intrinsically complex.

Pure PyTorch — uses native ``torch.complex64`` arithmetic. PyTorch's autograd
supports complex parameters cleanly; we avoid feeding complex tensors to
RMSNorm (which doesn't support complex) by keeping all complex math inside
the SSM update and casting to real before the residual add.

Honest gaps relative to the official ``lindermanlab/S5`` (do not block
production for the per-step Dreamer loop):

- **No parallel associative scan during training.** Per-step Python loop
  over T env steps. The official JAX implementation uses an associative
  scan for O(log T) parallel depth; we use the natural O(T) recurrent form.
  Identical output for the per-step path; slower throughput on long replay
  chunks.
- **HiPPO-N approximation init**, not the full HiPPO-LegS diagonalization.
  Sets ``Re(A) = -0.5``, ``Im(A) = π · n``. The official ``init_columns``
  routine in lindermanlab/S5 uses the LegS matrix's eigenvalue
  decomposition; the approximation we use is what most simpler S5 ports
  (and several published S5 follow-ups) adopt.
- **B and C are random-init complex Gaussians** scaled by ``1/sqrt(N)``.
  Official S5 has multiple init choices including B = column of LegS;
  ours is the simplest valid choice.

Cross-checked against:
- ``lindermanlab/S5`` (JAX, official) — for math and init scheme.
- Smith et al. 2023, sections 3-4 (the SSM update equations).
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from rssm_base import CategoricalRSSM


class S5Block(nn.Module):
    """S5 MIMO complex-diagonal SSM block with persistent complex state.

    Each head runs a single complex-diagonal SSM cell of state size
    ``d_state`` with input/output width ``head_dim = dim // heads``. State
    shape per layer is ``(B, heads, d_state)`` complex64.

    HiPPO-N approximation init: ``A_n = -0.5 + i · π · n`` per head.
    Parameterize ``Re(A) = -exp(A_re_log)`` so the real part is always
    negative (Hurwitz/stable). ``Im(A)`` is unconstrained.
    """

    def __init__(
        self,
        dim: int,
        d_state: int,
        heads: int,
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
    ):
        super().__init__()
        assert dim % heads == 0, (dim, heads)
        self.dim = int(dim)
        self.heads = int(heads)
        self.head_dim = self.dim // self.heads
        self.d_state = int(d_state)

        self.norm = nn.RMSNorm(dim, eps=1e-04, dtype=torch.float32)

        # A: complex-diagonal per head, parameterized as two real tensors.
        # HiPPO-N approximation: Re(A_n) = -0.5 (constant); Im(A_n) = π·n.
        # Stored as A_re_log = log(0.5) so runtime Re(A) = -exp(A_re_log) = -0.5
        # at init and is always negative (Hurwitz/stable).
        with torch.no_grad():
            a_re_log_init = torch.full(
                (self.heads, self.d_state), math.log(0.5), dtype=torch.float32
            )
            n = torch.arange(1, self.d_state + 1, dtype=torch.float32)
            a_im_init = (math.pi * n).unsqueeze(0).expand(self.heads, -1).contiguous()
        self.A_re_log = nn.Parameter(a_re_log_init)
        self.A_im = nn.Parameter(a_im_init)

        # B: complex (heads, d_state, head_dim). Stored as two real tensors.
        # Init Gaussian scaled by 1/sqrt(head_dim).
        b_scale = 1.0 / math.sqrt(self.head_dim)
        with torch.no_grad():
            B_re_init = torch.randn(self.heads, self.d_state, self.head_dim) * b_scale
            B_im_init = torch.randn(self.heads, self.d_state, self.head_dim) * b_scale
        self.B_re = nn.Parameter(B_re_init)
        self.B_im = nn.Parameter(B_im_init)

        # C: complex (heads, head_dim, d_state). Stored as two real tensors.
        # Init Gaussian scaled by 1/sqrt(d_state).
        c_scale = 1.0 / math.sqrt(self.d_state)
        with torch.no_grad():
            C_re_init = torch.randn(self.heads, self.head_dim, self.d_state) * c_scale
            C_im_init = torch.randn(self.heads, self.head_dim, self.d_state) * c_scale
        self.C_re = nn.Parameter(C_re_init)
        self.C_im = nn.Parameter(C_im_init)

        # D: real per-channel skip gain.
        self.D = nn.Parameter(torch.ones(self.dim))

        # Δ: per-pole-per-head learned timestep, init log-uniform [dt_min, dt_max].
        # Stored via softplus^{-1} so softplus(log_dt) recovers Δ at init.
        # (Standalone Parameter, not a Linear's bias, so the Dreamer global
        # weight_init_ doesn't touch it.)
        self._dt_min = float(dt_min)
        self._dt_max = float(dt_max)
        self.log_dt = nn.Parameter(torch.zeros(self.heads, self.d_state))
        self._init_log_dt()

    def _init_log_dt(self):
        """Initialize log_dt so softplus(log_dt) ∈ log-uniform [dt_min, dt_max]."""
        with torch.no_grad():
            dt = torch.exp(
                torch.rand(self.heads, self.d_state)
                * (math.log(self._dt_max) - math.log(self._dt_min))
                + math.log(self._dt_min)
            ).clamp(min=1e-4)
            inv_dt = dt + torch.log(-torch.expm1(-dt))
            self.log_dt.copy_(inv_dt)

    def _A_complex(self) -> torch.Tensor:
        """Complex-diagonal A of shape (heads, d_state)."""
        return torch.complex(-torch.exp(self.A_re_log), self.A_im)

    def _B_complex(self) -> torch.Tensor:
        """Complex B of shape (heads, d_state, head_dim)."""
        return torch.complex(self.B_re, self.B_im)

    def _C_complex(self) -> torch.Tensor:
        """Complex C of shape (heads, head_dim, d_state)."""
        return torch.complex(self.C_re, self.C_im)

    def discretize(self):
        """Precompute S5 recurrent tensors for one rollout."""
        A = self._A_complex()                                       # (H, N) complex
        B_param = self._B_complex()                                 # (H, N, P) complex
        C_param = self._C_complex()                                 # (H, P, N) complex
        dt = F.softplus(self.log_dt)                                # (H, N) real
        Ad = dt.to(A.dtype) * A                                     # (H, N) complex
        A_bar = torch.exp(Ad)                                       # (H, N) complex
        B_bar = ((A_bar - 1) / A).unsqueeze(-1) * B_param           # (H, N, P) complex
        D_h = self.D.view(self.heads, self.head_dim).unsqueeze(0)   # (1, H, P)
        return A_bar, B_bar, C_param, D_h

    def step(self, x_t: torch.Tensor, state: torch.Tensor, discrete=None):
        """One env-step S5 recurrent update.

        Args:
            x_t: ``(B, dim)`` real input.
            state: ``(B, heads, d_state)`` complex64 hidden state.

        Returns:
            ``(B, dim)`` real output (with residual), updated complex state.
        """
        x = self.norm(x_t)
        batch = x.shape[0]
        x_h = x.view(batch, self.heads, self.head_dim)             # (B, H, P) real

        if discrete is None:
            A_bar, B_bar, C_param, D_h = self.discretize()
        else:
            A_bar, B_bar, C_param, D_h = discrete

        # State update: state[t+1] = A_bar · state[t] + B_bar · x[t].
        Bx = torch.einsum(
            "hnp,bhp->bhn",
            B_bar,
            x_h.to(B_bar.dtype),
        )                                                           # (B, H, N) complex
        new_state = A_bar.unsqueeze(0) * state + Bx                 # (B, H, N) complex

        # Output: y_h = 2 · Re(C · new_state) + D · x.
        # The factor 2 accounts for the conjugate-paired structure that
        # ensures the projected output is real-valued in the original S5 paper.
        Ch = torch.einsum("hpn,bhn->bhp", C_param, new_state)       # (B, H, P) complex
        y_h = 2.0 * Ch.real                                         # (B, H, P) real
        y_h = y_h + D_h * x_h
        y = y_h.reshape(batch, self.dim)
        return x_t + y, new_state


class S5Deter(nn.Module):
    """Deter network with stacked S5 complex-diagonal SSM blocks.

    Each block carries its own complex-valued ``ssm_state_*`` in ``extra``,
    persistent across env steps. The deter representation is a rolling tape
    of the last ``tokens`` final-block outputs (kept for backward
    compatibility with the obs/img heads — the genuine recurrent memory is
    the complex SSM state).
    """

    def __init__(self, deter, stoch, act_dim, config, act="SiLU"):
        super().__init__()
        self.supports_step_context = True
        self.tokens = int(config.tokens)
        assert deter % self.tokens == 0, (deter, self.tokens)
        self.token_dim = deter // self.tokens
        self.heads = int(config.heads)
        self.d_state = int(config.state_size)
        act_cls = getattr(torch.nn, act)
        self.input_proj = nn.Sequential(
            nn.Linear(stoch + act_dim, self.token_dim, bias=True),
            nn.RMSNorm(self.token_dim, eps=1e-04, dtype=torch.float32),
            act_cls(),
        )
        self.blocks = nn.ModuleList(
            [
                S5Block(self.token_dim, self.d_state, self.heads)
                for _ in range(int(config.layers))
            ]
        )
        self.output_norm = nn.RMSNorm(self.token_dim, eps=1e-04, dtype=torch.float32)

    def initial_extra(self, batch_size, device):
        """Per-block complex SSM state + per-step ``x_t``.

        - ``ssm_state_*`` shape ``(B, heads, d_state)``, dtype **complex64**.
        - ``x_t`` shape ``(B, token_dim)``, dtype float32 — used by the SSSM
          hidden-state policy ``π(a | z_t, x_t)`` (proposal §3.1).
        """
        extras: dict = {}
        for i, block in enumerate(self.blocks):
            extras[f"ssm_state_{i}"] = torch.zeros(
                batch_size, block.heads, block.d_state,
                dtype=torch.complex64, device=device,
            )
        extras["x_t"] = torch.zeros(
            batch_size, self.token_dim, dtype=torch.float32, device=device,
        )
        return extras

    def prepare_step_context(self):
        """Precompute S5 discretization once per rollout."""
        return {
            "discrete": [block.discretize() for block in self.blocks]
        }

    def forward(self, stoch, deter, action, extra=None, step_context=None):
        if extra is None:
            extra = self.initial_extra(action.shape[0], action.device)
        batch_size = action.shape[0]
        action = action / torch.clip(torch.abs(action), min=1.0).detach()
        new_token = self.input_proj(torch.cat([stoch.reshape(batch_size, -1), action], dim=-1))

        new_extra: dict = {}
        x = new_token
        for i, block in enumerate(self.blocks):
            state = extra.get(f"ssm_state_{i}")
            if state is None or not state.is_complex():
                state = torch.zeros(
                    batch_size, block.heads, block.d_state,
                    dtype=torch.complex64, device=x.device,
                )
            discrete = None
            if step_context is not None:
                discrete = step_context["discrete"][i]
            x, state = block.step(x, state, discrete=discrete)
            new_extra[f"ssm_state_{i}"] = state

        # SSSM hidden-state policy input (proposal §3.1, R2I 2024).
        new_extra["x_t"] = x

        # Maintain a rolling token tape inside `deter` so obs/img heads see
        # the familiar fixed-length deter representation.
        tokens = deter.reshape(batch_size, self.tokens, self.token_dim)
        tokens = torch.cat([tokens[:, 1:], x.unsqueeze(1)], dim=1)
        tokens = self.output_norm(tokens)
        return tokens.reshape(batch_size, -1), new_extra


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
        # Re-init log_dt after the Dreamer global weight_init_ pass.
        # log_dt is a standalone Parameter (no `weight` attribute on its
        # parent module), so weight_init_ does NOT actually touch it; this
        # is defensive in case of any additional global init we add later.
        for block in deter_net.blocks:
            block._init_log_dt()

    @property
    def policy_feat_size(self):
        return self.flat_stoch + self._deter_net.token_dim

    def policy_feat(self, stoch, deter, extra=None):
        """SSSM hidden-state policy ``π(a | z_t, x_t)`` (proposal §3.1).

        Falls back to :py:meth:`get_feat` when ``extra`` lacks ``x_t``."""
        if extra is None or "x_t" not in extra:
            return self.get_feat(stoch, deter)
        stoch_flat = stoch.reshape(*stoch.shape[:-2], self.flat_stoch)
        return torch.cat([stoch_flat, extra["x_t"]], dim=-1)
