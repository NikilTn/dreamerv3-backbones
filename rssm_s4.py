"""S4D-style diagonal state-space backbone for the Dreamer RSSM (Phase 2c).

Replaces the Phase-1 single-pole sigmoid-decay diagonal recurrence with a
proper S4D (diagonal-state-space) block:

- **Per-channel diagonal A** parameterized as ``-exp(A_log)`` (always negative,
  stable). Init: ``A_n = -(1, 2, ..., state_size)`` per channel — S4D-Lin
  real-form init (cheaper than full HiPPO; standard published variant).
- **Per-channel learned timestep Δ**, log-uniform in [0.001, 0.1] at init via
  softplus^{-1} (matches the Mamba/S4 initialization conventions).
- **Per-step zero-order-hold discretization** for diagonal A:
  ``A_bar = exp(Δ * A)``, ``B_bar = Δ * B`` (the standard simplification).
- **Multi-channel SSM**: each of ``dim`` token channels has its own
  ``state_size`` diagonal SSM. State shape per layer is
  ``(B, dim, state_size)``.
- ``B`` is a shared ``(state_size,)`` parameter broadcast across channels.
  ``C`` is per-channel ``(dim, state_size)``. ``D`` is a per-channel skip
  ``(dim,)``.
- The SSM is **LTI**: ``A, B, C, D, Δ`` are all parameters, NOT input-dependent
  — this is the key architectural distinction from Mamba's selective SSM.

Persistent SSM hidden state lives in ``extra["ssm_state_*"]`` and persists
across env steps via the unified extra interface from Phase 1.

Honest gaps relative to the broader S4 family (left for future work):
- **Real-diagonal A** (S4D-Lin), not complex-diagonal (S4D-Inv) or full DPLR
  (the original S4). S4D-Lin is published and is the simplest faithful
  S4-family member; full DPLR / S4 is a substantially heavier rewrite.
- **No HiPPO-LegS init**; we use S4D-Lin which is the standard simpler init.
- **Per-step Python-loop recurrence during training**; no FFT-based parallel
  conv or parallel scan. For Dreamer's per-env-step training/inference path
  this matches the natural per-step shape and is what's actually run anyway.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from rssm_base import CategoricalRSSM


class S4Block(nn.Module):
    """S4D-Lin diagonal SSM block with persistent per-channel state.

    Each channel of the ``dim``-dim input has its own ``state_size``-pole
    diagonal SSM with shared ``B`` and per-channel ``C``. The state of shape
    ``(B, dim, state_size)`` is carried across env steps via ``extra``.

    Block structure: pre-norm → SSM (per-channel diagonal recurrence) +
    per-channel skip ``D`` → linear mixer → residual add.
    """

    def __init__(self, dim: int, state_size: int):
        super().__init__()
        self.dim = int(dim)
        self.state_size = int(state_size)

        self.norm = nn.RMSNorm(dim, eps=1e-04, dtype=torch.float32)

        # A: per-channel diagonal, S4D-Lin real init A_n = -(1, 2, ..., N).
        # Parameterize as -exp(A_log) so A is strictly negative (stable).
        a_init = torch.arange(1, self.state_size + 1, dtype=torch.float32)
        a_init = a_init.unsqueeze(0).expand(self.dim, -1).contiguous()
        self.A_log = nn.Parameter(torch.log(a_init))

        # Δ: per-channel learned timestep. Init: log-uniform [1e-3, 0.1] via
        # softplus^{-1} so softplus(log_dt) draws Δ from that distribution
        # (S4D / Mamba convention).
        with torch.no_grad():
            dt = torch.exp(
                torch.rand(self.dim) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
            ).clamp(min=1e-4)
            inv_dt = dt + torch.log(-torch.expm1(-dt))  # softplus^{-1}(dt)
        self.log_dt = nn.Parameter(inv_dt)

        # B: shared across channels, (state_size,). Init Normal(0, 1).
        self.B = nn.Parameter(torch.randn(self.state_size))

        # C: per-channel, (dim, state_size). Init Normal with 1/sqrt(N) scale.
        self.C = nn.Parameter(torch.randn(self.dim, self.state_size) / math.sqrt(self.state_size))

        # D: per-channel skip gain.
        self.D = nn.Parameter(torch.ones(self.dim))

        # Final per-block linear mixer that mixes across channels (standard
        # S4 block-output linear; provides expressivity beyond the per-channel
        # SSM).
        self.mixer = nn.Linear(self.dim, self.dim, bias=True)

    def _discretize(self):
        """Returns (A_bar, B_bar) of shape (dim, state_size) each."""
        dt = F.softplus(self.log_dt).unsqueeze(-1)        # (dim, 1)
        A = -torch.exp(self.A_log)                        # (dim, state_size)
        A_bar = torch.exp(dt * A)                         # (dim, state_size)
        B_bar = dt * self.B.unsqueeze(0)                  # (dim, state_size)
        return A_bar, B_bar

    def step(self, x_t: torch.Tensor, state: torch.Tensor, discrete=None):
        """One env-step recurrent update.

        Args:
            x_t: ``(B, dim)`` input at the current step.
            state: ``(B, dim, state_size)`` hidden from previous env step.

        Returns:
            ``(B, dim)`` block output (with residual), updated state.
        """
        x = self.norm(x_t)
        if discrete is None:
            A_bar, B_bar = self._discretize()
        else:
            A_bar, B_bar = discrete
        # Recurrent update; broadcast (dim, state_size) over batch.
        new_state = A_bar.unsqueeze(0) * state + B_bar.unsqueeze(0) * x.unsqueeze(-1)
        # Per-channel SSM output: y_n = sum_k C_n[k] * h_n[k].
        y = (new_state * self.C.unsqueeze(0)).sum(dim=-1)
        y = y + self.D.unsqueeze(0) * x
        out = self.mixer(y)
        return x_t + out, new_state

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Sequence-mode scan with zero initial state.

        Kept for legacy callers; the new persistent path uses :py:meth:`step`.
        """
        x = self.norm(tokens)
        batch_size, steps, _ = x.shape
        state = x.new_zeros(batch_size, self.dim, self.state_size)
        A_bar, B_bar = self._discretize()
        outs = []
        for t in range(steps):
            state = A_bar.unsqueeze(0) * state + B_bar.unsqueeze(0) * x[:, t].unsqueeze(-1)
            y = (state * self.C.unsqueeze(0)).sum(dim=-1)
            y = y + self.D.unsqueeze(0) * x[:, t]
            outs.append(self.mixer(y))
        return tokens + torch.stack(outs, dim=1)


class S4Deter(nn.Module):
    """Deter network whose S4D recurrence persists across env steps.

    ``deter`` retains its rolling-tape shape ``(B, tokens * token_dim)`` so
    obs/img heads continue to consume a fixed-shape ``(B, deter)`` vector.
    The genuine per-channel SSM hidden state lives in ``extra`` of shape
    ``(B, token_dim, state_size)`` per layer.
    """

    def __init__(self, deter, stoch, act_dim, config, act="SiLU"):
        super().__init__()
        self.supports_step_context = True
        self.tokens = int(config.tokens)
        assert deter % self.tokens == 0, (deter, self.tokens)
        self.token_dim = deter // self.tokens
        self.state_size = int(config.state_size)
        act_cls = getattr(torch.nn, act)
        self.input_proj = nn.Sequential(
            nn.Linear(stoch + act_dim, self.token_dim, bias=True),
            nn.RMSNorm(self.token_dim, eps=1e-04, dtype=torch.float32),
            act_cls(),
        )
        self.blocks = nn.ModuleList(
            [S4Block(self.token_dim, self.state_size) for _ in range(int(config.layers))]
        )
        self.output_norm = nn.RMSNorm(self.token_dim, eps=1e-04, dtype=torch.float32)

    def initial_extra(self, batch_size, device):
        """Per-block S4D hidden state + per-step ``x_t`` for SSSM policy."""
        extras = {
            f"ssm_state_{i}": torch.zeros(
                batch_size, block.dim, block.state_size, dtype=torch.float32, device=device
            )
            for i, block in enumerate(self.blocks)
        }
        # x_t: post-final-block per-step output, used by π(a | z, x).
        extras["x_t"] = torch.zeros(
            batch_size, self.token_dim, dtype=torch.float32, device=device
        )
        return extras

    def prepare_step_context(self):
        """Precompute S4D discretization once per rollout."""
        return {
            "discrete": [block._discretize() for block in self.blocks]
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
            if state is None:
                state = x.new_zeros(batch_size, block.dim, block.state_size)
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
        deter_net = S4Deter(
            int(config.deter),
            int(config.stoch) * int(config.discrete),
            act_dim,
            backbone_config,
            act=config.act,
        )
        super().__init__(config, embed_size, act_dim, deter_net)

    @property
    def policy_feat_size(self):
        # SSSM hidden-state policy: cat([stoch_flat, x_t]) where x_t is
        # the post-final-block per-step output of shape (B, token_dim).
        return self.flat_stoch + self._deter_net.token_dim

    def policy_feat(self, stoch, deter, extra=None):
        """SSSM hidden-state policy ``π(a | z_t, x_t)`` (proposal §3.1).

        Falls back to :py:meth:`get_feat` when ``extra`` lacks ``x_t``
        (only happens in construction-time sizing checks)."""
        if extra is None or "x_t" not in extra:
            return self.get_feat(stoch, deter)
        stoch_flat = stoch.reshape(*stoch.shape[:-2], self.flat_stoch)
        return torch.cat([stoch_flat, extra["x_t"]], dim=-1)
