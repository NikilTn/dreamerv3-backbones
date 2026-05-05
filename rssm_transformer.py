"""Transformer-XL deterministic backbone for the Dreamer RSSM (Phase 2a).

Replaces the prior rolling-token-tape `nn.TransformerEncoder` design with a
proper Transformer-XL-style backbone:

- Custom multi-head attention with **relative positional encoding** (Dai et al.,
  2019, eq. 4): the per-token attention score is decomposed into four terms
  (content-content, content-position, position-content, position-position) so
  the attention does not depend on absolute positions of tokens in the cache.
- **Causal self-attention mask** (only relevant when ``L_q > 1`` — for our
  per-step training the new query is a single token and the mask is implicit).
- **Per-layer segment cache** held in ``extra["cache_input_{l}"]``: layer ``l``
  reads its own past inputs as keys/values, so attention spans many env steps
  without re-encoding the full history at every step (O(L) per step instead of
  O(L^2)).
- A separate ``extra["summary_cache"]`` of length ``tokens`` tracks the most
  recent final-layer outputs, so ``deter`` continues to be a fixed-size
  ``(B, tokens * token_dim)`` vector that the obs/img heads can consume.

The previous absolute ``pos_emb`` parameter is gone; relative position is
computed on the fly from a sinusoidal table inside the attention.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from rssm_base import CategoricalRSSM


def sinusoidal_position_emb(length: int, dim: int, device) -> torch.Tensor:
    """Sinusoidal positional embeddings of shape ``(length, dim)``.

    Index ``i`` is the embedding for position ``i``. Uses the chunked
    sin-then-cos layout (rather than interleaved) for vectorization simplicity;
    both are valid positional encodings.
    """
    half = dim // 2
    position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)  # (L, 1)
    inv_freq = torch.exp(
        torch.arange(0, half, device=device, dtype=torch.float32) * (-math.log(10000.0) / max(half, 1))
    )
    sinusoid = position * inv_freq.unsqueeze(0)  # (L, half)
    pe = torch.zeros(length, dim, device=device)
    pe[:, :half] = sinusoid.sin()
    if dim - half > 0:
        pe[:, half : 2 * half] = sinusoid.cos()
    return pe


class TransformerXLAttention(nn.Module):
    """Multi-head attention with Transformer-XL-style relative positional encoding.

    Score (Dai et al. 2019, eq. 4):
      A_ij = (W_q x_i + u)^T (W_k x_j) + (W_q x_i + v)^T (W_kr R_{i-j})

    where:
    - ``W_q``, ``W_k``, ``W_v`` project query/key/value content,
    - ``W_kr`` projects sinusoidal relative-position embeddings ``R_k``,
    - ``u``, ``v`` are learned per-head bias vectors (content/position bias),
    - ``R_k`` is the embedding for relative offset ``k = i - j``.

    For our per-step usage L_q == 1 (the latest token) so the four terms reduce
    to two einsums of cost O(L_kv * D).
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        assert d_model % n_heads == 0, (d_model, n_heads)
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.r_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Per-head content/position bias vectors. Initialized to zero so the
        # bias terms start as no-ops; learned during training.
        self.u = nn.Parameter(torch.zeros(self.n_heads, self.head_dim))
        self.v = nn.Parameter(torch.zeros(self.n_heads, self.head_dim))

        self.scale = 1.0 / math.sqrt(self.head_dim)

    def project_relative(self, length: int, device) -> torch.Tensor:
        """Projected relative-position keys for a fixed cache length."""
        rel = sinusoidal_position_emb(length, self.d_model, device).flip(0)
        return self.r_proj(rel).reshape(length, self.n_heads, self.head_dim)

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        rel_proj: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x_q: ``(B, L_q, d_model)`` — current-step queries.
        x_kv: ``(B, L_kv, d_model)`` — concatenated [memory || current] for K/V.

        For our typical use ``L_q = 1``: the query is the latest position and
        attends over the full ``L_kv`` cache (causal-by-construction since
        ``L_kv`` includes only the current and past tokens).
        """
        B, L_q, _ = x_q.shape
        L_kv = x_kv.shape[1]

        Q = self.q_proj(x_q).reshape(B, L_q, self.n_heads, self.head_dim)        # (B, L_q, H, D)
        K = self.k_proj(x_kv).reshape(B, L_kv, self.n_heads, self.head_dim)      # (B, L_kv, H, D)
        V = self.v_proj(x_kv).reshape(B, L_kv, self.n_heads, self.head_dim)      # (B, L_kv, H, D)

        # Sinusoidal embedding for offsets 0..L_kv-1, projected by W_kr. The
        # projected table is constant across all steps in one rollout, so the
        # deter network precomputes it once when possible.
        if rel_proj is None or rel_proj.shape[0] != L_kv:
            R = self.project_relative(L_kv, x_q.device)                          # (L_kv, H, D)
        else:
            R = rel_proj

        # Content score: (Q + u)^T K -> (B, H, L_q, L_kv).
        AC = torch.einsum("bqhd,bkhd->bhqk", Q + self.u.unsqueeze(0).unsqueeze(0), K)
        # Content-position score: (Q + v)^T R -> (B, H, L_q, L_kv).
        BD = torch.einsum("bqhd,khd->bhqk", Q + self.v.unsqueeze(0).unsqueeze(0), R)

        attn = (AC + BD) * self.scale  # (B, H, L_q, L_kv)

        # Causal mask: only meaningful when L_q > 1 (multi-query). For L_q=1
        # the single query already sits at the latest position and may attend
        # to every key without leaking future info.
        # NOTE: when L_q > 1, the relative-position embedding R above is only
        # correct for the last query (the others would need different offsets
        # per (query, key) pair). For our per-step training/inference path the
        # query is always a single new token so this is not exercised; if a
        # future change vectorizes the chunk forward into multi-query blocks,
        # implement the Dai et al. 2019 "skew" trick to reuse position scores
        # across queries.
        if L_q > 1:
            i = torch.arange(L_q, device=attn.device).unsqueeze(1)
            j = torch.arange(L_kv, device=attn.device).unsqueeze(0)
            forbid = j > (L_kv - L_q + i)
            attn = attn.masked_fill(forbid, float("-inf"))

        attn = torch.softmax(attn, dim=-1)
        out = torch.einsum("bhqk,bkhd->bqhd", attn, V)
        out = out.reshape(B, L_q, self.d_model)
        return self.out_proj(out)


class TransformerXLBlock(nn.Module):
    """Pre-norm Transformer-XL block: RelAttn + FFN with residuals."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int):
        super().__init__()
        self.norm1 = nn.RMSNorm(d_model, eps=1e-04, dtype=torch.float32)
        self.attn = TransformerXLAttention(d_model, n_heads)
        self.norm2 = nn.RMSNorm(d_model, eps=1e-04, dtype=torch.float32)
        self.ff = nn.Sequential(
            nn.Linear(d_model, ff_mult * d_model),
            nn.GELU(),
            nn.Linear(ff_mult * d_model, d_model),
        )

    def forward(
        self,
        x_q: torch.Tensor,
        x_kv: torch.Tensor,
        rel_proj: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attn_out = self.attn(self.norm1(x_q), self.norm1(x_kv), rel_proj=rel_proj)
        x_q = x_q + attn_out
        x_q = x_q + self.ff(self.norm2(x_q))
        return x_q


class TransformerXLDeter(nn.Module):
    """Transformer-XL deter network with per-layer segment caches.

    Per env step:
    - One new token is projected from ``(stoch, action)``.
    - For each layer ``l``: read ``extra["cache_input_l"]`` of shape
      ``(B, cache_length, token_dim)`` (past *inputs* to layer ``l``), build
      ``kv = [cache || current_input]``, compute TXL attention, FFN, and
      residual. Append the input we just consumed to the cache (drop oldest).
    - The final layer output (``(B, 1, token_dim)``) is appended to a separate
      ``extra["summary_cache"]`` of length ``tokens``, which is then RMS-
      normed and flattened to populate ``deter`` of shape
      ``(B, tokens * token_dim)`` — same shape the obs/img heads have always
      consumed.

    Memory horizon for attention is ``cache_length`` env steps. Memory horizon
    for the deter representation is ``tokens`` env steps. They are independently
    configurable: keep ``tokens`` small so ``deter`` stays compact for the
    posterior/prior heads, and bump ``cache_length`` for paper-faithful long
    Transformer-XL memory.
    """

    def __init__(self, deter, stoch, act_dim, config, act="SiLU"):
        super().__init__()
        self.supports_step_context = True
        self.tokens = int(config.tokens)
        assert deter % self.tokens == 0, (deter, self.tokens)
        self.token_dim = deter // self.tokens
        self.heads = int(config.heads)
        assert self.token_dim % self.heads == 0, (self.token_dim, self.heads)
        self.layers = int(config.layers)
        # Length of per-layer attention context. Defaults to ``tokens`` so a
        # bare config still runs (with the same effective memory horizon as the
        # old rolling-tape transformer); production configs should bump this
        # to actually exercise the long-memory attention.
        self.cache_length = int(getattr(config, "cache_length", self.tokens))
        assert self.cache_length >= self.tokens, (
            f"cache_length {self.cache_length} must be >= tokens {self.tokens}"
        )

        act_cls = getattr(torch.nn, act)
        self.input_proj = nn.Sequential(
            nn.Linear(stoch + act_dim, self.token_dim, bias=True),
            nn.RMSNorm(self.token_dim, eps=1e-04, dtype=torch.float32),
            act_cls(),
        )

        self.blocks = nn.ModuleList(
            [TransformerXLBlock(self.token_dim, self.heads, int(config.ff_mult)) for _ in range(self.layers)]
        )
        self.output_norm = nn.RMSNorm(self.token_dim, eps=1e-04, dtype=torch.float32)

    def initial_extra(self, batch_size, device):
        """Per-layer input cache + a final-layer output cache for the deter tape."""
        extras = {
            f"cache_input_{l}": torch.zeros(
                batch_size, self.cache_length, self.token_dim, dtype=torch.float32, device=device
            )
            for l in range(self.layers)
        }
        extras["summary_cache"] = torch.zeros(
            batch_size, self.tokens, self.token_dim, dtype=torch.float32, device=device
        )
        return extras

    def prepare_step_context(self):
        """Precompute relative-position projections shared by every step."""
        device = self.input_proj[0].weight.device
        length = self.cache_length + 1
        return {
            "rel_proj": [
                block.attn.project_relative(length, device)
                for block in self.blocks
            ]
        }

    def forward(self, stoch, deter, action, extra=None, step_context=None):
        if extra is None:
            extra = self.initial_extra(action.shape[0], action.device)
        batch_size = action.shape[0]
        action = action / torch.clip(torch.abs(action), min=1.0).detach()
        # New token from the current step's (stoch, action). Shape (B, 1, token_dim).
        new_token = self.input_proj(
            torch.cat([stoch.reshape(batch_size, -1), action], dim=-1)
        ).unsqueeze(1)

        new_extra: dict = {}
        layer_input = new_token
        for layer_idx, block in enumerate(self.blocks):
            cache_key = f"cache_input_{layer_idx}"
            cache = extra.get(cache_key)
            if cache is None or cache.shape[1] != self.cache_length:
                cache = layer_input.new_zeros(batch_size, self.cache_length, self.token_dim)
            # K/V context: cached past inputs to this layer + the current input.
            kv_input = torch.cat([cache, layer_input], dim=1)  # (B, cache_length + 1, token_dim)
            rel_proj = None
            if step_context is not None:
                rels = step_context.get("rel_proj")
                if rels is not None:
                    rel_proj = rels[layer_idx]
            layer_output = block(layer_input, kv_input, rel_proj=rel_proj)  # (B, 1, token_dim)
            # Update cache: drop oldest, append the input we just consumed.
            new_extra[cache_key] = torch.cat([cache[:, 1:], layer_input], dim=1)
            layer_input = layer_output

        # Update the deter summary tape: drop oldest, append final-layer output.
        summary = extra.get("summary_cache")
        if summary is None or summary.shape[1] != self.tokens:
            summary = layer_input.new_zeros(batch_size, self.tokens, self.token_dim)
        new_summary = torch.cat([summary[:, 1:], layer_input], dim=1)
        new_extra["summary_cache"] = new_summary

        new_deter = self.output_norm(new_summary).reshape(batch_size, -1)
        return new_deter, new_extra


class RSSM(CategoricalRSSM):
    def __init__(self, config, embed_size, act_dim, backbone_config):
        deter_net = TransformerXLDeter(
            int(config.deter),
            int(config.stoch) * int(config.discrete),
            act_dim,
            backbone_config,
            act=config.act,
        )
        super().__init__(config, embed_size, act_dim, deter_net)
