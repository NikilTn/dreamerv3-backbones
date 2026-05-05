# Backbone Fidelity Status

Per-backbone audit of how close each implementation is to the cited paper. **This file is the answer to "is backbone X paper-faithful yet?"**

For the implementation plan and session log, see [backbone_work_plan.md](backbone_work_plan.md).

Status legend:
- ✅ **Faithful** — matches the cited paper's design within reasonable engineering tolerance.
- 🟡 **Approximate** — runs and trains, but missing key paper-design elements. Results not directly comparable to paper claims.
- ❌ **Stub / broken** — placeholder; should not be used for any scientific claim.

---

## Summary Table

Updated 2026-05-03 after Phase 2d (S5 complex-diagonal MIMO) and Phase 3 (SSSM hidden-state policy `π(a | z_t, x_t)`).

**Per proposal §3.1**: all SSSM backbones now use the hidden-state policy `π(a | z_t, x_t)` rather than DreamerV3's default `π(a | z_t, h_t)`. Implemented via `policy_feat()` on each SSSM RSSM, which returns `cat([stoch_flat, x_t])`. World-model heads (reward, cont, value, recon, CPC) still use `get_feat(stoch, deter)` — only the policy switches.

| Backbone | File | Construction | Persistent state | Paper internals | Long-memory test (step 99) | Overall |
|---|---|---|---|---|---|---|
| `gru` / `rssm` | `rssm.py` | ✅ | ✅ (deter is the recurrent state) | ✅ block-GRU per Hafner et al. | ✅ passes | ✅ |
| `transformer` / `storm` | `rssm_transformer.py` | ✅ | ✅ per-layer segment caches in `extra["cache_input_*"]` + `extra["summary_cache"]` | ✅ Transformer-XL relative pos + per-layer segment cache + causal mask | ✅ passes with `cache_length≥horizon` | ✅ |
| `mamba` / `mamba2` | `rssm_mamba.py` | ✅ | ✅ `extra["ssm_state_*"]` SSD multi-head state `(B, nheads, headdim, d_state)` + `extra["conv_state_*"]` xBC causal conv state `(B, hidden + 2·d_state, kernel-1)` | ✅ Mamba-2-inspired SSD recurrence + official xBC block layout: bias-free `in_proj` emits `[z, xBC, dt]`, depthwise conv on `xBC` (so B/C also receive short-time mixing), scalar A per head, per-head Δ, gated RMSNorm before `out_proj`. Cross-checked against `state-spaces/mamba`'s `Mamba2.step()`. **NOT** the chunked SSD parallel scan or fused CUDA/Triton kernels (per-step Python loop is slower than `mamba_ssm` but identical output). Pure PyTorch. | ✅ passes at step 99 (diff ≈ 1.87e-4 vs 1e-5 threshold) via private `_call_deter_net` path; full `observe()`-path validation still pending | 🟡 (architecturally close; speed and chunked-SSD scan still gaps) |
| `s4` / `s3m` | `rssm_s4.py` | ✅ | ✅ `extra["ssm_state_*"]` persistent per-channel S4D state of shape `(B, dim, state_size)` | ✅ S4D-Lin diagonal SSM with per-channel A (init `-(1..N)`), learned per-channel Δ (log-uniform [1e-3, 0.1]), ZOH discretization, fixed B/C/D | ✅ passes (step-99 marker diff ≈ 1.76 vs 1e-5 threshold) | ✅ |
| `s5` | `rssm_s5.py` | ✅ | ✅ `extra["ssm_state_*"]` complex-diagonal MIMO SSM, **complex64** state of shape `(B, heads, d_state)` | ✅ Complex-diagonal A with HiPPO-N init (`Re(A) = -0.5`, `Im(A_n) = π·n`), complex B (`(H, N, P_in)`) and C (`(H, P_out, N)`), real D, per-pole-per-head Δ, ZOH discretization. Output `y = 2·Re(C·h) + D·x`. Pure-PyTorch native complex64 arithmetic. No parallel associative scan (per-step Python loop). | ✅ passes (step-99 marker diff ≈ 1.9 vs 1e-5 threshold; seeded test retention ≈0.89 over 99 steps from the slowest pole) | ✅ |

---

## Per-Backbone Detail

### `gru` / `rssm` — ✅ Faithful

- Block-GRU cell from DreamerV3 (`rssm.py:8-73`).
- 8 independent groups via `BlockLinear`, GRU gating (`reset = sigmoid`, `update = sigmoid(x − 1)` initialized to keep state).
- The `deter` of size 2048 IS the recurrent hidden state and persists naturally across env steps.
- This is the baseline against which all other backbones are measured.

### `transformer` / `storm` — ✅ Faithful (Phase 2a)

Cited paper: STORM (Zhang et al., 2023), with proposal §3.3 requirement for **Transformer-XL relative positional encoding + segment caching**.

What's there now (`rssm_transformer.py`, after Phase 2a):
- `TransformerXLAttention` with relative positional encoding via Dai et al.
  2019's eq. 4 four-term decomposition: content-content + content-position +
  bias-content + bias-position. Sinusoidal `R_k` projected by `W_kr`. Per-head
  learned `u`, `v` bias vectors added to `Q` before the K and R dot products.
- `TransformerXLBlock` with pre-norm RMSNorm and GELU FFN.
- `TransformerXLDeter` with per-layer segment caches:
  `extra["cache_input_l"]` for each layer holds the past inputs to that layer
  (so layer `l` attends over its own historical inputs).
  `extra["summary_cache"]` holds the last `tokens` final-layer outputs to
  populate the fixed-shape `deter` vector.
- Causal self-attention mask: implicit for `L_q=1` per-step path; explicit
  lower-triangular when `L_q > 1`.
- `transformer.cache_length` config knob in `_base_.yaml` (default 8 for
  backward compat; bump to ~128 for paper-faithful long-memory runs).
- No `pos_emb` parameter — relative position is computed on the fly from a
  sinusoidal table inside attention.

Notes:
- Per-step processing is O(L_kv * d_model) per layer (single-query attention
  over the cache). Training-time `observe` runs this in a Python loop over
  T env steps for total cost O(T * L_kv * d_model).
- Gradients flow through the cache during training (no manual detach), giving
  proper backprop through time across the chunk's posterior rollout. Inference
  via `act` runs under `torch.no_grad()` so no autograd graph is built.

Honest caveats (Phase 2a known gaps, do not block production):
- **Default `cache_length` is 8** (backward-compat). For paper-faithful long
  memory you must set `model.transformer.cache_length=128` (or similar) at
  launch time.
- **Effective training context is `min(batch_length, cache_length)`** because
  during training the cache builds up from zero at the start of every replay
  chunk and only reaches `batch_length` by the chunk's end. To exploit a
  cache larger than `batch_length=64`, you must also bump `batch_length`.
- **Multi-query (`L_q>1`) attention is not paper-faithful**. Our per-step
  training/inference path uses `L_q=1`, so this never triggers. If someone
  later vectorizes the per-step loop into a chunked forward over multi-token
  query blocks, the Dai et al. 2019 "skew trick" must be added to reuse
  relative-position scores across queries.

Historical cluster note: the pre-Phase-2a Transformer implementation hit an SDPA
"invalid configuration argument" at `batch_size=16, batch_length=64`. The current
Transformer-XL attention path is custom PyTorch attention rather than SDPA, so
re-run throughput pilots before carrying that old limitation into production
settings.

### `mamba` / `mamba2` — 🟡 Mamba-2-inspired SSD recurrence + official xBC block layout (Phase 2b)

Cited papers: Drama (Wang et al., 2025) for the Dreamer integration, built on **Mamba-2** (Dao & Gu, ICML 2024) — reference [5] in our project proposal. Proposal §3.2: "drop-in replacement of GRU with Mamba-2 SSSM, without the architectural differences posed by Drama."

**Honest framing.** This is "Mamba-2-inspired SSD-style recurrent step + the official `Mamba2` block layout." The per-step recurrence math and the block layout (bias-free in_proj emitting `[z, xBC, dt]`, depthwise conv over xBC, gated RMSNorm before out_proj) match the official `state-spaces/mamba` source. The chunked SSD parallel scan and fused CUDA/Triton kernels are intentionally **not** implemented — we use a per-step Python loop instead, which is **slower than the official `mamba_ssm` package** but produces identical output for the per-env-step Dreamer path. Calling this "faithful Mamba-2" without that caveat would overstate.

What's there now (`rssm_mamba.py`, after the Phase 2b xBC-layout refactor):

- **Block layout matches official ``Mamba2`` per-step path:**
  - `in_proj` is **bias-free** (matches official default) and emits
    `[z, xBC, dt]` with widths `[hidden, hidden + 2·d_state, nheads]`.
  - Causal depthwise conv operates on the **full xBC** (so B and C also
    receive short-time mixing through the same kernel as x), not just x.
  - Post-conv xBC is split into `x, B, C` and passed into `MambaSSD.step`.
  - Gated RMSNorm: `out = OutProj(RMSNorm(SiLU(z) · y))`.
  - Persistent extras: `conv_state` shape `(B, hidden + 2·d_state, kernel-1)`
    (covers xBC, not just x); `ssm_state` shape `(B, nheads, headdim, d_state)`.

- **`MambaSSD` per-step recurrence (matches official `Mamba2.step()` math):**
  - **Scalar `A` per head**: `A_log` shape `(nheads,)`, runtime
    `A = -exp(A_log)`, init from uniform `[1, 16]` per head (the SSD
    restriction that distinguishes Mamba-2 from Mamba-1's per-channel A).
  - **Per-head `dt_bias`** init via `softplus^{-1}(Δ)` for Δ log-uniform in
    `[1e-3, 0.1]`. Re-applied via `MambaSSD.special_init()` after
    `CategoricalRSSM.apply(weight_init_)`.
  - **`B, C` shared across heads** (`ngroups=1`), shape `(B, d_state)` each,
    sliced from post-conv xBC.
  - Per-step update: `Δ = softplus(dt_raw + dt_bias)`;
    `dA = exp(Δ · A)`; `dBx = einsum("bh,bn,bhp->bhpn", Δ, B, x_h)`;
    `state = state · dA[..., None, None] + dBx`;
    `y = einsum("bhpn,bn->bhp", state, C) + D · x_h`.

- **Defaults matching Mamba-2 paper**: `nheads=8`, `d_state=64`,
  `headdim = (expand · token_dim) / nheads`.

- **No `mamba_ssm` dependency**. All ops are standard PyTorch — runs on any
  PyTorch version, no CUDA-build risk on academic clusters.

- Cross-checked against:
  - `state-spaces/mamba/mamba_ssm/modules/mamba2.py::Mamba2.step`
  - `tommyip/mamba2-minimal/mamba2.py`

**Persistence test passes at step 99** (same horizon as
GRU/Transformer-XL/S4/S5): marker injected at step 0 produces deter diff
≈ 1.87e-4 at step 99 (vs the 1e-5 threshold). The slowest-decay head
(smallest Δ × |A|) reliably retains the marker at random init.

A separate `Mamba2BlockLayoutTest` enforces the official-layout invariants:
bias-free `in_proj`, conv over xBC width, scalar A per head, post-Dreamer-
init dt_bias actually in the [1e-3, 0.1] range.

Known gaps relative to the official `mamba_ssm.Mamba2` (do not block
production for the proposal's per-step Dreamer use):

- **No SSD chunked parallel scan during training.** The official Mamba-2's
  distinguishing algorithmic contribution is the SSD reformulation enabling
  parallel scan via chunked matmuls. We use the per-step recurrence directly.
  Identical output, **slower** throughput on long replay chunks.
- **No fused CUDA/Triton kernels** (`selective_state_update`, `RMSNormGated`).
  Pure PyTorch eager execution. **Slower** than `mamba_ssm` for any given
  workload; we accept this to avoid CUDA-build risk on the cluster.
- **`ngroups` fixed at 1.** B and C shared across all heads (single group).
  Official allows `ngroups > 1` for parameter scaling at very large model
  sizes; not measurably relevant at world-model scale.
- **Gated norm is `RMSNorm(SiLU(z) · y)`** instead of fused `RMSNormGated`.
  Mathematically equivalent.
- **ZOH B-discretization uses standard `B_bar = Δ · B`** (matches
  `mamba_ssm` reference); the exact ZOH `B_bar = (Δ A)^{-1} (exp(Δ A) - I) Δ B`
  would be more general but production implementations use the simplification.
- **Persistence test path is private `_call_deter_net`** with
  `stoch_mode="onehot"`. Useful for isolating the recurrence, but not the
  same as a full training-time `observe()`-path validation. The
  `ObserveIntegrationTest` covers `observe()` for the long-memory backbones
  with `stoch_mode="zero"`; an analogous Mamba `observe()` test is future work.

### `s4` / `s3m` — ✅ Faithful (Phase 2c)

Cited paper: Recall2Imagine (Samsami et al., 2024), which uses an S4-variant called S3M.

What's there now (`rssm_s4.py`, after Phase 2c):
- `S4Block` is a paper-faithful **S4D-Lin** diagonal SSM:
  - **Per-channel diagonal `A`** parameterized as `-exp(A_log)` (always
    negative, stable). Init: `A_n = -(1, 2, ..., state_size)` per channel
    (S4D-Lin real-form init).
  - **Per-channel learned timestep `Δ`**, init log-uniform in [1e-3, 0.1] via
    softplus^{-1} (standard S4D / Mamba convention).
  - **Per-step ZOH discretization for diagonal A**: `A_bar = exp(Δ * A)`,
    `B_bar = Δ * B`.
  - **Multi-channel SSM**: each of `dim` token channels has its own
    `state_size`-pole diagonal SSM. State shape per layer is
    `(B, dim, state_size)`. `B` shared across channels; `C` per-channel;
    `D` per-channel skip.
  - The SSM is **LTI**: `A, B, C, D, Δ` are all parameters, NOT input-
    dependent — this is the architectural distinction from Mamba's
    selective SSM.
- `step(x_t, state)` runs the SSM one env-step at a time with hidden state
  carried in `extra["ssm_state_*"]` across env steps.
- The legacy `forward(tokens)` path is preserved for any caller still using
  it (zero initial state).
- Persistence test: marker injected at step 0 still produces a step-99 diff
  of ~1.76 (vs the 1e-5 threshold) — robust slow-decay channels at random
  init.

Honest gaps relative to the broader S4 family (do not block production):
- **Real-diagonal A (S4D-Lin)** is what we implement, not complex-diagonal
  (S4D-Inv) or full DPLR (the original S4 with diagonal + low-rank A).
  S4D-Lin is published and is the simplest faithful S4-family member;
  full DPLR is a substantially heavier rewrite and was never needed by
  Recall2Imagine's S3M either.
- **No HiPPO-LegS init**; we use S4D-Lin which is the standard simpler init.
- **Per-step Python-loop recurrence during training**; no FFT-based parallel
  conv or parallel scan. For Dreamer's per-env-step path this matches the
  natural shape (observe runs a per-step loop anyway).

### `s5` — ✅ Faithful S5 (Phase 2d)

Cited papers: Hieros (Mattes et al., 2024) for the Dreamer integration, built on **S5** (Smith, Warrington & Linderman, ICLR 2023) — reference [7] in the project proposal.

What's there now (`rssm_s5.py`, after Phase 2d):
- `S5Block` is a **paper-faithful S5 cell** with the architectural
  distinctives that separate S5 from S4D and Mamba-2:
  - **Complex-diagonal A**: shape `(heads, d_state)` complex64. Stored as
    two real Parameters `A_re_log` and `A_im`; runtime
    `A = -exp(A_re_log) + i·A_im` so `Re(A) < 0` always (Hurwitz/stable).
  - **HiPPO-N approximation init**: `Re(A) = -0.5` (constant);
    `Im(A_n) = π · n` for `n ∈ [1, d_state]`. This is the standard simpler
    S5 init from the paper (the full HiPPO-LegS diagonalization is what
    the official `lindermanlab/S5` uses; the approximation is published
    and is what most simpler S5 ports adopt).
  - **MIMO structure**: each head runs ONE shared SSM cell with `d_state`
    complex poles. B is complex `(heads, d_state, head_dim)` mapping
    `R^{head_dim} → C^{d_state}`; C is complex `(heads, head_dim, d_state)`
    mapping `C^{d_state} → R^{head_dim}` (via the `2·Re(...)` projection).
  - **Per-pole-per-head learned Δ**: shape `(heads, d_state)`, init
    log-uniform `[1e-3, 0.1]` via `softplus^{-1}`. Each pole gets its own
    timestep — strictly more expressive than a shared Δ.
  - **ZOH discretization** (closed form for diagonal A):
    `A_bar = exp(Δ · A)`, `B_bar = (1/A) · (A_bar - 1) · B` per pole.
  - **Output**: `y = 2 · Re(C · h) + D · x`. The factor 2 accounts for the
    conjugate-paired structure that makes the projected output real-valued.
  - Real per-channel skip `D`.
- `S5Deter` stacks `S5Block`. Persistent `extra["ssm_state_*"]` is
  **complex64** of shape `(B, heads, d_state)` — unlike S4 or Mamba whose
  states are real, S5's state is intrinsically complex.
- `extra["x_t"]` (real, shape `(B, token_dim)`) for the SSSM hidden-state
  policy `π(a | z_t, x_t)` (Phase 3).

**Persistence test passes at step 99** with very strong margin: marker
injected at step 0 produces deter diff ≈ 1.9 at step 99 (vs the 1e-5
threshold). In the seeded test config, HiPPO-N init's `Re(A) = -0.5` combined
with the slowest learned `Δ ≈ 2.3e-3` gives per-step decay
`exp(-0.5 · 2.3e-3) ≈ 0.9989`, or ≈0.89 after 99 steps — extremely robust
long memory.

A separate `S5BlockLayoutTest` enforces the architectural invariants:
HiPPO-N init shape (`Re(A) = -0.5`, `Im(A_n) = π·n`), `Re(A) < 0` at all
times, complex64 state, Δ in the [1e-3, 0.1] range after the Dreamer global
init pass.

Honest gaps relative to the official `lindermanlab/S5` (do not block
production for the per-step Dreamer loop):

- **No parallel associative scan during training.** Per-step Python loop
  over T env steps. The official JAX implementation uses associative scan
  for O(log T) parallel depth; we use the natural O(T) recurrent form.
  Identical output for the per-step path; **slower** throughput on long
  replay chunks. For Dreamer's per-env-step `observe()` loop the per-step
  form is the natural shape anyway.
- **HiPPO-N approximation init**, not the full HiPPO-LegS diagonalization.
  Sets `Re(A) = -0.5` analytically; the full LegS init computes `A` from
  the LegS matrix's eigenvalue decomposition. The approximation is what
  simpler S5 ports use; most published S5 follow-ups adopt it as well.
- **B and C are random-init complex Gaussians** scaled by `1/sqrt(N)` and
  `1/sqrt(d_state)` respectively. Official S5 has multiple init choices
  including B = column of LegS; ours is the simplest valid init.

---

## Aliases — verified working (Phase 0 complete)

`rssm_factory.py` now has a `BACKBONE_CONFIGS` mapping that resolves each alias
to its canonical config block:

```python
BACKBONE_CONFIGS = {
    "storm":  "transformer",
    "mamba2": "mamba",
    "s3m":    "s4",
    ...
}
```

`tests/test_persistent_state.py::ConstructionTest` constructs every alias and
runs one `img_step` to confirm. All pass locally.

---

## How to Update This File

After each phase lands:

1. Update the row in the Summary Table (status column).
2. Update the corresponding "Per-Backbone Detail" section: move the relevant items from "What's needed for fidelity" up to "What's there now."
3. Cross-reference the commit hash in the changelog inside [backbone_work_plan.md](backbone_work_plan.md).
4. Re-run the persistence test: `.venv/bin/python -m unittest tests.test_persistent_state -v`. Paste output into the work plan changelog.
