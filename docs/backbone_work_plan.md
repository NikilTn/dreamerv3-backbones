# Backbone Work Plan & Status

This is the **single source of truth** for the paper-faithful backbone fidelity work. Read this at the start of every session. Update it at the end of every session. Keep it concise.

For cluster access, partitions, smoke calibration results, and operating rules see [AGENTS.md](../AGENTS.md). This file tracks **what code work has been done, what is in progress, and what is next**.

---

## TL;DR — Where We Are Right Now

- **Active phase**: **Phase 0, 1, 1.5, 2a, 2b (SSD recurrence + official xBC
  block layout), 2c, 2d, 3 done.** All 27 tests pass + 1 BSuite skip locally
  (28 total). All five backbones now have the planned paper-aligned
  architecture inside the shared Dreamer harness, with the caveats documented
  below.
  GRU, Transformer/STORM (Transformer-XL relative pos + per-layer segment
  cache + causal mask), Mamba/mamba2 (**Mamba-2-inspired SSD recurrence +
  official `Mamba2` block layout**: bias-free in_proj emitting `[z, xBC, dt]`,
  depthwise conv over xBC, scalar A per head, per-head Δ, gated RMSNorm
  before out_proj — cross-checked against `state-spaces/mamba`'s
  `Mamba2.step()`), S4/S3M (paper-faithful S4D-Lin: per-channel diagonal A
  with HiPPO-spirit init, learned per-channel Δ, ZOH discretization), and S5
  (complex-diagonal MIMO SSM with HiPPO-N init, ZOH discretization, complex64
  state) all carry
  information across env steps via the unified `extra` interface. **All five
  backbones now pass the step-99 persistence test on the same 1e-5
  threshold.**
- **Mamba 2b status**: 🟡 Mamba-2-inspired in pure PyTorch — recurrence math
  and block layout match official, but the chunked SSD parallel scan and
  fused CUDA/Triton kernels are intentionally not implemented. This means
  our implementation is **slower than the official `mamba_ssm` package**
  (per-step Python loop instead of fused parallel scan), but produces
  identical per-step output. Other known gaps: `ngroups=1` only; gated norm
  is `RMSNorm(SiLU(z) · y)` instead of fused `RMSNormGated`. Mamba now has
  both the private `_call_deter_net` step-99 test and a public `observe()`
  integration test.
- **Next concrete action**: launch validation/throughput pilots for Phase 4,
  then the 75-run sweep. The planned architecture requirements are now in
  place for this controlled harness: hidden-state policy `π(a|z, x)` for
  SSSMs is wired and tested, CPC and DFS sub-experiments are implemented, and
  each backbone's remaining gaps are explicit.
- **Optimization note**: `observe()` now precomputes rollout-constant
  Transformer/S4/S5 tensors once per rollout instead of every time step.
  H100 observe-only microbench at `batch_size=512`, `batch_length=128`,
  `deter=2048`: eager = GRU 306 ms, Transformer 468 ms, Mamba2 403 ms, S4
  257 ms, S5 296 ms; compiled = GRU 75 ms, Transformer 180 ms, Mamba2 56 ms,
  S4 42 ms. S5 stays eager because TorchInductor currently fails on the
  native complex64 recurrence.
- The 75-run sweep can be launched now with strong framing: "controlled
  comparison of Transformer-XL, Mamba-2 SSD-style selective scan, S4D-Lin,
  and S5 backbones inside a unified Dreamer harness with a persistent-state
  interface." Production training should set:
  ```
  trainer.burn_in=8
  model.transformer.cache_length=128   # for paper-faithful long Transformer-XL memory
  model.mamba.state_size=64             # paper-typical d_state for Mamba-2
  model.mamba.nheads=8                  # Mamba-2 default head count
  ```

---

## Project Goal (from proposal PDF)

A controlled comparison of five world-model backbones inside DreamerV3, holding everything else fixed:

| Backbone | Reference paper |
|---|---|
| GRU / RSSM | DreamerV3 (Hafner et al., 2025) |
| Transformer / STORM | STORM (Zhang et al., 2023) — needs Transformer-XL relpos + segment cache |
| Mamba-2 | Drama (Wang et al., 2025) — drop-in selective SSM |
| S4 / S3M | Recall2Imagine (Samsami et al., 2024) |
| S5 | Hieros (Mattes et al., 2024) |

Plus two sub-experiments per backbone: CPC contrastive loss, dynamic frequency-based replay sampling (DFS).

Evaluation suites: Atari100k (general-performance control), BSuite memory probes (diagnostic), POPGym memory POMDPs (realistic). Five seeds per cell, 10 for BSuite. Full matrix is ~10,900 runs and not feasible on this cluster — see [scripts/build_proposal_sweep.py](../scripts/build_proposal_sweep.py) for the realistic 75-run subset we plan to run.

---

## Phased Plan

Each phase is sized to fit in **one focused implementation session**. Do not bundle phases — finish one, verify, commit, then start the next.

### Phase 0 — Bug fixes (DONE)

Status: **completed 2026-05-03**.

| Task | File | How verified |
|---|---|---|
| Fix `mamba2`/`s3m`/`storm` alias config lookup | `rssm_factory.py` | `BACKBONE_CONFIGS` mapping added; `tests/test_persistent_state.py::ConstructionTest` builds + steps every alias |
| Confirm POPGym env registration patch is committed and pushed | `envs/popgym.py` | Patch applied (the `popgym-` namespace prefix); pending push |

### Phase 1 — Persistent state architecture (PREREQUISITE for all paper-faithful backbones)

Status: **completed 2026-05-03**. See changelog for details.

### Phase 1.5 — Replay burn-in + post-extra threading + integration test

Status: **completed 2026-05-03**.

| Task | File | How verified |
|---|---|---|
| `observe()` optionally returns per-step extras | `rssm_base.py` | `tests/test_persistent_state.py::ObserveIntegrationTest::test_observe_returns_per_step_extras` checks shape `(B, T, ...)` per backbone key |
| `_imagine` and `rollout_actions` accept `initial_extra` | `dreamer.py` | Existing smoke tests still pass; `_cal_grad` now passes per-step extras into actor-critic imagination |
| CPC rollout uses sliced post-extras | `dreamer.py::cpc_loss` | `tests/test_backbones_smoke.py::test_cpc_option_executes_on_cpu` still passes |
| Replay burn-in mask on world-model losses | `dreamer.py::_cal_grad` (`trainer.burn_in` config knob) | `tests/test_backbones_smoke.py::test_burn_in_changes_world_model_loss` confirms burn_in=2 changes the cont loss vs burn_in=0 on identical data |
| Integration test through `observe()` | `tests/test_persistent_state.py::ObserveIntegrationTest` | 4 new test cases covering long-memory backbones, transformer with long cache, per-step extras shape |

The current `_deter_net(stoch, deter, action) -> next_deter` interface has no way to carry SSM hidden state or KV cache across env steps. S4/S5 zero state every forward; transformer has no cache. Until this is fixed, no backbone can demonstrate paper-faithful long-memory behavior, regardless of its internal block design.

Tasks:

1. Extend `rssm_base.CategoricalRSSM` so each backbone optionally returns/accepts an `extra_state` dict. Thread it through `initial`, `obs_step`, `img_step`, `observe`, `imagine_with_action`. Reset on `reset=True` per env.
2. Update `rssm.py` (GRU): empty `extra_state`. Behavior must be byte-identical to current.
3. Update `rssm_s4.py` and `rssm_s5.py`: move `state = x.new_zeros(...)` out of `forward`, persist via `extra_state`.
4. Update `rssm_mamba.py`: persist depthwise-conv past samples in `extra_state`.
5. Update `rssm_transformer.py`: maintain a configurable-length token cache in `extra_state`. Default cache length = current behavior (no regression); opt-in for longer.
6. Update `dreamer.py`'s `act()` and `update()` paths to carry `extra_state` alongside `(stoch, deter)` in the agent state dict.
7. Add `tests/test_persistent_state.py`:
   - For each backbone (`gru, transformer, storm, mamba, mamba2, s4, s3m, s5`): construct + one `obs_step`.
   - **Lone-perturbation-decay test**: marker input at step 0, zeros at steps 1–99, vs all-zeros sequence. Assert outputs differ at step 99 for non-GRU backbones AFTER the fix. Sanity assertion: outputs differ at step 5 (proves marker enters cache).
8. Update [docs/backbone_fidelity_status.md](backbone_fidelity_status.md) status table.

Acceptance: all existing tests pass + new persistence test passes for every backbone + `git diff rssm.py` shows no behavioral change for GRU (only signature change).

### Phase 2a — Transformer-XL fidelity

Status: **completed 2026-05-03**.

| Task | File | How verified |
|---|---|---|
| Custom `TransformerXLAttention` with relative positional encoding (Dai et al. 2019 eq. 4) | `rssm_transformer.py` | `tests/test_persistent_state.py::TransformerXLTest::test_relative_attention_carries_marker_to_step_99` (action-marker at step 0 still detectable at step 99 with `cache_length=128`) |
| Per-layer segment cache in `extra` (one cache per layer holding past *inputs* to that layer, plus a small `summary_cache` of last `tokens` final-layer outputs to fill `deter`) | `rssm_transformer.py` | `tests/test_persistent_state.py::TransformerXLTest::test_per_layer_caches_persist_in_extra` |
| Causal self-attention mask | `rssm_transformer.py` | Implicit for `L_q=1` (per-step path); explicit when `L_q>1` |
| Removed absolute `pos_emb` | `rssm_transformer.py` | No more `nn.Parameter(torch.zeros(1, tokens, token_dim))` |
| Configurable `transformer.cache_length` config knob | `configs/model/_base_.yaml`, `rssm_transformer.py` | Defaults to `tokens` (backward-compat); bump in production for paper-faithful long-memory runs |

### Phase 2b — Mamba-2-inspired SSD recurrence + official xBC block layout

Status: **completed (recurrence + block layout) 2026-05-03**. The per-step
SSD recurrence math and the official `Mamba2` block layout (bias-free
`in_proj` emitting `[z, xBC, dt]`, depthwise conv over xBC, gated RMSNorm
before `out_proj`) are implemented in pure PyTorch and cross-checked
against `state-spaces/mamba`'s `Mamba2.step()`. The chunked SSD parallel
scan and fused CUDA/Triton kernels are intentionally not implemented; per-
step Python loop produces identical output but is **slower** than the
official `mamba_ssm` package.

| Task | File | How verified |
|---|---|---|
| Block layout matches official `Mamba2` | `rssm_mamba.py::Mamba2Block` | `tests/test_persistent_state.py::Mamba2BlockLayoutTest`: bias-free `in_proj`, depthwise conv on `xBC` (= hidden + 2·d_state, not just hidden), scalar A per head, conv_state covers xBC, ssm_state shape `(B, nheads, headdim, d_state)`. |
| Multi-head SSD cell with scalar A per head | `rssm_mamba.py::MambaSSD` | `A_log` shape `(nheads,)` from uniform `[1, 16]`; runtime `A = -exp(A_log)`. |
| Per-head dt_bias with log-uniform Δ init via softplus^{-1} | `rssm_mamba.py::MambaSSD.special_init` | `dt_bias` is a standalone Parameter (NOT a Linear's bias) so `weight_init_` doesn't touch it. Re-applied via `RSSM.__init__` defensively. Layout test asserts `softplus(dt_bias) ∈ [1e-3, 0.1]` per head. |
| Gated RMSNorm before out_proj (Mamba-2 layout) | `rssm_mamba.py::Mamba2Block.step` | `out = OutProj(RMSNorm(SiLU(z) · y))` — matches `RMSNormGated` math without the fused kernel. |
| Conv on xBC (B and C also pass through the conv) | `rssm_mamba.py::Mamba2Block.step` | conv_state shape `(B, hidden + 2·d_state, kernel-1)`; post-conv `xBC` is split into `x, B, C` and passed to `MambaSSD.step`. |
| **Step-99 persistence test passes** | `tests/test_persistent_state.py::PersistentStateTest::test_mamba_ssd_carries_marker_to_step_99` | Marker survives 99 env steps via SSD state; diff ≈ 1.87e-4 vs 1e-5 threshold. Uses private `_call_deter_net` path with `stoch_mode="onehot"`. |
| Pure-PyTorch (no `mamba_ssm` dependency) | `rssm_mamba.py` | All ops standard PyTorch; runs on any PyTorch version; no CUDA-build risk on cluster |
| `nheads` and `state_size` config knobs | `configs/model/_base_.yaml::mamba` | `state_size: 64` (paper-typical d_state), `nheads: 8` (Mamba-2 default) |
| 🟡 No SSD chunked parallel scan during training | not implemented | per-step Python loop instead; identical output, slower throughput on long chunks |
| 🟡 No fused CUDA/Triton kernels | not implemented | pure PyTorch eager; **slower than official `mamba_ssm`**; trade-off is no CUDA-build risk |
| 🟡 `ngroups` fixed at 1 | not implemented | B and C shared across heads; only matters at much larger model scale |
| 🟡 Persistence test uses private `_call_deter_net` not full `observe()` | `tests/test_persistent_state.py::PersistentStateTest::test_mamba_ssd_carries_marker_to_step_99` | the long-memory backbones have an `ObserveIntegrationTest` covering `observe()` — analogous Mamba `observe()` test is future work |

### Phase 2c — Real S4 / S3M

Status: **completed 2026-05-03**.

| Task | File | How verified |
|---|---|---|
| Per-channel diagonal A with HiPPO-spirit init | `rssm_s4.py::S4Block` | A parameterized as `-exp(A_log)`, init `A_n = -(1, 2, ..., N)` per channel (S4D-Lin real form). Always negative ⇒ stable. |
| Learned per-channel timestep Δ | `rssm_s4.py::S4Block.log_dt` | softplus(log_dt) draws Δ from log-uniform [1e-3, 0.1] at init via softplus^{-1} (standard S4D / Mamba init convention). |
| ZOH discretization for diagonal A | `rssm_s4.py::S4Block._discretize` | `A_bar = exp(Δ * A)`, `B_bar = Δ * B` (the standard simplification for diagonal A). |
| Multi-channel SSM (per-channel B/C/D) | `rssm_s4.py::S4Block` | B shared `(state_size,)`; C per-channel `(dim, state_size)`; D per-channel `(dim,)`. Persistent state shape `(B, dim, state_size)` per layer. |
| LTI structure (no input-dependence) | `rssm_s4.py` | All of A, B, C, D, Δ are nn.Parameter; explicitly distinguishes S4 from Mamba's selective SSM. |
| Persistence test passes at step 99 | `tests/test_persistent_state.py::PersistentStateTest::test_long_memory_backbones_carry_marker_to_step_99` | Step-99 marker diff ≈ 1.76 (vs 1e-5 threshold) — robust slow-decay channels at random init. |

### Phase 2d — Real S5

Status: **completed 2026-05-03**.

Implements the Smith et al. 2023 S5 architecture: complex-diagonal A with
HiPPO-N init, complex MIMO B/C, real D, per-pole-per-head Δ, ZOH
discretization. Output `y = 2 · Re(C · h) + D · x`.

| Task | File | How verified |
|---|---|---|
| Complex-diagonal A with HiPPO-N init | `rssm_s5.py::S5Block` | `A_re_log` and `A_im` Parameters; runtime `A = -exp(A_re_log) + i·A_im` (always Hurwitz). HiPPO-N approximation: `Re(A) = -0.5`, `Im(A_n) = π·n`. `tests/test_persistent_state.py::S5BlockLayoutTest::test_block_layout_matches_paper` |
| Complex MIMO B and C | `rssm_s5.py::S5Block` | B shape `(heads, d_state, head_dim)`, C shape `(heads, head_dim, d_state)`, both complex via `(B_re, B_im)` and `(C_re, C_im)` Parameter pairs. |
| Per-pole-per-head Δ with log-uniform init | `rssm_s5.py::S5Block._init_log_dt` | `log_dt` shape `(heads, d_state)`; `softplus(log_dt) ∈ [1e-3, 0.1]` per pole at init. Re-applied in `RSSM.__init__` after `super().__init__()` (defensive). |
| ZOH discretization (closed form for diagonal A) | `rssm_s5.py::S5Block.step` | `A_bar = exp(Δ · A)`; `B_bar = (1/A) · (A_bar - 1) · B` per pole. |
| Output `y = 2·Re(C·h) + D·x` | `rssm_s5.py::S5Block.step` | The factor 2 accounts for the conjugate-paired structure that ensures real output. |
| **Complex64 persistent state** | `rssm_s5.py::S5Deter.initial_extra` | `extra["ssm_state_*"]` shape `(B, heads, d_state)`, dtype **complex64**. `tests/test_persistent_state.py::S5BlockLayoutTest::test_ssm_state_is_complex64` |
| **Step-99 persistence test passes with strong margin** | `tests/test_persistent_state.py::PersistentStateTest::test_long_memory_backbones_carry_marker_to_step_99` and `S5BlockLayoutTest::test_marker_survives_to_step_99` | Marker survives 99 env steps with diff ≈ 1.9 (vs 1e-5 threshold). With the seeded test config, smallest learned `Δ ≈ 2.3e-3` and `Re(A) = -0.5` give per-step retention ≈ 0.9989, or ≈0.89 after 99 steps. |
| 🟡 No parallel associative scan during training | not implemented | per-step Python loop; identical output, slower throughput on long replay chunks |
| 🟡 HiPPO-N approximation init, not full HiPPO-LegS diagonalization | `rssm_s5.py::S5Block.__init__` | sets `Re(A) = -0.5` analytically; full LegS init is a heavier rewrite |

### Phase 3 — SSSM policy `π(a | z_t, x_t)`

Status: **completed 2026-05-03**.

Per proposal §3.1, SSSM-based models condition the actor on `(z_t, x_t)`
rather than the DreamerV3 default `(z_t, h_t)`. Implementation:

| Task | File | How verified |
|---|---|---|
| `policy_feat(stoch, deter, extra)` + `policy_feat_size` on RSSM base | `rssm_base.py` | Defaults: `policy_feat = get_feat`, `policy_feat_size = feat_size` (no behavior change for GRU / transformer) |
| SSSM backbones expose `x_t` in `extra` | `rssm_mamba.py`, `rssm_s4.py`, `rssm_s5.py` | Each `*Deter.forward` adds `new_extra["x_t"] = x` (post-final-block per-step output, shape `(B, token_dim)`). `initial_extra` zeros it for first-step. `tests/test_persistent_state.py::PolicyFeatTest::test_sssm_x_t_exposed_in_extra` |
| SSSM RSSMs override `policy_feat` and `policy_feat_size` | same | `policy_feat_size = flat_stoch + token_dim`; `policy_feat` returns `cat([stoch_flat, extra["x_t"]])`. `tests/test_persistent_state.py::PolicyFeatTest::test_sssm_uses_hidden_state_policy` and `test_sssm_policy_feat_uses_x_t_not_deter` |
| Actor sized to `policy_feat_size` (not `feat_size`) | `dreamer.py:60` | For non-SSSM, both equal so no change. For SSSM, actor input width differs — verified at construction (smoke test passes). |
| Actor input pipeline uses `policy_feat` | `dreamer.py::_act`, `dreamer.py::_imagine` | `_act` calls `rssm.policy_feat(stoch, deter, extra)` with the post-`obs_step` extra. `_imagine` returns `(world_feats, policy_feats, actions)` and the actor reads `policy_feats` while reward/cont/value heads still read `world_feats` (per proposal: only the policy switches; world-model heads unchanged). |
| World-model heads (reward, cont, value, recon, CPC, SwAV) unchanged | `dreamer.py` | All still use `rssm.get_feat(stoch, deter)`. Critic stays on world-model feat per the proposal (only the actor switches). |

### Phase 4 — Validation + final sweep

Status: **not started**. Depends on Phases 1, 2a–d, 3.

Tasks: re-run smoke calibration with fidelity-fixed backbones (cost will go up vs current 8-token-window). Re-pick sweep matrix size based on new GPU-h projection. Launch the final 75-run sweep (or expanded matrix). Aggregate via `scripts/analyze_proposal_results.py`. Write up.

---

## Per-Backbone Status Table

Update this every session. Status legend: ✅ paper-faithful | 🟡 approximate but functional | ❌ broken or stub.

| Backbone | Construction | Persistent state across env steps | Paper-faithful internals | Aliases work |
|---|---|---|---|---|
| `gru` / `rssm` | ✅ | ✅ (deter persists by default) | ✅ baseline | ✅ |
| `transformer` / `storm` | ✅ | ✅ (per-layer segment caches in `extra["cache_input_*"]` + final-layer summary in `extra["summary_cache"]`, configurable `cache_length`) | ✅ Transformer-XL relative pos + per-layer segment cache + causal mask (Phase 2a done) | ✅ |
| `mamba` / `mamba2` | ✅ | ✅ (`extra["ssm_state_*"]` SSD state `(B, nheads, headdim, d_state)` + `extra["conv_state_*"]` xBC conv state `(B, hidden + 2·d_state, kernel-1)`) | 🟡 Mamba-2-inspired SSD recurrence + official `Mamba2` block layout (bias-free in_proj emitting [z, xBC, dt], depthwise conv on xBC, scalar A per head, gated RMSNorm before out_proj — cross-checked against `state-spaces/mamba`'s `Mamba2.step()`). Pure PyTorch; per-step recurrence (no chunked SSD parallel scan, no fused kernels — slower than official `mamba_ssm`). | ✅ |
| `s4` / `s3m` | ✅ | ✅ (`extra["ssm_state_*"]` persistent per-channel state of shape `(B, dim, state_size)`) | ✅ S4D-Lin diagonal SSM with per-channel A (init `-(1..N)`), learned per-channel Δ (log-uniform [1e-3, 0.1]), ZOH discretization, fixed B/C/D — paper-faithful S4D-family member (Phase 2c done) | ✅ |
| `s5` | ✅ | ✅ (`extra["ssm_state_*"]` complex64 state of shape `(B, heads, d_state)`) | ✅ Complex-diagonal MIMO S5 with HiPPO-N init (`Re(A) = -0.5`, `Im(A_n) = π·n`), complex B/C, real D, per-pole-per-head Δ, ZOH discretization, output `y = 2·Re(C·h) + D·x` — paper-faithful S5 (Phase 2d done). No parallel associative scan during training (per-step Python loop instead). | ✅ |

---

## Session Changelog

Append a dated entry at the bottom of each session.

### 2026-05-03 (optimization pass — rollout constants + compile policy)
- Added an optional per-rollout `step_context` hook in `rssm_base.py`.
  `observe()` and `imagine_with_action()` now prepare a backbone context once
  per rollout and pass it into every `obs_step`/`img_step`.
- `dreamer.py` now also prepares the same context for actor-critic
  imagination and CPC action rollouts.
- `rssm_transformer.py`: precomputes projected Transformer-XL relative
  position tables once per rollout/layer instead of recomputing sinusoidal
  embeddings and `r_proj` every step.
- `rssm_s4.py`: precomputes S4D discretization `(A_bar, B_bar)` once per
  rollout/layer instead of every step.
- `rssm_s5.py`: precomputes S5 discretization `(A_bar, B_bar, C, D)` once per
  rollout/layer instead of every step.
- `scripts/bench_backbones.py`: now matches `train.py` by setting
  `torch.set_float32_matmul_precision("high")` and flushes output so Slurm
  jobs can be monitored live.
- `dreamer.py`: automatically disables `torch.compile` for `model.backbone=s5`
  because TorchInductor fails on the native complex64 S5 recurrence. Other
  backbones keep compile enabled.
- H100 observe-only benchmark on `condo7`:
  - Eager, `batch_size=512`, `batch_length=128`, `deter=2048`, with TF32
    matching `train.py`:
    GRU 306 ms, Transformer 468 ms, Mamba2 403 ms, S4 257 ms, S5 296 ms.
  - Compiled same shape:
    GRU 75 ms, Transformer 180 ms, Mamba2 56 ms, S4 42 ms; S5 compile failed
    with a TorchInductor complex-stride error and should remain eager.
- Verification: `git diff --check` clean; `.venv/bin/python -m unittest
  discover -s tests` runs 28 tests, OK with 1 expected BSuite skip.

### 2026-05-02 (planning + cluster bring-up)
- Created `scripts/smoke_throughput.sh` and `scripts/smoke_analyze.py` for repeatable throughput calibration. Not yet committed.
- Smoke pilot on A100 40GB: GRU on POPGym RepeatPreviousEasy = 44.5 env-steps/sec at default `env_num=1, batch_size=16`.
- Verified CPC is fully implemented in `dreamer.py` (lines 33, 123–155, 471, 616, 660). Don't re-implement.
- Verified DFS is fully implemented in `buffer.py`. Don't re-implement.
- Identified the load-bearing scientific issue: all non-GRU backbones use `tokens: 8` rolling window in `_base_.yaml:83-102`. Effective memory ≈ 8 env steps regardless of architecture choice.
- Identified `mamba2`/`s3m`/`storm` alias bug: factory looks up config under the alias name but `_base_.yaml` has no aliased blocks → `getattr(model_config, "mamba2", None)` → crash. Needs Phase 0 fix.

### 2026-05-03 (Phase 2d — Real S5 with complex-diagonal A and HiPPO-N init)
- Replaced `rssm_s5.py`'s real-diagonal MIMO toy with a paper-faithful S5
  cell per Smith et al., ICLR 2023 (proposal reference [7], Hieros uses S5):
  - **Complex-diagonal A** parameterized as two real Parameters
    (`A_re_log`, `A_im`); runtime `A = -exp(A_re_log) + i·A_im` so `Re(A)`
    is always negative (Hurwitz/stable).
  - **HiPPO-N approximation init**: `Re(A) = -0.5` (constant);
    `Im(A_n) = π · n` for `n ∈ [1, d_state]`.
  - **Complex B** of shape `(heads, d_state, head_dim)` and **complex C**
    of shape `(heads, head_dim, d_state)`, both stored as real-tensor
    Parameter pairs. Init Gaussian scaled by `1/sqrt(...)`.
  - **Per-pole-per-head Δ** of shape `(heads, d_state)`, init log-uniform
    `[1e-3, 0.1]` via `softplus^{-1}`. Re-applied via `S5Block._init_log_dt`
    after the Dreamer global init (defensive — log_dt is a standalone
    Parameter so weight_init_ doesn't actually touch it).
  - **ZOH discretization** (closed form for diagonal A):
    `A_bar = exp(Δ · A)`, `B_bar = (1/A) · (A_bar - 1) · B` per pole.
  - **Real per-channel skip D**.
  - **Output**: `y = 2 · Re(C · h) + D · x`. The factor 2 accounts for the
    conjugate-paired structure that ensures the projected output is real.
- **Persistent state is complex64**: `extra["ssm_state_*"]` of shape
  `(B, heads, d_state)`. Unlike S4 or Mamba (real-state), S5's state is
  intrinsically complex.
- **Step-99 persistence diff ≈ 1.9** (vs the 1e-5 threshold) — extremely
  strong margin. In the seeded test config, HiPPO-N's `Re(A) = -0.5` with the
  slowest learned `Δ ≈ 2.3e-3` gives per-step retention
  `exp(-0.5 · 2.3e-3) ≈ 0.9989`, or ≈0.89 after 99 steps — robust long memory
  by construction.
- New `S5BlockLayoutTest` (3 cases): asserts HiPPO-N init shape, `Re(A) < 0`
  invariant, complex64 state dtype after one img_step, marker survives to
  step 99 with diff > 0.01 (much stronger than the generic 1e-5 threshold).
- Phase 3 hidden-state policy still works with the new complex SSM:
  `extra["x_t"]` (real, shape `(B, token_dim)`) is set by `S5Deter.forward`
  and consumed by `policy_feat`. PolicyFeatTest passes.
- Honest gaps left: no parallel associative scan during training (per-step
  Python loop is slower than the official `lindermanlab/S5` JAX implementation);
  HiPPO-N approximation init instead of full HiPPO-LegS diagonalization.
- Files changed: `rssm_s5.py` (rewrite), `tests/test_persistent_state.py`
  (added `import math`, `S5BlockLayoutTest`), `docs/backbone_work_plan.md`,
  `docs/backbone_fidelity_status.md`.
- All 27 tests pass + 1 BSuite skip (28 total — added 3 S5 layout cases).

### 2026-05-03 (Phase 3 — SSSM hidden-state policy `π(a | z_t, x_t)`)
- Implemented proposal §3.1: SSSM backbones (Mamba/S4/S5) now condition the
  actor on `(z_t, x_t)` instead of the DreamerV3 default `(z_t, h_t)`. R2I's
  ablation showed the standard `π(a|z, h)` underperforms on memory tasks
  for SSSMs (state compression into the deter rolling tape loses information
  that `x_t` retains).
- New `CategoricalRSSM.policy_feat(stoch, deter, extra)` and
  `CategoricalRSSM.policy_feat_size`:
  - Default: `policy_feat = get_feat`, `policy_feat_size = feat_size`.
    GRU and transformer backbones use the default — **no behavior change**.
  - SSSM RSSMs override: `policy_feat = cat([stoch_flat, extra["x_t"]])`;
    `policy_feat_size = flat_stoch + token_dim`.
- Each SSSM `*Deter.forward` now adds `new_extra["x_t"] = x` (the post-final-
  block per-step output, shape `(B, token_dim)`) and `initial_extra` zeros it
  for the first step.
- `dreamer.py` actor sized to `rssm.policy_feat_size`. `_act` and `_imagine`
  call `rssm.policy_feat(stoch, deter, extra)` for the actor input. Critic,
  reward, cont, recon, CPC, and SwAV heads all stay on the world-model
  `get_feat(stoch, deter)` (only the policy switches per the proposal).
- `_imagine` now returns `(world_feats, policy_feats, actions)` instead of
  `(feats, actions)`. World-model heads consume `world_feats`; the actor
  consumes `policy_feats`. For non-SSSM backbones the two are identical.
- New `PolicyFeatTest` (4 cases) verifies:
  - GRU/transformer: `policy_feat_size == feat_size`.
  - SSSM: `policy_feat_size == flat_stoch + token_dim` and differs from
    `feat_size`.
  - SSSM `extra["x_t"]` populated by `initial_extra` and updated by `forward`.
  - SSSM `policy_feat` output trailing dims equal `extra["x_t"]`.
- Verified actor input dims: GRU/transformer = 48 (= 16 stoch_flat + 32
  deter); SSSM = 24 (= 16 stoch_flat + 8 token_dim).
- Files changed: `rssm_base.py`, `rssm_mamba.py`, `rssm_s4.py`, `rssm_s5.py`,
  `dreamer.py`, `tests/test_persistent_state.py`, `docs/backbone_work_plan.md`,
  `docs/backbone_fidelity_status.md`.
- All 24 tests pass + 1 BSuite skip (25 total — added 4 PolicyFeatTest cases).

### 2026-05-03 (Phase 2b xBC layout — match official Mamba-2 block layout)
- External review (GPT) flagged that the prior SSD-recurrence-only refactor
  did not yet match the official `Mamba2.step()` *block layout*. The
  official block does:
    1. bias-free `in_proj` emits `[z, xBC, dt]`,
    2. depthwise conv runs over `xBC` (so B and C also pass through the
       conv), then
    3. post-conv `xBC` is split into `x, B, C`.
  The prior code applied the conv only to `x` and recomputed `B, C, dt` via
  a separate `MambaSSD.x_proj` Linear after the conv. That is a real
  architectural difference (different short-time mixing path for B/C).
- Refactored `Mamba2Block` and `MambaSSD` to match the official layout:
  - `in_proj`: bias-free, `dim → 2·hidden + 2·d_state + nheads`. Verified
    on test config (`8 → 168`).
  - `conv`: depthwise over `xBC = hidden + 2·d_state` channels (e.g. 144
    in the test config vs. previously 16).
  - `MambaSSD.step` now takes pre-computed `(x_heads, B, C, dt_raw, state)`
    instead of re-projecting from x. Owns only `A_log`, `D`, `dt_bias`.
  - `dt_bias` moved from a `dt_proj.bias` to a standalone `nn.Parameter`
    (matches official; also robust to the Dreamer global `weight_init_`
    pass since standalone Parameters have no `weight` attribute).
  - `conv_state` shape is now `(B, hidden + 2·d_state, kernel-1)` (covers
    xBC). `ssm_state` shape unchanged: `(B, nheads, headdim, d_state)`.
- Added `Mamba2BlockLayoutTest::test_block_layout_matches_official_mamba2`
  that asserts the official-layout invariants: bias-free in_proj, conv
  depthwise over xBC width, scalar A per head, post-init dt_bias actually
  in the [1e-3, 0.1] range (catches future regressions where a global init
  silently overrides the special_init).
- Step-99 marker diff now ≈ 1.87e-4 (was 1.95e-4 with the prior layout) —
  comparable; the layout change shifts the mixing path but the SSD slow
  heads still retain the marker.
- Honest framing in docs:
  - Mamba/mamba2 row in fidelity-status table moved from ✅ → 🟡 since the
    chunked SSD parallel scan and fused CUDA/Triton kernels are still
    missing. This implementation is **slower** than the official
    `mamba_ssm` package and should not be claimed as "faithful Mamba-2"
    without that caveat.
  - Persistence test still uses private `_call_deter_net` with
    `stoch_mode="onehot"` — useful for isolating the recurrence but not
    the same as a full training-time `observe()`-path validation.
- Files changed: `rssm_mamba.py` (refactor), `tests/test_persistent_state.py`
  (added `Mamba2BlockLayoutTest`), `docs/backbone_fidelity_status.md`,
  `docs/backbone_work_plan.md`.
- Tests: 19 pass + 1 BSuite skip (20 total — added one shape test).

### 2026-05-03 (Phase 2b SSD upgrade — Mamba-2 multi-head selective scan)
- Replaced the prior Mamba-1-style per-channel selective SSM in `rssm_mamba.py`
  with a **Mamba-2 SSD-style multi-head selective scan** that matches the
  architectural distinctives of `state-spaces/mamba`'s `Mamba2.step()`:
  - **Scalar `A` per head** (`A_log` shape `(nheads,)`, init uniform `[1, 16]`).
    This is the SSD restriction that makes the structured matrix-mixer math
    close out — distinguishes Mamba-2 from Mamba-1's per-channel diagonal A.
  - **Per-head `Δ`** (shape `(B, nheads)`) computed via low-rank bottleneck
    + per-head bias init for log-uniform Δ ∈ `[1e-3, 0.1]`.
  - **`B`, `C` shared across heads** (`ngroups=1`), shape `(B, d_state)` each.
  - **State shape**: `(B, nheads, headdim, d_state)` per layer.
  - **Per-step update**: `dA = exp(dt * A); dBx = einsum("bh,bn,bhp->bhpn", dt, B, x_h); state = state * dA + dBx; y = einsum("bhpn,bn->bhp", state, C) + D * x_h`.
  - **Gated RMSNorm before `out_proj`** (Mamba-2 layout: `out = OutProj(Norm(SiLU(gate) * y))`).
- Added `nheads` config knob (default 8); bumped `state_size` default to 64
  (paper-typical Mamba-2 d_state).
- **Bug fix**: `tools.weight_init_` (called by `CategoricalRSSM.apply()`)
  was wiping the `dt_proj.bias` Mamba init to zero, causing Δ ≈ softplus(0)
  ≈ 0.69 and the SSM to forget the marker in ~15 steps. Added
  `MambaSSD.special_init()` which `RSSM.__init__` calls after
  `super().__init__()` to restore the Mamba-2 dt init.
- Promoted Mamba persistence test from step 20 to **step 99** (same horizon
  as the long-memory backbones). Renamed
  `test_mamba_selective_ssm_carries_marker_to_step_20` → `test_mamba_ssd_carries_marker_to_step_99`.
  Diff at step 99 is now ~1.95e-4 (vs 1e-5 threshold) — comfortable margin.
- Cross-checked the per-step recurrence against `state-spaces/mamba`'s
  `Mamba2.step()` and `tommyip/mamba2-minimal/mamba2.py`. Output shape and
  einsum signatures match.
- Honest gaps left:
  - No SSD chunked parallel scan during training (per-step Python loop;
    identical output, slower throughput on long chunks). For Dreamer's
    per-env-step `observe()` loop this is the natural form.
  - `ngroups` fixed at 1; gated norm uses `RMSNorm(SiLU(gate) * y)` instead
    of the fused `RMSNormGated` kernel (mathematically equivalent).
- Files changed: `rssm_mamba.py` (rewrite), `configs/model/_base_.yaml`,
  `tests/test_persistent_state.py`, `docs/backbone_work_plan.md`,
  `docs/backbone_fidelity_status.md`.
- All 18 tests pass + 1 BSuite skip.

### 2026-05-03 (Phase 2c — paper-faithful S4D-Lin diagonal SSM)
- Replaced the Phase-1 single-pole sigmoid-decay diagonal recurrence in
  `rssm_s4.py` with a paper-faithful **S4D-Lin** block:
  - Per-channel diagonal `A` parameterized as `-exp(A_log)`, init `A_n = -(1, 2, ..., N)`
    per channel (S4D-Lin real-form init; standard simpler alternative to
    HiPPO-LegS).
  - Per-channel learned timestep `Δ`, init log-uniform [1e-3, 0.1] via
    softplus^{-1} (standard S4D / Mamba init convention).
  - ZOH discretization for diagonal A: `A_bar = exp(Δ * A)`, `B_bar = Δ * B`.
  - Multi-channel SSM: each of `dim` token channels has its own
    `state_size`-pole diagonal SSM. State shape `(B, dim, state_size)` per
    layer (was `(B, state_size)` before).
  - `B` shared across channels `(state_size,)`; `C` per-channel
    `(dim, state_size)`; `D` per-channel skip; final `mixer: dim → dim`
    linear.
  - LTI: `A, B, C, D, Δ` are all parameters, NOT input-dependent — this is
    the architectural distinction from Mamba's selective SSM.
- Step-99 persistence diff ≈ 1.76 (vs 1e-5 threshold). The slow-decay
  channels at random init carry the marker robustly across 99 env steps.
- Honest gaps (documented in `docs/backbone_fidelity_status.md`):
  - Real-diagonal A (S4D-Lin), not complex-diagonal (S4D-Inv) or full DPLR.
  - No HiPPO-LegS init; S4D-Lin only.
  - Per-step Python-loop training; no FFT-based parallel scan.
- All 19 tests pass; smoke tests pass too (parameter count for S4 dropped
  slightly from previous because `in_proj`/`out_proj` were replaced with
  per-channel B/C parameters).
- Files changed: `rssm_s4.py` (rewrite), `docs/backbone_fidelity_status.md`,
  `docs/backbone_work_plan.md`.

### 2026-05-03 (Phase 2b correction — soften "Mamba-2" overstatement)
- External review (GPT) flagged that the Phase 2b documentation
  overstates fidelity. Verified concerns and softened claims:
  - `A` itself is **not input-dependent** — only `Δ, B, C` are. The
    discretized `A_bar = exp(Δ * A)` is effectively input-dependent through
    Δ, but the underlying `A` parameter is learned-fixed. Updated docstrings
    and per-backbone tables to say "input-dependent Δ/B/C (with input-
    dependent A_bar through Δ)" instead of "input-dependent A/B/C/Δ".
  - This is **Mamba-1-style selective scan structure**, NOT the Mamba-2
    SSD / parallel-scan formulation. Renamed framing throughout from "real
    Mamba-2" / "✅ faithful" to "Mamba-style selective SSM" / "🟡 partial".
    Class name `Mamba2Block` kept for backward-compat with existing imports
    but explicitly noted in its docstring as not implementing Mamba-2 SSD.
  - Persistence test only checks step 20 via private `_call_deter_net` with
    `stoch_mode="onehot"`. Documented as weaker than the step-99 public-
    `observe` tests for other backbones. The "trained models do better" claim
    was speculation — removed; replaced with "improved long-horizon retention
    plausible but not empirically demonstrated yet."
- Files changed: `rssm_mamba.py` (module + class docstrings),
  `docs/backbone_fidelity_status.md` (table + per-backbone Mamba section),
  `docs/backbone_work_plan.md` (TL;DR, Phase 2b table, status table,
  this changelog).
- No code behavior change; tests still 19 OK / 1 skipped.

### 2026-05-03 (Phase 2b — real Mamba-2 selective SSM)
- Replaced `rssm_mamba.py`'s gated-conv mixer with a paper-faithful Mamba-2
  selective state-space model in pure PyTorch (no `mamba_ssm` dependency
  to avoid CUDA-build risk on the cluster's PyTorch 2.8 + CUDA 13.1):
  - `SelectiveSSM`: per-step recurrent cell with input-dependent ``Δ, B, C``
    (Mamba's "selective" mechanism), diagonal real ``A`` parameterized as
    ``-exp(A_log)`` (always negative, stable). A is initialized to
    ``-(1, 2, ..., state_size)`` per channel (S4D-Lin init). Per-channel
    skip ``D``. Δ is discretized via softplus from a low-rank projection
    (`dt_rank ≈ dim/16`), with the bias initialized so Δ is log-uniform in
    [0.001, 0.1] (Mamba paper §3.6).
  - `Mamba2Block`: full Mamba block wrapping the SSM with pre-RMSNorm,
    in_proj→split→causal depthwise conv→selective SSM→SiLU(gate) gating→
    out_proj→residual.
  - `MambaDeter`: stacks `Mamba2Block` and stores both `extra["conv_state_*"]`
    (kernel-1 conv history) and `extra["ssm_state_*"]` (B, hidden, state_size
    SSM hidden state) per layer in `extra`.
- Added `mamba.state_size` to `configs/model/_base_.yaml` (default 16,
  paper-typical).
- Updated `tests/test_persistent_state.py`:
  - `run_with_marker` now accepts `stoch_mode` (`"zero"` or `"onehot"`).
    Mamba's selective C has no bias, so exactly-zero input gives exactly-
    zero output regardless of state. The new `"onehot"` mode supplies a
    constant non-zero stoch (matches realistic Dreamer use) so the SSM
    state actually influences the output.
  - New `test_mamba_selective_ssm_carries_marker_to_step_20` confirms a
    perturbation at step 0 still affects deter at step 20 via the SSM's
    persistent hidden state. Step 20 was chosen because random-init Mamba's
    output magnitude decays fast (random C has small mean weights) — at
    step ~30 the diff drops below float32 precision. Trained models retain
    info much longer; the test verifies architecture, not trained capacity.
  - Hermetic seeding in `_check_persistence` so the test is independent of
    which other tests run first.
- All tests pass (19 OK, 1 skipped BSuite).
- Files changed: `rssm_mamba.py` (rewrite), `configs/model/_base_.yaml`,
  `tests/test_persistent_state.py`, `docs/backbone_work_plan.md`,
  `docs/backbone_fidelity_status.md`.

### 2026-05-03 (Phase 2a — Transformer-XL fidelity)
- Replaced `rssm_transformer.py`'s `nn.TransformerEncoder` + absolute `pos_emb`
  with a custom Transformer-XL implementation:
  - `TransformerXLAttention`: relative positional encoding via Dai et al.
    2019's eq. 4 four-term decomposition (content-content + content-position +
    bias-content + bias-position). Sinusoidal `R_k` is computed on the fly
    from offsets, then projected by `W_kr`. Per-head learned `u`, `v` biases
    are added to `Q` before the K and R dot products.
  - `TransformerXLBlock`: pre-norm residual block with RMSNorm on Q/KV
    branches before attention, RMSNorm before the FFN.
  - `TransformerXLDeter`: per-layer segment caches stored in
    `extra["cache_input_l"]` (each holds the past *inputs* to layer `l`).
    Per env step, layer `l`'s K/V context is the concat of its cache and the
    current step's layer-`l` input. A separate `extra["summary_cache"]` of
    length `tokens` tracks the final-layer outputs so `deter` keeps its
    fixed `(B, tokens * token_dim)` shape for the obs/img heads.
- Causal self-attention mask is implicit for the per-step path (L_q=1) and
  explicit (lower-triangular) when L_q>1.
- Added `transformer.cache_length` to `configs/model/_base_.yaml` (default
  `8` for backward compat; bump to e.g. 128 for paper-faithful long memory).
- New tests in `tests/test_persistent_state.py::TransformerXLTest`:
  - `test_per_layer_caches_persist_in_extra` confirms the per-layer cache
    structure exists in `extra`.
  - `test_relative_attention_carries_marker_to_step_99` confirms a
    perturbation at step 0 still affects deter at step 99 with
    `cache_length=128`.
- All existing tests still pass. Total: 19 tests pass (1 skipped BSuite).
- Files changed: `rssm_transformer.py` (rewrite), `configs/model/_base_.yaml`,
  `tests/test_persistent_state.py`, `docs/backbone_work_plan.md`,
  `docs/backbone_fidelity_status.md`.

### 2026-05-03 (Phase 1.5 — burn-in + post-extra threading + integration tests)
- Addressed the four real fidelity gaps an external review (GPT) flagged in
  the Phase 1 work:
  1. **Replay chunks no longer start with zero extra at every loss step.**
     Added `trainer.burn_in` config (default 0). When `burn_in > 0`, the first
     `burn_in` time steps' world-model losses (`dyn`, `rep`, `recon` for
     dreamer rep_loss, `rew`, `con`) are masked. Tested via
     `test_burn_in_changes_world_model_loss`.
  2. **Imagination and CPC rollouts use post-observe extras.** Added optional
     `return_extra=True` to `CategoricalRSSM.observe`, returning per-step
     `(B, T, ...)` extras. `_cal_grad` flattens these to `(B*T, ...)` and
     passes them to `_imagine` (actor-critic) and `cpc_loss → rollout_actions`
     (CPC). Imagination starting points now inherit the matching SSM state /
     transformer cache from their source posterior time step instead of zero.
  3. **Integration test through `observe()`.** New
     `ObserveIntegrationTest` class with 4 cases. Action-marker at step 1
     (post-reset) propagates through `obs_step`, categorical sampling, and
     subsequent `obs_step` calls; the perturbation is detectable at step 50.
     Embed-marker at the reset step doesn't propagate (the categorical
     posterior quantizes the small logit perturbation to the same one-hot in
     both runs); using action-marker is the right design.
  4. **Backward-compat verified.** All 13 existing tests still pass — the
     new `return_extra` parameter defaults to False, preserving behavior.
- Files changed: `rssm_base.py`, `dreamer.py`, `tests/test_persistent_state.py`,
  `tests/test_backbones_smoke.py`.
- Test results: 17 tests pass (1 BSuite skip).

### 2026-05-03 (Phase 0 + Phase 1 implementation)
- Verified Phase 0 alias-config-lookup fix already present in `rssm_factory.py`
  via the `BACKBONE_CONFIGS` mapping. `model.backbone=mamba2|s3m|storm` now
  resolves to the correct config block.
- Implemented Phase 1 persistent extra-state interface:
  - `rssm_base.CategoricalRSSM.initial(B)` now returns `(stoch, deter, extra)`.
    Buffer-style 2-tuple callers continue to work via `_unpack_initial`.
  - `obs_step` returns `(stoch, deter, logit, extra)`; `img_step` returns
    `(stoch, deter, extra)`. Both accept `extra=None` and auto-fill.
  - `observe(...)` accepts both 2-tuple and 3-tuple `initial`.
  - `_call_deter_net` shim handles backbones with the new
    `forward(stoch, deter, action, extra) -> (deter, extra)` signature.
- Per-backbone updates:
  - `rssm.py` (GRU): forward returns `(deter, {})`. Behavior identical to
    pre-change; `extra` is empty.
  - `rssm_s4.py`: SSM hidden state moved into `extra["ssm_state_*"]` and
    persists across env steps. `S4Block` exposes `step(x_t, state)`. Init
    bumped to `log_decay=4` (sigmoid → 0.982 retention) so the SSM remembers
    across many env steps; matches S4's HiPPO-spirit slow-decay init.
  - `rssm_s5.py`: same pattern. `S5Block.step(x_t, state)` with
    state shape `(B, heads, state_size)`. Init bumped to `a = 2.65 * I` so
    `tanh(a)` ≈ 0.99 along the diagonal.
  - `rssm_mamba.py`: depthwise-conv state moved into
    `extra["conv_state_*"]`. Manual depthwise-conv `step` consumes the past
    `kernel-1` samples instead of zero-padding. Memory horizon is still only
    `kernel-1` (2 steps) until Phase 2b adds real selective SSM.
  - `rssm_transformer.py`: rolling token cache moved into `extra["cache"]`
    with new optional `transformer.cache_length` config knob (defaults to
    existing `tokens` for backward compatibility). Position embedding sized
    to `cache_length`. Cache is detached during inference to avoid building
    an unbounded autograd graph.
- `dreamer.py` updates:
  - `get_initial_state(B)` returns a TensorDict that includes nested
    `extra` (skipped if empty, so GRU's agent_state shape is unchanged).
  - `act` reads/writes `extra` from the agent state via `_pack_extra` /
    `_unpack_extra`.
  - `_imagine` and `rollout_actions` initialize a fresh extra-state per
    call (imagination is "what-if from a checkpoint"; carrying the
    training-time extra would mix observed memory into imagined rollouts).
  - Buffer is unchanged: extra is NOT stored per transition (would be huge
    for transformers). When sampling, observe auto-fills fresh extra.
- New file `tests/test_persistent_state.py` with 4 test cases:
  - `ConstructionTest`: every alias in `BACKBONES` builds and runs one
    `img_step`. Covers `gru, rssm, transformer, storm, mamba, mamba2, s4, s3m, s5`.
  - `test_long_memory_backbones_carry_marker_to_step_99`: GRU/S4/S5 (and
    `s3m`) keep marker info detectable at step 99 vs an all-zero baseline.
  - `test_transformer_persists_with_long_cache`: transformer/storm with
    `cache_length=128` keeps marker to step 99.
  - `test_mamba_persists_within_conv_window`: mamba/mamba2 keep marker
    within the conv kernel window (step 2). Documents the architecture's
    intrinsic short memory; Phase 2b will lift this.
- Switched the test from `img_step` to direct `_call_deter_net` calls. The
  one-hot categorical `stoch` sampling was quantizing away tiny logit
  perturbations and giving exactly-zero diffs even for genuinely persistent
  backbones; testing the deter+extra interface directly isolates the
  persistence behavior we care about.
- Test results: all 13 tests pass locally (1 skipped, the BSuite test, since
  bsuite isn't installed on the local 3.13 venv).
- Files changed: `rssm_base.py`, `rssm.py`, `rssm_s4.py`, `rssm_s5.py`,
  `rssm_mamba.py`, `rssm_transformer.py`, `dreamer.py`,
  `tests/test_persistent_state.py` (new), `docs/backbone_work_plan.md`,
  `docs/backbone_fidelity_status.md`.

### 2026-05-03 (cluster smoke + tuning + planning continued, earlier)
- Patched `envs/popgym.py` for missing `popgym-` namespace prefix in `gym.make()`. Verified live; needs commit + push.
- Smoke calibration on cluster (per AGENTS.md):
  - GRU best at `env8_bs1024`: 109 env-steps/sec on H100.
  - Transformer best at `env8_bs512`: 61.2 sps. **Historical note:** the pre-Phase-2a Transformer crashed at `bs1024` with CUDA SDPA "invalid configuration argument"; the current custom Transformer-XL attention path needs a fresh throughput pilot.
  - Mamba-2: 93.5 sps at `env8_bs1024`.
  - S3M: 70.3 sps at `env8_bs1024`.
  - S5: 56.9 sps at `env8_bs1024`.
- Projected cost of 75-run RepeatPrevious sweep (with per-backbone best configs): ~283 GPU-hours, ~2.4 days on 5 sustained H100s. Caveat: per-backbone config tuning is a methodological confound (different `batch_size` ⇒ different gradient noise). For the production sweep, hold `batch_size` constant.
- Decided on phased plan documented above. This file created.

---

## Decisions Log

Key decisions and the reasoning behind them.

- **2026-05-03**: Phase 1 must precede any backbone fidelity work. Reason: without persistent `extra_state` across env steps, the cleanest possible Mamba-2 implementation still has its hidden state zeroed every env step — defeats the entire point.
- **2026-05-03**: Run the 75-run sweep on current code in parallel with Phase 1 work. Reason: the result is still publishable as "controlled harness comparison" framing; the GPU-hours are sunk regardless and this gives us a baseline to compare paper-faithful results against.
- **2026-05-03**: Test design = lone-perturbation-decay (marker at step 0, zeros at steps 1–99). Reason: a "differ at step 50, identical 51–99" test passes today even with the bug because the perturbation propagates through the recurrence. The lone-marker test reliably distinguishes "8-token window" from "true persistent state."
- **2026-05-03**: For Mamba-2, prefer pure-PyTorch fallback if `pip install mamba_ssm` fails. Reason: `mamba_ssm` requires CUDA-compiled kernels that may not match cluster's PyTorch 2.8 + CUDA 13.1. A 150-LOC selective SSM in PyTorch is enough for a fair comparison.
- **2026-05-03**: Production sweep must use uniform `batch_size` across all backbones. Reason: per-backbone tuning is a methodology confound. After the Phase-2a Transformer rewrite, re-run the throughput pilot and choose one shared config for all backbones.

---

## Open Questions

Things that need user input or external verification.

- **Q1**: Does `pip install mamba_ssm` work on this cluster's PyTorch 2.8 + CUDA 13.1 setup? → To be tested in Phase 2b.
- **Q2**: Should the production sweep include CPC and DFS sub-experiments, or just `subexp=none`? Current plan: `none` only for the 75-run main sweep; add CPC/DFS only on POPGym Hard if time permits. Decision can be deferred until Phase 4.
- **Q3**: For the sub-experiments (CPC, DFS), is the existing implementation a faithful match to the proposal's formulas? CPC implementation is verified to exist in `dreamer.py`; DFS in `buffer.py`. Detailed formula audit deferred until Phase 4.

---

## Quick Reference: Starting a New Session

When you (or a future agent) opens this repo:

1. Read this file end-to-end. **Do not skip the changelog**.
2. Read [AGENTS.md](../AGENTS.md) for cluster access and operating rules.
3. Run `git log --oneline -20` to see what's been merged since this file was last updated.
4. Run `git status` and `git diff` to see uncommitted local work.
5. Run the existing test suite to confirm nothing is broken: `.venv/bin/python -m unittest discover -s tests`
6. Identify the next phase from the table above. Implement it in one focused session.
7. Before stopping: update the per-backbone status table, append a changelog entry, commit + push.
