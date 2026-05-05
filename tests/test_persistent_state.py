"""Persistent-state behavior tests for the Dreamer RSSM backbones.

Each backbone exposes the new ``extra`` interface (see ``rssm_base.py``).
This test verifies two things per backbone:

1. *Construction*. Every alias in :data:`rssm_factory.BACKBONES` builds, runs
   one ``img_step``, and returns the expected shapes. Catches the historical
   alias-config-lookup crash.

2. *Persistent state across env steps*. A "marker" perturbation injected at
   step 0 must still influence the deterministic state at step 99, when
   compared against an all-zero control sequence. This is the
   lone-perturbation-decay test described in
   ``docs/backbone_work_plan.md`` and is what distinguishes a true persistent
   recurrent state from an 8-token rolling-window approximation.

   The test also asserts the marker has *some* visible effect at step 5
   (sanity: proves the marker entered the state at all).
"""

from __future__ import annotations

import math
import unittest

import torch
from omegaconf import OmegaConf

from rssm_factory import BACKBONES, build_rssm


def make_rssm_config(backbone, *, transformer_cache_length=None):
    """Compact CPU-friendly config covering every backbone block."""
    transformer_block = {"tokens": 4, "layers": 1, "heads": 2, "ff_mult": 2}
    if transformer_cache_length is not None:
        transformer_block["cache_length"] = int(transformer_cache_length)
    return OmegaConf.create(
        {
            "backbone": backbone,
            "rssm": {
                "stoch": 4,
                "deter": 32,
                "hidden": 32,
                "discrete": 4,
                "img_layers": 1,
                "obs_layers": 1,
                "dyn_layers": 1,
                "blocks": 4,
                "act": "SiLU",
                "norm": True,
                "unimix_ratio": 0.01,
                "initial": "learned",
                "device": "cpu",
            },
            "transformer": transformer_block,
            "mamba": {"tokens": 4, "layers": 1, "expand": 2, "conv_kernel": 3},
            "s4": {"tokens": 4, "layers": 1, "state_size": 8},
            "s5": {"tokens": 4, "layers": 1, "heads": 2, "state_size": 8},
        }
    )


def run_with_marker(rssm, n_steps, *, marker, act_dim, batch_size=1, stoch_mode="zero"):
    """Roll the backbone's deterministic transition for ``n_steps``.

    We call :py:meth:`CategoricalRSSM._call_deter_net` directly with a fixed
    ``stoch`` rather than going through :py:meth:`img_step`. ``img_step``
    samples a new categorical ``stoch`` from the prior at each call, and a
    one-hot sample quantizes away the small logit perturbation we want to
    track — leaving exactly zero observable difference at step 99 for two
    near-identical runs. By holding ``stoch`` fixed and feeding only
    ``action`` we isolate whether the deter + extra interface itself carries
    information across env steps.

    ``stoch_mode``:
    - ``"zero"`` (default): all-zero stoch. With zero-action filler steps the
      GRU update gate keeps the marker for many steps; suitable for GRU /
      S4 / S5 / Transformer.
    - ``"onehot"``: constant one-hot on class 0 — non-zero throughout. Mamba's
      selective SSM has an input-dependent ``C`` (no bias), so exactly-zero
      input gives ``C=0`` and the SSM-state contribution vanishes. A non-zero
      ``stoch`` keeps ``C`` non-zero so the persistent state actually
      influences the output.

    Returns the per-step deter tensor stack of shape ``(n_steps, B, deter)``.
    """
    torch.manual_seed(0)
    stoch, deter, extra = rssm.initial(batch_size)
    if stoch_mode == "onehot":
        stoch = torch.zeros_like(stoch)
        stoch[..., 0] = 1.0
    elif stoch_mode != "zero":
        raise ValueError(f"unknown stoch_mode: {stoch_mode!r}")
    deters = []
    zero_action = torch.zeros(batch_size, act_dim)
    for t in range(n_steps):
        action = torch.full((batch_size, act_dim), float(marker)) if t == 0 else zero_action
        deter, extra = rssm._call_deter_net(stoch, deter, action, extra)
        deters.append(deter.detach().clone())
    return torch.stack(deters, dim=0)


class ConstructionTest(unittest.TestCase):
    """Every alias in the factory must build and run one ``img_step``."""

    def test_all_aliases_construct_and_step(self):
        act_dim = 3
        for backbone in sorted(BACKBONES.keys()):
            with self.subTest(backbone=backbone):
                config = make_rssm_config(backbone)
                rssm = build_rssm(config, embed_size=8, act_dim=act_dim)

                stoch, deter, extra = rssm.initial(2)
                self.assertEqual(tuple(stoch.shape), (2, 4, 4))
                self.assertEqual(tuple(deter.shape), (2, 32))
                self.assertIsInstance(extra, dict)

                action = torch.zeros(2, act_dim)
                next_stoch, next_deter, next_extra = rssm.img_step(stoch, deter, action, extra=extra)
                self.assertEqual(tuple(next_stoch.shape), (2, 4, 4))
                self.assertEqual(tuple(next_deter.shape), (2, 32))
                # extra dict must have the same keys (stable shape).
                self.assertEqual(set(next_extra.keys()), set(extra.keys()))


class PersistentStateTest(unittest.TestCase):
    """Marker at step 0 should influence deter at a later step, with a horizon
    appropriate to each backbone's intrinsic memory mechanism."""

    # Long-memory recurrences: GRU's deter, S4/S5's persistent SSM state.
    # All should carry the marker to step 99 with random-init weights.
    LONG_MEMORY_BACKBONES = ("gru", "rssm", "s4", "s3m", "s5")

    # Transformer aliases: long memory only with a cache_length >= horizon.
    TRANSFORMER_BACKBONES = ("transformer", "storm")

    # Mamba-2 (Phase 2b SSD-style upgrade) carries memory across env steps
    # via its multi-head SSD hidden state. With Mamba-2 init (per-head Δ
    # log-uniform [1e-3, 0.1], scalar A per head uniform [-16, -1]), the
    # slowest-decay head reliably retains the marker to step 99 even at
    # random init. Uses ``stoch_mode=onehot`` because the input-dependent C
    # projection has no bias and exactly-zero input produces exactly-zero
    # SSM output regardless of state.
    MAMBA_BACKBONES = ("mamba", "mamba2")

    HORIZON = 100
    SANITY_STEP = 5
    PERSIST_STEP = 99
    EPSILON = 1e-5

    def _check_persistence(self, backbone, persist_step, *, transformer_cache_length=None, stoch_mode="zero"):
        # Seed before build so the test is hermetic regardless of which other
        # tests have run first (and consumed RNG via their own constructors).
        torch.manual_seed(0)
        config = make_rssm_config(backbone, transformer_cache_length=transformer_cache_length)
        rssm = build_rssm(config, embed_size=8, act_dim=3)

        with_marker = run_with_marker(rssm, self.HORIZON, marker=1.0, act_dim=3, stoch_mode=stoch_mode)
        without_marker = run_with_marker(rssm, self.HORIZON, marker=0.0, act_dim=3, stoch_mode=stoch_mode)

        diff_sanity = (with_marker[self.SANITY_STEP] - without_marker[self.SANITY_STEP]).abs().max()
        diff_persist = (with_marker[persist_step] - without_marker[persist_step]).abs().max()
        return diff_sanity, diff_persist

    def test_long_memory_backbones_carry_marker_to_step_99(self):
        """GRU, S4, S5 carry the marker for 99 steps via persistent recurrent state."""
        for backbone in self.LONG_MEMORY_BACKBONES:
            with self.subTest(backbone=backbone):
                diff_sanity, diff_persist = self._check_persistence(backbone, self.PERSIST_STEP)
                self.assertGreater(
                    float(diff_sanity), self.EPSILON,
                    f"{backbone}: marker did not enter the state at all (step {self.SANITY_STEP}).",
                )
                self.assertGreater(
                    float(diff_persist), self.EPSILON,
                    f"{backbone}: marker did not survive to step {self.PERSIST_STEP}; "
                    f"persistent extra-state interface is not actually carrying memory.",
                )

    def test_mamba_ssd_carries_marker_to_step_99(self):
        """Mamba-2 SSD-style multi-head selective scan carries the marker
        across 99 env steps via the persistent ``(B, nheads, headdim,
        d_state)`` hidden state. The slowest-decay head (smallest Δ × |A|)
        retains the marker far better than Mamba-1's per-channel SSM.
        Uses ``stoch_mode=onehot`` because the input-dependent C projection
        has no bias and exactly-zero input would produce exactly-zero SSM
        output regardless of state."""
        for backbone in self.MAMBA_BACKBONES:
            with self.subTest(backbone=backbone):
                diff_sanity, diff_persist = self._check_persistence(
                    backbone, self.PERSIST_STEP, stoch_mode="onehot"
                )
                self.assertGreater(float(diff_sanity), self.EPSILON, backbone)
                self.assertGreater(
                    float(diff_persist), self.EPSILON,
                    f"{backbone}: SSD state did not influence output at step "
                    f"{self.PERSIST_STEP}",
                )

    def test_transformer_persists_with_long_cache(self):
        """Transformer/storm carry the marker for 99 steps when cache_length=128."""
        for backbone in self.TRANSFORMER_BACKBONES:
            with self.subTest(backbone=backbone):
                diff_sanity, diff_persist = self._check_persistence(
                    backbone, self.PERSIST_STEP, transformer_cache_length=128
                )
                self.assertGreater(float(diff_sanity), self.EPSILON, backbone)
                self.assertGreater(float(diff_persist), self.EPSILON, backbone)


class Mamba2BlockLayoutTest(unittest.TestCase):
    """Verify ``rssm_mamba.Mamba2Block`` matches the official ``state-spaces/mamba``
    ``Mamba2`` block layout: bias-free in_proj emitting ``[z, xBC, dt]``,
    causal depthwise conv over the full ``xBC`` (not just ``x``), scalar A
    per head, and persistent extras of the right shapes."""

    def test_block_layout_matches_official_mamba2(self):
        torch.manual_seed(0)
        config = make_rssm_config("mamba2")
        rssm = build_rssm(config, embed_size=8, act_dim=3)
        block = rssm._deter_net.blocks[0]

        # Block dimensions.
        dim = block.dim
        hidden = block.hidden
        nheads = block.nheads
        headdim = block.headdim
        d_state = block.d_state
        conv_dim = block.conv_dim
        kernel = block.conv_kernel

        # in_proj: bias-free per official Mamba-2; out width = hidden + conv_dim + nheads.
        self.assertIsNone(
            block.in_proj.bias,
            "in_proj should be bias-free (matches official Mamba-2 default)",
        )
        self.assertEqual(block.in_proj.in_features, dim)
        self.assertEqual(
            block.in_proj.out_features, hidden + conv_dim + nheads,
            f"in_proj out_features must equal hidden ({hidden}) + conv_dim "
            f"({conv_dim}) + nheads ({nheads}) — emits [z, xBC, dt]",
        )

        # Conv operates on xBC = hidden + 2*d_state, depthwise (groups = conv_dim).
        self.assertEqual(conv_dim, hidden + 2 * d_state)
        self.assertEqual(block.conv.in_channels, conv_dim)
        self.assertEqual(block.conv.out_channels, conv_dim)
        self.assertEqual(
            block.conv.groups, conv_dim,
            "Conv must be depthwise (groups = conv_dim)",
        )
        self.assertEqual(tuple(block.conv.kernel_size), (kernel,))

        # SSD parameters: A_log, D, dt_bias all shape (nheads,) — scalar per head.
        self.assertEqual(tuple(block.ssm.A_log.shape), (nheads,),
                         "A_log must be scalar per head (SSD restriction)")
        self.assertEqual(tuple(block.ssm.D.shape), (nheads,))
        self.assertEqual(tuple(block.ssm.dt_bias.shape), (nheads,))

        # dt_bias init: softplus(dt_bias) should be in roughly log-uniform
        # [1e-3, 0.1] per head (not 0 — would mean the special_init was overridden).
        delta = torch.nn.functional.softplus(block.ssm.dt_bias).detach()
        self.assertTrue(
            (delta > 1e-4).all() and (delta < 0.5).all(),
            f"softplus(dt_bias) out of expected [1e-3, 0.1] range: {delta.tolist()}. "
            "If all near 0.693 (= log(2)), the Dreamer global weight_init_ "
            "wiped the dt_bias; RSSM.__init__ must call ssm.special_init() "
            "after super().__init__().",
        )

        # Persistent extras: conv_state covers xBC (not just x); ssm_state is multi-head.
        extras = rssm._deter_net.initial_extra(2, "cpu")
        self.assertEqual(
            tuple(extras["conv_state_0"].shape), (2, conv_dim, kernel - 1),
            f"conv_state must cover xBC ({conv_dim} channels), not just x ({hidden})",
        )
        self.assertEqual(
            tuple(extras["ssm_state_0"].shape), (2, nheads, headdim, d_state),
            "ssm_state must be (B, nheads, headdim, d_state)",
        )


class ObserveIntegrationTest(unittest.TestCase):
    """End-to-end test: a marker injected as a one-step ``action`` perturbs the
    posterior trajectory and the perturbation must still be detectable many
    steps later when the rollout goes through the full :py:meth:`observe` path
    (reset masking -> obs_step loop -> per-step extras -> sampled categorical
    stoch -> next obs_step).

    The marker is delivered as ``action`` (not ``embed``) at the first
    post-reset step. ``action`` is fed into ``_deter_net`` directly, so even if
    the categorical stoch sample at the next step happens to be the same one-
    hot for the marker and no-marker runs, the deter and extra still differ.

    This complements :class:`PersistentStateTest` by exercising the public
    training entry point (``observe`` + reset + obs_step + categorical
    sampling) rather than just the private ``_call_deter_net``.
    """

    HORIZON = 60
    SANITY_STEP = 5
    PERSIST_STEP = 50
    EPSILON = 1e-5

    LONG_MEMORY_BACKBONES = ("gru", "rssm", "s4", "s3m", "s5")
    TRANSFORMER_BACKBONES = ("transformer", "storm")
    MAMBA_BACKBONES = ("mamba", "mamba2")

    def _run_observe(self, backbone, *, transformer_cache_length=None, marker=1.0):
        """Construct an RSSM, run ``observe`` for ``HORIZON`` steps with an
        action-marker at step 1 (post-reset), and return the
        ``(B, T, deter)`` trajectory plus per-step extras."""
        torch.manual_seed(0)
        config = make_rssm_config(backbone, transformer_cache_length=transformer_cache_length)
        rssm = build_rssm(config, embed_size=8, act_dim=3)
        B, T = 1, self.HORIZON
        embed = torch.zeros(B, T, 8)
        action = torch.zeros(B, T, 3)
        action[:, 1, :] = float(marker)  # marker at step 1 (post-reset)
        reset = torch.zeros(B, T, dtype=torch.bool)
        reset[:, 0] = True  # realistic episode start
        # 2-tuple initial; observe auto-fills extra to backbone defaults.
        initial = rssm.initial(B)
        post_stoch, post_deter, post_logit, extras = rssm.observe(
            embed, action, initial, reset, return_extra=True
        )
        return post_deter, extras

    def _check_observe_persistence(self, backbone, persist_step, **kwargs):
        with_marker, _ = self._run_observe(backbone, marker=1.0, **kwargs)
        without_marker, _ = self._run_observe(backbone, marker=0.0, **kwargs)
        diff_sanity = (with_marker[:, self.SANITY_STEP] - without_marker[:, self.SANITY_STEP]).abs().max().detach()
        diff_persist = (with_marker[:, persist_step] - without_marker[:, persist_step]).abs().max().detach()
        return diff_sanity, diff_persist

    def test_observe_long_memory_backbones(self):
        """observe() carries an action-marker at step 1 through 50 posterior steps."""
        for backbone in self.LONG_MEMORY_BACKBONES:
            with self.subTest(backbone=backbone):
                diff_sanity, diff_persist = self._check_observe_persistence(backbone, self.PERSIST_STEP)
                self.assertGreater(float(diff_sanity), self.EPSILON, backbone)
                self.assertGreater(
                    float(diff_persist), self.EPSILON,
                    f"{backbone}: marker did not survive observe to step {self.PERSIST_STEP}",
                )

    def test_observe_transformer_with_long_cache(self):
        """observe() carries the action-marker for transformer when cache_length>=horizon."""
        for backbone in self.TRANSFORMER_BACKBONES:
            with self.subTest(backbone=backbone):
                diff_sanity, diff_persist = self._check_observe_persistence(
                    backbone, self.PERSIST_STEP, transformer_cache_length=64
                )
                self.assertGreater(float(diff_sanity), self.EPSILON, backbone)
                self.assertGreater(float(diff_persist), self.EPSILON, backbone)

    def test_observe_mamba_carries_marker(self):
        """observe() carries the action-marker through Mamba's full public path
        (reset → obs_step → SSD update → categorical sample → next obs_step).

        Closes the gap that ``PersistentStateTest::test_mamba_ssd_carries_marker_to_step_99``
        only validates the private ``_call_deter_net`` route. This test
        exercises the same public training entry point used by the long-
        memory backbones above."""
        for backbone in self.MAMBA_BACKBONES:
            with self.subTest(backbone=backbone):
                diff_sanity, diff_persist = self._check_observe_persistence(
                    backbone, self.PERSIST_STEP
                )
                self.assertGreater(float(diff_sanity), self.EPSILON, backbone)
                self.assertGreater(
                    float(diff_persist), self.EPSILON,
                    f"{backbone}: marker did not survive observe() to step "
                    f"{self.PERSIST_STEP}; SSD state should carry it via "
                    f"the public training path, not just _call_deter_net.",
                )

    def test_observe_returns_per_step_extras(self):
        """The new return_extra=True path returns (B, T, ...) per backbone key."""
        for backbone in ("gru", "s4", "s5", "mamba", "transformer"):
            with self.subTest(backbone=backbone):
                _, extras = self._run_observe(backbone)
                if backbone == "gru":
                    self.assertEqual(extras, {})
                else:
                    self.assertGreater(len(extras), 0)
                    for key, val in extras.items():
                        self.assertEqual(val.shape[0], 1, f"{backbone}/{key} batch dim")
                        self.assertEqual(val.shape[1], self.HORIZON, f"{backbone}/{key} time dim")


class TransformerXLTest(unittest.TestCase):
    """Phase 2a: tests specific to the Transformer-XL backbone.

    The TXL backbone uses relative positional encoding rather than absolute
    learned ``pos_emb``, so attention behavior must depend on relative offsets
    between tokens, not on which absolute slot each token occupies.
    """

    def _build(self, cache_length=64):
        config = make_rssm_config("transformer", transformer_cache_length=cache_length)
        return build_rssm(config, embed_size=8, act_dim=3)

    def test_per_layer_caches_persist_in_extra(self):
        """Each layer of the TXL backbone owns its own cache in extra."""
        rssm = self._build(cache_length=32)
        _, _, extra = rssm.initial(2)
        # 1 layer in the test config -> cache_input_0 + summary_cache.
        self.assertIn("cache_input_0", extra)
        self.assertIn("summary_cache", extra)
        self.assertEqual(tuple(extra["cache_input_0"].shape), (2, 32, 8))
        self.assertEqual(tuple(extra["summary_cache"].shape), (2, 4, 8))

    def test_relative_attention_carries_marker_to_step_99(self):
        """With cache_length=128, an action-marker at step 0 must still
        affect deter at step 99 (the perturbation enters layer-0's cache and
        every later step's attention sees it via the relative-position-encoded
        K/V context)."""
        config = make_rssm_config("transformer", transformer_cache_length=128)
        rssm = build_rssm(config, embed_size=8, act_dim=3)

        with_marker = run_with_marker(rssm, 100, marker=1.0, act_dim=3)
        without_marker = run_with_marker(rssm, 100, marker=0.0, act_dim=3)
        diff_5 = (with_marker[5] - without_marker[5]).abs().max()
        diff_99 = (with_marker[99] - without_marker[99]).abs().max()
        self.assertGreater(float(diff_5), 1e-5, "TXL: marker did not enter layer cache")
        self.assertGreater(
            float(diff_99), 1e-5,
            "TXL: marker did not survive 99 steps via the cache+relpos attention",
        )


class S5BlockLayoutTest(unittest.TestCase):
    """Phase 2d: verify ``rssm_s5.S5Block`` matches the Smith et al. 2023 S5
    architectural distinctives — complex-diagonal A with HiPPO-N init,
    complex B/C, real D, per-pole Δ, complex-state persistence."""

    def test_block_layout_matches_paper(self):
        torch.manual_seed(0)
        config = make_rssm_config("s5")
        rssm = build_rssm(config, embed_size=8, act_dim=3)
        block = rssm._deter_net.blocks[0]

        H, P, N = block.heads, block.head_dim, block.d_state

        # Parameter shapes.
        self.assertEqual(tuple(block.A_re_log.shape), (H, N))
        self.assertEqual(tuple(block.A_im.shape), (H, N))
        self.assertEqual(tuple(block.B_re.shape), (H, N, P))
        self.assertEqual(tuple(block.B_im.shape), (H, N, P))
        self.assertEqual(tuple(block.C_re.shape), (H, P, N))
        self.assertEqual(tuple(block.C_im.shape), (H, P, N))
        self.assertEqual(tuple(block.D.shape), (block.dim,))
        self.assertEqual(tuple(block.log_dt.shape), (H, N))

        # HiPPO-N approximation init: Re(A) = -0.5 everywhere; Im(A_n) = π·n.
        A = torch.complex(-torch.exp(block.A_re_log), block.A_im).detach()
        self.assertTrue(
            torch.allclose(A.real, torch.full_like(A.real, -0.5), atol=1e-5),
            "HiPPO-N init: Re(A) must be -0.5 at init",
        )
        # Im(A_n) = π · (1, 2, ..., d_state) per head.
        expected_im = math.pi * torch.arange(1, N + 1, dtype=torch.float32)
        self.assertTrue(
            torch.allclose(A[0].imag, expected_im, atol=1e-5),
            f"HiPPO-N init: Im(A) must be π·(1..N); got {A[0].imag.tolist()}",
        )

        # A is always Hurwitz at runtime: Re(A) = -exp(A_re_log), strictly negative.
        self.assertTrue(
            (A.real < 0).all(),
            "A must always have negative real part (Hurwitz) — parameterization "
            "as -exp(A_re_log) should guarantee this",
        )

        # Δ init in log-uniform [1e-3, 0.1] per pole.
        delta = torch.nn.functional.softplus(block.log_dt).detach()
        self.assertTrue(
            (delta > 5e-4).all() and (delta < 0.5).all(),
            f"softplus(log_dt) out of [1e-3, 0.1] range: {delta.min():.3e}–{delta.max():.3e}. "
            "If all near 0.693 (= log(2)), the Dreamer global init wiped log_dt; "
            "RSSM.__init__ must call _init_log_dt() after super().__init__().",
        )

    def test_ssm_state_is_complex64(self):
        """S5's hidden state is intrinsically complex (one of the paper's
        distinctive design choices). The persistent state extra must carry
        a complex64 tensor, not real."""
        config = make_rssm_config("s5")
        rssm = build_rssm(config, embed_size=8, act_dim=3)
        extras = rssm._deter_net.initial_extra(2, "cpu")
        self.assertIn("ssm_state_0", extras)
        self.assertEqual(extras["ssm_state_0"].dtype, torch.complex64,
                         "S5 SSM state must be complex64")
        # After one forward, the state must still be complex.
        stoch, deter, extra = rssm.initial(2)
        action = torch.zeros(2, 3)
        _, _, new_extra = rssm.img_step(stoch, deter, action, extra=extra)
        self.assertEqual(new_extra["ssm_state_0"].dtype, torch.complex64)

    def test_marker_survives_to_step_99(self):
        """With HiPPO-N init (Re(A) = -0.5) and a slow learned Δ, the
        decay rate remains close to 1 over a 99-step test horizon.
        S5 should retain the marker very robustly — much better than the
        prior tanh-based real-diagonal toy."""
        config = make_rssm_config("s5")
        rssm = build_rssm(config, embed_size=8, act_dim=3)
        with_m = run_with_marker(rssm, 100, marker=1.0, act_dim=3)
        without_m = run_with_marker(rssm, 100, marker=0.0, act_dim=3)
        diff_99 = (with_m[99] - without_m[99]).abs().max().item()
        self.assertGreater(
            diff_99, 0.01,
            f"S5 with HiPPO-N init should retain marker very strongly at "
            f"step 99 (expected diff > 0.01, got {diff_99:.3e})",
        )


class PolicyFeatTest(unittest.TestCase):
    """Phase 3: SSSM hidden-state policy ``π(a | z_t, x_t)`` (proposal §3.1).

    Verifies that:
    - GRU and Transformer backbones use the default actor input
      ``cat([stoch_flat, deter])`` (``policy_feat == get_feat``,
      ``policy_feat_size == feat_size``).
    - SSSM backbones (Mamba, S4, S5) override to use
      ``cat([stoch_flat, x_t])`` where x_t is the per-step pre-tape SSM
      output of shape ``(B, token_dim)``. Their ``policy_feat_size``
      differs from ``feat_size``.
    - x_t is actually exposed in ``extra`` after a forward pass.
    """

    GRU_LIKE = ("gru", "rssm", "transformer", "storm")
    SSSM_LIKE = ("mamba", "mamba2", "s4", "s3m", "s5")

    def test_gru_and_transformer_use_default_policy_feat(self):
        for backbone in self.GRU_LIKE:
            with self.subTest(backbone=backbone):
                config = make_rssm_config(backbone)
                rssm = build_rssm(config, embed_size=8, act_dim=3)
                self.assertEqual(
                    rssm.policy_feat_size, rssm.feat_size,
                    f"{backbone}: non-SSSM backbones should keep "
                    f"policy_feat_size == feat_size (DreamerV3 default π(a|z, h))",
                )

    def test_sssm_uses_hidden_state_policy(self):
        """SSSM backbones must override policy_feat to use x_t per §3.1."""
        for backbone in self.SSSM_LIKE:
            with self.subTest(backbone=backbone):
                config = make_rssm_config(backbone)
                rssm = build_rssm(config, embed_size=8, act_dim=3)
                # Sizes must differ: feat = stoch + deter, policy_feat = stoch + token_dim.
                token_dim = rssm._deter_net.token_dim
                expected_policy_size = rssm.flat_stoch + token_dim
                self.assertEqual(
                    rssm.policy_feat_size, expected_policy_size,
                    f"{backbone}: SSSM policy_feat_size must equal "
                    f"flat_stoch ({rssm.flat_stoch}) + token_dim ({token_dim}); "
                    f"got {rssm.policy_feat_size}",
                )
                self.assertNotEqual(
                    rssm.policy_feat_size, rssm.feat_size,
                    f"{backbone}: SSSM should use hidden-state policy "
                    f"(different actor input width than world-model feat)",
                )

    def test_sssm_x_t_exposed_in_extra(self):
        """After a forward pass, every SSSM backbone must expose x_t in extra."""
        for backbone in self.SSSM_LIKE:
            with self.subTest(backbone=backbone):
                config = make_rssm_config(backbone)
                rssm = build_rssm(config, embed_size=8, act_dim=3)
                stoch, deter, extra = rssm.initial(2)
                self.assertIn(
                    "x_t", extra,
                    f"{backbone}: initial_extra must include 'x_t' (zeros) so "
                    f"policy_feat works at construction time",
                )
                action = torch.zeros(2, 3)
                _, _, new_extra = rssm.img_step(stoch, deter, action, extra=extra)
                self.assertIn("x_t", new_extra, f"{backbone}: forward must update x_t")
                # Shape: (B, token_dim).
                self.assertEqual(
                    tuple(new_extra["x_t"].shape),
                    (2, rssm._deter_net.token_dim),
                    f"{backbone}: x_t shape mismatch",
                )

    def test_sssm_policy_feat_uses_x_t_not_deter(self):
        """policy_feat output must equal cat([stoch, x_t]), not cat([stoch, deter])."""
        for backbone in self.SSSM_LIKE:
            with self.subTest(backbone=backbone):
                torch.manual_seed(0)
                config = make_rssm_config(backbone)
                rssm = build_rssm(config, embed_size=8, act_dim=3)
                stoch, deter, extra = rssm.initial(2)
                # Run one forward to populate a real x_t.
                stoch2, deter2, extra2 = rssm.img_step(
                    stoch, deter, torch.ones(2, 3), extra=extra
                )
                pf = rssm.policy_feat(stoch2, deter2, extra2)
                self.assertEqual(
                    tuple(pf.shape), (2, rssm.policy_feat_size),
                    f"{backbone}: policy_feat output shape mismatch",
                )
                # The trailing slice should be x_t, not deter.
                token_dim = rssm._deter_net.token_dim
                trailing = pf[:, -token_dim:]
                self.assertTrue(
                    torch.allclose(trailing, extra2["x_t"], atol=1e-6),
                    f"{backbone}: policy_feat trailing dims should be x_t",
                )


if __name__ == "__main__":
    unittest.main()
