import torch
from torch import distributions as torchd
from torch import nn

import distributions as dists
from networks import LambdaLayer
from tools import rpad, weight_init_


class CategoricalRSSM(nn.Module):
    """Shared categorical Dreamer-style RSSM scaffold.

    Subclasses provide a deterministic transition module ``_deter_net`` that
    implements:

    - ``initial_extra(batch_size, device) -> dict[str, Tensor]``
        Returns the per-environment recurrent state that the backbone needs to
        carry across env steps in addition to ``deter`` itself. May return an
        empty dict (GRU backbone uses this).
    - ``forward(stoch, deter, action, extra) -> (next_deter, next_extra)``
        Computes the next deterministic state and updates the per-env recurrent
        state in ``extra``.

    The ``extra`` dict lets backbones carry KV caches, SSM hidden state, conv
    history, etc. across ``obs_step`` / ``img_step`` calls without packing them
    into ``deter``. ``extra`` is always present in :py:meth:`initial`'s return
    value; for GRU and other extra-state-free backbones it is ``{}``.
    """

    def __init__(self, config, embed_size, act_dim, deter_net):
        super().__init__()
        self._stoch = int(config.stoch)
        self._deter = int(config.deter)
        self._hidden = int(config.hidden)
        self._discrete = int(config.discrete)
        act = getattr(torch.nn, config.act)
        self._unimix_ratio = float(config.unimix_ratio)
        self._initial = str(config.initial)
        self._device = torch.device(config.device)
        self._act_dim = act_dim
        self._obs_layers = int(config.obs_layers)
        self._img_layers = int(config.img_layers)
        self.flat_stoch = self._stoch * self._discrete
        self.feat_size = self.flat_stoch + self._deter
        self._deter_net = deter_net

        self._obs_net = nn.Sequential()
        inp_dim = self._deter + embed_size
        for i in range(self._obs_layers):
            self._obs_net.add_module(f"obs_net_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
            self._obs_net.add_module(f"obs_net_n_{i}", nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
            self._obs_net.add_module(f"obs_net_a_{i}", act())
            inp_dim = self._hidden
        self._obs_net.add_module("obs_net_logit", nn.Linear(inp_dim, self._stoch * self._discrete, bias=True))
        self._obs_net.add_module(
            "obs_net_lambda",
            LambdaLayer(lambda x: x.reshape(*x.shape[:-1], self._stoch, self._discrete)),
        )

        self._img_net = nn.Sequential()
        inp_dim = self._deter
        for i in range(self._img_layers):
            self._img_net.add_module(f"img_net_{i}", nn.Linear(inp_dim, self._hidden, bias=True))
            self._img_net.add_module(f"img_net_n_{i}", nn.RMSNorm(self._hidden, eps=1e-04, dtype=torch.float32))
            self._img_net.add_module(f"img_net_a_{i}", act())
            inp_dim = self._hidden
        self._img_net.add_module("img_net_logit", nn.Linear(inp_dim, self._stoch * self._discrete))
        self._img_net.add_module(
            "img_net_lambda",
            LambdaLayer(lambda x: x.reshape(*x.shape[:-1], self._stoch, self._discrete)),
        )
        self.apply(weight_init_)

    # ------------------------------------------------------------------ state

    def initial(self, batch_size):
        """Return an initial latent state ``(stoch, deter, extra)``.

        ``extra`` is always present (possibly empty). Callers that only need
        ``(stoch, deter)`` can ignore the third element.
        """
        deter = torch.zeros(batch_size, self._deter, dtype=torch.float32, device=self._device)
        stoch = torch.zeros(batch_size, self._stoch, self._discrete, dtype=torch.float32, device=self._device)
        extra = self._initial_extra(batch_size)
        return stoch, deter, extra

    def _initial_extra(self, batch_size):
        """Backbone-supplied extra recurrent state, falling back to {}."""
        get_extra = getattr(self._deter_net, "initial_extra", None)
        if get_extra is None:
            return {}
        return get_extra(batch_size, self._device)

    def _unpack_initial(self, initial, batch_size):
        """Accept 2-tuple ``(stoch, deter)`` OR 3-tuple ``(stoch, deter, extra)``.

        Backwards-compatible with callers (e.g. the replay buffer) that only
        store ``stoch`` and ``deter``: in that case we fill ``extra`` with the
        backbone's zero-initialized state.
        """
        if len(initial) == 3:
            return initial
        stoch, deter = initial
        return stoch, deter, self._initial_extra(batch_size)

    def _reset_extra(self, extra, reset):
        """Zero ``extra`` tensors at batch positions where ``reset`` is True."""
        if not extra:
            return extra
        out = {}
        for key, value in extra.items():
            mask = rpad(reset, value.dim() - int(reset.dim()))
            out[key] = torch.where(mask, torch.zeros_like(value), value)
        return out

    def _prepare_step_context(self):
        """Optional backbone hook for constants shared across a rollout.

        Some deterministic backbones have tensors that are constant for every
        step of a single ``observe`` / ``imagine`` call, but still need
        gradients with respect to the current parameters (for example S4/S5
        discretized matrices or Transformer-XL relative-position projections).
        Computing those once per rollout avoids launching the same tiny kernels
        at every time step while keeping the per-step ``extra`` state small.
        """
        prepare = getattr(self._deter_net, "prepare_step_context", None)
        if prepare is None:
            return None
        return prepare()

    # --------------------------------------------------------------- rollouts

    def observe(self, embed, action, initial, reset, return_extra=False):
        """Posterior rollout using observations.

        ``initial`` may be ``(stoch, deter)`` or ``(stoch, deter, extra)``.
        ``extra`` is threaded through the time loop and reset per-env where
        ``reset[:, i]`` is True.

        By default the final ``extra`` is discarded — callers that don't need
        it (e.g. ``_video_pred``, dreamerpro's augmented observe) keep their
        existing 3-tuple unpack. If ``return_extra=True``, the call returns
        ``(stochs, deters, logits, extras)`` where ``extras`` is a dict mapping
        each backbone-specific key to a ``(B, T, ...)`` tensor stacked along
        the time dim — appropriate for downstream imagination starts that need
        per-time-step recurrent state.
        """
        length = action.shape[1]
        stoch, deter, extra = self._unpack_initial(initial, action.shape[0])
        step_context = self._prepare_step_context()
        stochs, deters, logits = [], [], []
        extras_per_step = {key: [] for key in extra.keys()} if return_extra else None
        for i in range(length):
            stoch, deter, logit, extra = self.obs_step(
                stoch, deter, action[:, i], embed[:, i], reset[:, i],
                extra=extra, step_context=step_context,
            )
            stochs.append(stoch)
            deters.append(deter)
            logits.append(logit)
            if extras_per_step is not None:
                for key in extras_per_step:
                    extras_per_step[key].append(extra[key])
        stochs = torch.stack(stochs, dim=1)
        deters = torch.stack(deters, dim=1)
        logits = torch.stack(logits, dim=1)
        if return_extra:
            stacked_extras = {key: torch.stack(vs, dim=1) for key, vs in extras_per_step.items()}
            return stochs, deters, logits, stacked_extras
        return stochs, deters, logits

    def obs_step(self, stoch, deter, prev_action, embed, reset, extra=None, step_context=None):
        """Single posterior step. Returns ``(stoch, deter, logit, extra)``."""
        if extra is None:
            extra = self._initial_extra(stoch.shape[0])
        stoch = torch.where(rpad(reset, stoch.dim() - int(reset.dim())), torch.zeros_like(stoch), stoch)
        deter = torch.where(rpad(reset, deter.dim() - int(reset.dim())), torch.zeros_like(deter), deter)
        prev_action = torch.where(
            rpad(reset, prev_action.dim() - int(reset.dim())), torch.zeros_like(prev_action), prev_action
        )
        extra = self._reset_extra(extra, reset)

        deter, extra = self._call_deter_net(stoch, deter, prev_action, extra, step_context=step_context)
        logit = self._obs_net(torch.cat([deter, embed], dim=-1))
        stoch = self.get_dist(logit).rsample()
        return stoch, deter, logit, extra

    def img_step(self, stoch, deter, prev_action, extra=None, step_context=None):
        """Single prior step without observation. Returns ``(stoch, deter, extra)``."""
        if extra is None:
            extra = self._initial_extra(stoch.shape[0])
        deter, extra = self._call_deter_net(stoch, deter, prev_action, extra, step_context=step_context)
        stoch, _ = self.prior(deter)
        return stoch, deter, extra

    def _call_deter_net(self, stoch, deter, action, extra, step_context=None):
        """Invoke ``_deter_net`` with backwards-compatible signature handling.

        Older backbones may not yet accept ``extra`` and return only ``deter``;
        treat those as having no extra state.
        """
        try:
            if step_context is not None and getattr(self._deter_net, "supports_step_context", False):
                result = self._deter_net(stoch, deter, action, extra, step_context)
            else:
                result = self._deter_net(stoch, deter, action, extra)
        except TypeError:
            # Backbone not yet upgraded to the new (stoch, deter, action, extra)
            # signature; fall back to the original 3-arg call.
            return self._deter_net(stoch, deter, action), extra
        if isinstance(result, tuple):
            new_deter, new_extra = result
        else:
            new_deter, new_extra = result, extra
        return new_deter, new_extra

    def prior(self, deter):
        """Compute prior distribution parameters and sample stochastic state."""
        logit = self._img_net(deter)
        stoch = self.get_dist(logit).rsample()
        return stoch, logit

    def imagine_with_action(self, stoch, deter, actions, extra=None):
        """Roll out prior dynamics given a sequence of actions.

        ``extra`` is optional; when omitted, a fresh zero-initialized state is
        used (appropriate for "what-if" imagination from a checkpoint).
        Returns ``(stochs, deters)``; the trailing ``extra`` is discarded.
        """
        if extra is None:
            extra = self._initial_extra(stoch.shape[0])
        length = actions.shape[1]
        step_context = self._prepare_step_context()
        stochs, deters = [], []
        for i in range(length):
            stoch, deter, extra = self.img_step(
                stoch, deter, actions[:, i], extra=extra, step_context=step_context
            )
            stochs.append(stoch)
            deters.append(deter)
        stochs = torch.stack(stochs, dim=1)
        deters = torch.stack(deters, dim=1)
        return stochs, deters

    # ------------------------------------------------------------------ feats

    def get_feat(self, stoch, deter):
        """Flatten stochastic state and concatenate with deterministic state.

        This is the **world-model feature**: the input to reward/cont/value/
        recon heads. Always ``cat([stoch_flat, deter])`` regardless of
        backbone. For the actor (policy) input, see :py:meth:`policy_feat`.
        """
        stoch = stoch.reshape(*stoch.shape[:-2], self.flat_stoch)
        return torch.cat([stoch, deter], -1)

    def policy_feat(self, stoch, deter, extra=None):
        """Actor (policy) input features.

        Default: same as :py:meth:`get_feat` — ``cat([stoch_flat, deter])``.
        This is the standard DreamerV3 ``π(a | z_t, h_t)`` "output state policy."

        SSSM backbones (Mamba/S4/S5) **override** this to return
        ``cat([stoch_flat, x_t])``, the "hidden state policy" ``π(a | z_t, x_t)``
        from R2I (Samsami et al., 2024). Per the project proposal §3.1, all
        SSSM backbones in this comparison use the hidden-state policy because
        the standard ``π(a | z_t, h_t)`` underperforms on memory tasks for
        SSSMs (state compression into the deter rolling tape loses
        information that ``x_t`` retains).

        ``extra`` is the persistent extra-state dict from the most recent
        ``obs_step`` / ``img_step``. SSSM backbones read ``extra["x_t"]``;
        the base implementation ignores ``extra`` entirely.
        """
        return self.get_feat(stoch, deter)

    @property
    def policy_feat_size(self):
        """Dimension of :py:meth:`policy_feat`'s output.

        Defaults to :py:attr:`feat_size`. SSSM backbones override when their
        actor sees ``cat([stoch, x_t])`` instead of ``cat([stoch, deter])``.
        """
        return self.feat_size

    def get_dist(self, logit):
        return torchd.independent.Independent(dists.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1)

    def kl_loss(self, post_logit, prior_logit, free):
        rep_loss = dists.kl(post_logit, prior_logit.detach()).sum(-1)
        dyn_loss = dists.kl(post_logit.detach(), prior_logit).sum(-1)
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        return dyn_loss, rep_loss
