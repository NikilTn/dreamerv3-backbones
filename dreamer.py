import copy
import math
from collections import OrderedDict

import torch
import torch.nn.functional as F
from tensordict import TensorDict
from torch import nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import LambdaLR

import networks
import tools
from networks import Projector
from optim import LaProp, clip_grad_agc_
from rssm_factory import build_rssm
from tools import to_f32


class Dreamer(nn.Module):
    def __init__(self, config, obs_space, act_space):
        super().__init__()
        self.device = torch.device(config.device)
        self.act_entropy = float(config.act_entropy)
        self.kl_free = float(config.kl_free)
        self.imag_horizon = int(config.imag_horizon)
        self.horizon = int(config.horizon)
        self.lamb = float(config.lamb)
        self.return_ema = networks.ReturnEMA(device=self.device)
        self.act_dim = act_space.n if hasattr(act_space, "n") else sum(act_space.shape)
        self.rep_loss = str(config.rep_loss)
        self._amp_enabled = self.device.type == "cuda"
        self.cpc_enabled = bool(config.cpc.enabled)
        # Replay-chunk burn-in: number of leading time steps whose world-model
        # losses are masked out so the backbone's persistent extra-state has a
        # chance to warm up from the chunk's zero-initialized seed before its
        # predictions are graded. See `docs/backbone_work_plan.md` Phase 1.5.
        self.burn_in = int(getattr(config, "burn_in", 0))

        # World model components
        shapes = {k: tuple(v.shape) for k, v in obs_space.spaces.items()}
        self.encoder = networks.MultiEncoder(config.encoder, shapes)
        self.embed_size = self.encoder.out_dim
        self.rssm = build_rssm(config, self.embed_size, self.act_dim)
        self.reward = networks.MLPHead(config.reward, self.rssm.feat_size)
        self.cont = networks.MLPHead(config.cont, self.rssm.feat_size)

        config.actor.shape = (act_space.n,) if hasattr(act_space, "n") else tuple(map(int, act_space.shape))
        self.act_discrete = False
        if hasattr(act_space, "multi_discrete"):
            config.actor.dist = config.actor.dist.multi_disc
            self.act_discrete = True
        elif hasattr(act_space, "discrete"):
            config.actor.dist = config.actor.dist.disc
            self.act_discrete = True
        else:
            config.actor.dist = config.actor.dist.cont

        # Actor-critic components.
        # The actor uses ``policy_feat`` (proposal §3.1: SSSM backbones use
        # the hidden-state policy ``π(a | z_t, x_t)`` instead of the default
        # output-state policy ``π(a | z_t, h_t)``). For GRU/transformer
        # backbones, ``policy_feat_size == feat_size`` and behavior is
        # unchanged; for SSSM backbones (Mamba/S4/S5), the actor sees a
        # different input width.
        self.actor = networks.MLPHead(config.actor, self.rssm.policy_feat_size)
        # Critic stays on the world-model feat (z, h) — only the policy
        # input changes per the proposal.
        self.value = networks.MLPHead(config.critic, self.rssm.feat_size)
        self.slow_target_update = int(config.slow_target_update)
        self.slow_target_fraction = float(config.slow_target_fraction)
        self._slow_value = copy.deepcopy(self.value)
        for param in self._slow_value.parameters():
            param.requires_grad = False
        self._slow_value_updates = 0

        self._loss_scales = dict(config.loss_scales)
        self._log_grads = bool(config.log_grads)

        modules = {
            "rssm": self.rssm,
            "actor": self.actor,
            "value": self.value,
            "reward": self.reward,
            "cont": self.cont,
            "encoder": self.encoder,
        }

        if self.rep_loss == "dreamer":
            self.decoder = networks.MultiDecoder(
                config.decoder,
                self.rssm._deter,
                self.rssm.flat_stoch,
                shapes,
            )
            recon = self._loss_scales.pop("recon")
            self._loss_scales.update({k: recon for k in self.decoder.all_keys})
            modules.update({"decoder": self.decoder})
        elif self.rep_loss == "r2dreamer" or self.rep_loss == "infonce":
            # add projector for latent to embedding
            self.prj = Projector(self.rssm.feat_size, self.embed_size)
            modules.update({"projector": self.prj})
            self.barlow_lambd = float(config.r2dreamer.lambd)
        elif self.rep_loss == "dreamerpro":
            dpc = config.dreamer_pro
            self.warm_up = int(dpc.warm_up)
            self.num_prototypes = int(dpc.num_prototypes)
            self.proto_dim = int(dpc.proto_dim)
            self.temperature = float(dpc.temperature)
            self.sinkhorn_eps = float(dpc.sinkhorn_eps)
            self.sinkhorn_iters = int(dpc.sinkhorn_iters)
            self.ema_update_every = int(dpc.ema_update_every)
            self.ema_update_fraction = float(dpc.ema_update_fraction)
            self.freeze_prototypes_iters = int(dpc.freeze_prototypes_iters)
            self.aug_max_delta = float(dpc.aug.max_delta)
            self.aug_same_across_time = bool(dpc.aug.same_across_time)
            self.aug_bilinear = bool(dpc.aug.bilinear)

            self._prototypes = nn.Parameter(torch.randn(self.num_prototypes, self.proto_dim))
            self.obs_proj = nn.Linear(self.embed_size, self.proto_dim)
            self.feat_proj = nn.Linear(self.rssm.feat_size, self.proto_dim)
            self._ema_encoder = copy.deepcopy(self.encoder)
            self._ema_obs_proj = copy.deepcopy(self.obs_proj)
            for param in self._ema_encoder.parameters():
                param.requires_grad = False
            for param in self._ema_obs_proj.parameters():
                param.requires_grad = False
            self._ema_updates = 0
            modules.update({
                "prototypes": self._prototypes,
                "obs_proj": self.obs_proj,
                "feat_proj": self.feat_proj,
                "ema_encoder": self._ema_encoder,
                "ema_obs_proj": self._ema_obs_proj,
            })
        if self.cpc_enabled:
            cpc = config.cpc
            hidden_dim = int(cpc.hidden_dim)
            proj_dim = int(cpc.proj_dim)
            act = getattr(torch.nn, config.act)
            self.cpc_horizon = int(cpc.horizon)
            self.cpc_temperature = float(cpc.temperature)
            self.cpc_aug_max_delta = float(cpc.aug.max_delta)
            self.cpc_aug_same_across_time = bool(cpc.aug.same_across_time)
            self.cpc_aug_bilinear = bool(cpc.aug.bilinear)
            self.cpc_pred_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.rssm.feat_size, hidden_dim),
                        nn.RMSNorm(hidden_dim, eps=1e-04, dtype=torch.float32),
                        act(),
                        nn.Linear(hidden_dim, proj_dim),
                    )
                    for _ in range(self.cpc_horizon)
                ]
            )
            self.cpc_target_heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.embed_size, hidden_dim),
                        nn.RMSNorm(hidden_dim, eps=1e-04, dtype=torch.float32),
                        act(),
                        nn.Linear(hidden_dim, proj_dim),
                    )
                    for _ in range(self.cpc_horizon)
                ]
            )
            modules.update({"cpc_pred_heads": self.cpc_pred_heads, "cpc_target_heads": self.cpc_target_heads})
        # count number of parameters in each module
        for key, module in modules.items():
            if isinstance(module, nn.Parameter):
                print(f"{module.numel():>14,}: {key}")
            else:
                print(f"{sum(p.numel() for p in module.parameters()):>14,}: {key}")
        self._named_params = OrderedDict()
        for name, module in modules.items():
            if isinstance(module, nn.Parameter):
                self._named_params[name] = module
            else:
                for param_name, param in module.named_parameters():
                    self._named_params[f"{name}.{param_name}"] = param
        print(f"Optimizer has: {sum(p.numel() for p in self._named_params.values())} parameters.")

        def _agc(params):
            clip_grad_agc_(params, float(config.agc), float(config.pmin), foreach=True)

        self._agc = _agc
        self._optimizer = LaProp(
            self._named_params.values(),
            lr=config.lr,
            betas=(config.beta1, config.beta2),
            eps=config.eps,
        )
        self._scaler = GradScaler(enabled=self._amp_enabled)

        def lr_lambda(step):
            if config.warmup:
                return min(1.0, (step + 1) / config.warmup)
            return 1.0

        self._scheduler = LambdaLR(self._optimizer, lr_lambda=lr_lambda)

        self.train()
        self.clone_and_freeze()
        compile_requested = bool(config.compile) and self.device.type == "cuda"
        backbone = str(getattr(config, "backbone", "")).lower()
        if compile_requested and backbone == "s5":
            # TorchInductor currently does not compile S5's native complex64
            # recurrence reliably. Keep S5 eager while allowing the other
            # backbones to use compile's large steady-state speedup.
            print("torch.compile disabled for S5 complex-state backbone; using eager mode.")
            compile_requested = False
        self._compile_enabled = compile_requested
        if self._compile_enabled:
            print("Compiling update function with torch.compile...")
            self._cal_grad = torch.compile(self._cal_grad, mode="reduce-overhead")

    def _mark_step_begin(self):
        if self.device.type == "cuda" and hasattr(torch.compiler, "cudagraph_mark_step_begin"):
            torch.compiler.cudagraph_mark_step_begin()

    def _update_slow_target(self):
        """Update slow-moving value target network."""
        if self._slow_value_updates % self.slow_target_update == 0:
            with torch.no_grad():
                mix = self.slow_target_fraction
                for v, s in zip(self.value.parameters(), self._slow_value.parameters()):
                    s.data.copy_(mix * v.data + (1 - mix) * s.data)
        self._slow_value_updates += 1

    def train(self, mode=True):
        super().train(mode)
        # slow_value should be always eval mode
        self._slow_value.train(False)
        return self

    def clone_and_freeze(self):
        # NOTE: "requires_grad" affects whether a parameter is updated
        # not whether gradients flow through its operations
        self._frozen_encoder = copy.deepcopy(self.encoder)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.encoder.named_parameters(), self._frozen_encoder.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_rssm = copy.deepcopy(self.rssm)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.rssm.named_parameters(), self._frozen_rssm.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_reward = copy.deepcopy(self.reward)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.reward.named_parameters(), self._frozen_reward.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_cont = copy.deepcopy(self.cont)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.cont.named_parameters(), self._frozen_cont.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_actor = copy.deepcopy(self.actor)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.actor.named_parameters(), self._frozen_actor.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_value = copy.deepcopy(self.value)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self.value.named_parameters(), self._frozen_value.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

        self._frozen_slow_value = copy.deepcopy(self._slow_value)
        for (name_orig, param_orig), (name_new, param_new) in zip(
            self._slow_value.named_parameters(), self._frozen_slow_value.named_parameters()
        ):
            assert name_orig == name_new
            param_new.data = param_orig.data
            param_new.requires_grad_(False)

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        # Re-establish shared memory after moving the model to a new device
        self.clone_and_freeze()
        return self

    @torch.no_grad()
    def act(self, obs, state, eval=False):
        """Policy inference step.

        ``state`` carries ``stoch``, ``deter``, ``prev_action`` (always) plus an
        optional nested ``extra`` TensorDict holding per-env recurrent state for
        backbones that need it (S4/S5 SSM hidden state, Mamba conv state,
        Transformer KV/token cache). For GRU ``extra`` is empty.
        """
        # obs: dict of (B, *), state: (stoch: (B, S, K), deter: (B, D), prev_action: (B, A))
        self._mark_step_begin()
        p_obs = self.preprocess(obs)
        # (B, E)
        embed = self._frozen_encoder(p_obs)
        prev_stoch, prev_deter, prev_action = (
            state["stoch"],
            state["deter"],
            state["prev_action"],
        )
        prev_extra = self._unpack_extra(state, prev_stoch.shape[0])
        # (B, S, K), (B, D)
        stoch, deter, _, extra = self._frozen_rssm.obs_step(
            prev_stoch, prev_deter, prev_action, embed, obs["is_first"], extra=prev_extra
        )
        # (B, F_policy) — SSSM hidden-state policy uses extra["x_t"], others
        # default to cat([stoch, deter]).
        policy_feat = self._frozen_rssm.policy_feat(stoch, deter, extra)
        action_dist = self._frozen_actor(policy_feat)
        # (B, A)
        action = action_dist.mode if eval else action_dist.rsample()
        new_state = {"stoch": stoch, "deter": deter, "prev_action": action}
        new_state.update(self._pack_extra(extra, prev_stoch.shape[0]))
        return action, TensorDict(new_state, batch_size=state.batch_size)

    @torch.no_grad()
    def get_initial_state(self, B):
        stoch, deter, extra = self.rssm.initial(B)
        action = torch.zeros(B, self.act_dim, dtype=torch.float32, device=self.device)
        td = {"stoch": stoch, "deter": deter, "prev_action": action}
        td.update(self._pack_extra(extra, B))
        return TensorDict(td, batch_size=(B,))

    def _pack_extra(self, extra, batch_size):
        """Wrap an extra-state dict for storage in the agent_state TensorDict.

        Returns a dict mapping the single key ``"extra"`` to a nested
        TensorDict, or an empty dict if there is no extra state to carry.
        Empty dicts skip the ``"extra"`` key entirely so GRU's agent_state
        stays unchanged.
        """
        if not extra:
            return {}
        return {"extra": TensorDict(dict(extra), batch_size=(batch_size,))}

    def _unpack_extra(self, state, batch_size):
        """Pull the per-env extra-state dict out of the agent_state TensorDict.

        Falls back to the backbone's zero-initialized state if absent (e.g. for
        the very first step before any extra has been written, or for GRU which
        never stores anything).
        """
        if "extra" in state.keys():
            extra_td = state["extra"]
            return {key: extra_td[key] for key in extra_td.keys()}
        return self._frozen_rssm._initial_extra(batch_size)

    @torch.no_grad()
    def video_pred(self, data, initial):
        self._mark_step_begin()
        p_data = self.preprocess(data)
        return self._video_pred(p_data, initial)

    def _video_pred(self, data, initial):
        """Video prediction utility."""
        if self.rep_loss != "dreamer":
            raise NotImplementedError("video_pred requires decoder and is only supported when rep_loss == 'dreamer'.")

        B = min(data["action"].shape[0], 6)
        # (B, T, E)
        embed = self.encoder(data)

        post_stoch, post_deter, _ = self.rssm.observe(
            embed[:B, :5],
            data["action"][:B, :5],
            tuple(val[:B] for val in initial),
            data["is_first"][:B, :5],
        )
        recon = self.decoder(post_stoch, post_deter)["image"].mode()[:B]
        init_stoch, init_deter = post_stoch[:, -1], post_deter[:, -1]
        prior_stoch, prior_deter = self.rssm.imagine_with_action(
            init_stoch,
            init_deter,
            data["action"][:B, 5:],
        )
        openl = self.decoder(prior_stoch, prior_deter)["image"].mode()
        model = torch.cat([recon[:, :5], openl], 1)
        truth = data["image"][:B]
        error = (model - truth + 1.0) / 2.0
        return torch.cat([truth, model, error], 2)

    def update(self, replay_buffer):
        """Sample a batch from replay and perform one optimization step."""
        data, index, initial, replay_metrics = replay_buffer.sample()
        self._mark_step_begin()
        p_data = self.preprocess(data)
        self._update_slow_target()
        if self.rep_loss == "dreamerpro":
            self.ema_update()
        policy_start = None
        policy_metrics = {}
        if getattr(replay_buffer, "strategy", "uniform") == "dfs":
            batch_shape = p_data.shape
            policy_start, policy_metrics = replay_buffer.sample_policy_starts(int(batch_shape[0] * batch_shape[1]))
        metrics = {}
        with autocast(device_type=self.device.type, dtype=torch.float16, enabled=self._amp_enabled):
            (stoch, deter), mets = self._cal_grad(p_data, initial, policy_start)
        self._scaler.unscale_(self._optimizer)  # unscale grads in params
        if self.rep_loss == "dreamerpro" and self._ema_updates < self.freeze_prototypes_iters:
            self._prototypes.grad.zero_()
        if self._log_grads:
            old_params = [p.data.clone().detach() for p in self._named_params.values()]
            grads = [p.grad for p in self._named_params.values() if p.grad is not None]  # log grads before clipping
            grad_norm = tools.compute_global_norm(grads)
            grad_rms = tools.compute_rms(grads)
            mets["opt/grad_norm"] = grad_norm
            mets["opt/grad_rms"] = grad_rms
        self._agc(self._named_params.values())  # clipping
        self._scaler.step(self._optimizer)  # update params
        self._scaler.update()  # adjust scale
        self._scheduler.step()  # increment scheduler
        self._optimizer.zero_grad(set_to_none=True)  # reset grads
        mets["opt/lr"] = self._scheduler.get_lr()[0]
        mets["opt/grad_scale"] = self._scaler.get_scale()
        if self._log_grads:
            updates = [(new - old) for (new, old) in zip(self._named_params.values(), old_params)]
            update_rms = tools.compute_rms(updates)
            params_rms = tools.compute_rms(self._named_params.values())
            mets["opt/param_rms"] = params_rms
            mets["opt/update_rms"] = update_rms
        metrics.update(replay_metrics)
        metrics.update(policy_metrics)
        metrics.update(mets)
        # update latent vectors in replay buffer
        replay_buffer.update(index, stoch.detach(), deter.detach())
        return metrics

    def _cal_grad(self, data, initial, policy_start=None):
        """Compute gradients for one batch.

        Notes
        -----
        This function computes:
        1) World model loss (dynamics + representation)
        2) Optional representation loss variants (Dreamer, R2-Dreamer, InfoNCE, DreamerPro)
        3) Imagination rollouts for actor-critic updates
        4) Replay-based value learning
        """
        # data: dict of (B, T, *), initial: (stoch: (B, S, K), deter: (B, D))
        losses = {}
        metrics = {}
        B, T = data.shape

        # === World model: posterior rollout and KL losses ===
        # (B, T, E)
        embed = self.encoder(data)
        # Burn-in mask: zero out the first `burn_in` time steps so the
        # backbone's persistent extra-state can warm up before its predictions
        # are graded. `min(burn_in, T-1)` keeps at least one step contributing.
        eff_burn = min(int(self.burn_in), max(int(T) - 1, 0))
        time_mask = data["action"].new_ones(B, T)
        if eff_burn > 0:
            time_mask[:, :eff_burn] = 0.0
        mask_norm = time_mask.sum().clamp(min=1.0)
        # (B, T, S, K), (B, T, D), (B, T, S, K), per-step extras
        post_stoch, post_deter, post_logit, post_extras = self.rssm.observe(
            embed, data["action"], initial, data["is_first"], return_extra=True
        )
        # (B, T, S, K)
        _, prior_logit = self.rssm.prior(post_deter)
        dyn_loss, rep_loss = self.rssm.kl_loss(post_logit, prior_logit, self.kl_free)
        losses["dyn"] = (dyn_loss * time_mask).sum() / mask_norm
        losses["rep"] = (rep_loss * time_mask).sum() / mask_norm
        # === Representation / auxiliary losses ===
        # (B, T, F)
        feat = self.rssm.get_feat(post_stoch, post_deter)
        if self.rep_loss == "dreamer":
            recon_losses = {}
            for key, dist in self.decoder(post_stoch, post_deter).items():
                # log_prob over the data; mask burn-in time steps before averaging.
                lp = dist.log_prob(data[key])
                # `lp` may have a trailing feature dim; broadcast time_mask over it.
                while lp.dim() > time_mask.dim():
                    lp = lp.sum(dim=-1)
                recon_losses[key] = (-lp * time_mask).sum() / mask_norm
            losses.update(recon_losses)
        elif self.rep_loss == "r2dreamer":
            # R2-Dreamer: Barlow Twins style redundancy reduction between latent features and encoder embeddings.
            # Flatten batch/time dims for a single cross-correlation matrix.
            # (B, T, F) -> (B*T, F)
            x1 = self.prj(feat[:, :].reshape(B * T, -1))
            # (B, T, E) -> (B*T, E)
            x2 = embed.reshape(B * T, -1).detach()  # this detach is important

            x1_norm = (x1 - x1.mean(0)) / (x1.std(0) + 1e-8)
            x2_norm = (x2 - x2.mean(0)) / (x2.std(0) + 1e-8)

            c = torch.mm(x1_norm.T, x2_norm) / (B * T)
            invariance_loss = (torch.diagonal(c) - 1.0).pow(2).sum()
            off_diag_mask = ~torch.eye(x1.shape[-1], dtype=torch.bool, device=x1.device)
            redundancy_loss = c[off_diag_mask].pow(2).sum()
            losses["barlow"] = invariance_loss + self.barlow_lambd * redundancy_loss
        elif self.rep_loss == "infonce":
            # Contrastive (InfoNCE) objective between projected latent features and encoder embeddings.
            # (B, T, F) -> (B*T, F)
            x1 = self.prj(feat[:, :].reshape(B * T, -1))
            # (B, T, E) -> (B*T, E)
            x2 = embed.reshape(B * T, -1).detach()  # this detach is important
            logits = torch.matmul(x1, x2.T)
            norm_logits = logits - torch.max(logits, 1)[0][:, None]
            labels = torch.arange(norm_logits.shape[0]).long().to(self.device)
            losses["infonce"] = torch.nn.functional.cross_entropy(norm_logits, labels)
        elif self.rep_loss == "dreamerpro":
            # DreamerPro uses augmentation + EMA targets + Sinkhorn assignment.
            with torch.no_grad():
                data_aug = self.augment_data(data)
                initial_aug = (
                    # (B, ...) -> (2B, ...)
                    torch.cat([initial[0], initial[0]], dim=0),
                    torch.cat([initial[1], initial[1]], dim=0),
                )
                ema_proj = self.ema_proj(data_aug)

            embed_aug = self.encoder(data_aug)
            post_stoch_aug, post_deter_aug, _ = self.rssm.observe(
                embed_aug, data_aug["action"], initial_aug, data_aug["is_first"]
            )
            proto_losses = self.proto_loss(post_stoch_aug, post_deter_aug, embed_aug, ema_proj)
            losses.update(proto_losses)
        else:
            raise NotImplementedError
        if self.cpc_enabled:
            losses["cpc"] = self.cpc_loss(data, feat, post_stoch, post_deter, post_extras)

        # reward and continue
        rew_lp = self.reward(feat).log_prob(to_f32(data["reward"]))
        cont = 1.0 - to_f32(data["is_terminal"])
        cont_lp = self.cont(feat).log_prob(cont)
        # Apply burn-in mask consistently with the world-model losses above.
        while rew_lp.dim() > time_mask.dim():
            rew_lp = rew_lp.sum(dim=-1)
        while cont_lp.dim() > time_mask.dim():
            cont_lp = cont_lp.sum(dim=-1)
        losses["rew"] = (-rew_lp * time_mask).sum() / mask_norm
        losses["con"] = (-cont_lp * time_mask).sum() / mask_norm
        # log
        metrics["dyn_entropy"] = torch.mean(self.rssm.get_dist(prior_logit).entropy())
        metrics["rep_entropy"] = torch.mean(self.rssm.get_dist(post_logit).entropy())

        # === Imagination rollout for actor-critic ===
        # (B*T, S, K), (B*T, D)
        if policy_start is None:
            start = (
                post_stoch.reshape(-1, *post_stoch.shape[2:]).detach(),
                post_deter.reshape(-1, *post_deter.shape[2:]).detach(),
            )
            # Per-time-step posterior extras: (B, T, ...) -> (B*T, ...). Each
            # imagination starting point inherits the matching SSM/cache state
            # from its source time step instead of starting from zeros.
            initial_extra = {
                key: val.reshape(-1, *val.shape[2:]).detach()
                for key, val in post_extras.items()
            }
        else:
            start = (policy_start[0].detach(), policy_start[1].detach())
            # `policy_start` comes from DFS sampling and only carries (stoch, deter)
            # for now; imagination starts from a fresh extra-state in that path.
            initial_extra = None
        # (B, T, ...) -> (B*T, ...)
        imag_feat, imag_policy_feat, imag_action = self._imagine(
            start, self.imag_horizon + 1, initial_extra=initial_extra
        )
        imag_feat = imag_feat.detach()
        imag_policy_feat = imag_policy_feat.detach()
        imag_action = imag_action.detach()

        # (B*T, T_imag, 1) — reward/cont/value all use the world-model feat.
        imag_reward = self._frozen_reward(imag_feat).mode()
        # (B*T, T_imag, 1)  probability of continuation
        imag_cont = self._frozen_cont(imag_feat).mean
        # (B*T, T_imag, 1)
        imag_value = self._frozen_value(imag_feat).mode()
        imag_slow_value = self._frozen_slow_value(imag_feat).mode()
        disc = 1 - 1 / self.horizon
        # (B*T, T_imag, 1)
        weight = torch.cumprod(imag_cont * disc, dim=1)
        last = torch.zeros_like(imag_cont)
        term = 1 - imag_cont
        ret = self._lambda_return(
            last, term, imag_reward, imag_value, imag_value, disc, self.lamb
        )  # (B*T, T_imag-1, 1)
        ret_offset, ret_scale = self.return_ema(ret)
        # (B*T, T_imag-1, 1)
        adv = (ret - imag_value[:, :-1]) / ret_scale

        # Actor sees policy feat (cat([z, x_t]) for SSSM, cat([z, h]) otherwise).
        policy = self.actor(imag_policy_feat)
        # (B*T, T_imag-1, 1)
        logpi = policy.log_prob(imag_action)[:, :-1].unsqueeze(-1)
        entropy = policy.entropy()[:, :-1].unsqueeze(-1)
        losses["policy"] = torch.mean(weight[:, :-1].detach() * -(logpi * adv.detach() + self.act_entropy * entropy))

        imag_value_dist = self.value(imag_feat)
        # (B*T, T_imag, 1)
        tar_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)
        losses["value"] = torch.mean(
            weight[:, :-1].detach()
            * (-imag_value_dist.log_prob(tar_padded.detach()) - imag_value_dist.log_prob(imag_slow_value.detach()))[
                :, :-1
            ].unsqueeze(-1)
        )
        # log
        ret_normed = (ret - ret_offset) / ret_scale
        metrics["ret"] = torch.mean(ret_normed)
        metrics["ret_005"] = self.return_ema.ema_vals[0]
        metrics["ret_095"] = self.return_ema.ema_vals[1]
        metrics["adv"] = torch.mean(adv)
        metrics["adv_std"] = torch.std(adv)
        metrics["con"] = torch.mean(imag_cont)
        metrics["rew"] = torch.mean(imag_reward)
        metrics["val"] = torch.mean(imag_value)
        metrics["tar"] = torch.mean(ret)
        metrics["slowval"] = torch.mean(imag_slow_value)
        metrics["weight"] = torch.mean(weight)
        metrics["action_entropy"] = torch.mean(entropy)
        metrics.update(tools.tensorstats(imag_action, "action"))

        # === Replay-based value learning (keep gradients through world model) ===
        last, term, reward = (
            to_f32(data["is_last"]),
            to_f32(data["is_terminal"]),
            to_f32(data["reward"]),
        )
        feat = self.rssm.get_feat(post_stoch, post_deter)
        boot = ret[:, 0].reshape(B, T, 1)
        value = self._frozen_value(feat).mode()
        slow_value = self._frozen_slow_value(feat).mode()
        disc = 1 - 1 / self.horizon
        weight = 1.0 - last
        ret = self._lambda_return(last, term, reward, value, boot, disc, self.lamb)
        ret_padded = torch.cat([ret, 0 * ret[:, -1:]], 1)

        # Keep this attached to the world model so gradients can flow through
        value_dist = self.value(feat)
        losses["repval"] = torch.mean(
            weight[:, :-1]
            * (-value_dist.log_prob(ret_padded.detach()) - value_dist.log_prob(slow_value.detach()))[:, :-1].unsqueeze(
                -1
            )
        )
        # log
        metrics.update(tools.tensorstats(ret, "ret_replay"))
        metrics.update(tools.tensorstats(value, "value_replay"))
        metrics.update(tools.tensorstats(slow_value, "slow_value_replay"))

        total_loss = sum([v * self._loss_scales[k] for k, v in losses.items()])
        self._scaler.scale(total_loss).backward()

        metrics.update({f"loss/{name}": loss for name, loss in losses.items()})
        metrics.update({"opt/loss": total_loss})
        return (post_stoch, post_deter), metrics

    @torch.no_grad()
    def _imagine(self, start, imag_horizon, initial_extra=None):
        """Roll out the policy in latent space.

        ``initial_extra`` is the per-env recurrent state to seed the rollout
        with (matching SSM hidden state / transformer cache for the
        corresponding posterior time step). When omitted, the rollout starts
        from a fresh zero extra-state — appropriate for "what-if from a
        checkpoint" calls where no real prior context is available.

        Returns ``(world_feats, policy_feats, actions)`` of shapes
        ``(B, T, feat_size)``, ``(B, T, policy_feat_size)``, ``(B, T, act_dim)``.
        For GRU/transformer, ``policy_feats == world_feats`` (they share dim);
        for SSSM backbones the actor sees ``policy_feats = cat([z, x_t])``.
        """
        # (B, S, K), (B, D)
        world_feats = []
        policy_feats = []
        actions = []
        stoch, deter = start
        if initial_extra is None:
            extra = self._frozen_rssm._initial_extra(stoch.shape[0])
        else:
            extra = initial_extra
        step_context = self._frozen_rssm._prepare_step_context()
        for _ in range(imag_horizon):
            # World-model feat (used by reward/cont/value heads).
            world_feat = self._frozen_rssm.get_feat(stoch, deter)
            # Policy feat (used by actor — equals world_feat for non-SSSM).
            policy_feat = self._frozen_rssm.policy_feat(stoch, deter, extra)
            action = self._frozen_actor(policy_feat).rsample()
            world_feats.append(world_feat)
            policy_feats.append(policy_feat)
            actions.append(action)
            stoch, deter, extra = self._frozen_rssm.img_step(
                stoch, deter, action, extra=extra, step_context=step_context
            )

        # Stack along sequence dim T_imag.
        return (
            torch.stack(world_feats, dim=1),
            torch.stack(policy_feats, dim=1),
            torch.stack(actions, dim=1),
        )

    @torch.no_grad()
    def _lambda_return(self, last, term, reward, value, boot, disc, lamb):
        """
        lamb=1 means discounted Monte Carlo return.
        lamb=0 means fixed 1-step return.
        """
        assert last.shape == term.shape == reward.shape == value.shape == boot.shape
        live = (1 - to_f32(term))[:, 1:] * disc
        cont = (1 - to_f32(last))[:, 1:] * lamb
        interm = reward[:, 1:] + (1 - cont) * live * boot[:, 1:]
        out = [boot[:, -1]]
        for i in reversed(range(live.shape[1])):
            out.append(interm[:, i] + live[:, i] * cont[:, i] * out[-1])
        return torch.stack(list(reversed(out))[:-1], 1)

    def cpc_loss(self, data, feat, post_stoch, post_deter, post_extras=None):
        """Action-conditioned CPC over real latent states and augmented encodings.

        ``post_extras`` (per-time-step recurrent state from the posterior
        observe pass) is sliced and flattened so each ``k``-step rollout
        starts with the matching extra-state. When ``None``, rollouts start
        from a fresh zero extra-state (legacy behavior).
        """
        aug_data = self.augment_cpc_data(data)
        target_embed = self.encoder(aug_data)
        B, T = target_embed.shape[:2]
        horizon = min(self.cpc_horizon, T)
        losses = []

        for k in range(horizon):
            if k == 0:
                pred_feat = feat
                target = target_embed
            else:
                roll_extra = None
                if post_extras:
                    roll_extra = {
                        key: val[:, : T - k].reshape(-1, *val.shape[2:])
                        for key, val in post_extras.items()
                    }
                pred_feat = self.rollout_actions(
                    post_stoch[:, : T - k], post_deter[:, : T - k], data["action"], k,
                    initial_extra=roll_extra,
                )
                target = target_embed[:, k:]

            pred = self.cpc_pred_heads[k](pred_feat.reshape(-1, pred_feat.shape[-1]))
            target = self.cpc_target_heads[k](target.reshape(-1, target.shape[-1]))
            pred = F.normalize(pred, p=2, dim=-1)
            target = F.normalize(target, p=2, dim=-1)
            logits = torch.matmul(pred, target.T) / self.cpc_temperature
            labels = torch.arange(logits.shape[0], device=logits.device)
            losses.append(F.cross_entropy(logits, labels))

        return torch.stack(losses).mean()

    def rollout_actions(self, stoch, deter, actions, horizon, initial_extra=None):
        """Roll out the world model for `horizon` real actions from every valid start state.

        ``initial_extra`` (per-time-step posterior extras flattened to
        ``(B*valid_steps, ...)``) seeds the imagined rollout with the matching
        recurrent state. Defaults to a fresh zero extra-state for callers that
        don't provide it (e.g. callers outside ``_cal_grad``).
        """
        B, valid_steps = stoch.shape[:2]
        stoch = stoch.reshape(-1, *stoch.shape[2:])
        deter = deter.reshape(-1, deter.shape[-1])
        if initial_extra is None:
            extra = self.rssm._initial_extra(stoch.shape[0])
        else:
            extra = initial_extra
        step_context = self.rssm._prepare_step_context()
        for step in range(horizon):
            action = actions[:, step : step + valid_steps].reshape(-1, actions.shape[-1])
            stoch, deter, extra = self.rssm.img_step(
                stoch, deter, action, extra=extra, step_context=step_context
            )
        feat = self.rssm.get_feat(stoch, deter)
        return feat.reshape(B, valid_steps, -1)

    @torch.no_grad()
    def preprocess(self, data):
        if "image" in data:
            data["image"] = to_f32(data["image"]) / 255.0
        return data

    @torch.no_grad()
    def augment_cpc_data(self, data):
        data_aug = {k: v.clone() for k, v in data.items()}
        if "image" in data_aug:
            image = data_aug["image"].permute(0, 1, 4, 2, 3)
            data_aug["image"] = self.random_translate(
                image,
                self.cpc_aug_max_delta,
                same_across_time=self.cpc_aug_same_across_time,
                bilinear=self.cpc_aug_bilinear,
            )
            data_aug["image"] = data_aug["image"].permute(0, 1, 3, 4, 2)
        return data_aug

    @torch.no_grad()
    def augment_data(self, data):
        data_aug = {k: torch.cat([v, v], axis=0) for k, v in data.items()}
        # (B, T, H, W, C) -> (B, T, C, H, W)
        image = data_aug["image"].permute(0, 1, 4, 2, 3)
        data_aug["image"] = self.random_translate(
            image,
            self.aug_max_delta,
            same_across_time=self.aug_same_across_time,
            bilinear=self.aug_bilinear,
        )
        # (B, T, C, H, W) -> (B, T, H, W, C)
        data_aug["image"] = data_aug["image"].permute(0, 1, 3, 4, 2)
        return data_aug

    @torch.no_grad()
    def ema_proj(self, data):
        with torch.no_grad():
            embed = self._ema_encoder(data)
            proj = self._ema_obs_proj(embed)
        return F.normalize(proj, p=2, dim=-1)

    @torch.no_grad()
    def ema_update(self):
        prototypes = F.normalize(self._prototypes, p=2, dim=-1)
        self._prototypes.data.copy_(prototypes)
        if self._ema_updates % self.ema_update_every == 0:
            mix = self.ema_update_fraction if self._ema_updates > 0 else 1.0
            for s, d in zip(self.encoder.parameters(), self._ema_encoder.parameters()):
                d.data.copy_(mix * s.data + (1 - mix) * d.data)
            for s, d in zip(self.obs_proj.parameters(), self._ema_obs_proj.parameters()):
                d.data.copy_(mix * s.data + (1 - mix) * d.data)
        self._ema_updates += 1

    def sinkhorn(self, scores):
        """Sinkhorn-Knopp normalization.

        Notes
        -----
        Given a score matrix, we iteratively normalize rows and columns in log
        space so that the resulting assignment matrix is approximately doubly
        stochastic.
        """
        shape = scores.shape
        K = shape[0]
        scores = scores.reshape(-1)
        log_Q = F.log_softmax(scores / self.sinkhorn_eps, dim=0)
        log_Q = log_Q.reshape(K, -1)
        N = log_Q.shape[1]
        for _ in range(self.sinkhorn_iters):
            log_row_sums = torch.logsumexp(log_Q, dim=1, keepdim=True)
            log_Q = log_Q - log_row_sums - math.log(K)
            log_col_sums = torch.logsumexp(log_Q, dim=0, keepdim=True)
            log_Q = log_Q - log_col_sums - math.log(N)
        log_Q = log_Q + math.log(N)
        Q = torch.exp(log_Q)
        return Q.reshape(shape)

    def proto_loss(self, post_stoch, post_deter, embed, ema_proj):
        prototypes = F.normalize(self._prototypes, p=2, dim=-1)

        obs_proj = self.obs_proj(embed)
        obs_norm = torch.norm(obs_proj, dim=-1)
        obs_proj = F.normalize(obs_proj, p=2, dim=-1)

        B, T = obs_proj.shape[:2]
        # (B, T, P) -> (B*T, P)
        obs_proj = obs_proj.reshape(B * T, -1)
        obs_scores = torch.matmul(obs_proj, prototypes.T)
        # (B*T, K) -> (B, T, K) -> (K, B, T)
        obs_scores = obs_scores.reshape(B, T, -1).permute(2, 0, 1)
        obs_scores = obs_scores[:, :, self.warm_up :]
        obs_logits = F.log_softmax(obs_scores / self.temperature, dim=0)
        obs_logits_1, obs_logits_2 = torch.chunk(obs_logits, 2, dim=1)

        # (B, T, P) -> (B*T, P)
        ema_proj = ema_proj.reshape(B * T, -1)
        ema_scores = torch.matmul(ema_proj, prototypes.T)
        # (B*T, K) -> (B, T, K) -> (K, B, T)
        ema_scores = ema_scores.reshape(B, T, -1).permute(2, 0, 1)
        ema_scores = ema_scores[:, :, self.warm_up :]
        ema_scores_1, ema_scores_2 = torch.chunk(ema_scores, 2, dim=1)

        with torch.no_grad():
            ema_targets_1 = self.sinkhorn(ema_scores_1)
            ema_targets_2 = self.sinkhorn(ema_scores_2)
        ema_targets = torch.cat([ema_targets_1, ema_targets_2], dim=1)

        feat = self.rssm.get_feat(post_stoch, post_deter)
        feat_proj = self.feat_proj(feat)
        feat_norm = torch.norm(feat_proj, dim=-1)
        feat_proj = F.normalize(feat_proj, p=2, dim=-1)

        # (B, T, P) -> (B*T, P)
        feat_proj = feat_proj.reshape(B * T, -1)
        feat_scores = torch.matmul(feat_proj, prototypes.T)
        # (B*T, K) -> (B, T, K) -> (K, B, T)
        feat_scores = feat_scores.reshape(B, T, -1).permute(2, 0, 1)
        feat_scores = feat_scores[:, :, self.warm_up :]
        feat_logits = F.log_softmax(feat_scores / self.temperature, dim=0)

        swav_loss = -0.5 * torch.mean(torch.sum(ema_targets_2 * obs_logits_1, dim=0)) - 0.5 * torch.mean(
            torch.sum(ema_targets_1 * obs_logits_2, dim=0)
        )
        temp_loss = -torch.mean(torch.sum(ema_targets * feat_logits, dim=0))
        norm_loss = torch.mean(torch.square(obs_norm - 1)) + torch.mean(torch.square(feat_norm - 1))

        return {
            "swav": swav_loss,
            "temp": temp_loss,
            "norm": norm_loss,
        }

    @torch.no_grad()
    def random_translate(self, x, max_delta, same_across_time=False, bilinear=False):
        B, T, C, H, W = x.shape
        x_flat = x.reshape(B * T, C, H, W)
        pad = int(max_delta)

        # Pad
        x_padded = F.pad(x_flat, (pad, pad, pad, pad), "replicate")
        h_padded, w_padded = H + 2 * pad, W + 2 * pad

        # Create base grid
        eps_h = 1.0 / h_padded
        eps_w = 1.0 / w_padded
        arange_h = torch.linspace(-1.0 + eps_h, 1.0 - eps_h, h_padded, device=x.device, dtype=x.dtype)[:H]
        arange_w = torch.linspace(-1.0 + eps_w, 1.0 - eps_w, w_padded, device=x.device, dtype=x.dtype)[:W]
        arange_h = arange_h.unsqueeze(1).repeat(1, W).unsqueeze(2)
        arange_w = arange_w.unsqueeze(0).repeat(H, 1).unsqueeze(2)
        base_grid = torch.cat([arange_w, arange_h], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(B * T, 1, 1, 1)

        # Create shift
        if same_across_time:
            shift = torch.randint(0, 2 * pad + 1, size=(B, 1, 1, 1, 2), device=x.device, dtype=x.dtype)
            shift = shift.repeat(1, T, 1, 1, 1).reshape(B * T, 1, 1, 2)
        else:
            shift = torch.randint(0, 2 * pad + 1, size=(B * T, 1, 1, 2), device=x.device, dtype=x.dtype)

        shift = shift * 2.0 / torch.tensor([w_padded, h_padded], device=x.device, dtype=x.dtype)

        # Apply shift and sample
        grid = base_grid + shift
        mode = "bilinear" if bilinear else "nearest"
        x_translated = F.grid_sample(x_padded, grid, mode=mode, padding_mode="zeros", align_corners=False)

        return x_translated.reshape(B, T, C, H, W)
