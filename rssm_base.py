import torch
from torch import distributions as torchd
from torch import nn

import distributions as dists
from networks import LambdaLayer
from tools import rpad, weight_init_


class CategoricalRSSM(nn.Module):
    """Shared categorical Dreamer-style RSSM scaffold.

    Subclasses provide a deterministic transition module that maps
    `(stoch, deter, memory, action) -> (next_deter, next_memory)` and an
    `initial_memory(batch_size)` method. Backbones with no auxiliary cache
    return an empty `(B, 0)` tensor and a no-op pass-through for memory.
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
        self._recurrent_posterior = bool(getattr(config, "recurrent_posterior", True))
        self.flat_stoch = self._stoch * self._discrete
        self.feat_size = self.flat_stoch + self._deter
        self._deter_net = deter_net

        self._obs_net = nn.Sequential()
        inp_dim = (self._deter + embed_size) if self._recurrent_posterior else embed_size
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

    def initial(self, batch_size):
        """Return an initial latent state (stoch, deter, memory)."""
        deter = torch.zeros(batch_size, self._deter, dtype=torch.float32, device=self._device)
        stoch = torch.zeros(batch_size, self._stoch, self._discrete, dtype=torch.float32, device=self._device)
        memory = self._deter_net.initial_memory(batch_size)
        return stoch, deter, memory

    def observe(self, embed, action, initial, reset):
        """Posterior rollout using observations.

        Dispatches to a parallel implementation when the posterior does not
        depend on `deter` (`recurrent_posterior=False`) AND the backbone exposes
        a `forward_parallel` method. Otherwise falls back to the per-step loop.
        """
        if len(initial) == 2:
            stoch_init, deter_init = initial
            memory_init = self._deter_net.initial_memory(stoch_init.shape[0])
        else:
            stoch_init, deter_init, memory_init = initial

        if (not self._recurrent_posterior) and hasattr(self._deter_net, "forward_parallel"):
            return self._observe_parallel(embed, action, stoch_init, deter_init, memory_init, reset)

        length = action.shape[1]
        stoch, deter, memory = stoch_init, deter_init, memory_init
        stochs, deters, memories, logits = [], [], [], []
        for i in range(length):
            stoch, deter, memory, logit = self.obs_step(stoch, deter, memory, action[:, i], embed[:, i], reset[:, i])
            stochs.append(stoch)
            deters.append(deter)
            memories.append(memory)
            logits.append(logit)
        stochs = torch.stack(stochs, dim=1)
        deters = torch.stack(deters, dim=1)
        memories = torch.stack(memories, dim=1)
        logits = torch.stack(logits, dim=1)
        return stochs, deters, memories, logits

    def _observe_parallel(self, embed, action, stoch_init, deter_init, memory_init, reset):
        """Parallel posterior rollout for non-recurrent posterior backbones.

        Steps:
            1. Sample all stochs in one shot from q(z_t | o_t).
            2. Build (prev_stoch, prev_action) sequences by shifting the sampled
               stochs and the given actions by one step, with the initial state
               at position 0.
            3. Call backbone.forward_parallel to get per-position deters and
               memory snapshots.
        """
        B, T = action.shape[:2]
        # Sample posteriors for the whole segment in parallel.
        post_logit = self._obs_net(embed)  # (B, T, S, K)
        stochs = self.get_dist(post_logit).rsample()

        # Build prev_stoch / prev_action sequences (shifted by 1).
        prev_stoch = torch.cat([stoch_init.unsqueeze(1), stochs[:, :-1]], dim=1)
        prev_action = torch.cat([torch.zeros_like(action[:, :1]), action[:, :-1]], dim=1)

        deter_seq, memory_seq = self._deter_net.forward_parallel(prev_stoch, prev_action, memory_init, reset)
        return stochs, deter_seq, memory_seq, post_logit

    def obs_step(self, stoch, deter, memory, prev_action, embed, reset):
        """Single posterior step."""
        stoch = torch.where(rpad(reset, stoch.dim() - int(reset.dim())), torch.zeros_like(stoch), stoch)
        deter = torch.where(rpad(reset, deter.dim() - int(reset.dim())), torch.zeros_like(deter), deter)
        memory = self._deter_net.reset_memory(memory, reset)
        prev_action = torch.where(
            rpad(reset, prev_action.dim() - int(reset.dim())), torch.zeros_like(prev_action), prev_action
        )

        deter, memory = self._deter_net(stoch, deter, memory, prev_action)
        if self._recurrent_posterior:
            logit = self._obs_net(torch.cat([deter, embed], dim=-1))
        else:
            logit = self._obs_net(embed)
        stoch = self.get_dist(logit).rsample()
        return stoch, deter, memory, logit

    def img_step(self, stoch, deter, memory, prev_action):
        """Single prior step without observation."""
        deter, memory = self._deter_net(stoch, deter, memory, prev_action)
        stoch, _ = self.prior(deter)
        return stoch, deter, memory

    def prior(self, deter):
        """Compute prior distribution parameters and sample stochastic state."""
        logit = self._img_net(deter)
        stoch = self.get_dist(logit).rsample()
        return stoch, logit

    def imagine_with_action(self, stoch, deter, memory, actions):
        """Roll out prior dynamics given a sequence of actions."""
        length = actions.shape[1]
        stochs, deters, memories = [], [], []
        for i in range(length):
            stoch, deter, memory = self.img_step(stoch, deter, memory, actions[:, i])
            stochs.append(stoch)
            deters.append(deter)
            memories.append(memory)
        stochs = torch.stack(stochs, dim=1)
        deters = torch.stack(deters, dim=1)
        memories = torch.stack(memories, dim=1)
        return stochs, deters, memories

    def get_feat(self, stoch, deter):
        """Flatten stochastic state and concatenate with deterministic state."""
        stoch = stoch.reshape(*stoch.shape[:-2], self.flat_stoch)
        return torch.cat([stoch, deter], -1)

    def get_dist(self, logit):
        return torchd.independent.Independent(dists.OneHotDist(logit, unimix_ratio=self._unimix_ratio), 1)

    def kl_loss(self, post_logit, prior_logit, free):
        rep_loss = dists.kl(post_logit, prior_logit.detach()).sum(-1)
        dyn_loss = dists.kl(post_logit.detach(), prior_logit).sum(-1)
        rep_loss = torch.clip(rep_loss, min=free)
        dyn_loss = torch.clip(dyn_loss, min=free)
        return dyn_loss, rep_loss
