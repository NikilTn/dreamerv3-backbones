import torch
from tensordict import TensorDict
from torchrl.data.replay_buffers import LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import PrioritizedSliceSampler, SliceSampler


class Buffer:
    def __init__(self, config):
        self.device = torch.device(config.device)
        self.storage_device = torch.device(config.storage_device)
        self.batch_size = int(config.batch_size)
        self.batch_length = int(config.batch_length)
        self.slice_batch = self.batch_size * (self.batch_length + 1)
        self.num_eps = 0
        self.strategy = str(config.sampling.strategy).lower()
        self._dfs_enabled = self.strategy == "dfs"
        self._dfs_world_temp = float(config.sampling.dfs.world_temperature)
        self._dfs_policy_temp = float(config.sampling.dfs.policy_temperature)
        self._dfs_alpha = float(config.sampling.dfs.alpha)
        self._dfs_beta = float(config.sampling.dfs.beta)
        self._dfs_eps = float(config.sampling.dfs.eps)
        self._dfs_priority_clip = float(config.sampling.dfs.priority_clip)
        self._buffer = ReplayBuffer(
            storage=LazyTensorStorage(max_size=config.max_size, device=self.storage_device, ndim=2),
            sampler=SliceSampler(num_slices=self.batch_size, end_key=None, traj_key="episode", truncated_key=None, strict_length=True),
            prefetch=0,
            batch_size=self.slice_batch,  # +1 for context
        )
        self._uniform_seq_sampler = SliceSampler(
            num_slices=self.batch_size, end_key=None, traj_key="episode", truncated_key=None, strict_length=True
        )
        self._storage_capacity_shape = None
        self._world_counts = None
        self._policy_counts = None
        if self._dfs_enabled:
            self._world_sampler = PrioritizedSliceSampler(
                max_capacity=int(config.max_size),
                alpha=self._dfs_alpha,
                beta=self._dfs_beta,
                eps=self._dfs_eps,
                num_slices=self.batch_size,
                end_key=None,
                traj_key="episode",
                truncated_key=None,
                strict_length=True,
            )
        else:
            self._world_sampler = None

    def _ensure_dfs_state(self):
        if not self._dfs_enabled or self._storage_capacity_shape is not None:
            return
        self._storage_capacity_shape = tuple(int(x) for x in self._buffer.storage._storage.shape[:2])
        self._world_counts = torch.zeros(self._storage_capacity_shape, dtype=torch.float32)
        self._policy_counts = torch.zeros(self._storage_capacity_shape, dtype=torch.float32)

    def _to_device(self, sample_td):
        src_dev = sample_td.device
        if src_dev.type == "cpu" and self.device.type == "cuda":
            sample_td = sample_td.pin_memory().to(self.device, non_blocking=True)
        elif src_dev != self.device:
            sample_td = sample_td.to(self.device, non_blocking=True)
        return sample_td

    def _stack_index(self, time_idx, env_idx):
        return torch.stack([time_idx, env_idx], dim=-1).to(dtype=torch.int64, device="cpu")

    def _split_index(self, index_tensor):
        index_tensor = index_tensor.to(device="cpu", dtype=torch.int64)
        return index_tensor[:, 0], index_tensor[:, 1]

    def _world_priority(self, counts):
        logits = -counts / max(self._dfs_world_temp, 1e-6)
        return torch.exp(torch.clamp(logits, min=-self._dfs_priority_clip)) + self._dfs_eps

    def _policy_priority(self, world_counts, policy_counts):
        imbalance = world_counts - policy_counts
        logits = (imbalance - torch.clamp(imbalance, min=0.0)) / max(self._dfs_policy_temp, 1e-6)
        return torch.exp(torch.clamp(logits, min=-self._dfs_priority_clip)) + self._dfs_eps

    def _update_priorities(self, index_tensor):
        if not self._dfs_enabled or index_tensor.numel() == 0:
            return
        time_idx, env_idx = self._split_index(index_tensor)
        world_counts = self._world_counts[time_idx, env_idx]
        policy_counts = self._policy_counts[time_idx, env_idx]
        world_priority = self._world_priority(world_counts)
        self._world_sampler.update_priority(index_tensor, world_priority, storage=self._buffer.storage)

    def _current_storage_shape(self):
        shape = self._buffer.storage.shape
        if shape is None:
            return None
        return tuple(int(x) for x in shape[:2])

    def _sample_with_sampler(self, sampler, batch_size):
        index, _ = sampler.sample(self._buffer.storage, batch_size=batch_size)
        sample_td = self._buffer.storage[index[0], index[1]]
        return sample_td, index

    def _sample_world_batch(self):
        if self._dfs_enabled:
            sample_td, full_index = self._sample_with_sampler(self._world_sampler, self.slice_batch)
        else:
            sample_td, full_index = self._sample_with_sampler(self._uniform_seq_sampler, self.slice_batch)
        sample_td = sample_td.view(-1, self.batch_length + 1)
        sample_td = self._to_device(sample_td)
        initial = (sample_td["stoch"][:, 0], sample_td["deter"][:, 0])
        data = sample_td[:, 1:]
        data.set_("action", sample_td["action"][:, :-1])  # action is 1 step back
        index = [ind.view(-1, self.batch_length + 1)[:, 1:] for ind in full_index]
        metrics = {}
        if self._dfs_enabled:
            sampled_index = self._stack_index(index[0].reshape(-1), index[1].reshape(-1))
            self._world_counts[sampled_index[:, 0], sampled_index[:, 1]] += 1
            self._update_priorities(sampled_index)
            start_time = full_index[0].view(-1, self.batch_length + 1)[:, 0]
            start_env = full_index[1].view(-1, self.batch_length + 1)[:, 0]
            start_index = self._stack_index(start_time.reshape(-1), start_env.reshape(-1))
            seq_counts = self._world_counts[sampled_index[:, 0], sampled_index[:, 1]].view(self.batch_size, self.batch_length)
            start_priority = self._world_priority(seq_counts.mean(dim=-1))
            self._world_sampler.update_priority(start_index, start_priority, storage=self._buffer.storage)
            metrics["buffer/world_count_mean"] = self._world_counts[sampled_index[:, 0], sampled_index[:, 1]].mean()
            metrics["buffer/policy_count_mean"] = self._policy_counts[sampled_index[:, 0], sampled_index[:, 1]].mean()
        return data, index, initial, metrics

    def add_transition(self, data):
        # This is batched data and lifted for storage.
        # (B, ...) -> (B, 1, ...)
        index = self._buffer.extend(data.unsqueeze(1))
        if self._dfs_enabled:
            self._ensure_dfs_state()
            index = index.to(device="cpu", dtype=torch.int64)
            self._world_counts[index[:, 0], index[:, 1]] = 0.0
            self._policy_counts[index[:, 0], index[:, 1]] = 0.0
            self._world_sampler.extend(index)
            self._update_priorities(index)

    def sample(self):
        return self._sample_world_batch()

    def sample_policy_starts(self, batch_size):
        if not self._dfs_enabled:
            return None, {}
        current_shape = self._current_storage_shape()
        weights = self._policy_priority(
            self._world_counts[: current_shape[0], : current_shape[1]],
            self._policy_counts[: current_shape[0], : current_shape[1]],
        ).reshape(-1)
        weights = weights / torch.clamp(weights.sum(), min=self._dfs_eps)
        flat_index = torch.multinomial(weights, num_samples=int(batch_size), replacement=True)
        time_idx = torch.div(flat_index, current_shape[1], rounding_mode="floor")
        env_idx = flat_index % current_shape[1]
        sample_td = self._buffer.storage[time_idx, env_idx]
        sample_td = self._to_device(sample_td)
        sampled_index = self._stack_index(time_idx.reshape(-1), env_idx.reshape(-1))
        self._policy_counts[sampled_index[:, 0], sampled_index[:, 1]] += 1
        self._update_priorities(sampled_index)
        metrics = {
            "buffer/policy_start_count_mean": self._policy_counts[sampled_index[:, 0], sampled_index[:, 1]].mean(),
            "buffer/policy_start_world_count_mean": self._world_counts[sampled_index[:, 0], sampled_index[:, 1]].mean(),
        }
        return (sample_td["stoch"], sample_td["deter"]), metrics

    def update(self, index, stoch, deter):
        # Flatten the data
        index = [ind.reshape(-1) for ind in index]
        # (B, T, S, K) -> (B*T, S, K)
        stoch = stoch.reshape(-1, *stoch.shape[2:])
        # (B, T, D) -> (B*T, D)
        deter = deter.reshape(-1, *deter.shape[2:])
        # In storage, the length is the first dimension, and the batch (number of environments) is the second dimension.
        stoch = stoch.to(self.storage_device, non_blocking=True)
        deter = deter.to(self.storage_device, non_blocking=True)
        self._buffer.storage.set(
            (index[0], index[1]),
            TensorDict({"stoch": stoch, "deter": deter}, batch_size=(stoch.shape[0],), device=self.storage_device),
            set_cursor=False,
        )

    def count(self):
        if self._buffer.storage.shape is None:
            return 0
        return self._buffer.storage.shape.numel()
