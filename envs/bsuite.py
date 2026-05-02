import gymnasium as gym
import numpy as np


class BSuite(gym.Env):
    metadata = {}

    def __init__(self, task, seed=None):
        try:
            import bsuite
            from dm_env import specs as dm_specs
        except ImportError as exc:
            raise ImportError(
                "BSuite support requires the 'bsuite' package. Use Python 3.11 on the HPC and install "
                "the repo requirements so proposal benchmark runs can import it."
            ) from exc

        self._bsuite = bsuite
        self._dm_specs = dm_specs
        self._env = self._bsuite.load_from_id(task)
        self._seed = seed
        self._obs_spec = self._env.observation_spec()
        self.observation_space = gym.spaces.Dict(
            {
                "vector": self._vector_space(self._obs_spec),
                "is_first": gym.spaces.Box(0, 1, (), bool),
                "is_last": gym.spaces.Box(0, 1, (), bool),
                "is_terminal": gym.spaces.Box(0, 1, (), bool),
            }
        )
        self.action_space = self._action_space(self._env.action_spec())

    def reset(self):
        timestep = self._env.reset()
        return self._obs(timestep.observation, is_first=True)

    def step(self, action):
        timestep = self._env.step(action)
        done = bool(timestep.last())
        reward = np.float32(0.0 if timestep.reward is None else timestep.reward)
        return self._obs(timestep.observation, is_last=done, is_terminal=done), reward, done, {}

    def _action_space(self, spec):
        if isinstance(spec, self._dm_specs.DiscreteArray):
            return gym.spaces.Discrete(spec.num_values)
        if isinstance(spec, self._dm_specs.BoundedArray):
            return gym.spaces.Box(spec.minimum, spec.maximum, shape=spec.shape, dtype=spec.dtype)
        return gym.spaces.Box(-np.inf, np.inf, shape=spec.shape, dtype=spec.dtype)

    def _vector_space(self, spec):
        size = self._flatdim(spec)
        return gym.spaces.Box(-np.inf, np.inf, shape=(size,), dtype=np.float32)

    def _flatdim(self, spec):
        if isinstance(spec, self._dm_specs.DiscreteArray):
            return int(spec.num_values)
        if isinstance(spec, dict):
            return sum(self._flatdim(v) for _, v in sorted(spec.items()))
        if isinstance(spec, (tuple, list)):
            return sum(self._flatdim(v) for v in spec)
        return int(np.prod(spec.shape)) or 1

    def _flatten(self, spec, value):
        if isinstance(spec, self._dm_specs.DiscreteArray):
            out = np.zeros(spec.num_values, dtype=np.float32)
            out[int(value)] = 1.0
            return out
        if isinstance(spec, dict):
            parts = [self._flatten(spec[k], value[k]) for k in sorted(spec.keys())]
            return np.concatenate(parts, axis=0)
        if isinstance(spec, (tuple, list)):
            parts = [self._flatten(subspec, subvalue) for subspec, subvalue in zip(spec, value)]
            return np.concatenate(parts, axis=0)
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size == 0:
            return np.zeros(1, dtype=np.float32)
        return array

    def _obs(self, obs, is_first=False, is_last=False, is_terminal=False):
        return {
            "vector": self._flatten(self._obs_spec, obs),
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
        }

    def close(self):
        close = getattr(self._env, "close", None)
        if callable(close):
            close()
