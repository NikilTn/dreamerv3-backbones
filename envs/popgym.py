import gymnasium as gym
import numpy as np
import popgym  # noqa: F401
from gymnasium.spaces.utils import flatten, flatten_space


class POPGym(gym.Env):
    metadata = {}

    def __init__(self, task, seed=None):
        if not str(task).startswith("popgym-"):
            task = f"popgym-{task}"
        self._env = gym.make(task)
        self._seed = seed
        self._did_seed = False
        flat_space = flatten_space(self._env.observation_space)
        shape = flat_space.shape if flat_space.shape else (1,)
        self.observation_space = gym.spaces.Dict(
            {
                "vector": gym.spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float32),
                "is_first": gym.spaces.Box(0, 1, (), bool),
                "is_last": gym.spaces.Box(0, 1, (), bool),
                "is_terminal": gym.spaces.Box(0, 1, (), bool),
            }
        )
        self.action_space = self._env.action_space

    def reset(self):
        kwargs = {}
        if not self._did_seed and self._seed is not None:
            kwargs["seed"] = self._seed
            self._did_seed = True
        obs, _ = self._env.reset(**kwargs)
        return self._obs(obs, is_first=True)

    def step(self, action):
        outcome = self._env.step(action)
        if len(outcome) == 5:
            obs, reward, terminated, truncated, info = outcome
            done = terminated or truncated
        else:
            obs, reward, done, info = outcome
            terminated = done
        return self._obs(obs, is_last=done, is_terminal=terminated), reward, done, info

    def _obs(self, obs, is_first=False, is_last=False, is_terminal=False):
        vector = flatten(self._env.observation_space, obs).astype(np.float32)
        if vector.ndim == 0:
            vector = vector[None]
        return {
            "vector": vector,
            "is_first": is_first,
            "is_last": is_last,
            "is_terminal": is_terminal,
        }

    def close(self):
        self._env.close()
