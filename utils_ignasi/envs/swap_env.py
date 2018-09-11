import numpy as np


class SwapEnv(object):
    def __init__(self, list_envs):
        self.envs = list_envs
        self.num_envs = len(list_envs)
        self._idx = 0
        self.env = list_envs[self._idx]

    def reset_task(self):
        self._idx = np.random.randint(0, self.num_envs)
        self.env = self.envs[self._idx]
        self.env.reset_task()

    @property
    def _wrapped_env(self):
        return self.env._wrapped_env