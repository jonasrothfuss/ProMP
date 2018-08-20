from maml_zoo.envs.base import MetaEnv

import numpy as np
from gym.spaces import Box


class MetaPointEnv(MetaEnv):

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.

        Args:
            action : an action provided by the environment
        Returns:
            (observation, reward, done, info)
            observation : agent's observation of the current environment
            reward [Float] : amount of reward due to the previous action
            done : a boolean, indicating whether the episode has ended
            info : a dictionary containing other diagnostic information from the previous action
        """
        prev_state = self._state
        self._state = prev_state + np.clip(action, -0.1, 0.1)
        reward = self.reward(prev_state, action, self._state)
        done = self.done(self._state)
        next_observation = np.copy(self._state)
        return next_observation, reward, done, {}

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        self._state = np.random.uniform(-2, 2, size=(2,))
        observation = np.copy(self._state)
        return observation

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(2,))

    @property
    def action_space(self):
        return Box(low=-0.1, high=0.1, shape=(2,))

    def done(self, obs):
        if obs.ndim == 1:
            return abs(obs[0]) < 0.01 and abs(obs[1]) < 0.01
        elif obs.ndim == 2:
            return np.logical_and(np.abs(obs[:, 0]) < 0.01, np.abs(obs[:, 1]) < 0.01)

    def reward(self, obs, act, obs_next):
        if obs_next.ndim == 1:
            return - np.sqrt(obs_next[0]**2 + obs_next[1]**2)
        elif obs_next.ndim == 2:
            return - np.sqrt(obs_next[:, 0] ** 2 + obs_next[:, 1] ** 2)

    def log_diagnostics(self, paths):
        pass

    def sample_tasks(self, n_tasks):
        return [{}] * n_tasks

    def set_task(self, task):
        pass

    def get_task(self):
        return {}