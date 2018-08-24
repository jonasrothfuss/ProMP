from maml_zoo.envs.base import MetaEnv

import numpy as np
from gym.spaces import Box


class MetaPointEnvCorner(MetaEnv):
    """
    Simple 2D point meta environment. Each meta-task corresponds to a different goal / corner
    (one of the 4 points (-2,-2), (-2, 2), (2, -2), (2,2)) which are sampled with equal probability
    """

    def __init__(self, reward_type='dense', sparse_reward_radius=0.2):
        assert reward_type in ['dense', 'dense_squared', 'sparse']
        self.reward_type = reward_type
        self.sparse_reward_radius = sparse_reward_radius
        self.corners = [np.array([-2,-2]), np.array([2,-2]), np.array([-2,2]), np.array([2, 2])]
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = Box(low=-0.1, high=0.1, shape=(2,))

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
        self._state = np.random.uniform(-0.1, 0.1, size=(2,))
        observation = np.copy(self._state)
        return observation

    def done(self, obs):
        if obs.ndim == 1:
            return abs(obs[0]) < 0.01 and abs(obs[1]) < 0.01
        elif obs.ndim == 2:
            return np.logical_and(np.abs(obs[:, 0]) < 0.01, np.abs(obs[:, 1]) < 0.01)

    def reward(self, obs, act, obs_next):
        if obs_next.ndim == 2:
            goal_distance = np.linalg.norm(obs_next - self.goal[None,:], axis=1)
            if self.reward_type == 'dense':
                return - goal_distance
            elif self.reward_type == 'dense_squared':
                return - goal_distance**2
            elif self.reward_type == 'sparse':
                return np.max(self.sparse_reward_radius - goal_distance, 0)

        elif obs_next.ndim == 1:
            return self.reward(None, None, np.array([obs_next]))[0]
        else:
            raise NotImplementedError

    def log_diagnostics(self, *args):
        pass

    def sample_tasks(self, n_tasks):
        return [self.corners[idx] for idx in np.random.choice(range(len(self.corners)), size=n_tasks)]

    def set_task(self, task):
        self.goal = task

    def get_task(self):
        return self.goal

if __name__ == "__main__":
    env = MetaPointEnvCorner()
    task = env.sample_tasks(10)
    env.set_task(task[0])
