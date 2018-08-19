import numpy as np
from maml_zoo.utils.serializable import Serializable
from gym.spaces import Box


def normalize(EnvCls, env_args=None,
                scale_reward=1.,
                normalize_obs=False,
                normalize_reward=False,
                obs_alpha=0.001,
                reward_alpha=0.001
              ):
    """
    Normalizes the environment class.

    Args:
        EnvCls (gym.Env): class of the unnormalized gym environment
        env_args (dict or None): arguments of the environment
        scale_reward (float): scale of the reward
        normalize_obs (bool): whether normalize the observations or not
        normalize_reward (bool): whether normalize the reward or not
        obs_alpha (float): step size of the running mean and variance for the observations
        reward_alpha (float): step size of the running mean and variance for the observations

    Returns:
        Normalized environment

    """

    class NormalizedEnv(Serializable, EnvCls):
        def __init__(self):
            Serializable.quick_init(self, locals())
            self._scale_reward = scale_reward
            if env_args is None:
                EnvCls.__init__(self)
            else:
                EnvCls.__init__(self, env_args)

            self._normalize_obs = normalize_obs
            self._normalize_reward = normalize_reward
            self._obs_alpha = obs_alpha
            self._obs_mean = np.zeros(self.observation_space.shape)
            self._obs_var = np.ones(self.observation_space.shape)
            self._reward_alpha = reward_alpha
            self._reward_mean = 0.
            self._reward_var = 1.

        def _update_obs_estimate(self, obs):
            o_a = self._obs_alpha
            self._obs_mean = (1 - o_a) * self._obs_mean + o_a * obs
            self._obs_var = (1 - o_a) * self._obs_var + o_a * np.square(obs - self._obs_mean)

        def _update_reward_estimate(self, reward):
            r_a = self._reward_alpha
            self._reward_mean = (1 - r_a) * self._reward_mean + r_a * reward
            self._reward_var = (1 - r_a) * self._reward_var + r_a * np.square(reward - self._reward_mean)

        def _apply_normalize_obs(self, obs):
            self._update_obs_estimate(obs)
            return (obs - self._obs_mean) / (np.sqrt(self._obs_var) + 1e-8)

        def _apply_normalize_reward(self, reward):
            self._update_reward_estimate(reward)
            return reward / (np.sqrt(self._reward_var) + 1e-8)

        def reset(self):
            obs = super(NormalizedEnv, self).reset()
            if self._normalize_obs:
                return self._apply_normalize_obs(obs)
            else:
                return obs

        def __getstate__(self):
            d = Serializable.__getstate__(self)
            d["_obs_mean"] = self._obs_mean
            d["_obs_var"] = self._obs_var
            return d

        def __setstate__(self, d):
            Serializable.__setstate__(self, d)
            self._obs_mean = d["_obs_mean"]
            self._obs_var = d["_obs_var"]

        def step(self, action):
            if isinstance(self.action_space, Box):
                # rescale the action
                lb, ub = self.action_space.low, self.action_space.high
                scaled_action = lb + (action + 1.) * 0.5 * (ub - lb)
                scaled_action = np.clip(scaled_action, lb, ub)
            else:
                scaled_action = action
            wrapped_step = EnvCls.step(self, scaled_action)
            next_obs, reward, done, info = wrapped_step
            if getattr(self, "_normalize_obs", False):
                next_obs = self._apply_normalize_obs(next_obs)
            if getattr(self, "_normalize_reward", False):
                reward = self._apply_normalize_reward(reward)
            return next_obs, reward * self._scale_reward, done, info

        def __str__(self):
            return "Normalized: %s" % EnvCls

    return NormalizedEnv()
