import numpy as np
from maml_zoo.utils.serializable import Serializable
from gym.spaces import Box
from rand_param_envs.gym.spaces import Box as OldBox

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


class RL2Env(Serializable):
    """
    Normalizes the environment class.

    Args:
        Env (gym.Env): class of the unnormalized gym environment
        scale_reward (float): scale of the reward
        normalize_obs (bool): whether normalize the observations or not
        normalize_reward (bool): whether normalize the reward or not
        obs_alpha (float): step size of the running mean and variance for the observations
        reward_alpha (float): step size of the running mean and variance for the observations

    """
    def __init__(self,
                 env,
                 scale_reward=1.,
                 normalize_obs=False,
                 normalize_reward=False,
                 obs_alpha=0.001,
                 reward_alpha=0.001,
                 normalization_scale=10.,
                 ):
        Serializable.quick_init(self, locals())

        self._wrapped_env = env

    def __getattr__(self, attr):
        """
        If normalized env does not have the attribute then call the attribute in the wrapped_env
        Args:
            attr: attribute to get

        Returns:
            attribute of the wrapped_env

        """
        orig_attr = self._wrapped_env.__getattribute__(attr)

        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                return result

            return hooked
        else:
            return orig_attr

    def reset(self):
        obs = self._wrapped_env.reset()
        return np.concatenate([obs, np.zeros(self._wrapped_env.action_space.shape), [0], [0]])

    def __getstate__(self):
        d = Serializable.__getstate__(self)
        return d

    def __setstate__(self, d):
        Serializable.__setstate__(self, d)

    def step(self, action):
        wrapped_step = self._wrapped_env.step(action)
        next_obs, reward, done, info = wrapped_step
        next_obs = np.concatenate([next_obs, action, [reward], [done]])
        return next_obs, reward, done, info


rl2env = RL2Env