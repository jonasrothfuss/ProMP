from .env_spec import EnvSpec
import collections
from cached_property import cached_property

class Env(object):
    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.
        Args:
            action (obj) : an action provided by the environment       

        Returns: 
            observation (obj) : agent's observation of the current environment
            reward (float) : amount of reward due to the previous action
            done (bool) : a boolean indicating whether the episode has ended
            info (dict) : a dictionary containing other diagnostic information from the previous action
        """
        raise NotImplementedError

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Returns:
            observation (obj) : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        raise NotImplementedError

    @property
    def action_space(self):
        """
        Returns a Space object
        :rtype: rllab.spaces.base.Space
        """
        raise NotImplementedError

    @property
    def observation_space(self):
        """
        Returns a Space object
        :rtype: rllab.spaces.base.Space
        """
        raise NotImplementedError

    # Helpers that derive from Spaces
    @property
    def action_dim(self):
        return self.action_space.flat_dim

    def render(self):
        pass

    @property
    def horizon(self):
        """
        Returns:
            horizon (int) : Horizon of the environment, if it has one
        """
        raise NotImplementedError


    def terminate(self):
        """
        Clean up operation,
        """
        pass

class MetaEnv(Env):
    def sample_tasks(self, n_tasks):
        """ 
        Args:
            n_tasks (int) : number of different meta-tasks needed
        Returns:
            tasks (list) : an (n_tasks) length list of reset args
        """
        raise NotImplementedError

    def reset(self, reset_args=None):
        """
        Args: 
            reset_args (reset_args) : specification for how to reset the environment
        Returns:
            observation (obj) : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        if reset_args is None:
            return self.reset()
        raise NotImplementedError

_Step = collections.namedtuple("Step", ["observation", "reward", "done", "info"])

def Step(observation, reward, done, **kwargs):
    """
    Convenience method creating a namedtuple with the results of the
    environment.step method.
    Put extra diagnostic info in the kwargs
    """
    return _Step(observation, reward, done, kwargs)
