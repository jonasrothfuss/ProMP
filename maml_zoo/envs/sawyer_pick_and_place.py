from multiworld.envs.mujoco.sawyer_xyz.sawyer_pick_and_place import SawyerPickAndPlaceEnv as SawyerEnv
from multiworld.core.flat_goal_env import FlatGoalEnv
import numpy as np
from maml_zoo.envs.base import MetaEnv
from maml_zoo.logger import logger

class SawyerPickAndPlaceEnv(FlatGoalEnv, MetaEnv):
    """
    Wrapper for SawyerPickAndPlaceEnv from multiworld envs, using our method headers
    """
    def __init__(self, *args, **kwargs):
        self.quick_init(locals())
        FlatGoalEnv.__init__(self, SawyerEnv(*args, **kwargs), obs_keys=['observation'], goal_keys=['desired_goal'])

    def sample_tasks(self, n_tasks):
        return self.sample_goals(n_tasks)['state_desired_goal']

    def set_task(self, task):
        """
        Args:
            task: task of the meta-learning environment
        """
        return self.set_goal(dict(state_desired_goal=task))

    def get_task(self):
        """
        Returns:
            task: task of the meta-learning environment
        """
        return self.get_goal()

    def log_diagnostics(self, paths, prefix=''):
        self.get_diagnostics(paths)

    @property
    def action_space(self):
        return FlatGoalEnv.action_space(self)