from multiworld.envs.mujoco.sawyer_xyz.push.sawyer_push_simple import SawyerPushSimpleEnv as SawyerEnv
from multiworld.core.flat_goal_env import FlatGoalEnv
import numpy as np
from maml_zoo.envs.base import MetaEnv
from maml_zoo.logger import logger

class SawyerPushSimpleEnv(FlatGoalEnv, MetaEnv):
    """
    Wrapper for SawyerPushSimpleEnv from multiworld envs, using our method headers
    """
    def __init__(self, *args, **kwargs):
        self.quick_init(locals())
        sawyer_env = SawyerEnv(
            obj_low=(-0.0, 0.5, 0.02),
            obj_high=(0.0, 0.5, 0.02),
            goal_low=(-0.2, 0.6, 0.02),
            goal_high=(0.2, 0.8, 0.02),
            rew_mode='posPlace',
            *args, **kwargs)
        FlatGoalEnv.__init__(self, sawyer_env, obs_keys=['state_observation'], goal_keys=['state_desired_goal'])

    def sample_tasks(self, n_tasks):
        return self.sample_goals(n_tasks)

    def set_task(self, task):
        """
        Args:
            task: task of the meta-learning environment
        """
        return self.set_goal(task)

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

    def render(self):
        SawyerEnv.render(self)

    def log_diagnostics(self, paths, prefix=''):
        reach_dist = [path["env_infos"]['reachDist'] for path in paths]
        placing_dist = [path["env_infos"]['placeDist'] for path in paths]
        cos_dist = [path["env_infos"]['cosDist'] for path in paths]

        logger.logkv(prefix + 'AverageReachDistance', np.mean(reach_dist))
        logger.logkv(prefix + 'AveragePlaceDistance', np.mean(placing_dist))
        logger.logkv(prefix + 'AverageCosDistance', np.mean(cos_dist))