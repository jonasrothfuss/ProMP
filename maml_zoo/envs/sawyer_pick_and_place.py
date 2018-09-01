from multiworld.envs.mujoco.sawyer_xyz.pickPlace.sawyer_pick_and_place import SawyerPickPlaceEnv as SawyerEnv
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
        FlatGoalEnv.__init__(self, SawyerEnv(*args, **kwargs), obs_keys=['state_observation'], goal_keys=['state_desired_goal'])

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

if __name__ == "__main__":
    env = SawyerPickAndPlaceEnv()
    while True:
        task = env.sample_tasks(1)[0]
        env.set_task(task)
        env.reset()
        for _ in range(500):
            SawyerEnv.render(env)
            _, reward, _, _ = env.step(env.action_space.sample())  # take a random action