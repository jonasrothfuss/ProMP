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
            # obj_low=(-0.0, 0.5, 0.02),
            # obj_high=(0.0, 0.5, 0.02),
            # goal_low=(0, 0.7, 0.02),
            # goal_high=(0, 0.7, 0.02),
            # obj_type='block',
            rew_mode='posPlace',
            *args, **kwargs)
        SawyerEnv.compute_rewards = compute_rewards
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

def compute_rewards(self, actions, obs):   
    state_obs = obs['state_observation']
    endEffPos, objPos = state_obs[0:3], state_obs[3:6]
           
    placingGoal = self._state_goal
    rightFinger, leftFinger = self.get_site_pos('rightEndEffector'), self.get_site_pos('leftEndEffector')
    objPos = self.get_body_com("obj")
    fingerCOM = (rightFinger + leftFinger)/2

    reachDist = np.linalg.norm(objPos - fingerCOM)
    placeDist = np.linalg.norm(objPos - placingGoal)

    v1 = placingGoal - objPos
    v2 = objPos - fingerCOM
    cosDist = v1.dot(v2) / (reachDist * placeDist)

    if self.rew_mode == 'normal':
        reward = -reachDist - placeDist

    elif self.rew_mode == 'posPlace':
        reward = -reachDist + 100* max(0, self.origPlacingDist - placeDist)

    elif self.rew_mode == 'angle':
        reward = -1.0 * reachDist - placeDist + 0.5 * cosDist

    return reward, reachDist, placeDist, cosDist

if __name__ == "__main__":
    env = SawyerPushSimpleEnv()
    while True:
        task = env.sample_tasks(1)[0]
        env.set_task(task)
        env.reset()
        for _ in range(500):
            env.render()
            _, reward, _, _ = env.step(env.action_space.sample())  # take a random action