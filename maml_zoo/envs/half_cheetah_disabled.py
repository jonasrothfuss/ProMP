import numpy as np
from maml_zoo.envs.base import MetaEnv
from maml_zoo.logger import logger
import gym
from gym.envs.mujoco.mujoco_env import MujocoEnv

class HalfCheetahRandVelEnv(MetaEnv, MujocoEnv, gym.utils.EzPickle):
    def __init__(self):
        self.set_task(self.sample_tasks(1)[0])
        MujocoEnv.__init__(self, 'half_cheetah.xml', 5)
        gym.utils.EzPickle.__init__(self)

    def sample_tasks(self, n_tasks):
        return np.random.uniform(0.0, 3.0, (n_tasks, ))

    def set_task(self, task):
        """
        Args:
            task: task of the meta-learning environment
        """
        self.goal_velocity = task

    def get_task(self):
        """
        Returns:
            task: task of the meta-learning environment
        """
        return self.goal_velocity

    def step(self, action):
        xposbefore = self.sim.data.qpos[0]
        self.do_simulation(action, self.frame_skip)
        xposafter = self.sim.data.qpos[0]
        ob = self._get_obs()
        reward_ctrl = - 0.5 * 0.1 * np.square(action).sum()
        forward_vel = (xposafter - xposbefore) / self.dt
        reward_run = - np.abs(forward_vel - self.goal_velocity)
        reward = reward_ctrl + reward_run
        done = False
        return ob, reward, done, dict(forward_vel=forward_vel, reward_run=reward_run, reward_ctrl=reward_ctrl)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            self.sim.data.qvel.flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.1, high=.1, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def log_diagnostics(self, paths, prefix=''):
        fwrd_vel = [path["env_infos"]['forward_vel'] for path in paths]
        final_fwrd_vel = [path["env_infos"]['forward_vel'][-1] for path in paths]
        ctrl_cost = [-path["env_infos"]['reward_ctrl'] for path in paths]

        logger.logkv(prefix + 'AvgForwardVel', np.mean(fwrd_vel))
        logger.logkv(prefix + 'AvgFinalForwardVel', np.mean(final_fwrd_vel))
        logger.logkv(prefix + 'AvgCtrlCost', np.std(ctrl_cost))
