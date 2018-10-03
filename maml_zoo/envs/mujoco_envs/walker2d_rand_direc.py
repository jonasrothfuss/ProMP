import numpy as np
from maml_zoo.envs.base import MetaEnv
from maml_zoo.logger import logger
import gym
from gym.envs.mujoco.mujoco_env import MujocoEnv

class Walker2DRandDirecEnv(MetaEnv, gym.utils.EzPickle, MujocoEnv):
    def __init__(self):
        self.set_task(self.sample_tasks(1)[0])
        MujocoEnv.__init__(self, 'walker2d.xml', 8)
        gym.utils.EzPickle.__init__(self)
    
    def sample_tasks(self, n_tasks):
        return np.random.choice((-1.0, 1.0), (n_tasks, ))

    def set_task(self, task):
        """
        Args:
            task: task of the meta-learning environment
        """
        self.goal_direction = task

    def get_task(self):
        """
        Returns:
            task: task of the meta-learning environment
        """
        return self.goal_direction

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (self.goal_direction * (posafter - posbefore) / self.dt)
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        done = not (height > 0.8 and height < 2.0 and
                    ang > -1.0 and ang < 1.0)
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        qpos = self.sim.data.qpos
        qvel = self.sim.data.qvel
        return np.concatenate([qpos[1:], np.clip(qvel, -10, 10)]).ravel()

    def reset_model(self):
        self.set_state(
            self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq),
            self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        )
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.5

if __name__ == "__main__":
    env = Walker2DRandDirecEnv()
    import time
    while True:
        env.reset()
        for _ in range(200):
            env.render()
            _, reward, _, _ = env.step(env.action_space.sample())  # take a random action