import numpy as np
from maml_zoo.envs.base import MetaEnv
from gym.envs.mujoco.mujoco_env import MujocoEnv
from maml_zoo.logger import logger
import gym


class AntRandDirec2DEnv(MetaEnv, MujocoEnv, gym.utils.EzPickle):
    def __init__(self):
        self.set_task(self.sample_tasks(1)[0])
        MujocoEnv.__init__(self, 'ant.xml', 5)
        gym.utils.EzPickle.__init__(self)

    def sample_tasks(self, n_tasks):
        # for fwd/bwd env, goal direc is backwards if - 1.0, forwards if + 1.0
        directions = np.random.normal(size=(n_tasks, 2))
        directions /= np.linalg.norm(directions, axis=1)[..., np.newaxis]
        return directions

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
        posbefore = np.copy(self.get_body_com("torso")[:2])
        self.do_simulation(a, self.frame_skip)
        posafter = self.get_body_com("torso")[:2]
        forward_reward = np.sum(self.goal_direction * (posafter - posbefore))/self.dt
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and 1.0 >= state[2] >= 0.
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[2:],
            self.sim.data.qvel.flat,
            np.clip(self.sim.data.cfrc_ext, -1, 1).flat,
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-.1, high=.1)
        qvel = self.init_qvel + self.np_random.randn(self.model.nv) * .1
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.distance = self.model.stat.extent * 0.5

    def log_diagnostics(self, paths, prefix=''):
        progs = [np.mean(path["env_infos"]["reward_forward"]) for path in paths]
        ctrl_cost = [-np.mean(path["env_infos"]["reward_ctrl"]) for path in paths]

        logger.logkv(prefix+'AverageForwardReturn', np.mean(progs))
        logger.logkv(prefix+'MaxForwardReturn', np.max(progs))
        logger.logkv(prefix+'MinForwardReturn', np.min(progs))
        logger.logkv(prefix+'StdForwardReturn', np.std(progs))

        logger.logkv(prefix + 'AverageCtrlCost', np.mean(ctrl_cost))


if __name__ == "__main__":
    env = AntRandDirec2DEnv()
    while True:
        task = env.sample_tasks(1)[0]
        env.set_task(task)
        env.reset()
        for _ in range(100):
            env.render()
            _, reward, _, _ = env.step(env.action_space.sample())  # take a random action