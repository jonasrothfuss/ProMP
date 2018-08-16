import numpy as np
from gym import utils
from maml_zoo.envs.base import MetaEnv

class AntRandGoalEnv(MetaEnv, utils.EzPickle):
    def __init__(self):
        self.goal_pos = np.random.uniform((-3.0, 3.0), size=(2))
        MetaEnv.__init__(self, 'ant.xml', 5)
        utils.EzPickle.__init__(self)

    def sample_tasks(self, n_tasks):
        # random target location
        return np.random.uniform((-3.0, 3.0), size=(n_tasks, 2))

    def set_task(self, task):
        """
        Args:
            task: task of the meta-learning environment
        """
        self.goal_pos = task

    def get_task(self):
        """
        Returns:
            task: task of the meta-learning environment
        """
        return self.goal_pos

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        xposafter = self.get_body_com("torso")
        goal_reward = np.exp(-np.sum(np.abs(xposafter[:2] - self.goal_pos))) # make it happy, not suicidal
        ctrl_cost = .5 * np.square(a).sum()
        contact_cost = 0.5 * 1e-3 * np.sum(
            np.square(np.clip(self.sim.data.cfrc_ext, -1, 1)))
        survive_reward = 1.0
        reward = goal_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = not notdone
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=goal_reward,
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
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular(prefix+'AverageForwardProgress', np.mean(progs))
        logger.record_tabular(prefix+'MaxForwardProgress', np.max(progs))
        logger.record_tabular(prefix+'MinForwardProgress', np.min(progs))
        logger.record_tabular(prefix+'StdForwardProgress', np.std(progs))
        
if __name__ == "__main__":
    env = AntRandGoalEnv()
    while True:
        env.reset()
        for _ in range(100):
            env.render()
            _, reward, _, _ = env.step(env.action_space.sample())  # take a random action
