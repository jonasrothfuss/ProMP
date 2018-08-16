import numpy as np
from maml_zoo.envs.base import MetaEnv
from maml_zoo.envs.base import RandomEnv
from gym import utils

class HopperRandParamsEnv(RandomEnv, utils.EzPickle):
    def __init__(self, log_scale_limit=3.0):
        RandomEnv.__init__(self, log_scale_limit, 'hopper.xml', 4)
        utils.EzPickle.__init__(self)

    def step(self, a):
        posbefore = self.sim.data.qpos[0]
        self.do_simulation(a, self.frame_skip)
        posafter, height, ang = self.sim.data.qpos[0:3]
        alive_bonus = 1.0
        reward = (posafter - posbefore) / self.dt
        reward += alive_bonus
        reward -= 1e-3 * np.square(a).sum()
        s = self.state_vector()
        done = not (np.isfinite(s).all() and (np.abs(s[2:]) < 100).all() and
                    (height > .7) and (abs(ang) < .2))
        ob = self._get_obs()
        return ob, reward, done, {}

    def _get_obs(self):
        return np.concatenate([
            self.sim.data.qpos.flat[1:],
            np.clip(self.sim.data.qvel.flat, -10, 10)
        ])

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(low=-.005, high=.005, size=self.model.nq)
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 2
        self.viewer.cam.distance = self.model.stat.extent * 0.75
        # self.viewer.cam.lookat[2] += .8
        self.viewer.cam.elevation = -20

    def log_diagnostics(self, paths, prefix=''):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
        ]
        logger.record_tabular(prefix + 'AverageForwardProgress', np.mean(progs))
        logger.record_tabular(prefix + 'MaxForwardProgress', np.max(progs))
        logger.record_tabular(prefix + 'MinForwardProgress', np.min(progs))
        logger.record_tabular(prefix + 'StdForwardProgress', np.std(progs))

if __name__ == "__main__":

    env = HopperRandParamsEnv()
    tasks = env.sample_tasks(40)
    while True:
        env.reset()
        env.set_task(np.random.choice(tasks))
        print(env.model.body_mass)
        for _ in range(100):
            env.render()
            env.step(env.action_space.sample())  # take a random action

