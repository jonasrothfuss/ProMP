import numpy as np
from gym import utils

class HopperRandParamsEnv(MetaEnv, utils.EzPickle):
    def __init__(self):
        MetaEnv.__init__(self, 'hopper.xml', 4)
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
        self.viewer.cam.lookat[2] += .8
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

    env = HopperEnvRandParams()
    while True:
        env.reset()
        print(env.model.body_mass)
        for _ in range(100):
            env.render()
            env.step(env.action_space.sample())  # take a random action



    def __init__(self, *args, log_scale_limit=3.0, fix_params=False, rand_params=BaseEnvRandParams.RAND_PARAMS, random_seed=None, max_path_length=None, **kwargs):
        """
        Half-Cheetah environment with randomized mujoco parameters
        :param log_scale_limit: lower / upper limit for uniform sampling in logspace of base 2
        :param random_seed: random seed for sampling the mujoco model params
        :param fix_params: boolean indicating whether the mujoco parameters shall be fixed
        :param rand_params: mujoco model parameters to sample
        """

        args_all, kwargs_all = get_all_function_arguments(self.__init__, locals())
        BaseEnvRandParams.__init__(*args_all, **kwargs_all)
        HopperEnv.__init__(self, *args, **kwargs)
        Serializable.__init__(*args_all, **kwargs_all)
        self._obs_bounds()

    def reward(self, obs, action, obs_next):
        alive_bonus = 1.0
        if obs.ndim == 2 and action.ndim == 2:
            assert obs.shape == obs_next.shape and action.shape[0] == obs.shape[0]
            vel = obs_next[:, 5]
            ctrl_cost = 1e-3 * np.sum(np.square(action), axis=1)
            reward =  vel + alive_bonus - ctrl_cost
        else:
            reward = self.reward(np.array([obs]), np.array([action]), np.array([obs_next]))[0]
        return np.minimum(np.maximum(-1000.0, reward), 1000.0)

    def done(self, obs):
        if obs.ndim == 2:
            notdone = np.all(np.isfinite(obs), axis=1) * (np.abs(obs[:, 3:]) < 100).all(axis=1) * (obs[:, 0] > .7) * (np.abs(obs[:, 1]) < .2)
            return np.logical_not(notdone)
        else:
            notdone = np.isfinite(obs).all() and \
                      (np.abs(obs[3:]) < 100).all() and (obs[0] > .7) and \
                      (abs(obs[1]) < .2)
            return not notdone


    @overrides


