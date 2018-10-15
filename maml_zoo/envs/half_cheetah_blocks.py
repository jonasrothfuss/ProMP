import numpy as np
from rand_param_envs.base import RandomEnv
from rand_param_envs.gym import utils
from maml_zoo.logger import logger



class AntDisabledEnv(RandomEnv, utils.EzPickle):
    def __init__(self):
        self._crippled_leg = None
        self._cripple_mask = np.ones(8)
        RandomEnv.__init__(self, 0, 'ant.xml', 2)
        utils.EzPickle.__init__(self)
        self._init_geom_size = self.model.geom_size.copy()
        self._init_geom_pos = self.model.geom_pos.copy()

    def sample_tasks(self, n_tasks):
        # for fwd/bwd env, goal direc is backwards if - 1.0, forwards if + 1.0
        return np.random.randint(0, 3, size=n_tasks)

    def set_task(self, task):
        """
        Args:
            task: task of the meta-learning environment
        """
        # pick which leg to remove (0 1 2 are train... 3 is test)
        # print("\n\nREMOVED LEG: ", self.crippled_leg, "\n\n")

        # pick which actuators to disable
        self._cripple_mask = np.ones(8)

        if self._crippled_leg == 0:
            self._cripple_mask[2] = 0
            self._cripple_mask[3] = 0

        elif self._crippled_leg == 1:
            self._cripple_mask[4] = 0
            self._cripple_mask[5] = 0

        elif self._crippled_leg == 2:
            self._cripple_mask[6] = 0
            self._cripple_mask[7] = 0

        elif self._crippled_leg == 3:
            self._cripple_mask[0] = 0
            self._cripple_mask[1] = 0

        # make the removed leg not affect anything
        temp_size = self._init_geom_size.copy()
        temp_pos = self._init_geom_pos.copy()

        if (self._crippled_leg == 0):
            # top half
            temp_size[3, 0] = temp_size[3, 0] / 2
            temp_size[3, 1] = temp_size[3, 1] / 2
            # bottom half
            temp_size[4, 0] = temp_size[4, 0] / 2
            temp_size[4, 1] = temp_size[4, 1] / 2
            temp_pos[4, :] = temp_pos[3, :]
        elif (self._crippled_leg == 1):
            # top half
            temp_size[6, 0] = temp_size[6, 0] / 2
            temp_size[6, 1] = temp_size[6, 1] / 2
            # bottom half
            temp_size[7, 0] = temp_size[7, 0] / 2
            temp_size[7, 1] = temp_size[7, 1] / 2
            temp_pos[7, :] = temp_pos[6, :]
        elif (self._crippled_leg == 2):
            # top half
            temp_size[9, 0] = temp_size[9, 0] / 2
            temp_size[9, 1] = temp_size[9, 1] / 2
            # bottom half
            temp_size[10, 0] = temp_size[10, 0] / 2
            temp_size[10, 1] = temp_size[10, 1] / 2
            temp_pos[10, :] = temp_pos[9, :]
        elif (self._crippled_leg == 3):
            # top half
            temp_size[12, 0] = temp_size[12, 0] / 2
            temp_size[12, 1] = temp_size[12, 1] / 2
            # bottom half
            temp_size[13, 0] = temp_size[13, 0] / 2
            temp_size[13, 1] = temp_size[13, 1] / 2
            temp_pos[13, :] = temp_pos[12, :]

        self.model.geom_size = temp_size
        self.model.geom_pos = temp_pos

    def get_task(self):
        """
        Returns:
            task: task of the meta-learning environment
        """
        return self._crippled_leg

    def _step(self, a):
        xposbefore = self.get_body_com("torso")[0]
        self.do_simulation(a * self._cripple_mask, self.frame_skip)
        xposafter = self.get_body_com("torso")[0]
        forward_reward = self.get_body_comvel("torso")[0]
        # forward_reward = self.goal_direction * (xposafter - xposbefore)/self.dt
        ctrl_cost = .5 * 1e-2 * np.square(a).sum()
        contact_cost = 0
        survive_reward = 0.05
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self.state_vector()
        notdone = np.isfinite(state).all() and 1.0 >= state[2] >= 0.
        done = False
        ob = self._get_obs()
        return ob, reward, done, dict(
            reward_forward=forward_reward,
            reward_ctrl=-ctrl_cost,
            reward_contact=-contact_cost,
            reward_survive=survive_reward)

    def _get_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            self.get_body_xmat("torso").flat,
            self.get_body_com("torso"),
        ]).reshape(-1)

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