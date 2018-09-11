import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
#from rllab.envs.mujoco.mujoco_env import MujocoEnv
from sandbox.ignasi.envs.mujoco_env import MujocoEnv
from rllab.misc import logger
from rllab.misc.overrides import overrides


def smooth_abs(x, param):
    return np.sqrt(np.square(x) + np.square(param)) - param


class HalfCheetahEnv(MujocoEnv, Serializable):

    FILE = '../../sandbox/ignasi/vendor/mujoco_models/half_cheetah_hfield.xml'

    def __init__(self, task='hfield', reset_every_episode=False, reward=True, *args, **kwargs):
        self.reset_every_episode = reset_every_episode
        self.first = True
        self.id_torso = 0
        super(HalfCheetahEnv, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)
        self._init_geom_rgba = self.model.geom_rgba.copy()
        self.init_body_mass = self.model.body_mass.copy()
        self.id_torso = self.model.body_names.index('torso')
        self._reward = reward
        self._action_bounds = self.action_bounds
        self.x_walls = np.array([250, 260, 261, 270, 280, 285])
        self.height_walls = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        self.height = 0.8
        # self.x_walls = np.array([255, 270, 285, 300, 315, 330])
        ### TASK 1: BASIN ###
        # self.height_walls = np.array([-1, 1, 0., 0., 0., 0.])  # basin
        # self.height = 0.55
        ### TASK 2: HILL ###
        # self.height_walls = np.array([1, -1, 0, 0., 0, 0])   # hill
        # self.height  = 0.6
        ### TASK 3: GENTLE SLOPE
        # self.height_walls = np.array([1, 1, 1, 1, 1, 1]) # low slope
        # self.height = 1
        ### TASK 4: CRAZY SLOPE
        # self.height_walls = np.array([1, 1, 1, 1, 1, 1]) # high slope
        # self.height = 4
        self.width = 15
        if task in [None, 'None', 'hfield', 'same', 'hill', 'gentle', 'crazy', 'basin']:
            self.task = task
            self.sign = 1
            self.cripple_mask = np.ones(self.action_space.shape)
        else:
            raise NameError

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flatten(),
            self.model.data.qvel.flat,
            self.get_body_com("torso").flat,
        ])

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.xmat[idx].reshape((3, 3))

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.com_subtree[idx]

    def step(self, action):
        action = self.cripple_mask * action
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        action = np.clip(action, *self._action_bounds)
        reward = 0
        if self._reward:
            ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action))
            run_cost = self.sign * -1 * self.get_body_comvel("torso")[0]
            cost = ctrl_cost + run_cost
            reward = -cost
        done = False
        return Step(next_obs, reward, done)

    def get_reward(self, observation, next_observation, action):
        dt = 0.01
        ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action), axis=1)
        vel = self.sign * (next_observation[:,-3] - observation[:,-3])/dt
        reward = vel - ctrl_cost
        # Limit the legs from moving back
        # Add penalty for actions
        return reward

    def reset_mujoco(self, init_state=None):
        super(HalfCheetahEnv, self).reset_mujoco(init_state=init_state)
        if self.reset_every_episode and not self.first:
            self.reset_task()
        if self.first:
            self.first = False

    def reset_task(self, value=None):
        if self.task == 'hfield':
            height = np.random.uniform(0.2, 1)
            width = 10
            n_walls = 6
            self.model.hfield_size = np.array([50, 5, height, 0.1])
            x_walls = np.random.choice(np.arange(255, 310, width), replace=False, size=n_walls)
            x_walls.sort()
            sign = np.random.choice([1, -1], size=n_walls)
            sign[:2] = 1
            height_walls = np.random.uniform(0.2, 0.6, n_walls) * sign
            row = np.zeros((500,)) # (np.sin(x / x[-1] * np.pi * freq) + 1) / 2
            for i, x in enumerate(x_walls):
                terrain = np.cumsum([height_walls[i]] * width)
                row[x:x + width] += terrain
                row[x+width:] = row[x+width - 1]
            row = (row - np.min(row))/(np.max(row) - np.min(row))

            hfield = np.tile(row.reshape(-1, 1), (1, 528)).T.reshape(-1, 1)
            self.model.hfield_data = hfield

        elif self.task == 'same':
            height = self.height
            width = self.width
            self.model.hfield_size = np.array([50, 5, height, 0.1])
            row = np.zeros((500,))  # (np.sin(x / x[-1] * np.pi * freq) + 1) / 2
            for i, x in enumerate(self.x_walls):
                terrain = np.cumsum([self.height_walls[i]] * width)
                row[x:x + width] += terrain
                row[x + width:] = row[x + width - 1]
            row = (row - np.min(row)) / (np.max(row) - np.min(row))
            hfield = np.tile(row.reshape(-1, 1), (1, 528)).T.reshape(-1, 1)
            self.model.hfield_data = hfield

        elif self.task == 'basin':
            ### TASK 1: BASIN ###
            self.height_walls = np.array([-1, 1, 0., 0., 0., 0.])  # basin
            self.height = 0.55
            height = self.height
            width = self.width
            self.x_walls = np.array([255, 270, 285, 300, 315, 330]) - 5
            self.model.hfield_size = np.array([50, 5, height, 0.1])
            row = np.zeros((500,))  # (np.sin(x / x[-1] * np.pi * freq) + 1) / 2
            for i, x in enumerate(self.x_walls):
                terrain = np.cumsum([self.height_walls[i]] * width)
                row[x:x + width] += terrain
                row[x + width:] = row[x + width - 1]
            row = (row - np.min(row)) / (np.max(row) - np.min(row))
            hfield = np.tile(row.reshape(-1, 1), (1, 528)).T.reshape(-1, 1)
            self.model.hfield_data = hfield

        elif self.task == 'hill':
            ### TASK 2: HILL ###
            self.height_walls = np.array([1, -1, 0, 0., 0, 0])   # hill
            self.height  = 0.6
            height = self.height
            width = self.width
            self.x_walls = np.array([255, 270, 285, 300, 315, 330]) - 5
            self.model.hfield_size = np.array([50, 5, height, 0.1])
            row = np.zeros((500,))  # (np.sin(x / x[-1] * np.pi * freq) + 1) / 2
            for i, x in enumerate(self.x_walls):
                terrain = np.cumsum([self.height_walls[i]] * width)
                row[x:x + width] += terrain
                row[x + width:] = row[x + width - 1]
            row = (row - np.min(row)) / (np.max(row) - np.min(row))
            hfield = np.tile(row.reshape(-1, 1), (1, 528)).T.reshape(-1, 1)
            self.model.hfield_data = hfield


        elif self.task == 'gentle':
            ### TASK 3: GENTLE SLOPE
            self.height_walls = np.array([1, 1, 1, 1, 1, 1]) # low slope
            self.height = 1
            height = self.height
            width = self.width
            self.x_walls = np.array([255, 270, 285, 300, 315, 330]) - 5
            self.model.hfield_size = np.array([50, 5, height, 0.1])
            row = np.zeros((500,))  # (np.sin(x / x[-1] * np.pi * freq) + 1) / 2
            for i, x in enumerate(self.x_walls):
                terrain = np.cumsum([self.height_walls[i]] * width)
                row[x:x + width] += terrain
                row[x + width:] = row[x + width - 1]
            row = (row - np.min(row)) / (np.max(row) - np.min(row))
            hfield = np.tile(row.reshape(-1, 1), (1, 528)).T.reshape(-1, 1)
            self.model.hfield_data = hfield

        elif self.task == 'crazy':
            ### TASK 4: CRAZY SLOPE
            self.height_walls = np.array([1, 1, 1, 1, 1, 1]) # low slope
            self.height = 4
            height = self.height
            width = self.width
            self.x_walls = np.array([255, 270, 285, 300, 315, 330]) - 5
            self.model.hfield_size = np.array([50, 5, height, 0.1])
            row = np.zeros((500,))  # (np.sin(x / x[-1] * np.pi * freq) + 1) / 2
            for i, x in enumerate(self.x_walls):
                terrain = np.cumsum([self.height_walls[i]] * width)
                row[x:x + width] += terrain
                row[x + width:] = row[x + width - 1]
            row = (row - np.min(row)) / (np.max(row) - np.min(row))
            hfield = np.tile(row.reshape(-1, 1), (1, 528)).T.reshape(-1, 1)
            self.model.hfield_data = hfield

        elif self.task == 'None' or self.task is None:
            pass
        else:
            raise NotImplementedError

        # set 1 joint to be crippled
        # this joint will not be functional in step func
        # crippled_joint = 1  # value if value is not None else np.random.randint(1, self.action_dim)
        # self.cripple_mask = np.ones(self.action_space.shape)
        # self.cripple_mask[crippled_joint] = 0.
        #
        # geom_idx = self.model.geom_names.index(self.model.joint_names[crippled_joint + 3])
        # geom_rgba = self._init_geom_rgba.copy()
        # geom_rgba[geom_idx, :3] = np.array([1, 0, 0])
        # self.model.geom_rgba = geom_rgba
        # value = crippled_joint

        self.model.forward()

    def __getstate__(self):
        state = super(HalfCheetahEnv, self).__getstate__()
        state['task'] = self.task
        state['reset_every_episode'] = self.reset_every_episode
        state['reward'] = self._reward
        return state

    def __setstate__(self, d):
        super(HalfCheetahEnv, self).__setstate__(d)
        self.task = d['task']
        self.reset_every_episode = d['reset_every_episode']
        self._reward = d.get('reward', True)

    @overrides
    def log_diagnostics(self, paths):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
            ]
        logger.record_tabular('AverageForwardProgress', np.mean(progs))
        logger.record_tabular('MaxForwardProgress', np.max(progs))
        logger.record_tabular('MinForwardProgress', np.min(progs))
        logger.record_tabular('StdForwardProgress', np.std(progs))




