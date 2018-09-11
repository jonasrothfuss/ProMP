import numpy as np

from rllab.core.serializable import Serializable
from rllab.envs.base import Step
#from rllab.envs.mujoco.mujoco_env import MujocoEnv
from sandbox.ignasi.envs.mujoco_env import MujocoEnv
from rllab.misc import logger
from rllab.misc.overrides import overrides


def smooth_abs(x, param):
    return np.sqrt(np.square(x) + np.square(param)) - param


class HalfCheetahBlocksEnv(MujocoEnv, Serializable):

    FILE = '../../sandbox/ignasi/vendor/mujoco_models/half_cheetah_blocks.xml'

    def __init__(self, task=None, reset_every_episode=False, reward=True, *args, **kwargs):
        self.reset_every_episode = reset_every_episode
        self.first = True
        self.id_torso = 0
        super(HalfCheetahBlocksEnv, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)
        self._init_geom_rgba = self.model.geom_rgba.copy()
        self.init_body_mass = self.model.body_mass.copy()
        self.id_torso = self.model.body_names.index('torso')
        self._reward = reward
        self._action_bounds = self.action_bounds
        if task in [None, 'damping', 'forwback', 'mass', 'gravity', 'cripple', 'None']:
            self.task = task
            self.sign = 1
            self.cripple_mask = np.ones(self.action_space.shape)
        else:
            raise NameError

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flatten()[9:],
            self.model.data.qvel.flat[8:],
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
        super(HalfCheetahBlocksEnv, self).reset_mujoco(init_state=init_state)
        if self.reset_every_episode and not self.first:
            self.reset_task()
        if self.first:
            self.first = False

    def reset_task(self, value=None):

        #goal changes between moving forward and moving backward
        if self.task == 'forwback':
            self.sign = value if value is not None else np.random.choice([-1, 1])

        if self.task == 'damping':
            damping = self.model.dof_damping.copy()
            damping[:8, 0] = np.random.uniform(0, 10, size=8)
            self.model.dof_damping = damping

        #each task uses a cheetah with different mass for torso (torso includes all the other parts in it)
        elif self.task == 'mass':
            # self.model.body_names has ['world', 'torso', 'bthigh', 'bshin', 'bfoot', 'fthigh', 'fshin', 'ffoot']
            idx = self.model.body_names.index("torso")
            body_mass = self.init_body_mass.copy() #initial torso mass is 6.36
            body_mass[idx] = value if value is not None else np.random.uniform(1, 50)
            value = body_mass[idx][0]
            self.model.body_mass = body_mass

        #each task sets a different gravity value (currently, by applying z force on all 8 bodies)
            # why are we not just changing self.model.opt.gravity?
        elif self.task == 'gravity':
            g = value if value is not None else np.random.uniform(0.1, 20)
            value = g
            xfrc = np.zeros_like(self.model.data.xfrc_applied)
            #set xfrc
                # this is 8x6 (8 bodies, and each one has xfrc of dim 6)
                # first 3 are forces and second 3 are torques
                # so here, set the 3rd force (z) = m*a
            xfrc[:, 2] += (9.81 - g) * self.model.body_mass.copy().reshape(-1)
            self.model.data.xfrc_applied = xfrc

        #set one of the 6 joints to be crippled (and color it red)
        elif self.task == 'cripple':
            #set 1 joint to be crippled
                #this joint will not be functional in step func
            crippled_joint = value if value is not None else np.random.randint(1, self.action_dim)
            self.cripple_mask = np.ones(self.action_space.shape)
            self.cripple_mask[crippled_joint] = 0

            #self.model.geom_names has ['floor','torso','head','bthigh','bshin','bfoot','fthigh','fshin','ffoot']
                #add 3 to the index above, because 6 joints from above corresp to the last 6 things in geom_names
            geom_idx = self.model.geom_names.index(self.model.joint_names[crippled_joint+3])
            geom_rgba = self._init_geom_rgba.copy()
            #geom_rgba is 9x4
                #make the crippled joint be shown as "red"
            geom_rgba[geom_idx, :3] = np.array([1, 0, 0])
            self.model.geom_rgba = geom_rgba
            value = crippled_joint

        elif self.task in [None, 'None']:
            pass
        else:
            raise NotImplementedError

        self.model.forward()
        # print(self.task, value)

    def __getstate__(self):
        state = super(HalfCheetahBlocksEnv, self).__getstate__()
        state['task'] = self.task
        state['reset_every_episode'] = self.reset_every_episode
        state['reward'] = self._reward
        return state

    def __setstate__(self, d):
        super(HalfCheetahBlocksEnv, self).__setstate__(d)
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




