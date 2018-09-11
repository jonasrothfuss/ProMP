from rllab.envs.base import Step
from rllab.misc.overrides import overrides
#from .mujoco_env import MujocoEnv
from rllab.envs.mujoco.mujoco_env import MujocoEnv
import numpy as np
from rllab.core.serializable import Serializable
from rllab.misc import logger
from rllab.misc import autoargs
import numpy as np

class SwimmerBlocksEnv(MujocoEnv, Serializable):

    FILE = '../../sandbox/ignasi/vendor/mujoco_models/swimmer_blocks.xml'
    ORI_IND = 2

    def __init__(self, task=None, reset_every_episode=False, switch_task=False, switch_only_once=False, *args, **kwargs):

        if(task=='None'):
            task=None
        self.task=task
        self.ctrl_cost_coeff = 0 ###########1e-2
        self.reset_every_episode = reset_every_episode
        self.switch_task = switch_task
        self.switch_only_once = switch_only_once
        self.first = True
        self.crippled_leg= 0
        self.num_steps=0
        self.stopSwitching=False

        super(SwimmerBlocksEnv, self).__init__(*args, **kwargs)
        Serializable.quick_init(self, locals())

        self.dt = self.model.opt.timestep

        if task in [None, 'force', 'rand_const_fric', 'low_fric', 'high_fric', 'one_fric']:
            self.task = task
            self.cripple_mask = np.ones(self.action_space.shape)
        else:
            raise NameError

        self.crippled_leg= 0
        self.stopSwitching=False
        self.num_steps = 0 
        self.switch_task = switch_task
        self.switch_only_once = switch_only_once

        #setup vars for blocks
        self.block_height = -0.55
        self.num_sections = 1 ##################8
        self.init_position_curr = -2
        self.position_inc = 2
        self.block_size = 80 ########################self.position_inc/2.0-0.01
        self.block_depth = 0.5

        #setup blocks sizes/depths
        sizes = self.model.geom_size.copy()
        body_positions = self.model.body_pos.copy()
        position_curr=self.init_position_curr
        for section in range(self.num_sections):
            body_positions[section+1][2] = self.block_height
            body_positions[section+1][0]= position_curr
            sizes[section][0]= self.block_size
            sizes[section][2] = self.block_depth
            position_curr+= self.position_inc
        self.model.geom_size= sizes
        self.model.body_pos= body_positions

        #setup blocks frictions and colors
        frictions = self.model.geom_friction.copy()
        colors = self.model.geom_rgba.copy()
        # for section in range(self.num_sections):
        #     selected_fric = 1
        #     frictions[section][0]= selected_fric
        #     colors[section]=np.array([1,0,0,1])
        self.model.geom_friction = frictions
        self.model.geom_rgba = colors

        #init geom values
        self._init_geom_rgba = self.model.geom_rgba.copy()
        self._init_geom_contype = self.model.geom_contype.copy()
        self._init_geom_size = self.model.geom_size.copy()
        self._init_body_pos = self.model.body_pos.copy()
        self.init_body_masses = self.model.body_mass.copy()
        self._init_geom_friction = self.model.geom_friction.copy()

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            self.get_body_com("torso").flat,
        ]).reshape(-1)

    def get_ori(self):
        return self.model.data.qpos[self.__class__.ORI_IND]

    def step(self, action):
        action = self.cripple_mask * action

        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * self.ctrl_cost_coeff * np.sum(
            np.square(action / scaling))
        forward_reward = self.get_body_comvel("torso")[0]
        reward = forward_reward - ctrl_cost
        done = False

        self.num_steps+=1
        if(self.num_steps%100==0):
            if(self.switch_task):
                self.reset_task(switch=True)
        return Step(next_obs, reward, done)

    '''@overrides
    def log_diagnostics(self, paths):
        if len(paths) > 0:
            progs = [
                path["observations"][-1][-3] - path["observations"][0][-3]
                for path in paths
            ]
            logger.record_tabular('AverageForwardProgress', np.mean(progs))
            logger.record_tabular('MaxForwardProgress', np.max(progs))
            logger.record_tabular('MinForwardProgress', np.min(progs))
            logger.record_tabular('StdForwardProgress', np.std(progs))
        else:
            logger.record_tabular('AverageForwardProgress', np.nan)
            logger.record_tabular('MaxForwardProgress', np.nan)
            logger.record_tabular('MinForwardProgress', np.nan)
            logger.record_tabular('StdForwardProgress', np.nan)'''

    def get_reward(self, obs, next_obs, action):
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * self.ctrl_cost_coeff * np.sum(
            np.square(action / scaling))
        forward_reward = (next_obs[:, -3] - obs[:, -3]) / self.dt #vx
        reward = forward_reward - ctrl_cost
        return reward

    def reset_mujoco(self, init_state=None):
        super(SwimmerBlocksEnv, self).reset_mujoco(init_state=init_state)
        self.num_steps=0
        if self.reset_every_episode and not self.first:
            self.reset_task(switch=self.switch_task, calling_reset=True)
        if self.first:
            self.first = False

    def reset_task(self, value=None, switch=False, calling_reset=False):

        if(self.task=='rand_const_fric'):

            #select a random friction (and color), same for each block
            selected_fric = np.random.uniform(1, 3.5)
            color_use = (selected_fric-1)/(3)*0.7+0.3

            frictions = self._init_geom_friction
            colors = self._init_geom_rgba
            for section in range(self.num_sections):
                frictions[section][0]= selected_fric
                colors[section]=np.array([1,0,0,color_use])
            self.model.geom_friction = frictions
            self.model.geom_rgba = colors

        elif(self.task=='low_fric'):
            #select a random friction (and color), same for each block
            selected_fric = 2.0
            color_use = (selected_fric-1)/(3)*0.7+0.3

            frictions = self._init_geom_friction
            colors = self._init_geom_rgba
            for section in range(self.num_sections):
                frictions[section][0]= selected_fric
                colors[section]=np.array([1,0,0,color_use])
            self.model.geom_friction = frictions
            self.model.geom_rgba = colors

        elif(self.task=='high_fric'):
            #select a random friction (and color), same for each block
            selected_fric = 5.1
            color_use = (selected_fric-1)/(3)*0.7+0.3

            frictions = self._init_geom_friction
            colors = self._init_geom_rgba
            for section in range(self.num_sections):
                frictions[section][0]= selected_fric
                colors[section]=np.array([1,0,0,color_use])
            self.model.geom_friction = frictions
            self.model.geom_rgba = colors

        elif(self.task=='one_fric'):
            #select a random friction (and color), same for each block
            selected_fric = 1.0
            color_use = (selected_fric-1)/(3)*0.7+0.3

            frictions = self._init_geom_friction
            colors = self._init_geom_rgba
            for section in range(self.num_sections):
                frictions[section][0]= selected_fric
                colors[section]=np.array([1,0,0,color_use])
            self.model.geom_friction = frictions
            self.model.geom_rgba = colors


        elif 'force' in self.task:
            masses = self.init_body_masses
            xfrc = np.zeros_like(self.model.data.xfrc_applied)
            g = value if value is not None else np.random.uniform(-2, 2, size=(xfrc.shape[0], 2))
            # this is 8x6 (8 bodies, and each one has xfrc of dim 6)
            # first 3 are forces and second 3 are torques
            # so here, set the 3rd force (z) = m*a
            xfrc[:, :2] = masses * g
            self.model.data.xfrc_applied = xfrc

        elif self.task is None:
            pass
        else:
            raise NotImplementedError
        self.model.forward()

    def __getstate__(self):
        state = super(SwimmerBlocksEnv, self).__getstate__()
        state['task'] = self.task
        state['reset_every_episode'] = self.reset_every_episode
        return state

    def __setstate__(self, d):
        super(SwimmerBlocksEnv, self).__setstate__(d)
        self.task = d['task']
        self.reset_every_episode = d['reset_every_episode']

    ##########################################################

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

