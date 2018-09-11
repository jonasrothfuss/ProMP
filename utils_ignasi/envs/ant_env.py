from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger

from rllab.envs.mujoco.mujoco_env import q_mult, q_inv
import numpy as np
import math


class AntEnv(MujocoEnv, Serializable):

    FILE = 'ant.xml'
    ORI_IND = 3

    def __init__(self, task=None, reset_every_episode=False, *args, **kwargs):

        if(task=='None'):
            task=None

        
        self.reset_every_episode = reset_every_episode
        self.first = True

        super(AntEnv, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)

        self._init_geom_rgba = self.model.geom_rgba.copy()
        self._init_geom_contype = self.model.geom_contype.copy()
        self._init_geom_size = self.model.geom_size.copy()
        self._init_geom_pos = self.model.geom_pos.copy()

        if task in [None, 'forwback', 'mass', 'gravity', 'cripple','remove_leg', 'shrink_leg', 'LegNothing']:
            self.task = task
            self.sign = 1
            self.cripple_mask = np.ones(self.action_space.shape)
        else:
            raise NameError

        self.dt = self.model.opt.timestep

        self.crippled_leg=0

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            #######np.clip(self.model.data.cfrc_ext, -1, 1).flat,
            self.get_body_xmat("torso").flat,
            self.get_body_com("torso"),
        ]).reshape(-1)

    def step(self, action):
        action = self.cripple_mask * action
        self.forward_dynamics(action)
        comvel = self.get_body_comvel("torso")
        forward_reward = comvel[0]
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
        contact_cost = 0 #########0.5 * 1e-3 * np.sum(np.square(np.clip(self.model.data.cfrc_ext, -1, 1))),
        survive_reward = 0.05
        reward = forward_reward - ctrl_cost - contact_cost + survive_reward
        state = self._state
        notdone = np.isfinite(state).all() \
            and state[2] >= 0.2 and state[2] <= 1.0
        done = False #########not notdone
        ob = self.get_current_obs()
        return Step(ob, float(reward), done)

    @overrides
    def get_ori(self):
        ori = [0, 1, 0, 0]
        rot = self.model.data.qpos[self.__class__.ORI_IND:self.__class__.ORI_IND + 4]  # take the quaternion
        ori = q_mult(q_mult(rot, ori), q_inv(rot))[1:3]  # project onto x-y plane
        ori = math.atan2(ori[1], ori[0])
        return ori

    def get_reward(self, obs, next_obs, action):
        lb, ub = self.action_bounds
        scaling = (ub - lb) * 0.5
        ctrl_cost = 0 ##############5e-3 * np.sum(np.square(action / scaling), axis=1)
        vel = (next_obs[:, -3] - obs[:, -3]) / self.dt
        survive_reward = 0.05
        reward = vel - ctrl_cost + survive_reward
        return reward

    ##########################################################

    def reset_mujoco(self, init_state=None):
        super(AntEnv, self).reset_mujoco(init_state=init_state)
        if self.reset_every_episode and not self.first:
            self.reset_task()
        if self.first:
            self.first = False

    '''
    our "front" is in +x direction, to the right side of screen

    LEG 4 (they call this back R)
    action0: front-right leg, top joint 
    action1: front-right leg, bottom joint
    
    LEG 1 (they call this front L)
    action2: front-left leg, top joint
    action3: front-left leg, bottom joint 
    
    LEG 2 (they call this front R)
    action4: back-left leg, top joint
    action5: back-left leg, bottom joint 
    
    LEG 3 (they call this back L)
    action6: back-right leg, top joint
    action7: back-right leg, bottom joint 

    geom_names has 
            ['floor','torso_geom',
            'aux_1_geom','left_leg_geom','left_ankle_geom', --1
            'aux_2_geom','right_leg_geom','right_ankle_geom', --2
            'aux_3_geom','back_leg_geom','third_ankle_geom', --3
            'aux_4_geom','rightback_leg_geom','fourth_ankle_geom'] --4
    '''

    def reset_task(self, value=None, switch=False):

        if(self.task=='remove_leg'):

            #pick which leg to remove (0 1 2 are train... 3 is test)
            self.crippled_leg = 1 #######value if value is not None else np.random.randint(0, 3)
            #print("\n\nREMOVED LEG: ", self.crippled_leg, "\n\n")

            #pick which actuators to disable
            self.cripple_mask = np.ones(self.action_space.shape)
            if(self.crippled_leg==0):
                self.cripple_mask[2] = 0
                self.cripple_mask[3] = 0
            elif(self.crippled_leg==1):
                self.cripple_mask[4] = 0
                self.cripple_mask[5] = 0
            elif(self.crippled_leg==2):
                self.cripple_mask[6] = 0
                self.cripple_mask[7] = 0
            elif(self.crippled_leg==3):
                self.cripple_mask[0] = 0
                self.cripple_mask[1] = 0

            #make the removed leg look red
            geom_rgba = self._init_geom_rgba.copy()
            if(self.crippled_leg==0):
                #geom_rgba[3, :3] = np.array([1, 0, 0])
                geom_rgba[4, :3] = np.array([1, 0, 0])
            elif(self.crippled_leg==1):
                #geom_rgba[6, :3] = np.array([1, 0, 0])
                geom_rgba[7, :3] = np.array([1, 0, 0])
            elif(self.crippled_leg==2):
                #geom_rgba[9, :3] = np.array([1, 0, 0])
                geom_rgba[10, :3] = np.array([1, 0, 0])
            elif(self.crippled_leg==3):
                #geom_rgba[12, :3] = np.array([1, 0, 0])
                geom_rgba[13, :3] = np.array([1, 0, 0])
            self.model.geom_rgba = geom_rgba

            #make the removed leg not affect anything
            temp = self._init_geom_contype.copy()
            if(self.crippled_leg==0):
                #temp[3, :] = np.array([0])
                temp[4, :] = np.array([0])
            elif(self.crippled_leg==1):
                #temp[6, :] = np.array([0])
                temp[7, :] = np.array([0])
            elif(self.crippled_leg==2):
                #temp[9, :] = np.array([0])
                temp[10, :] = np.array([0])
            elif(self.crippled_leg==3):
                #temp[12, :] = np.array([0])
                temp[13, :] = np.array([0])
            self.model.geom_contype = temp

            value = self.crippled_leg

        elif(self.task=='shrink_leg'):

            #pick which leg to remove (0 1 2 are train... 3 is test)
            self.crippled_leg = value if value is not None else np.random.randint(0, 3)
            #print("\n\nREMOVED LEG: ", self.crippled_leg, "\n\n")

            #pick which actuators to disable
            self.cripple_mask = np.ones(self.action_space.shape)
            if(self.crippled_leg==0):
                self.cripple_mask[2] = 0
                self.cripple_mask[3] = 0
            elif(self.crippled_leg==1):
                self.cripple_mask[4] = 0
                self.cripple_mask[5] = 0
            elif(self.crippled_leg==2):
                self.cripple_mask[6] = 0
                self.cripple_mask[7] = 0
            elif(self.crippled_leg==3):
                self.cripple_mask[0] = 0
                self.cripple_mask[1] = 0

            #make the removed leg look red
            geom_rgba = self._init_geom_rgba.copy()
            if(self.crippled_leg==0):
                geom_rgba[3, :3] = np.array([1, 0, 0])
                geom_rgba[4, :3] = np.array([1, 0, 0])
            elif(self.crippled_leg==1):
                geom_rgba[6, :3] = np.array([1, 0, 0])
                geom_rgba[7, :3] = np.array([1, 0, 0])
            elif(self.crippled_leg==2):
                geom_rgba[9, :3] = np.array([1, 0, 0])
                geom_rgba[10, :3] = np.array([1, 0, 0])
            elif(self.crippled_leg==3):
                geom_rgba[12, :3] = np.array([1, 0, 0])
                geom_rgba[13, :3] = np.array([1, 0, 0])
            self.model.geom_rgba = geom_rgba

            #make the removed leg not affect anything
            temp_size = self._init_geom_size.copy()
            temp_pos = self._init_geom_pos.copy()

            if(self.crippled_leg==0):
                #top half
                temp_size[3, 0] = temp_size[3, 0]/2
                temp_size[3, 1] = temp_size[3, 1]/2
                #bottom half
                temp_size[4, 0] = temp_size[4, 0]/2
                temp_size[4, 1] = temp_size[4, 1]/2
                temp_pos[4, :] = temp_pos[3, :]
            elif(self.crippled_leg==1):
                #top half
                temp_size[6, 0] = temp_size[6, 0]/2
                temp_size[6, 1] = temp_size[6, 1]/2
                #bottom half
                temp_size[7, 0] = temp_size[7, 0]/2
                temp_size[7, 1] = temp_size[7, 1]/2
                temp_pos[7, :] = temp_pos[6, :]
            elif(self.crippled_leg==2):
                #top half
                temp_size[9, 0] = temp_size[9, 0]/2
                temp_size[9, 1] = temp_size[9, 1]/2
                #bottom half
                temp_size[10, 0] = temp_size[10, 0]/2
                temp_size[10, 1] = temp_size[10, 1]/2
                temp_pos[10, :] = temp_pos[9, :]
            elif(self.crippled_leg==3):
                #top half
                temp_size[12, 0] = temp_size[12, 0]/2
                temp_size[12, 1] = temp_size[12, 1]/2
                #bottom half
                temp_size[13, 0] = temp_size[13, 0]/2
                temp_size[13, 1] = temp_size[13, 1]/2
                temp_pos[13, :] = temp_pos[12, :]

            self.model.geom_size = temp_size
            self.model.geom_pos = temp_pos
            value = self.crippled_leg

        elif(self.task=='LegNothing'):

            #pick which leg to remove (0 1 2 are train... 3 is test)
            self.crippled_leg = 0
            #print("\n\nREMOVED LEG: ", self.crippled_leg, "\n\n")

            #pick which actuators to disable
            self.cripple_mask = np.ones(self.action_space.shape)
            if(self.crippled_leg==0):
                self.cripple_mask[2] = 0
                self.cripple_mask[3] = 0
            elif(self.crippled_leg==1):
                self.cripple_mask[4] = 0
                self.cripple_mask[5] = 0
            elif(self.crippled_leg==2):
                self.cripple_mask[6] = 0
                self.cripple_mask[7] = 0
            elif(self.crippled_leg==3):
                self.cripple_mask[0] = 0
                self.cripple_mask[1] = 0

            #make the removed leg look red
            geom_rgba = self._init_geom_rgba.copy()
            if(self.crippled_leg==0):
                geom_rgba[3, :3] = np.array([1, 0, 0])
                geom_rgba[4, :3] = np.array([1, 0, 0])
            elif(self.crippled_leg==1):
                geom_rgba[6, :3] = np.array([1, 0, 0])
                geom_rgba[7, :3] = np.array([1, 0, 0])
            elif(self.crippled_leg==2):
                geom_rgba[9, :3] = np.array([1, 0, 0])
                geom_rgba[10, :3] = np.array([1, 0, 0])
            elif(self.crippled_leg==3):
                geom_rgba[12, :3] = np.array([1, 0, 0])
                geom_rgba[13, :3] = np.array([1, 0, 0])
            self.model.geom_rgba = geom_rgba

            #make the removed leg not affect anything
            temp_size = self._init_geom_size.copy()
            temp_pos = self._init_geom_pos.copy()

            if(self.crippled_leg==0):
                #top half
                temp_size[3, 0] = temp_size[3, 0]/2
                temp_size[3, 1] = temp_size[3, 1]/2
                #bottom half
                temp_size[4, 0] = temp_size[4, 0]/2
                temp_size[4, 1] = temp_size[4, 1]/2
                temp_pos[4, :] = temp_pos[3, :]
            elif(self.crippled_leg==1):
                #top half
                temp_size[6, 0] = temp_size[6, 0]/2
                temp_size[6, 1] = temp_size[6, 1]/2
                #bottom half
                temp_size[7, 0] = temp_size[7, 0]/2
                temp_size[7, 1] = temp_size[7, 1]/2
                temp_pos[7, :] = temp_pos[6, :]
            elif(self.crippled_leg==2):
                #top half
                temp_size[9, 0] = temp_size[9, 0]/2
                temp_size[9, 1] = temp_size[9, 1]/2
                #bottom half
                temp_size[10, 0] = temp_size[10, 0]/2
                temp_size[10, 1] = temp_size[10, 1]/2
                temp_pos[10, :] = temp_pos[9, :]
            elif(self.crippled_leg==3):
                #top half
                temp_size[12, 0] = temp_size[12, 0]/2
                temp_size[12, 1] = temp_size[12, 1]/2
                #bottom half
                temp_size[13, 0] = temp_size[13, 0]/2
                temp_size[13, 1] = temp_size[13, 1]/2
                temp_pos[13, :] = temp_pos[12, :]

            self.model.geom_size = temp_size
            self.model.geom_pos = temp_pos
            value = self.crippled_leg

        elif self.task == 'cripple':

            if(switch):
                if(self.crippled_leg==0):
                    self.crippled_leg=3
                else:
                    self.crippled_leg=0

            else:
                #pick which leg to cripple (0 1 2 are train... 3 is test)
                self.crippled_leg = value if value is not None else np.random.randint(0, 3)

            ####print("\n\nCRIPPLED LEG: ", self.crippled_leg, "\n\n")

            #map from leg to elements_of_action
            self.cripple_mask = np.ones(self.action_space.shape)
            if(self.crippled_leg==0):
                self.cripple_mask[2] = 0
                self.cripple_mask[3] = 0
            elif(self.crippled_leg==1):
                self.cripple_mask[4] = 0
                self.cripple_mask[5] = 0
            elif(self.crippled_leg==2):
                self.cripple_mask[6] = 0
                self.cripple_mask[7] = 0
            elif(self.crippled_leg==3):
                self.cripple_mask[0] = 0
                self.cripple_mask[1] = 0

            #map from leg to elements_of_geomNames (make the crippled leg look red)
            geom_rgba = self._init_geom_rgba.copy()
            if(self.crippled_leg==0):
                geom_rgba[3, :3] = np.array([1, 0, 0])
                geom_rgba[4, :3] = np.array([1, 0, 0])
            elif(self.crippled_leg==1):
                geom_rgba[6, :3] = np.array([1, 0, 0])
                geom_rgba[7, :3] = np.array([1, 0, 0])
            elif(self.crippled_leg==2):
                geom_rgba[9, :3] = np.array([1, 0, 0])
                geom_rgba[10, :3] = np.array([1, 0, 0])
            elif(self.crippled_leg==3):
                geom_rgba[12, :3] = np.array([1, 0, 0])
                geom_rgba[13, :3] = np.array([1, 0, 0])
            
            
            self.model.geom_rgba = geom_rgba
            value = self.crippled_leg

        elif self.task is None:
            pass
        else:
            raise NotImplementedError
        self.model.forward()

    def __getstate__(self):
        state = super(AntEnv, self).__getstate__()
        state['task'] = self.task
        state['reset_every_episode'] = self.reset_every_episode
        return state

    def __setstate__(self, d):
        super(AntEnv, self).__setstate__(d)
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

