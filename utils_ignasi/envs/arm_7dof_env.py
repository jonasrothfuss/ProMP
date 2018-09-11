from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger
import numpy as np


class Arm7DofEnv(MujocoEnv, Serializable):

    FILE = '../../sandbox/ignasi/vendor/mujoco_models/arm_7dof.xml'

    def __init__(self, task=None, reset_every_episode=False, fixed_goal=False, *args, **kwargs):
        self.task = task
        self.sign = 1
        self.first = True
        self.fixed_goal = fixed_goal
        self.reset_every_episode = reset_every_episode
        super(Arm7DofEnv, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)
        self.penalty_ctrl = 0 ###########0.01
        self.init_geom_size = self.model.geom_size.copy()
        self.init_body_pos = self.model.body_pos.copy()
        self.init_geom_pos = self.model.geom_pos.copy()
        self.init_body_masses = self.model.body_mass.copy()
        self.cripple_mask = np.ones(self.action_space.shape)
        self._init_geom_rgba = self.model.geom_rgba.copy()

    '''ACTION SPACE:
    a[0] rotate LEFT
    a[1] shoulder bends down
    a[2] shoulder rotates right
    a[3] elbow bends down
    a[4] elbow (to wrist) rotates right
    a[5] wrist bends down
    a[6] wrist rotates right'''
    def step(self, action):

        action = self.cripple_mask * action
        
        #distance between end of reacher and the goal
        vec = self.get_body_com("object")-self.get_body_com("target")

        #calculate reward
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(action).sum()
        reward = reward_dist + self.penalty_ctrl * reward_ctrl

        ###print(reward_dist)

        #take a step
        self.forward_dynamics(action)
        ob = self.get_current_obs()
        done = False
        return Step(ob, float(reward), done, reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    @overrides
    def reset_mujoco(self, init_state=None, evaluating=None):

        if(init_state is None):
            '''#set random joint starting positions
            jnt_range_low = self.model.jnt_range[:, 0]
            jnt_range_high = self.model.jnt_range[:, 1]
            qpos = np.random.uniform(size=self.init_qpos.shape) *(jnt_range_high - jnt_range_low) + jnt_range_low'''
            low = np.array([-0.1, -.2, .5])
            high=np.array([0.4, .2, -.5])

            ##################
            qpos = np.ones(self.init_qpos.shape)*0.5 ##np.copy(self.init_qpos)
            ###self.goal = np.array([0.3, 0.15, 0])
            ##################

            #set random goal position
            self.fixed_goal=True
            if(self.fixed_goal):
                self.goal = np.array([0.3, 0.15, 0])
            else:
                self.goal = np.random.uniform(size=3)*(high - low) + low
            qpos[-3:,0] = self.goal

            #set random starting joint velocities
            qvel = self.init_qvel + np.random.uniform(low=-.005, high=.005, size=self.init_qvel.shape)
            #set 0 vel for the goal
            qvel[-3:,0] = 0

            #reset task, if supposed to
            if self.reset_every_episode and not self.first:
                self.reset_task()
            else:
                self.first = False

            #set vars
            self.model.data.qpos = qpos
            self.model.data.qvel = qvel
            self.model.data.qacc = self.init_qacc
            self.model.data.ctrl = self.init_ctrl

        else:
            start = 0
            for datum_name in ["qpos", "qvel", "qacc", "ctrl"]:
                datum = getattr(self.model.data, datum_name)
                datum_dim = datum.shape[0]
                datum = init_state[start: start + datum_dim]
                setattr(self.model.data, datum_name, datum)
                start += datum_dim


    @overrides
    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flat,
            self.model.data.qvel.flat,
            (self.get_body_com('object') - self.get_body_com("target"))
        ])

    def get_reward(self, obs, next_obs, action):
        vec = next_obs[:,-3:]
        reward_dist = -np.linalg.norm(vec, axis=1)
        reward_ctrl = -np.sum(np.square(action), axis=1)
        reward = reward_dist + self.penalty_ctrl * reward_ctrl
        return reward

    def reset_task(self, value=None):

        if self.task in [None, 'None']:
            pass

        elif 'cripple' in self.task:

            #pick which joint to cripple (0-6)
            crippled_joint = value if value is not None else np.random.randint(0, 7)
            self.cripple_mask = np.ones(self.action_space.shape)
            self.cripple_mask[crippled_joint]=0

            print("\n\nCRIPPLED JOINT: ", crippled_joint, "\n\n")

            #map from joint to elements_of_geomNames (make the crippled joint look red)
            #['floor','e1','e2','e1p','e2p','sp'--0,'sl'-- 1,'uar','ua','ef','fr','fa','wf','wr','pl','object','target']
            geom_rgba = self._init_geom_rgba.copy()
            geom_idx = crippled_joint+5
            geom_rgba[geom_idx, :3] = np.array([1, 0, 0])
            self.model.geom_rgba = geom_rgba
            value = crippled_joint

        elif 'damping' in self.task:
            #set damping
            #pick random damping
            damping = np.ones(self.model.dof_damping.shape)*1 ###np.random.uniform(0, 10, self.model.dof_damping.shape)

            #damping on 1st 2 joints is random... the last 2 are goal position
            damping[-2:,:] = 0

            self.model.dof_damping = damping

        elif 'size' in self.task:
            geom_size = self.init_geom_size.copy()
            body_pos = self.init_body_pos.copy()
            geom_pos = self.init_geom_pos.copy()
            links = np.random.choice([0, 1, 2, 3, 4], size=2, replace=False)
            links.sort()
            for link in links:
                geom_idx = self.model.geom_names.index('link' + str(link))
                body_son_idx = self.model.body_names.index('body'+str(link)) + 1
                size = np.random.uniform(0.01, 0.075)
                geom_size[geom_idx][1] = size
                body_pos[body_son_idx][0] = 2 * size
                geom_pos[geom_idx][0] = size
            self.model.geom_size = geom_size
            self.model.body_pos = body_pos
            self.model.geom_pos = geom_pos

        elif 'mass' in self.task:

            mass_multiplier = value if value is not None else np.random.randint(1, 4)
            masses = self.init_body_masses
            object_mass = masses[-2]
            masses[-2] = object_mass*mass_multiplier
            self.model.body_mass = masses

        elif 'force' in self.task:
            g = value if value is not None else np.random.uniform(.1, 2)
            masses = self.init_body_masses
            xfrc = np.zeros_like(self.model.data.xfrc_applied)
            #set xfrc
            # this is 8x6 (8 bodies, and each one has xfrc of dim 6)
            # first 3 are forces and second 3 are torques
            # so here, set the 3rd force (z) = m*a
            object_mass = masses[-2]
            xfrc[-2, 2] -= object_mass * g
            self.model.data.xfrc_applied = xfrc


        self.model.forward()

    def __getstate__(self):
        state = super(Arm7DofEnv, self).__getstate__()
        state['task'] = self.task
        return state

    def __setstate__(self, d):
        super(Arm7DofEnv, self).__setstate__(d)
        self.task = d['task']
