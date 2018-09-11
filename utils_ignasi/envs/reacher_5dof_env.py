from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger
import numpy as np


class Reacher5DofEnv(MujocoEnv, Serializable):

    FILE = '../../sandbox/ignasi/vendor/mujoco_models/reacher_5dof.xml'

    def __init__(self, task=None, reset_every_episode=False, fixed_goal=True, *args, **kwargs):
        self.task = task
        self.sign = 1
        self.first = True
        self.fixed_goal = fixed_goal
        self.reset_every_episode = reset_every_episode
        super(Reacher5DofEnv, self).__init__(*args, **kwargs)
        Serializable.__init__(self, *args, **kwargs)
        self.penalty_ctrl = 0
        self.init_geom_size = self.model.geom_size.copy()
        self.init_body_pos = self.model.body_pos.copy()
        self.init_geom_pos = self.model.geom_pos.copy()

    def step(self, action):
        
        #distance between end of reacher and the goal
        vec = self.get_body_com("fingertip")-self.get_body_com("target")

        #calculate reward
        reward_dist = - np.linalg.norm(vec)
        reward_ctrl = - np.square(action).sum()
        reward = reward_dist + self.penalty_ctrl * reward_ctrl

        #take a step
        self.forward_dynamics(action)
        ob = self.get_current_obs()
        done = False
        return Step(ob, float(reward), done, reward_dist=reward_dist, reward_ctrl=reward_ctrl)

    @overrides
    def reset_mujoco(self, init_state=None, evaluating=None):

        if(init_state is None):

            #set random joint starting positions
            qpos = np.random.uniform(low=-1.5, high=1.5, size=self.init_qpos.shape) + self.init_qpos

            #set random goal position
            if(self.fixed_goal):
                self.goal = np.array([0.15, 0.1])
            else:
                while True:
                    potential_goal = np.random.uniform(low=-.15, high=.15, size=2)
                    if np.linalg.norm(potential_goal) > 0.03:
                        self.goal = potential_goal
                        break
            qpos[-2:,0] = self.goal

            #set random starting joint velocities
            qvel = self.init_qvel + np.random.uniform(low=-.005, high=.005, size=self.init_qvel.shape)
            #set 0 vel for the goal
            qvel[-2:,0] = 0

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

    #def get_current_obs(self):
    #    theta = self.model.data.qpos.flat[:-4]
    #    return np.concatenate([
    #        np.cos(theta),
    #        np.sin(theta),
    #        self.model.data.qpos.flat[-4:-2],
    #        self.model.data.qvel.flat[:-2],
    #        (self.get_body_com('fingertip') - self.get_body_com("target"))[:2]
    #    ])

    def get_current_obs(self):
        #qpos 7 things (5pos, goalx, goaly)
        #qvel 7 things (5pos, goalx, goaly)
        #OBS: 5,5,2,5,2
        theta = self.model.data.qpos.flat[:-2]
        return np.concatenate([
            np.cos(theta),
            np.sin(theta),
            self.model.data.qpos.flat[-2:],
            self.model.data.qvel.flat[:-2],
            (self.get_body_com('fingertip') - self.get_body_com("target"))[:2]
        ])

    def get_reward(self, obs, next_obs, action):
        vec = next_obs[:,-2:]
        reward_dist = -np.linalg.norm(vec, axis=1)
        reward_ctrl = -np.sum(np.square(action), axis=1)
        reward = reward_dist + self.penalty_ctrl * reward_ctrl
        return reward

    def reset_task(self):

        #if self.task == 'forwback':
        #    self.sign = np.random.choice([-1, 1])
        if self.task in [None, 'None']:
            pass

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


        self.model.forward()

    def __getstate__(self):
        state = super(Reacher5DofEnv, self).__getstate__()
        state['task'] = self.task
        return state

    def __setstate__(self, d):
        super(Reacher5DofEnv, self).__setstate__(d)
        self.task = d['task']