from rllab.envs.mujoco.mujoco_env import MujocoEnv
from rllab.core.serializable import Serializable
from rllab.envs.base import Step
from rllab.misc.overrides import overrides
from rllab.misc import logger

from rllab.envs.mujoco.mujoco_env import q_mult, q_inv
import numpy as np
import math


class RoachEnv(MujocoEnv, Serializable):
  FILE = 'roach.xml'
  ORI_IND = 3

  def __init__(self,  task=None, reset_every_episode=False, *args, **kwargs):
    self.reset_every_episode = reset_every_episode
    self.first = True
    super(RoachEnv, self).__init__(*args, **kwargs)
    Serializable.__init__(self, *args, **kwargs)
    self._init_geom_rgba = self.model.geom_rgba.copy()
    self.init_body_mass = self.model.body_mass.copy()
    self.id_torso = self.model.body_names.index('torso')
    if task in [None, 'None', 'forwback', 'mass', 'gravity', 'cripple', 'leg']:
      self.task = task
      self.sign = 1
      self.cripple_mask = np.ones(self.action_space.shape)
    else:
      raise (NameError, 'task ' + str(task) + ' not implemented')

  def get_current_obs(self):
    obs = np.concatenate([
      self.model.data.qpos.flat,
      self.model.data.qvel.flat,
      self.get_body_xmat("torso").flat,
      self.get_body_com("torso"),
    ]).reshape(-1)
    return obs

  def step(self, action):
    action = self.cripple_mask * action
    self.forward_dynamics(action)
    comvel = self.get_body_comvel("torso")
    forward_reward = comvel[0]
    lb, ub = self.action_bounds
    scaling = (ub - lb) * 0.5
    ctrl_cost = 0.5 * 1e-2 * np.sum(np.square(action / scaling))
    contact_cost = 0.5 * 1e-3 * np.sum(
      np.square(np.clip(self.model.data.cfrc_ext, -1, 1))),
    survive_reward = 0.05
    reward = forward_reward - ctrl_cost - contact_cost + survive_reward
    done = False
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
    dt = 0.03
    lb, ub = self.action_bounds
    scaling = (ub - lb) * 0.5
    ctrl_cost = 5e-3 * np.sum(np.square(action / scaling), axis=1)
    vel = (next_obs[:, -3] - obs[:, -3]) / dt
    reward = vel - ctrl_cost
    return reward

  def reset_mujoco(self, init_state=None):
    super(RoachEnv, self).reset_mujoco(init_state=init_state)
    if self.reset_every_episode and not self.first:
      self.reset_task()
    if self.first:
      self.first = False

  def reset_task(self, value=None):

    # goal changes between moving forward and moving backward
    if self.task == 'forwback':
      self.sign = value if value is not None else np.random.choice([-1, 1])

    # each task uses a cheetah with different mass for torso (torso includes all the other parts in it)
    elif self.task == 'mass':
      idx = self.model.body_names.index("torso")
      body_mass = self.init_body_mass.copy()  # initial torso mass is 6.36
      body_mass[idx] = value if value is not None else np.random.uniform(1, 50)
      value = body_mass[idx][0]
      self.model.body_mass = body_mass

      # each task sets a different gravity value (currently, by applying z force on all 8 bodies)
      # why are we not just changing self.model.opt.gravity?
    elif self.task == 'gravity':
      g = value if value is not None else np.random.uniform(0.1, 20)
      value = g
      xfrc = np.zeros_like(self.model.data.xfrc_applied)
      # set xfrc
      # this is 8x6 (8 bodies, and each one has xfrc of dim 6)
      # first 3 are forces and second 3 are torques
      # so here, set the 3rd force (z) = m*a
      xfrc[:, 2] += (9.81 - g) * self.model.body_mass.copy().reshape(-1)
      self.model.data.xfrc_applied = xfrc

    # set one of the 6 joints to be crippled (and color it red)
    elif self.task == 'cripple':
      # set 1 joint to be crippled
      # this joint will not be functional in step func
      crippled_joint = value if value is not None else np.random.randint(0, self.action_dim)
      self.cripple_mask = np.ones(self.action_space.shape)
      self.cripple_mask[crippled_joint] = 0

      # self.model.geom_names has ['floor','torso','head','bthigh','bshin','bfoot','fthigh','fshin','ffoot']
      # add 3 to the index above, because 6 joints from above corresp to the last 6 things in geom_names
      geom_idx = self.model.geom_names.index(self.model.joint_names[crippled_joint + 1])
      geom_rgba = self._init_geom_rgba.copy()
      # geom_rgba is 9x4
      # make the crippled joint be shown as "red"
      geom_rgba[geom_idx, :3] = np.array([1, 0, 0])
      self.model.geom_rgba = geom_rgba
      value = crippled_joint

    elif self.task == 'leg':
      # set 1 joint to be crippled
      # this joint will not be functional in step func
      crippled_joint = value if value is not None else np.random.randint(0, 6)
      self.cripple_mask = np.ones(self.action_space.shape)
      self.cripple_mask[crippled_joint*3:(crippled_joint+1)*3] = 0

      # self.model.geom_names has ['floor','torso','head','bthigh','bshin','bfoot','fthigh','fshin','ffoot']
      # add 3 to the index above, because 6 joints from above corresp to the last 6 things in geom_names
      geom_idx = self.model.geom_names.index(self.model.joint_names[crippled_joint*3 + 1])
      geom_rgba = self._init_geom_rgba.copy()
      # geom_rgba is 9x4
      # make the crippled joint be shown as "red"
      geom_rgba[geom_idx:geom_idx+3, :3] = np.array([1, 0, 0])
      self.model.geom_rgba = geom_rgba
      value = crippled_joint

    elif self.task in [None, 'None']:
      pass

    else:
      raise NotImplementedError

    self.model.forward()
    # print(self.task, value)


    def __getstate__(self):
        state = super(RoachEnv, self).__getstate__()
        state['task'] = self.task
        state['reset_every_episode'] = self.reset_every_episode
        return state

    def __setstate__(self, d):
        super(RoachEnv, self).__setstate__(d)
        self.task = d['task']
        self.reset_every_episode = d['reset_every_episode']

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
