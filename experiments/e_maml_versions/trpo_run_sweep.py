import os
import json
import tensorflow as tf
import numpy as np
from experiment_utils.run_sweep import run_sweep
from maml_zoo.utils.utils import set_seed, ClassEncoder
from maml_zoo.baselines.linear_baseline import LinearFeatureBaseline, LinearTimeBaseline
from maml_zoo.envs.mujoco_envs.half_cheetah_rand_direc import HalfCheetahRandDirecEnv
from maml_zoo.envs.mujoco_envs.ant_rand_direc import AntRandDirecEnv
from maml_zoo.envs.mujoco_envs.ant_rand_goal import AntRandGoalEnv
from maml_zoo.envs.mujoco_envs.half_cheetah_rand_vel import HalfCheetahRandVelEnv
from maml_zoo.envs.mujoco_envs.swimmer_rand_vel import SwimmerRandVelEnv
from maml_zoo.envs.point_envs.point_env_2d_corner import MetaPointEnvCorner
from maml_zoo.envs.sawyer_envs.sawyer_pick_and_place import SawyerPickAndPlaceEnv
from rand_param_envs.hopper_rand_params import HopperRandParamsEnv
from rand_param_envs.walker2d_rand_params import Walker2DRandParamsEnv
from maml_zoo.envs.normalized_env import normalize
from maml_zoo.meta_algos.trpo_maml import TRPOMAML
from maml_zoo.meta_trainer import Trainer
from maml_zoo.samplers.maml_sampler import MAMLSampler
from maml_zoo.samplers.maml_sample_processor import MAMLSampleProcessor
from maml_zoo.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from maml_zoo.logger import logger

INSTANCE_TYPE = 'c4.4xlarge'
EXP_NAME = 'trpo-exploration-v2'

def run_experiment(**kwargs):
    exp_dir = os.getcwd() + '/data/' + EXP_NAME
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last_gap', snapshot_gap=50)
    json.dump(kwargs, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

    # Instantiate classes
    set_seed(kwargs['seed'])

    baseline = kwargs['baseline']()

    env = normalize(kwargs['env']()) # Wrappers?

    policy = MetaGaussianMLPPolicy(
        name="meta-policy",
        obs_dim=np.prod(env.observation_space.shape), # Todo...?
        action_dim=np.prod(env.action_space.shape),
        meta_batch_size=kwargs['meta_batch_size'],
        hidden_sizes=kwargs['hidden_sizes'],
        learn_std=kwargs['learn_std'],
        hidden_nonlinearity=kwargs['hidden_nonlinearity'],
        output_nonlinearity=kwargs['output_nonlinearity'],
    )

    # Load policy here

    sampler = MAMLSampler(
        env=env,
        policy=policy,
        rollouts_per_meta_task=kwargs['rollouts_per_meta_task'],
        meta_batch_size=kwargs['meta_batch_size'],
        max_path_length=kwargs['max_path_length'],
        parallel=kwargs['parallel'],
        envs_per_task=1,
    )

    sample_processor = MAMLSampleProcessor(
        baseline=baseline,
        discount=kwargs['discount'],
        gae_lambda=kwargs['gae_lambda'],
        normalize_adv=kwargs['normalize_adv'],
        positive_adv=kwargs['positive_adv'],
    )

    algo = TRPOMAML(
        policy=policy,
        step_size=kwargs['step_size'],
        inner_type=kwargs['inner_type'],
        inner_lr=kwargs['inner_lr'],
        meta_batch_size=kwargs['meta_batch_size'],
        num_inner_grad_steps=kwargs['num_inner_grad_steps'],
        exploration=kwargs['exploration'],
    )

    trainer = Trainer(
        algo=algo,
        policy=policy,
        env=env,
        sampler=sampler,
        sample_processor=sample_processor,
        n_itr=kwargs['n_itr'],
        num_inner_grad_steps=kwargs['num_inner_grad_steps'],
    )

    trainer.train()

if __name__ == '__main__':    

    sweep_params = {
        'seed' : [1, 2, 3],

        'baseline': [LinearFeatureBaseline],

        'env': [HalfCheetahRandDirecEnv, HalfCheetahRandVelEnv],

        'rollouts_per_meta_task': [40],
        'max_path_length': [100],
        'parallel': [True],

        'discount': [0.99],
        'gae_lambda': [1],
        'normalize_adv': [True],
        'positive_adv': [False],

        'hidden_sizes': [(64, 64)],
        'learn_std': [True],
        'hidden_nonlinearity': [tf.tanh],
        'output_nonlinearity': [None],

        'inner_lr': [0.1],
        'inner_type': ['log_likelihood'],
        'step_size': [0.01],
        'exploration': [True, False],

        'n_itr': [501],
        'meta_batch_size': [20],
        'num_inner_grad_steps': [1],
        'scope': [None],
    }
    
    run_sweep(run_experiment, sweep_params, EXP_NAME, INSTANCE_TYPE)