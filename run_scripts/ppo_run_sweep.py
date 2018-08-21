import os
import sys
import argparse
import json
import itertools
import doodad as dd
import doodad.mount as mount
import doodad.easy_sweep.launcher as launcher
from doodad.easy_sweep.hyper_sweep import run_sweep_doodad
import tensorflow as tf
import numpy as np
from maml_zoo.utils.utils import set_seed, ClassEncoder
from maml_zoo.baselines.linear_feature_baseline import LinearFeatureBaseline
from maml_zoo.envs.half_cheetah_rand_direc import HalfCheetahRandDirecEnv
from maml_zoo.envs.normalized_env import normalize
from maml_zoo.meta_algos.ppo_maml import PPOMAML
from maml_zoo.meta_trainer import Trainer
from maml_zoo.samplers.maml_sampler import MAMLSampler
from maml_zoo.samplers.maml_sample_processor import MAMLSampleProcessor
from maml_zoo.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from maml_zoo.logger import logger

INSTANCE_TYPE = 'c4.2xlarge'
EXP_NAME = 'ppo/ppo-hyperparams'

def run_experiment(**kwargs):
    logger.configure(dir='./data', format_strs=['stdout', 'log', 'csv'], snapshot_mode='last_gap', snapshot_gap=50)
    json.dump(kwargs, open('./data' + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

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
    )

    sample_processor = MAMLSampleProcessor(
        baseline=baseline,
        discount=kwargs['discount'],
        gae_lambda=kwargs['gae_lambda'],
        normalize_adv=kwargs['normalize_adv'],
        positive_adv=kwargs['positive_adv'],
    )

    algo = PPOMAML(
        policy=policy,
        inner_lr=kwargs['inner_lr'],
        meta_batch_size=kwargs['meta_batch_size'],
        num_inner_grad_steps=kwargs['num_inner_grad_steps'],
        learning_rate=kwargs['learning_rate'],
        max_epochs=kwargs['max_epochs'],
        num_minibatches=kwargs['num_minibatches'],
        clip_eps=kwargs['clip_eps'], 
        clip_outer=kwargs['clip_outer'],
        target_outer_step=kwargs['target_outer_step'],
        target_inner_step=kwargs['target_inner_step'],
        init_outer_kl_penalty=kwargs['init_outer_kl_penalty'],
        init_inner_kl_penalty=kwargs['init_inner_kl_penalty'],
        adaptive_outer_kl_penalty=kwargs['adaptive_outer_kl_penalty'],
        adaptive_inner_kl_penalty=kwargs['adaptive_inner_kl_penalty'],
        anneal_factor=kwargs['anneal_factor'],
        entropy_bonus=kwargs['entropy_bonus'],
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='local',
                        help='Mode for running the experiments - local: runs on local machine, '
                             'ec2: runs on AWS ec2 cluster (requires a proper configuration file)')

    args = parser.parse_args(sys.argv[1:])

    local_mount = mount.MountLocal(local_dir='~/maml_zoo', pythonpath=True)

    sweep_params = {
        'seed': [1, 2, 3, 4, 5],

        'baseline': [LinearFeatureBaseline],

        'env': [HalfCheetahRandDirecEnv],

        'rollouts_per_meta_task': [20],
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
        'learning_rate': [1e-3],
        'max_epochs': [5],
        'num_minibatches': [1],
        'clip_eps': [0.5],
        'clip_outer': [True],
        'target_outer_step': [0],
        'target_inner_step': [2e-2],
        'init_outer_kl_penalty': [0],
        'init_inner_kl_penalty': [1e-3],
        'adaptive_outer_kl_penalty': [False],
        'adaptive_inner_kl_penalty': [True],
        'anneal_factor': [1.0],
        'entropy_bonus': [0.0],

        'n_itr': [301],
        'meta_batch_size': [40],
        'num_inner_grad_steps': [1],
        'scope': [None],
    }
    
    sweeper = launcher.DoodadSweeper([local_mount], docker_img="dennisl88/maml_zoo", docker_output_dir='./data')
    if args.mode == 'ec2':
        # print("\n" + "**********" * 10 + "\nexp_prefix: {}\nvariants: {}".format(EXP_NAME, len(itertools.product(sweep_params))))
        sweeper.run_sweep_ec2(run_experiment, sweep_params, bucket_name='rllab-experiments', instance_type=INSTANCE_TYPE, s3_log_name=EXP_NAME, add_date_to_logname=False)
    elif args.mode == 'local_docker':
        mode_docker = dd.mode.LocalDocker(
            image=sweeper.image,
        )
        run_sweep_doodad(run_experiment, sweep_params, run_mode=mode_docker, 
                mounts=sweeper.mounts)
    elif args.mode == 'local':
        sweeper.run_sweep_serial(run_experiment, sweep_params)
    else:
        raise NotImplementedError