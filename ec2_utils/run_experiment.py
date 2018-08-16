import os
import sys
import argparse
import doodad as dd
import doodad.mount as mount
import doodad.easy_sweep.launcher as launcher
from doodad.easy_sweep.hyper_sweep import run_sweep_doodad
import tensorflow as tf
import numpy as np
from maml_zoo.baselines.linear_feature_baseline import LinearFeatureBaseline
from maml_zoo.envs.half_cheetah_rand_direc import HalfCheetahRandDirecEnv
from maml_zoo.meta_algos.ppo_maml import MAMLPPO
from maml_zoo.meta_trainer import Trainer
from maml_zoo.samplers.maml_sampler import MAMLSampler
from maml_zoo.samplers.maml_sample_processor import MAMLSampleProcessor
from maml_zoo.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from maml_zoo.logger import logger

def run_experiment(**kwargs):
    # Todo: better path naming
    maml_zoo_path = '/'.join(os.path.realpath(os.path.dirname(__file__)).split('/')[:-1]) + '/data'
    logger.configure(dir=maml_zoo_path, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last_gap', snapshot_gap=50)
    # Todo: Log variant

    baseline = kwargs['baseline']()

    env = kwargs['env']() # Wrappers? normalization?

    policy = MetaGaussianMLPPolicy(
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
        batch_size=kwargs['batch_size'],
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

    algo = MAMLPPO(
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

    local_mount = mount.MountLocal(local_dir='/maml_zoo', pythonpath=True)

    sweep_params = {
        'baseline': [LinearFeatureBaseline],

        'env': [HalfCheetahRandDirecEnv],

        'batch_size': [20],
        'max_path_length': [100],
        'parallel': [True],

        'discount': [0.99],
        'gae_lambda': [1],
        'normalize_adv': [False],
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

        'n_itr': [100],
        'meta_batch_size': [40],
        'num_inner_grad_steps': [1],
        'scope': [None],
    }
    
    sweeper = launcher.DoodadSweeper([local_mount], docker_img="jonasrothfuss/rllab3")

    if args.mode == 'ec2':
        sweeper.run_sweep_ec2(run_experiment, sweep_params, bucket_name='rllab-experiments')
    elif args.mode == 'local_docker':
        mode_docker = dd.mode.LocalDocker(
            image=sweeper.image,
        )
        run_sweep_doodad(run_method, params, run_mode=mode_docker, 
                mounts=sweeper.mounts)
    elif args.mode == 'local':
        sweeper.run_sweep_serial(run_experiment, sweep_params)
    else:
        raise NotImplementedError