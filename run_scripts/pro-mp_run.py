from meta_policy_search.baselines.linear_baseline import LinearFeatureBaseline
from meta_policy_search.envs.mujoco_envs.half_cheetah_rand_direc import HalfCheetahRandDirecEnv
from meta_policy_search.envs.normalized_env import normalize
from meta_policy_search.meta_algos.pro_mp import ProMP
from meta_policy_search.meta_trainer import Trainer
from meta_policy_search.samplers.meta_sampler import MetaSampler
from meta_policy_search.samplers.meta_sample_processor import MetaSampleProcessor
from meta_policy_search.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from meta_policy_search.utils import logger
from meta_policy_search.utils.utils import set_seed, ClassEncoder

import numpy as np
import tensorflow as tf
import os
import json
import argparse
import time

meta_policy_search_path = '/'.join(os.path.realpath(os.path.dirname(__file__)).split('/')[:-1])

def main(config):
    set_seed(config['seed'])


    baseline =  globals()[config['baseline']]() #instantiate baseline

    env = globals()[config['env']]() # instantiate env
    env = normalize(env) # apply normalize wrapper to env

    policy = MetaGaussianMLPPolicy(
            name="meta-policy",
            obs_dim=np.prod(env.observation_space.shape),
            action_dim=np.prod(env.action_space.shape),
            meta_batch_size=config['meta_batch_size'],
            hidden_sizes=config['hidden_sizes'],
        )

    sampler = MetaSampler(
        env=env,
        policy=policy,
        rollouts_per_meta_task=config['rollouts_per_meta_task'],  # This batch_size is confusing
        meta_batch_size=config['meta_batch_size'],
        max_path_length=config['max_path_length'],
        parallel=config['parallel'],
    )

    sample_processor = MetaSampleProcessor(
        baseline=baseline,
        discount=config['discount'],
        gae_lambda=config['gae_lambda'],
        normalize_adv=config['normalize_adv'],
    )

    algo = ProMP(
        policy=policy,
        inner_lr=config['inner_lr'],
        meta_batch_size=config['meta_batch_size'],
        num_inner_grad_steps=config['num_inner_grad_steps'],
        learning_rate=config['learning_rate'],
        num_ppo_steps=config['num_promp_steps'],
        clip_eps=config['clip_eps'],
        target_inner_step=config['target_inner_step'],
        init_inner_kl_penalty=config['init_inner_kl_penalty'],
        adaptive_inner_kl_penalty=config['adaptive_inner_kl_penalty'],
    )

    trainer = Trainer(
        algo=algo,
        policy=policy,
        env=env,
        sampler=sampler,
        sample_processor=sample_processor,
        n_itr=config['n_itr'],
        num_inner_grad_steps=config['num_inner_grad_steps'],
    )

    trainer.train()

if __name__=="__main__":
    idx = int(time.time())

    parser = argparse.ArgumentParser(description='ProMP: Proximal Meta-Policy Search')
    parser.add_argument('--config_file', type=str, default='', help='json file with run specifications')
    parser.add_argument('--dump_path', type=str, default=meta_policy_search_path + '/data/pro-mp/run_%d' % idx)

    args = parser.parse_args()


    if args.config_file: # load configuration from json file
        with open(args.config_file, 'r') as f:
            config = json.load(f)

    else: # use default config

        config = {
            'seed': 1,

            'baseline': 'LinearFeatureBaseline',

            'env': 'HalfCheetahRandDirecEnv',

            # sampler config
            'rollouts_per_meta_task': 20,
            'max_path_length': 100,
            'parallel': True,

            # sample processor config
            'discount': 0.99,
            'gae_lambda': 1,
            'normalize_adv': True,

            # policy config
            'hidden_sizes': (64, 64),
            'learn_std': True, # whether to learn the standard deviation of the gaussian policy

            # ProMP config
            'inner_lr': 0.1, # adaptation step size
            'learning_rate': 1e-3, # meta-policy gradient step size
            'num_promp_steps': 5, # number of ProMp steps without re-sampling
            'clip_eps': 0.3, # clipping range
            'target_inner_step': 0.01,
            'init_inner_kl_penalty': 5e-4,
            'adaptive_inner_kl_penalty': False, # whether to use an adaptive or fixed KL-penalty coefficient
            'n_itr': 1001, # number of overall training iterations
            'meta_batch_size': 40, # number of sampled meta-tasks per iterations
            'num_inner_grad_steps': 1, # number of inner / adaptation gradient steps

        }

    # configure logger
    logger.configure(dir=args.dump_path, format_strs=['stdout', 'log', 'csv'],
                     snapshot_mode='last_gap')

    # dump run configuration before starting training
    json.dump(config, open(args.dump_path + '/params.json', 'w'), cls=ClassEncoder)

    # start the actual algorithm
    main(config)