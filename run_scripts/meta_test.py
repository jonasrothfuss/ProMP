import os
import json
import tensorflow as tf
import numpy as np
from experiment_utils.run_sweep import run_sweep
from maml_zoo.utils.utils import set_seed, ClassEncoder
from maml_zoo.baselines.linear_baseline import LinearFeatureBaseline, LinearTimeBaseline
from maml_zoo.envs.normalized_env import normalize
from maml_zoo.meta_algos.ppo_maml import PPOMAML
from maml_zoo.algos.vpg import VPG
from maml_zoo.meta_tester import Tester
from maml_zoo.samplers.maml_sampler import MAMLSampler
from maml_zoo.samplers import SampleProcessor
from maml_zoo.policies.gaussian_mlp_policy import GaussianMLPPolicy
from maml_zoo.logger import logger
import joblib
import os.path as osp

INSTANCE_TYPE = 'c4.4xlarge'
EXP_NAME = 'ppo-humanoid-large-batch'


def run_experiment(**kwargs):
    exp_dir = os.getcwd() + '/data/' + EXP_NAME
    logger.configure(dir=exp_dir, format_strs=['stdout', 'log', 'csv'], snapshot_mode='last_gap', snapshot_gap=50)
    json.dump(kwargs, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)

    # Instantiate classes
    set_seed(kwargs['seed'])
    sess = tf.Session()


    with sess.as_default() as sess:
        config = json.load(open(osp.join(kwargs['path'], 'params.json'), 'r'))
        data = joblib.load(osp.join(kwargs['path'], 'params.pkl'))
        policy = data['policy']
        env = data['env']
        baseline = data['baseline']

        if kwargs['rollouts_per_meta_task'] is None:
            rollouts_per_meta_task = int(np.ceil(config['rollouts_per_meta_task']/config['meta_batch_size']))
        else:
            rollouts_per_meta_task = kwargs['rollouts_per_meta_task']


        sampler = MAMLSampler(
            env=env,
            policy=policy,
            rollouts_per_meta_task=rollouts_per_meta_task,
            meta_batch_size=config['meta_batch_size'],
            max_path_length=kwargs['max_path_length'],
            parallel=kwargs['parallel'],
        )

        sample_processor = SampleProcessor(
            baseline=baseline,
            discount=config['discount'],
            normalize_adv=config['normalize_adv'],
            positive_adv=config['positive_adv'],
        )

        algo = VPG(
            policy=policy,
            learning_rate=config['inner_lr'],
        )

        tester = Tester(
            algo=algo,
            policy=policy,
            env=env,
            sampler=sampler,
            sample_processor=sample_processor,
            n_itr=kwargs['n_itr'],
            sess=sess,
            task=None,
        )

        tester.train()

if __name__ == '__main__':

    sweep_params = {
        'seed': [1, 2, 3],
        'path': ['/home/ignasi/GitRepos/ProMP/data/s3/dice-eval-2/dice-eval-2-1542681462667'],

        'rollouts_per_meta_task': [None],
        'max_path_length': [200],
        'parallel': [False],

        'n_itr': [1001],
        'meta_batch_size': [40],
        'scope': [None],

        'exp_tag': ['v0']
    }

    run_sweep(run_experiment, sweep_params, EXP_NAME, INSTANCE_TYPE)