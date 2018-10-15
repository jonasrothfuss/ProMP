import os
import json
from experiment_utils.run_sweep import run_sweep
from maml_zoo.utils.utils import set_seed, ClassEncoder
from experiments.env_1d.point_1d_hand_grads import run_1d_experiment
from doodad.easy_sweep.hyper_sweep import Sweeper, kwargs_wrapper
import multiprocessing
import random
import time
INSTANCE_TYPE = 'c4.2xlarge'
EXP_NAME = 'one-dimensional-idn-larger-samples-longer-exp'



def run_experiment(**kwargs):
    exp_dir = os.getcwd() + '/data/' + EXP_NAME
    exp_dir += "/" + kwargs['opt_type'] + "_" + kwargs['exp_name']
    os.makedirs(exp_dir)
    json.dump(kwargs, open(exp_dir + '/params.json', 'w'), indent=2, sort_keys=True, cls=ClassEncoder)
    set_seed(kwargs['seed'])

    # Instantiate classes

    t0 = time.time()
    run_1d_experiment(
        opt_type=kwargs['opt_type'],
        num_meta_tasks=kwargs['num_meta_tasks'],
        horizon=kwargs['horizon'],
        num_samples=kwargs['num_samples'],
        num_itr=kwargs['num_itr'],
        lr=kwargs['lr'],
        inner_lr=kwargs['inner_lr'],
        init_state_std=kwargs['init_state_std'],
        exp_dir=exp_dir,
    )

    print("time taken:  ", time.time() - t0)


if __name__ == '__main__':
    sweep_params = {
        'seed': [0],
        'opt_type': ['exploration', 'dice'],
        'num_meta_tasks': [40],
        'horizon': [5],
        'num_samples': [100],
        'num_itr': [200],
        'lr': [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1],
        'inner_lr': [0.1],
        'init_state_std': [0.1],
    }

    sweeper = Sweeper(sweep_params, 1, include_name=True)

    num_cpu = 12
    pool = multiprocessing.Pool(num_cpu)
    exp_args = []
    for config in sweeper:
        exp_args.append((config, run_experiment))
    random.shuffle(exp_args)
    pool.map(kwargs_wrapper, exp_args)
    run_sweep(run_experiment, sweep_params, EXP_NAME, INSTANCE_TYPE)





