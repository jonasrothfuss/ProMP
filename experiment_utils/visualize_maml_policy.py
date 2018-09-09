import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import joblib
import tensorflow as tf
from maml_zoo.logger import logger
from maml_zoo.envs.normalized_env import normalize
from maml_zoo.envs.half_cheetah_rand_direc import HalfCheetahRandDirecEnv
from maml_zoo.envs.ant_rand_direc import AntRandDirecEnv
from maml_zoo.envs.ant_rand_goal import AntRandGoalEnv
from maml_zoo.envs.half_cheetah_rand_vel import HalfCheetahRandVelEnv
from maml_zoo.envs.swimmer_rand_vel import SwimmerRandVelEnv
from rand_param_envs.hopper_rand_params import HopperRandParamsEnv
from rand_param_envs.walker2d_rand_params import Walker2DRandParamsEnv
from maml_zoo.envs.sawyer_pick_and_place import SawyerPickAndPlaceEnv
from maml_zoo.envs.sawyer_push import SawyerPushEnv
from maml_zoo.samplers.vectorized_env_executor import MAMLIterativeEnvExecutor
from maml_zoo.baselines.linear_baseline import LinearFeatureBaseline, LinearTimeBaseline
from maml_zoo.samplers.maml_sampler import MAMLSampler
from maml_zoo.samplers.maml_sample_processor import MAMLSampleProcessor
from maml_zoo.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from maml_zoo.meta_algos.ppo_maml import PPOMAML

META_BATCH_SIZE = 40
PATH_LENGTH = 200

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default=None,
                        help='policy to load')
    args = parser.parse_args(sys.argv[1:])

    sess = tf.InteractiveSession()

    policy = joblib.load(args.policy)['policy']
    policy.switch_to_pre_update()

    baseline = LinearFeatureBaseline()

    env = normalize(AntRandGoalEnv())

    sampler = MAMLSampler(
        env=env,
        policy=policy,
        rollouts_per_meta_task=20,
        meta_batch_size=META_BATCH_SIZE,
        max_path_length=PATH_LENGTH,
        parallel=True,
        # envs_per_task=1,
    )

    sample_processor = MAMLSampleProcessor(
        baseline=baseline,
        discount=0.99,
        gae_lambda=1,
        normalize_adv=True,
        positive_adv=False,
    )

    # Doesn't matter which algo
    algo = PPOMAML(
        policy=policy,
        inner_lr=0.1,
        meta_batch_size=40,
        num_inner_grad_steps=1,
    )

    uninit_vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
    sess.run(tf.variables_initializer(uninit_vars))

    fig = plt.figure()
    
    tasks = env.sample_tasks(META_BATCH_SIZE)
    sampler.vec_env.set_tasks(tasks)
    
    # Preupdate:
    paths = sampler.obtain_samples(log=False)
    
    # Update:
    samples_data = sample_processor.process_samples(paths, log=True, log_prefix='0_')
    env.log_diagnostics(sum(list(paths.values()), []), prefix='0_')
    # samples_data *= 8
    algo._adapt(samples_data)

    paths = sampler.obtain_samples(log=False)
    samples_data = sample_processor.process_samples(paths, log=True, log_prefix='1_')
    env.log_diagnostics(sum(list(paths.values()), []), prefix='1_')
    logger.dumpkvs()
    # Postupdate:
    while True:
        task_i = np.random.choice(range(META_BATCH_SIZE))
        env.set_task(tasks[task_i])
        obs = env.reset()
        for _ in range(PATH_LENGTH):
            env.render()
            action, _ = policy.get_action(obs, task_i)
            obs, reward, _, _ = env.step(action)
        print(np.sqrt(-reward), env.get_goal())