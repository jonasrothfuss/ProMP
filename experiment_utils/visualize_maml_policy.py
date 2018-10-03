import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import joblib
import tensorflow as tf
import time
from maml_zoo.logger import logger
from maml_zoo.envs.normalized_env import normalize
from maml_zoo.envs.mujoco_envs.half_cheetah_rand_direc import HalfCheetahRandDirecEnv
from maml_zoo.envs.mujoco_envs.ant_rand_direc import AntRandDirecEnv
from maml_zoo.envs.mujoco_envs.ant_rand_direc_2d import AntRandDirec2DEnv
from maml_zoo.envs.mujoco_envs.ant_rand_goal import AntRandGoalEnv
from maml_zoo.envs.mujoco_envs.half_cheetah_rand_vel import HalfCheetahRandVelEnv
from maml_zoo.envs.mujoco_envs.swimmer_rand_vel import SwimmerRandVelEnv
from maml_zoo.envs.mujoco_envs.humanoid_rand_direc import HumanoidRandDirecEnv
from maml_zoo.envs.mujoco_envs.humanoid_rand_direc_2d import HumanoidRandDirec2DEnv
from maml_zoo.envs.mujoco_envs.walker2d_rand_direc import Walker2DRandDirecEnv
from maml_zoo.envs.mujoco_envs.walker2d_rand_vel import Walker2DRandVelEnv
from maml_zoo.envs.point_envs.point_env_2d_corner import MetaPointEnvCorner
from maml_zoo.envs.point_envs.point_env_2d_walls import MetaPointEnvWalls
from maml_zoo.envs.point_envs.point_env_2d_momentum import MetaPointEnvMomentum
from maml_zoo.envs.sawyer_envs.sawyer_pick_and_place import SawyerPickAndPlaceEnv
from maml_zoo.envs.sawyer_envs.sawyer_push import SawyerPushEnv
from maml_zoo.envs.sawyer_envs.sawyer_push_simple import SawyerPushSimpleEnv
from rand_param_envs.hopper_rand_params import HopperRandParamsEnv
from rand_param_envs.walker2d_rand_params import Walker2DRandParamsEnv
from maml_zoo.samplers.vectorized_env_executor import MAMLIterativeEnvExecutor
from maml_zoo.baselines.linear_baseline import LinearFeatureBaseline, LinearTimeBaseline
from maml_zoo.samplers.maml_sampler import MAMLSampler
from maml_zoo.samplers.maml_sample_processor import MAMLSampleProcessor
from maml_zoo.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from maml_zoo.meta_algos.vpg_maml import VPGMAML

BATCH_SIZE = 80
META_BATCH_SIZE = 40
PATH_LENGTH = 200
NUM_INNER_GRAD_STEPS = 2

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
        rollouts_per_meta_task=BATCH_SIZE,
        meta_batch_size=META_BATCH_SIZE,
        max_path_length=PATH_LENGTH,
        parallel=True,
        envs_per_task=20,
    )

    sample_processor = MAMLSampleProcessor(
        baseline=baseline,
        discount=0.99,
        gae_lambda=1,
        normalize_adv=True,
        positive_adv=False,
    )

    # Doesn't matter which algo
    algo = VPGMAML(
        policy=policy,
        inner_lr=0.1,
        meta_batch_size=META_BATCH_SIZE,
        inner_type='likelihood_ratio',
        num_inner_grad_steps=NUM_INNER_GRAD_STEPS,
    )

    uninit_vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
    sess.run(tf.variables_initializer(uninit_vars))
    
    # Preupdate:
    tasks = env.sample_tasks(META_BATCH_SIZE)
    sampler.vec_env.set_tasks(tasks)
    
    # Preupdate:
    for i in range(NUM_INNER_GRAD_STEPS):
        paths = sampler.obtain_samples(log=False)
        samples_data = sample_processor.process_samples(paths, log=True, log_prefix='%i_' % i)
        env.log_diagnostics(sum(list(paths.values()), []), prefix='%i_' % i)
        algo._adapt(samples_data)

    paths = sampler.obtain_samples(log=False)
    samples_data = sample_processor.process_samples(paths, log=True, log_prefix='%i_' % NUM_INNER_GRAD_STEPS)
    env.log_diagnostics(sum(list(paths.values()), []), prefix='%i_' % NUM_INNER_GRAD_STEPS)
    logger.dumpkvs()

    # Postupdate:
    while True:
        task_i = np.random.choice(range(META_BATCH_SIZE))
        env.set_task(tasks[task_i])
        print(tasks[task_i])
        obs = env.reset()
        for _ in range(PATH_LENGTH):
            env.render()
            action, _ = policy.get_action(obs, task_i)
            obs, reward, done, _ = env.step(action)
            time.sleep(0.001)
            if done:
                break