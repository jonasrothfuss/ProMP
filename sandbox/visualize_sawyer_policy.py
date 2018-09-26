import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import joblib
import tensorflow as tf
from maml_zoo.logger import logger
from maml_zoo.envs.normalized_env import normalize
from maml_zoo.envs.sawyer_pick_and_place import SawyerPickAndPlaceEnv
from maml_zoo.envs.sawyer_push import SawyerPushEnv
from maml_zoo.envs.sawyer_push_simple import SawyerPushSimpleEnv
from maml_zoo.samplers.vectorized_env_executor import MAMLIterativeEnvExecutor
from maml_zoo.baselines.linear_baseline import LinearFeatureBaseline, LinearTimeBaseline
from maml_zoo.samplers.maml_sampler import MAMLSampler
from maml_zoo.samplers.maml_sample_processor import MAMLSampleProcessor
from maml_zoo.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from maml_zoo.meta_algos.ppo_maml import PPOMAML

RUN_META_BATCH_SIZE = 20
ROLLOUTS_PER_META_TASK = 80
META_BATCH_SIZE = 10
PATH_LENGTH = 150
NUM_INNER_GRAD_STEPS = 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default=None,
                        help='policy to load')
    args = parser.parse_args(sys.argv[1:])

    sess = tf.InteractiveSession()

    policy = joblib.load(args.policy)['policy']
    policy.switch_to_pre_update()

    def get_actions(self, observations):
        """
        Hack to allow running high memory sawyer env locally
        Args:
            observations (list): List of numpy arrays of shape (meta_batch_size, batch_size, obs_dim)

        Returns:
            (tuple) : A tuple containing a list of numpy arrays of action, and a list of list of dicts of agent infos
        """
        observations *= int(RUN_META_BATCH_SIZE / META_BATCH_SIZE)
        assert len(observations) == self.meta_batch_size

        if self._pre_update_mode:
            actions, agent_infos = self._get_pre_update_actions(observations)
        else:
            actions, agent_infos = self._get_post_update_actions(observations)

        assert len(actions) == self.meta_batch_size
        actions = actions[:META_BATCH_SIZE]
        agent_infos = agent_infos[:META_BATCH_SIZE]
        return actions, agent_infos

    def get_action(self, observation, task=0):
        """
        Runs a single observation through the specified policy and samples an action

        Args:
            observation (ndarray) : single observation - shape: (obs_dim,)

        Returns:
            (ndarray) : single action - shape: (action_dim,)
        """
        observation = np.repeat(np.expand_dims(np.expand_dims(observation, axis=0), axis=0), META_BATCH_SIZE, axis=0)
        action, agent_infos = self.get_actions(observation)
        action, agent_infos = action[task][0], dict(mean=agent_infos[task][0]['mean'], log_std=agent_infos[task][0]['log_std'])
        return action, agent_infos

    MetaGaussianMLPPolicy.get_actions = get_actions
    MetaGaussianMLPPolicy.get_action = get_action

    baseline = LinearFeatureBaseline()

    env = normalize(SawyerPushSimpleEnv())

    sampler = MAMLSampler(
        env=env,
        policy=policy,
        rollouts_per_meta_task=ROLLOUTS_PER_META_TASK,
        meta_batch_size=META_BATCH_SIZE,
        max_path_length=PATH_LENGTH,
        parallel=True,
        envs_per_task=1,
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
        inner_lr=0.05,
        meta_batch_size=RUN_META_BATCH_SIZE,
        num_inner_grad_steps=1,
    )

    uninit_vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
    sess.run(tf.variables_initializer(uninit_vars))
    
    tasks = env.sample_tasks(META_BATCH_SIZE)
    sampler.vec_env.set_tasks(tasks)
    
    # Preupdate:
    for i in range(NUM_INNER_GRAD_STEPS):
        paths = sampler.obtain_samples(log=False)
        samples_data = sample_processor.process_samples(paths, log=True, log_prefix='%i_' % i)
        env.log_diagnostics(sum(list(paths.values()), []), prefix='%i_' % i)
        samples_data *= int(RUN_META_BATCH_SIZE / META_BATCH_SIZE)
        algo._adapt(samples_data)

    paths = sampler.obtain_samples(log=False)
    samples_data = sample_processor.process_samples(paths, log=True, log_prefix='%i_' % NUM_INNER_GRAD_STEPS)
    env.log_diagnostics(sum(list(paths.values()), []), prefix='%i_' % NUM_INNER_GRAD_STEPS)
    logger.dumpkvs()

    while True:
        task_i = np.random.choice(range(META_BATCH_SIZE))
        env.set_task(tasks[task_i])
        obs = env.reset()
        samples = samples_data[task_i]
        traj_index = np.random.choice(range(ROLLOUTS_PER_META_TASK)) * PATH_LENGTH
        for i in range(PATH_LENGTH):
            env.render()
            prev_obs = samples['observations'][traj_index + i]
            action = samples['actions'][traj_index + i]
            obs, reward, _, env_info = env.step(action)

    # Postupdate:
    # while True:
    #     task_i = np.random.choice(range(META_BATCH_SIZE))
    #     env.set_task(tasks[task_i])
    #     obs = env.reset()
    #     for _ in range(PATH_LENGTH):
    #         env.render()
    #         action, _ = policy.get_action(obs, task_i)
    #         obs, reward, _, action_info = env.step(action)
    #         # for key, value in action_info.items():
    #         #     print(key, value)
    #         # print('\n')