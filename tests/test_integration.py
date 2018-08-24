from maml_zoo.baselines.linear_feature_baseline import LinearFeatureBaseline
from maml_zoo.envs.point_env_2d import MetaPointEnv
from maml_zoo.meta_algos.ppo_maml import MAMLPPO
from maml_zoo.meta_trainer import Trainer
from maml_zoo.samplers.maml_sampler import MAMLSampler
from maml_zoo.samplers.maml_sample_processor import MAMLSampleProcessor
from maml_zoo.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from maml_zoo.logger import logger

import tensorflow as tf
import numpy as np
import unittest

class TestLikelihoodRation(unittest.TestCase):
    """
    Assure that likelihhood ratio at first gradient step is approx. one since pi_old = pi_new
    """

    def setUp(self):
        self.env = env = MetaPointEnv()

        self.baseline = baseline = LinearFeatureBaseline()

        self.policy = policy = MetaGaussianMLPPolicy(
            name="meta-policy",
            obs_dim=np.prod(env.observation_space.shape),
            action_dim=np.prod(env.action_space.shape),
            meta_batch_size=10,
            hidden_sizes=(16, 16),
            learn_std=True,
            hidden_nonlinearity=tf.tanh,
            output_nonlinearity=None,
        )

        self.sampler = MAMLSampler(
            env=env,
            policy=policy,
            rollouts_per_meta_task=2,
            meta_batch_size=10,
            max_path_length=50,
            parallel=False,
        )

        self.sample_processor = sample_processor = MAMLSampleProcessor(
            baseline=baseline,
            discount=0.99,
            gae_lambda=1.0,
            normalize_adv=True,
            positive_adv=False,
        )

        self.algo = MAMLPPO(
            policy=policy,
            inner_lr=0.1,
            meta_batch_size=10,
            num_inner_grad_steps=2,
            learning_rate=1e-3,
            max_epochs=300,
            num_minibatches=1,
            clip_eps=0.5,
            clip_outer=True,
            target_outer_step=0,
            target_inner_step=2e-2,
            init_outer_kl_penalty=0,
            init_inner_kl_penalty=1e-3,
            adaptive_outer_kl_penalty=False,
            adaptive_inner_kl_penalty=True,
            anneal_factor=1.0,
            entropy_bonus=0.0,
        )

    def test_likelihood_ratio(self):
        with tf.Session() as sess:

            # initialize uninitialized vars  (only initialize vars that were not loaded)
            uninit_vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
            sess.run(tf.variables_initializer(uninit_vars))

            self.sampler.update_tasks()
            self.policy.switch_to_pre_update()  # Switch to pre-update policy

            all_samples_data, all_paths = [], []
            for step in range(1):

                """ -------------------- Sampling --------------------------"""
                paths = self.sampler.obtain_samples(log_prefix=str(step))
                all_paths.append(paths)

                """ ----------------- Processing Samples ---------------------"""
                samples_data = self.sample_processor.process_samples(paths, log=False)
                all_samples_data.append(samples_data)

                """ ------------------- Inner Policy Update --------------------"""
                obs_phs, action_phs, adv_phs, dist_info_phs, all_phs = self.algo.make_input_placeholders('')

                for i in range(self.algo.meta_batch_size):
                    obs = samples_data[i]['observations']
                    actions = samples_data[i]['actions']
                    agent_infos = samples_data[i]['agent_infos']
                    param_vals = self.policy.get_param_values()

                    likelihood_ratio_sym = self.policy.likelihood_ratio_sym(obs_phs[i], action_phs[i],
                                                                          dist_info_phs[i],
                                                                          self.policy.policies_params_phs[i])

                    feed_dict_params = dict(zip(self.policy.policies_params_phs[i].values(), param_vals.values()))

                    feed_dict_dist_infos = dict(zip(dist_info_phs[i].values(), agent_infos.values()))

                    feed_dict = {obs_phs[i]: obs,
                                 action_phs[i]: actions
                                 }

                    feed_dict.update(feed_dict_params)
                    feed_dict.update(feed_dict_dist_infos)

                    lr = sess.run(likelihood_ratio_sym, feed_dict=feed_dict)

                    self.assertTrue(np.allclose(lr, 1))
