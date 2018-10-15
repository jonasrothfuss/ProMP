from meta_policy_search.baselines.linear_baseline import LinearFeatureBaseline
from meta_policy_search.meta_algos.pro_mp import ProMP
from meta_policy_search.samplers.meta_sampler import MetaSampler
from meta_policy_search.samplers.meta_sample_processor import MetaSampleProcessor
from meta_policy_search.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy

import tensorflow as tf
import numpy as np
import unittest

from gym.spaces import Box


class MetaPointEnv():

    def step(self, action):
        """
        Run one timestep of the environment's dynamics. When end of episode
        is reached, reset() should be called to reset the environment's internal state.

        Args:
            action : an action provided by the environment
        Returns:
            (observation, reward, done, info)
            observation : agent's observation of the current environment
            reward [Float] : amount of reward due to the previous action
            done : a boolean, indicating whether the episode has ended
            info : a dictionary containing other diagnostic information from the previous action
        """
        prev_state = self._state
        self._state = prev_state + np.clip(action, -0.1, 0.1)
        reward = self.reward(prev_state, action, self._state)
        done = self.done(self._state)
        next_observation = np.copy(self._state)
        return next_observation, reward, done, {}

    def reset(self):
        """
        Resets the state of the environment, returning an initial observation.
        Outputs
        -------
        observation : the initial observation of the space. (Initial reward is assumed to be 0.)
        """
        self._state = np.random.uniform(-2, 2, size=(2,))
        observation = np.copy(self._state)
        return observation

    @property
    def observation_space(self):
        return Box(low=-np.inf, high=np.inf, shape=(2,))

    @property
    def action_space(self):
        return Box(low=-0.1, high=0.1, shape=(2,))

    def done(self, obs):
        if obs.ndim == 1:
            return abs(obs[0]) < 0.01 and abs(obs[1]) < 0.01
        elif obs.ndim == 2:
            return np.logical_and(np.abs(obs[:, 0]) < 0.01, np.abs(obs[:, 1]) < 0.01)

    def reward(self, obs, act, obs_next):
        if obs_next.ndim == 1:
            return - np.sqrt(obs_next[0]**2 + obs_next[1]**2)
        elif obs_next.ndim == 2:
            return - np.sqrt(obs_next[:, 0] ** 2 + obs_next[:, 1] ** 2)

    def log_diagnostics(self, paths):
        pass

    def sample_tasks(self, n_tasks):
        return [{}] * n_tasks

    def set_task(self, task):
        pass

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

        self.sampler = MetaSampler(
            env=env,
            policy=policy,
            rollouts_per_meta_task=2,
            meta_batch_size=10,
            max_path_length=50,
            parallel=False,
        )

        self.sample_processor = MetaSampleProcessor(
            baseline=baseline,
            discount=0.99,
            gae_lambda=1.0,
            normalize_adv=True,
            positive_adv=False,
        )

        self.algo = ProMP(
            policy=policy,
            inner_lr=0.1,
            meta_batch_size=10,
            num_inner_grad_steps=2,
            learning_rate=1e-3,
            num_ppo_steps=5,
            num_minibatches=1,
            clip_eps=0.5,
            target_inner_step=2e-2,
            init_inner_kl_penalty=1e-3,
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
                obs_phs, action_phs, adv_phs, dist_info_phs, all_phs = self.algo._make_input_placeholders('')

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
