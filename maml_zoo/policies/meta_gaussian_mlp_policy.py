from maml_zoo.policies.base import MetaPolicy
from maml_zoo.policies.gaussian_mlp_policy import GaussianMLPPolicy
import numpy as np
import tensorflow as tf
from maml_zoo.policies.networks.mlp import forward_mlp


class MetaGaussianMLPPolicy(GaussianMLPPolicy, MetaPolicy):
    def __init__(self, *args, **kwargs):
        super(MetaGaussianMLPPolicy, self).__init__(*args, **kwargs)
        self.num_tasks = None

        self.pre_update_action_var = None
        self.pre_update_mean_var = None
        self.pre_update_log_std_var = None

        self.post_update_action_var = None
        self.post_update_mean_var = None
        self.post_update_log_std_var = None

        self.policies_params_ph = None
        self.policy_params_keys = None
        self.policies_params_vals = None

        self._pre_update_mode = True

    def build_graph(self, env_spec, num_tasks=1):
        self.num_tasks = num_tasks

        # Create pre-update policy graph
        super(MetaGaussianMLPPolicy, self).build_graph(env_spec)
        self.pre_update_action_var = tf.split(self.action_var, self.num_tasks)
        self.pre_update_mean_var = tf.split(self.mean_var, self.num_tasks)
        self.pre_update_log_std_var = [self.log_std_var for _ in range(self.num_tasks)]

        # Create post-update policy graph
        with tf.variable_scope(self.name):

            current_scope = tf.get_default_graph().get_name_scope()
            scopes = [current_scope + '/mean_network', current_scope + '/log_std_network']

            mean_network_ph, log_std_network_ph = self._create_placeholders_for_vars(scopes, num_tasks=num_tasks)

            assert len(log_std_network_ph[0]) == 1

            self.policies_params_ph = [odict.update(log_std_network_ph[idx])
                                       for idx, odict in enumerate(mean_network_ph)]

            self.policy_params_keys = list(self.policies_params_ph[0].keys())

            self.post_update_action_var = []
            self.post_update_mean_var = []
            self.post_update_log_std_var = []

            obs_var = tf.split(self.obs_var, self.num_tasks, axis=0)
            for idx in range(self.num_tasks):
                obs_var, mean_var = forward_mlp(output_dim=self.obs_dim,
                                                hidden_sizes=self.hidden_sizes,
                                                hidden_nonlinearity=self.hidden_nonlinearity,
                                                output_nonlinearity=self.output_nonlinearity,
                                                input_var=obs_var,
                                                mlp_params=list(mean_network_ph[idx].values()),
                                                )
                log_std_var = log_std_network_ph[idx][0]
                action_var = mean_var + tf.random_normal(shape=mean_var.shape) * tf.exp(log_std_var)

                self.post_update_action_var.append(action_var)
                self.post_update_mean_var.append(mean_var)
                self.post_update_log_std_var.append(log_std_var)

    def get_actions(self, observations):
        """
        Args:
            observations (list): List of size meta-batch size with numpy arrays of shape batch_size x obs_dim

        """
        if self._pre_update_mode:
            actions, agent_infos = self._get_pre_update_actions(observations)
        else:
            actions, agent_infos = self._get_post_update_actions(observations)

        return actions, agent_infos

    def _get_pre_update_actions(self, observations):
        """
        Args:
            observations (list): List of size meta-batch size with numpy arrays of shape batch_size x obs_dim

        """
        batch_size = observations[0].shape[0]
        assert all([obs.shape[0] == batch_size for obs in observations])
        assert len(observations) == self.num_tasks

        obs_stack = np.concatenate(observations, axis=0)
        feed_dict = {self.obs_var, obs_stack}

        sess = tf.get_default_session()
        actions, means, log_stds = sess.run([self.pre_update_action_var,
                                             self.pre_update_mean_var,
                                             self.pre_update_log_std_var],
                                            feed_dict=feed_dict)

        agent_infos = [dict(mean=means[idx], log_std=log_stds[idx]) for idx in range(self.num_tasks)]
        return actions, agent_infos

    def _get_post_update_actions(self, observations):
        """
        Args:
            observations (list): List of size meta-batch size with numpy arrays of shape batch_size x obs_dim

        """
        assert self.policies_params_vals is not None
        obs_stack = np.concatenate(observations, axis=0)
        feed_dict = {self.obs_var, obs_stack}
        feed_dict_params = dict([(self.policies_params_ph[idx][key], self.policies_params_vals[idx][key])
                                for key in self.policy_params_keys for idx in range(self.num_tasks)])
        feed_dict.update(feed_dict_params)

        sess = tf.get_default_session()
        actions, means, log_stds = sess.run([self.post_update_action_var,
                                             self.post_update_mean_var,
                                             self.post_update_log_std_var],
                                            feed_dict=feed_dict)

        agent_infos = [dict(mean=means[idx], log_std=log_stds[idx]) for idx in range(self.num_tasks)]
        return actions, agent_infos

