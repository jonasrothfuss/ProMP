from maml_zoo.policies.networks.mlp import create_mlp, forward_mlp
from maml_zoo.policies.distributions.diagonal_gaussian import DiagonalGaussian
from maml_zoo.policies.base import Policy
from maml_zoo.utils.utils import get_original_tf_name

import tensorflow as tf
import numpy as np
from collections import OrderedDict
import gym

class GaussianMLPPolicy(Policy):
    """
    Gaussian multi-layer perceptron policy (diagonal covariance matrix)
    Provides functions for executing and updating policy parameters
    A container for storing the current pre and post update policies

    Args:
        env (gym.Env): gym environment
        name (str): name of the policy used as tf variable scope
        hidden_sizes (tuple): tuple of integers specifying the hidden layer sizes of the MLP
        hidden_nonlinearity (tf.op): nonlinearity function of the hidden layers
        output_nonlinearity (tf.op or None): nonlinearity function of the output layer
        learn_std (boolean): whether the standard_dev / variance is a trainable or fixed variable
        init_std (float): initial policy standard deviation
        min_std( float): minimal policy standard deviation

    """

    def __init__(self,
                 env,
                 name='gaussian_mlp_policy',
                 hidden_sizes=(32, 32),
                 learn_std=True,
                 hidden_nonlinearity=tf.tanh,
                 output_nonlinearity=None,
                 init_std=1,
                 min_std=1e-6,
                 ):
        assert isinstance(env.observation_space, gym.spaces.Box), 'observation space must be continous'
        assert isinstance(env.action_space, gym.spaces.Box), 'action space must be continous'
        self.obs_dim = int(np.prod(env.observation_space.shape))
        self.action_dim = int(np.prod(env.action_space.shape))

        # Assert is instance Box
        self.obs_dim = np.prod(env.observation_space.shape)
        self.action_dim = np.prod(env.action_space.shape)
        self.name = name
        self.hidden_sizes = hidden_sizes
        self.learn_std = learn_std
        self.hidden_nonlinearity = hidden_nonlinearity
        self.output_nonlinearity = output_nonlinearity
        self.min_log_std = np.log(min_std)
        self.init_log_std = np.log(init_std)

        self.init_policy = None
        self.policy_params = None
        self.obs_var = None
        self.mean_var = None
        self.log_std_var = None
        self.action_var = None
        self._dist = None

        self.build_graph()

    def build_graph(self):
        """
        Builds computational graph for policy
        """
        with tf.variable_scope(self.name):
            obs_var, mean_var = create_mlp(name='mean_network',
                                           output_dim=self.action_dim,
                                           hidden_sizes=self.hidden_sizes,
                                           hidden_nonlinearity=self.hidden_nonlinearity,
                                           output_nonlinearity=self.output_nonlinearity,
                                           input_dim=(None, self.obs_dim,)
                                           )

            log_std_var = tf.get_variable(name='log_std_network',
                                          shape=(1, self.action_dim,),
                                          dtype=tf.float32,
                                          initializer=tf.constant_initializer(self.init_log_std),
                                          trainable=self.learn_std
                                          )

            log_std_var = tf.maximum(log_std_var, self.min_log_std)

            current_scope = tf.get_default_graph().get_name_scope()
            mean_network_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                  scope=current_scope + '/mean_network')
            log_std_network_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
                                                     scope=current_scope + '/log_std_network')
            assert len(log_std_network_vars) == 1

            mean_network_vars = OrderedDict([(get_original_tf_name(var.name), var) for var in mean_network_vars])
            log_std_network_vars = OrderedDict([(get_original_tf_name(var.name), var) for var in log_std_network_vars])

            # self._create_getter_setter()

        action_var = mean_var + tf.random_normal(shape=tf.shape(mean_var)) * tf.exp(log_std_var)

        self.obs_var = obs_var
        self.action_var = action_var
        self.mean_var = mean_var
        self.log_std_var = log_std_var
        self.policy_params = mean_network_vars.update(log_std_network_vars)

        self._dist = DiagonalGaussian(self.action_dim)

    def get_action(self, observation):
        """
        Runs a single observation through the specified policy and samples an action

        Args:
            observation (ndarray) : single observation - shape: (obs_dim,)

        Returns:
            (ndarray) : single action - shape: (action_dim,)
        """
        observation = np.expand_dims(observation, axis=0)
        action, agent_infos = self.get_actions(observation)
        action, agent_infos = action[0], dict(mean=agent_infos['mean'][0], log_std=agent_infos['log_std'][0])
        return action, agent_infos

    def get_actions(self, observations):
        """
        Runs each set of observations through each task specific policy

        Args:
            observations (ndarray) : array of observations - shape: (batch_size, obs_dim)

        Returns:
            (ndarray) : array of sampled actions - shape: (batch_size, action_dim)
        """
        assert observations.ndim == 2 and observations.shape[1] == self.obs_dim

        sess = tf.get_default_session()
        actions, means, logs_stds = sess.run([self.action_var, self.mean_var, self.log_std_var],
                                             feed_dict={self.obs_var: observations})
        rnd = np.random.normal(size=means.shape)
        actions = means + rnd * np.exp(logs_stds)

        assert actions.shape == (observations.shape[0], self.action_dim)
        return actions, dict(mean=means, log_std=logs_stds)

    def log_diagnostics(self, paths):
        """
        Log extra information per iteration based on the collected paths
        """
        raise NotImplementedError

    def load_params(self, policy_params):
        """
        Args:
            policy_params (ndarray): array of policy parameters for each task
        """
        raise NotImplementedError

    @property
    def distribution(self):
        """
        Returns this policy's distribution

        Returns:
            (Distribution) : this policy's distribution
        """
        return self._dist

    def distribution_info_sym(self, obs_var, params=None):
        """
        Return the symbolic distribution information about the actions.

        Args:
            obs_var (placeholder) : symbolic variable for observations
            params (dict) : a dictionary of placeholders or vars with the parameters of the MLP

        Returns:
            (dict) : a dictionary of tf placeholders for the policy output distribution
        """
        with tf.variable_scope(self.name):
            if params is None:
                obs_var, mean_var = create_mlp(name='mean_network',
                                               output_dim=self.action_dim,
                                               hidden_sizes=self.hidden_sizes,
                                               hidden_nonlinearity=self.hidden_nonlinearity,
                                               output_nonlinearity=self.output_nonlinearity,
                                               input_var=obs_var,
                                               reuse=True,
                                               )

                log_std_var = self.log_std_var
            else:
                mean_network_params = []
                log_std_network_params = []
                for param in params.values():
                    if 'mean_network' in param.name:
                        mean_network_params.append(param)
                    elif 'log_std_network' in param.name:
                        log_std_network_params.append(params)

                assert len(log_std_network_params) == 1

                obs_var, mean_var = forward_mlp(output_dim=self.obs_dim,
                                                hidden_sizes=self.hidden_sizes,
                                                hidden_nonlinearity=self.hidden_nonlinearity,
                                                output_nonlinearity=self.output_nonlinearity,
                                                input_var=obs_var,
                                                mlp_params=mean_network_params,
                                                )

                log_std_var = log_std_network_params[0]

        return dict(mean=mean_var, log_std=log_std_var)

    def distribution_info_keys(self, obs, state_infos):
        """
        Args:
            obs (placeholder) : symbolic variable for observations
            state_infos (dict) : a dictionary of placeholders that contains information about the
            state of the policy at the time it received the observation

        Returns:
            (dict) : a dictionary of tf placeholders for the policy output distribution
        """
        raise ["mean", "log_std"]


