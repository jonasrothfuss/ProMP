import tensorflow as tf
import numpy as np
from maml_zoo.optimizers.base import Optimizer
from collections import OrderedDict
from maml_zoo.utils.utils import extract


class Algo(object):
    """

    """

    def __init__(
            self,
            optimizer,
            inner_lr,
            entropy_bonus=0,
            ):

        assert isinstance(optimizer, Optimizer)
        self.optimizer = optimizer
        self.inner_lr = inner_lr
        self.entropy_bonus = entropy_bonus
        self.meta_batch_size = None
        self.policy = None

    def build_graph(self, policy, meta_batch_size):
        """
        Creates meta-learning computation graph

        Notes:
            Pseudocode:

            for task in meta_batch_size:
                make_vars
                init_dist_info_sym
            for step in num_grad_steps:
                for task in meta_batch_size:
                    make_vars
                    update_dist_info_sym
            set objectives for optimizer

        Args:
            policy: policy object
            meta_batch_size (int): number of meta-tasks

        """
        raise NotImplementedError

    def make_vars(self, prefix=''):
        """
        Args:
            prefix (str) : a string to prepend to the name of each variable
        Returns:
            (tuple) : a tuple containing lists of placeholders for each input type and meta task
        """
        raise NotImplementedError

    def init_dist_sym(self, obs_var, params_var, is_training=False):
        """
        Creates the symbolic representation of the current tf policy

        Args:
            obs_var (list) : list of obs placeholders split by env
            params_ph (dict) : dict of placeholders for initial policy params
            is_training (bool) : used for batch norm # (Do we support this?)

        Returns:
            (tf_op) : symbolic representation the policy's output for each obs
        """
        raise NotImplementedError

    def compute_updated_dist_sym(self, surr_obj, obs_var, params_var, is_training=False):
        """
        Creates the symbolic representation of the tf policy after one gradient step towards the surr_obj

        Args:
            surr_obj (tf_op) : tensorflow op for task specific (inner) objective
            obs_var (list) : list of obs placeholders split by env
            params_ph (dict) : dict of placeholders for current policy params
            is_training (bool) : used for batch norm # (Do we support this?)

        Returns:
            (tf_op) : symbolic representation the policy's output for each obs
        """
        raise NotImplementedError

    def compute_updated_dists(self, samples):
        """
        Performs MAML inner step for each task and stores resulting gradients # (in the policy?)

        Args:
            samples (list) : list of lists of samples (each is a dict) split by meta task

        Returns:
            None
        """
        raise NotImplementedError

    def optimize_policy(self, all_samples_data, log=True):
        """
        Performs MAML outer step for each task

        Args:
            all_samples_data (list) : list of lists of lists of samples (each is a dict) split by gradient update and meta task
            log (bool) : whether to log statistics

        Returns:
            None
        """
        raise NotImplementedError


class MAMLAlgo(Algo):
    """
    Provides some implementations shared between all MAML algorithms
    """
    def build_graph(self, policy, meta_batch_size, num_inner_grad_steps):
        raise NotImplementedError

    def make_placeholders(self, prefix='', scope=''):
        """
        Args:
            prefix (str) : a string to prepend to the name of each variable

        Returns:
            (tuple) : a tuple containing lists of placeholders for each input type and meta task
        """
        obs_phs, action_phs, adv_phs, dist_info_phs = [], [], [], []
        dist_info_specs = self.policy.distribution.dist_info_specs

        with tf.variable_scope(scope):
            for i in range(self.meta_batch_size):
                obs_phs.append(tf.placeholder(
                    dtype=tf.float32,
                    shape=[None, np.prod(self.env.observation_space.shape)],
                    name='obs' + prefix + '_' + str(i)
                ))
                action_phs.append(tf.placeholder(
                    dtype=tf.float32,
                    shape=[None, np.prod(self.env.action_space.shape)],
                    name='action' + prefix + '_' + str(i),
                ))
                adv_phs.append(tf.placeholder(
                    dtype=tf.float32,
                    shape=[None],
                    name='advantage' + prefix + '_' + str(i),
                ))
                dist_info_phs.append([tf.placeholder(
                    dtype=tf.float32,
                    shape=[None] + list(shape), name='%s%s_%i' % (k, prefix, i))
                    for k, shape in dist_info_specs
                ])
        return obs_phs, action_phs, adv_phs, dist_info_phs

    def init_dist_sym(self, obs_var, params=None, is_training=False):
        """
        Creates the symbolic representation of the current tf policy

        Args:
            obs_var (list) : list of obs placeholders split by env
            params_phs (dict or None) : dict of placeholders for initial policy params
            is_training (bool) : used for batch norm # (Do we support this?)

        Returns:
            (tf_op) : symbolic representation the policy's output for each obs
        """
        return self.policy.output_sym(obs_var, params=params)

    def compute_updated_dist_sym(self, surr_obj, obs_var, params_var, is_training=False):
        """
        Creates the symbolic representation of the tf policy after one gradient step towards the surr_obj

        Args:
            surr_obj (tf_op) : tensorflow op for task specific (inner) objective
            obs_var (list) : list of obs placeholders split by env
            params_ph (dict) : dict of placeholders for current policy params
            is_training (bool) : used for batch norm # (Do we support this?)

        Returns:
            (tf_op) : symbolic representation the policy's output for each obs
        """
        update_param_keys = params_var.keys()
        # TODO: Fix this if we want to learn the learning rate (it isn't supported right now).

        grads = tf.gradients(surr_obj, [params_var[key] for key in update_param_keys])

        gradients = dict(zip(update_param_keys, grads))
        params_dict = dict(zip(update_param_keys, [
            params_var[key] - tf.multiply(self.param_step_sizes[key + "_step_size"], gradients[key]) for key in
            update_param_keys]))

        return self.init_dist_sym(obs_var, params=params_dict)

    def compute_updated_dists(self, samples):
        """
        Performs MAML inner step for each task and performs an update with the resulting gradients

        Args:
            samples (list) : list of lists of samples (each is a dict) split by meta task

        Returns:
            None
        """
        sess = tf.get_default_session()
        num_tasks = len(samples)
        assert num_tasks == self.meta_batch_size

        obs_list, action_list, adv_list, dist_info_list = [], [], [], []
        for i in range(num_tasks):
            inputs = extract(
                samples[i], *self._optimization_keys
            )
            obs_list.append(inputs[0])
            action_list.append(inputs[1])
            adv_list.append(inputs[2])
            dist_info_list.append([inputs[3][k] for k in self.policy.distribution.dist_info_keys])

        input_list = obs_list + action_list + adv_list + sum(list(zip(*dist_info_list)))

        feed_dict_inputs = list(zip(self.input_list_for_grad, input_list))
        feed_dict_params = list((self.policy.policies_params_ph[i][key], self.policy.policies_params_vals[i][key])
                                for i in range(num_tasks) for key in self.policy.policy_params_keys)
        feed_dict = dict(feed_dict_inputs + feed_dict_params)

        new_param_vals = sess.run(self.fast_policy_params_var, feed_dict=feed_dict)

        self.policy.update_task_parameters(new_param_vals)

    def build_test_inner_obj(self, input_list, surr_objs_tensor):
        self.input_list_for_grad = input_list
        self.surr_objs = surr_objs_tensor
        self.fast_policy_params_var = []

        update_param_keys = self.policy.policy_params_keys
        with tf.variable_scope(self.name):
            # Create the symbolic graph for the one-step inner gradient update (It'll be called several times if
            # more gradient steps are needed
            # TODO: A tf.map would be faster
            for i in range(self.num_tasks):
                # compute gradients for a current task (symbolic)
                gradients = dict(zip(update_param_keys, tf.gradients(self.surr_objs[i],
                                                                     [self.policies_params_ph[i][key]
                                                                      for key in update_param_keys]
                                                                     )))

                # gradient update for params of current task (symbolic)
                fast_params_tensor = OrderedDict(zip(update_param_keys,
                                                     [self.policies_params_ph[i][key] - tf.multiply(
                                                         self.param_step_sizes[key + "_step_size"], gradients[key]) for
                                                      key in update_param_keys]))

                # tensors that represent the updated params for all of the tasks (symbolic)
                self.fast_policy_params_var.append(fast_params_tensor)
