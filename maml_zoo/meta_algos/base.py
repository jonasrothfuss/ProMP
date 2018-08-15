import tensorflow as tf
import numpy as np
from maml_zoo.policies.base import Policy
from collections import OrderedDict
from maml_zoo.utils.utils import extract

class Algo(object):
    """
    Base class for algorithms

    Args:
        policy (Policy) : policy object
        entropy_bonus (float) : scaling factor for policy entropy
    """

    def __init__(self, policy, entropy_bonus=0.0, *args, **kwargs):
        assert isinstance(policy, Policy)
        assert isinstance(entropy_bonus, float)
        self.policy = policy
        self.entropy_bonus = entropy_bonus
        self._optimization_keys = None

    def build_graph(self):
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

    def _extract_input_list(self, all_samples_data, keys):
        inputs = [list() for _ in range(len(keys))]
        extracted_data = extract(
            all_samples_data, keys
        )
        for j, data in enumerate(extracted_data):
            if isinstance(data, dict):
                inputs[j].extend([data[k] for k in data.keys()]) # FIXME: ?? Before was self.policy.distribution.dist_info_keys]
            elif isinstance(data, np.ndarray):
                inputs[j].append(data)
            else:
                raise NotImplementedError

        input_list = sum(inputs, [])
        return input_list


class MAMLAlgo(Algo):
    """
    Provides some implementations shared between all MAML algorithms
    
    Args:
        inner_lr (float) : gradient step size used for inner step
        meta_batch_size (int): number of metalearning tasks
        num_inner_grad_steps (int) : number of gradient updates taken per maml iteration
    """
    def __init__(self, inner_lr, meta_batch_size, num_inner_grad_steps, *args, **kwargs):
        super(MAMLAlgo, self).__init__(*args, **kwargs)
        assert (num_inner_grad_steps).is_integer()
        assert isinstance(inner_lr, float)
        assert (meta_batch_size).is_integer()
        self.inner_lr = inner_lr
        self.meta_batch_size = meta_batch_size
        self.num_inner_grad_steps = num_inner_grad_steps
        self.input_list_ph = None
        self.surr_objs_var = None
        self.adapted_policies_params = None
        self.step_sizes = None
        self.meta_batch_size = meta_batch_size
        self.policy = policy
        self.policies_params_ph = policy.policies_params_ph
        self.policy_params = policy.policy_params

    def build_graph(self):
        raise NotImplementedError

    def make_input_placeholders(self, prefix='', scope=''):
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

    def adapt_sym(self, surr_obj, obs_var, params_var):
        """
        Creates the symbolic representation of the tf policy after one gradient step towards the surr_obj

        Args:
            surr_obj (tf_op) : tensorflow op for task specific (inner) objective
            obs_var (list) : list of obs placeholders split by env
            params_ph (dict) : dict of placeholders for current policy params

        Returns:
            (tf_op) : symbolic representation the policy's output for each obs
        """
        # TODO: Fix this if we want to learn the learning rate (it isn't supported right now).
        update_param_keys = params_var.keys()

        grads = tf.gradients(surr_obj, [params_var[key] for key in update_param_keys])
        gradients = dict(zip(update_param_keys, grads))

        adapted_policy_params = [params_var[key] - tf.multiply(self.step_sizes[key], gradients[key])
                          for key in update_param_keys]

        adapted_policy_params_dict = OrderedDict(zip(update_param_keys, adapted_policy_params))

        return self.policy.distribution_info_sym(obs_var, params=adapted_policy_params_dict)

    def adapt_sym_test(self, input_list, surr_objs):
        """
        Create the symbolic graph for the one-step inner gradient update (It'll be called several times if
        more gradient steps are needed)
        
        Args:
            input_list (list) : a list of placeholders for sampled data
            surr_objs (list) : a list of operations to compute the loss for each task

        Returns:
            None
        """
        self.input_list_ph = input_list
        self.surr_objs_var = surr_objs
        self.adapted_policies_params = []

        update_param_keys = self.policy.policy_params_keys
        num_tasks = len(surr_objs)
        # Create the symbolic graph for the one-step inner gradient update (It'll be called several times if
        # more gradient steps are needed
        # TODO: A tf.map would be faster

        for i in range(num_tasks):
            # compute gradients for a current task (symbolic)
            grads = tf.gradients(surr_objs[i],[self.policies_params_ph[i][key] for key in update_param_keys])
            gradients = dict(zip(update_param_keys, grads))

            # gradient update for params of current task (symbolic)
            adapted_policy_params = [self.policies_params_ph[i][key] - tf.multiply(self.step_sizes[key], gradients[key])
                                     for key in update_param_keys]

            adapted_policy_params_dict = OrderedDict(zip(update_param_keys, adapted_policy_params))

            # tensors that represent the updated params for all of the tasks (symbolic)
            self.adapted_policies_params.append(adapted_policy_params_dict)

    def adapt(self, samples):
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

        input_list = self._extract_input_list(samples, self._optimization_keys)

        feed_dict_inputs = list(zip(self.input_list_ph, input_list))
        feed_dict_params = list((self.policy.policies_params_ph[i][key], self.policy.policies_params_vals[i][key])
                                for i in range(num_tasks) for key in self.policy.policy_params_keys)
        feed_dict = dict(feed_dict_inputs + feed_dict_params)

        adapted_policies_params_vals = sess.run(self.adapted_policies_params, feed_dict=feed_dict)

        self.policy.update_task_parameters(adapted_policies_params_vals)

    def _extract_input_list(self, all_samples_data, keys):
        """
        Extracts keys from each dict in a list of list of dicts, and 
        Todo: double check what this does and simplify

        Args:
            all_samples_data (list) : list of list of dicts (?)
            keys (list) : a list of keys that should exist in each dict

        Returns:
            (list) : list of input data of the form [key_1], [key_2], [key_3]
        """
        input_list = []
        for step in range(len(all_samples_data)):  # these are the gradient steps
            inputs = [list() for _ in range(len(keys))]
            for i in range(self.meta_batch_size):
                extracted_data = extract(
                    all_samples_data[step][i], keys
                )
                for j, data in enumerate(extracted_data):
                    if isinstance(data, dict):
                        inputs[j].extend([data[k] for k in data.keys()]) # FIXME: ?? Before was self.policy.distribution.dist_info_keys]
                    elif isinstance(data, np.ndarray):
                        inputs[j].append(data)
                    else:
                        raise NotImplementedError

            input_list += sum(inputs, [])
        return input_list