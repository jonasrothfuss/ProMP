from maml_zoo import utils
from maml_zoo.policies.base import Policy

from collections import OrderedDict
import tensorflow as tf
import numpy as np


class Algo(object):
    """
    Base class for algorithms

    Args:
        policy (Policy) : policy object
    """

    def __init__(self, policy):
        assert isinstance(policy, Policy)
        self.policy = policy
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

    def _make_input_placeholders(self, prefix='', recurrent=False):
        """
        Args:
            prefix (str) : a string to prepend to the name of each variable

        Returns:
            (tuple) : a tuple containing lists of placeholders for each input type and meta task,
            and for convenience, a list containing all placeholders created
        """
        dist_info_specs = self.policy.distribution.dist_info_specs

        all_phs_dict = OrderedDict()

        # observation ph
        obs_shape = [None, self.policy.obs_dim] if not recurrent else [None, None, self.policy.obs_dim]
        obs_ph = tf.placeholder(dtype=tf.float32, shape=obs_shape, name=prefix + '_obs')
        all_phs_dict['%s_%s' % (prefix, 'observations')] = obs_ph

        # action ph
        action_shape = [None, self.policy.action_dim] if not recurrent else [None, None, self.policy.action_dim]
        action_ph = tf.placeholder(dtype=tf.float32, shape=action_shape, name=prefix + '_action')
        all_phs_dict['%s_%s' % (prefix, 'actions')] = action_ph

        # advantage ph
        adv_shape = [None] if not recurrent else [None, None]
        adv_ph = tf.placeholder(dtype=tf.float32, shape=adv_shape, name=prefix + '_advantage')
        all_phs_dict['%s_%s' % (prefix, 'advantages')] = adv_ph

        # distribution / agent info
        dist_info_ph_dict = {}
        for info_key, shape in dist_info_specs:
            _shape = [None] + list(shape) if not recurrent else [None, None] + list(shape)
            ph = tf.placeholder(dtype=tf.float32, shape=_shape, name='%s_%s' % (info_key, prefix))
            all_phs_dict['%s_agent_infos/%s' % (prefix, info_key)] = ph
            dist_info_ph_dict[info_key] = ph

        return obs_ph, action_ph, adv_ph, dist_info_ph_dict, all_phs_dict

    def _extract_input_dict(self, samples_data, keys, prefix=''):
        """
        Re-arranges a list of dicts containing the processed sample data into a OrderedDict that can be matched
        with a placeholder dict for creating a feed dict

        Args:
            samples_data_meta_batch (list) : list of dicts containing the processed data corresponding to each meta-task
            keys (list) : a list of keys that should exist in each dict and whose values shall be extracted
            prefix (str): prefix to prepend the keys in the resulting OrderedDict

        Returns:
            OrderedDict containing the data from all_samples_data. The data keys follow the naming convention:
                '<prefix>_task<task_number>_<key_name>'
        """
        input_dict = OrderedDict()

        extracted_data = utils.extract(
            samples_data, *keys
        )

        # iterate over the desired data instances and corresponding keys
        for j, (data, key) in enumerate(zip(extracted_data, keys)):
            if isinstance(data, dict):
                # if the data instance is a dict -> iterate over the items of this dict
                for k, d in data.items():
                    assert isinstance(d, np.ndarray)
                    input_dict['%s_%s/%s' % (prefix, key, k)] = d

            elif isinstance(data, np.ndarray):
                input_dict['%s_%s' % (prefix, key)] = data
            else:
                raise NotImplementedError
        return input_dict

