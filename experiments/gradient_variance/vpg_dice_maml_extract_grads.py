from maml_zoo.logger import logger

from maml_zoo.meta_algos.dice_maml import DICEMAML
from maml_zoo.optimizers.maml_first_order_optimizer import MAMLFirstOrderOptimizer
from maml_zoo import utils

import tensorflow as tf
import numpy as np
from collections import OrderedDict


class VPG_DICEMAML(DICEMAML):
    """
    Algorithm for DICE VPG MAML

    Args:
        max_path_length (int): maximum path length
        policy (Policy) : policy object
        name (str): tf variable scope
        learning_rate (float): learning rate for the meta-objective
        inner_lr (float) : gradient step size used for inner step
        meta_batch_size (int): number of meta-learning tasks
        num_inner_grad_steps (int) : number of gradient updates taken per maml iteration
        trainable_inner_step_size (boolean): whether make the inner step size a trainable variable
    """
    def __init__(
            self,
            max_path_length,
            *args,
            name="vpg_dice_maml",
            **kwargs
            ):
        super(VPG_DICEMAML, self).__init__(max_path_length, *args, **kwargs)

        self._optimization_keys = ['observations', 'actions', 'advantages', 'adjusted_rewards', 'mask', 'agent_infos']
        self.name = name

        self.build_graph()

    def build_graph(self):
        """
        Creates the computation graph for DICE MAML

        Notes:
            Pseudocode:
            for task in meta_batch_size:
                make_vars
                init_init_dist_sym
            for step in num_inner_grad_steps:
                for task in meta_batch_size:
                    make_vars
                    update_init_dist_sym
            set objectives for optimizer
        """

        self.gradients = []

        """ Build graph for sampling """
        with tf.variable_scope(self.name + '_sampling'):
            self.step_sizes = self._create_step_size_vars()

            """ --- Build inner update graph for adapting the policy and sampling trajectories --- """
            # this graph is only used for adapting the policy and not computing the meta-updates
            self.adapted_policies_params, self.adapt_input_ph_dict = self._build_inner_adaption()


        """ Build graph for meta-update """
        meta_update_scope = tf.variable_scope(self.name + '_meta_update')

        with meta_update_scope:
            obs_phs, action_phs, adj_reward_phs, mask_phs, dist_info_old_phs, all_phs_dict = self._make_dice_input_placeholders('step0')
            self.meta_op_phs_dict = OrderedDict(all_phs_dict)

            distribution_info_vars, current_policy_params, all_surr_objs = [], [], []

        for i in range(self.meta_batch_size):
            obs_stacked = self._reshape_obs_phs(obs_phs[i])
            dist_info_sym = self.policy.distribution_info_sym(obs_stacked, params=None)
            distribution_info_vars.append(dist_info_sym)  # step 0
            current_policy_params.append(self.policy.policy_params) # set to real policy_params (tf.Variable)

        with meta_update_scope:
            """ Inner updates"""
            for step_id in range(1, self.num_inner_grad_steps+1):
                with tf.variable_scope("inner_update_%i"%step_id):
                    surr_objs, adapted_policy_params, gradient_vectors = [], [], []

                    # inner adaptation step for each task
                    for i in range(self.meta_batch_size):
                        action_stacked = self._reshape_action_phs(action_phs[i])
                        surr_loss = self._adapt_objective_sym(action_stacked, adj_reward_phs[i], mask_phs[i], distribution_info_vars[i])

                        adapted_params_var, gradient_vector = self._adapt_sym(surr_loss, current_policy_params[i])
                        gradient_vectors.append(gradient_vector)

                        adapted_policy_params.append(adapted_params_var)
                        surr_objs.append(surr_loss)

                    all_surr_objs.append(surr_objs)
                    # Create new placeholders for the next step
                obs_phs, action_phs, adj_reward_phs, mask_phs, dist_info_old_phs, all_phs_dict = self._make_dice_input_placeholders('step%i' % step_id)
                self.meta_op_phs_dict.update(all_phs_dict)

                # dist_info_vars_for_next_step
                distribution_info_vars = []
                for i in range(self.meta_batch_size):
                    obs_stacked = self._reshape_obs_phs(obs_phs[i])
                    distribution_info_vars.append(self.policy.distribution_info_sym(obs_stacked, params=adapted_policy_params[i]))

                current_policy_params = adapted_policy_params
                self.gradients.append(gradient_vectors)

            """ Outer (meta-)objective """
            with tf.variable_scope("outer_update"):
                adv_phs, phs_dict = self._make_advantage_phs('step%i' % self.num_inner_grad_steps)
                self.meta_op_phs_dict.update(phs_dict)

                surr_objs = []

                # meta-objective
                for i in range(self.meta_batch_size):
                    action_stacked = self._reshape_action_phs(action_phs[i])
                    log_likelihood = self.policy.distribution.log_likelihood_sym(action_stacked, distribution_info_vars[i])
                    log_likelihood = tf.reshape(log_likelihood, tf.shape(mask_phs[i]))
                    surr_obj = - tf.reduce_mean(log_likelihood * adv_phs[i] * mask_phs[i])
                    surr_objs.append(surr_obj)

                """ Mean over meta tasks """
                meta_objective = tf.reduce_mean(tf.stack(surr_objs, 0))

                # get meta gradients
                params_var = self.policy.get_params()
                meta_gradients = tf.gradients(meta_objective, [params_var[key] for key in sorted(params_var.keys())])
                meta_gradients = tf.concat([tf.reshape(grad, shape=(-1,)) for grad in meta_gradients],
                                           axis=0)  # flatten and concatenate
                self.gradients.append(meta_gradients)

                self.optimizer.build_graph(
                    loss=meta_objective,
                    target=self.policy,
                    input_ph_dict=self.meta_op_phs_dict,
                )

    def _make_advantage_phs(self, prefix=''):
        adv_phs = []
        all_phs_dict = OrderedDict()

        for task_id in range(self.meta_batch_size):
            # advantage ph
            ph = tf.placeholder(dtype=tf.float32, shape=[None, self.max_path_length], name='advantage' + '_' + prefix + '_' + str(task_id))
            all_phs_dict['%s_task%i_%s' % (prefix, task_id, 'advantages')] = ph
            adv_phs.append(ph)

        return adv_phs, all_phs_dict

    def _adapt_sym(self, surr_obj, params_var):
        """
        Creates the symbolic representation of the tf policy after one gradient step towards the surr_obj

        Args:
            surr_obj (tf_op) : tensorflow op for task specific (inner) objective
            params_var (dict) : dict of tf.Tensors for current policy params

        Returns:
            (dict):  dict of tf.Tensors for adapted policy params
        """
        # TODO: Fix this if we want to learn the learning rate (it isn't supported right now).
        update_param_keys = list(params_var.keys())

        grads = tf.gradients(surr_obj, [params_var[key] for key in update_param_keys])
        gradients = dict(zip(update_param_keys, grads))

        # gradient descent
        adapted_policy_params = [params_var[key] - tf.multiply(self.step_sizes[key], gradients[key])
                          for key in update_param_keys]

        adapted_policy_params_dict = OrderedDict(zip(update_param_keys, adapted_policy_params))

        # flattens and concatenates the gadients
        gradient_vector = tf.concat([tf.reshape(grad, shape=(-1,)) for grad in grads], axis=0)
        return adapted_policy_params_dict, gradient_vector

    def compute_gradients(self, all_samples_data, log=True):
        meta_op_input_dict = self._extract_input_dict_meta_op(all_samples_data, self._optimization_keys)
        feed_dict = utils.create_feed_dict(placeholder_dict=self.meta_op_phs_dict, value_dict=meta_op_input_dict)
        if log: logger.log("compute gradients")
        gradients_values = tf.get_default_session().run(self.gradients, feed_dict=feed_dict)
        return gradients_values

    def _build_inner_adaption(self):
        """
        Creates the (DICE) symbolic graph for the one-step inner gradient update (It'll be called several times if
        more gradient steps are needed)

        Args:
            some placeholders

        Returns:
            adapted_policies_params (list): list of Ordered Dict containing the symbolic post-update parameters
            adapt_input_list_ph (list): list of placeholders

        """
        obs_phs, action_phs, adj_reward_phs, mask_phs, dist_info_old_phs, adapt_input_ph_dict = self._make_dice_input_placeholders('adapt')

        adapted_policies_params = []

        for i in range(self.meta_batch_size):
            with tf.variable_scope("adapt_task_%i" % i):
                with tf.variable_scope("adapt_objective"):
                    obs_stacked = self._reshape_obs_phs(obs_phs[i])
                    action_stacked = self._reshape_action_phs(action_phs[i])
                    distribution_info_stacked = self.policy.distribution_info_sym(obs_stacked,
                                                                              params=self.policy.policies_params_phs[i])

                    # inner surrogate objective
                    adapt_loss = self._adapt_objective_sym(action_stacked, adj_reward_phs[i], mask_phs[i],
                                                           distribution_info_stacked)

                # get tf operation for adapted (post-update) policy
                with tf.variable_scope("adapt_step"):
                    adapted_policy_param, _ = self._adapt_sym(adapt_loss, self.policy.policies_params_phs[i])
                adapted_policies_params.append(adapted_policy_param)

        return adapted_policies_params, adapt_input_ph_dict