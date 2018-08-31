from maml_zoo.logger import logger
from maml_zoo.meta_algos.base import MAMLAlgo
from maml_zoo.optimizers.maml_first_order_optimizer import MAMLFirstOrderOptimizer
from maml_zoo import utils

import tensorflow as tf
import numpy as np
from collections import OrderedDict


class VPGMAML(MAMLAlgo):
    """
    Algorithm for PPO MAML

    Args:
        policy (Policy): policy object
        name (str): tf variable scope
        learning_rate (float): learning rate for the meta-objective
        inner_type (str): inner optimization objective - either log_likelihood or likelihood_ratio
        inner_lr (float) : gradient step size used for inner step
        meta_batch_size (int): number of meta-learning tasks
        num_inner_grad_steps (int) : number of gradient updates taken per maml iteration
        trainable_inner_step_size (boolean): whether make the inner step size a trainable variable
    """
    def __init__(
            self,
            *args,
            name="vpg_maml",
            learning_rate=1e-3,
            inner_type='likelihood_ratio',
            exploration=False,
            **kwargs
            ):
        super(VPGMAML, self).__init__(*args, **kwargs)
        assert inner_type in ["log_likelihood", "likelihood_ratio"]

        self.optimizer = MAMLFirstOrderOptimizer(learning_rate=learning_rate)
        self.inner_type = inner_type
        self._optimization_keys = ['observations', 'actions', 'advantages', 'agent_infos']
        self.name = name
        self.exploration = exploration

        self.build_graph()

    def _adapt_objective_sym(self, action_sym, adv_sym, dist_info_old_sym, dist_info_new_sym):
        if self.inner_type == 'likelihood_ratio':
            with tf.variable_scope("likelihood_ratio"):
                likelihood_ratio_adapt = self.policy.distribution.likelihood_ratio_sym(action_sym,
                                                                                       dist_info_old_sym, dist_info_new_sym)
            with tf.variable_scope("surrogate_loss"):
                surr_obj_adapt = -tf.reduce_mean(likelihood_ratio_adapt * adv_sym)

        elif self.inner_type == 'log_likelihood':
            with tf.variable_scope("log_likelihood"):
                log_likelihood_adapt = self.policy.distribution.log_likelihood_sym(action_sym, dist_info_new_sym)
            with tf.variable_scope("surrogate_loss"):
                surr_obj_adapt = -tf.reduce_mean(log_likelihood_adapt * adv_sym)

        else:
            raise NotImplementedError

        return surr_obj_adapt

    def build_graph(self):
        """
        Creates the computation graph

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

        """ Create Variables """
        with tf.variable_scope(self.name):
            self.step_sizes = self._create_step_size_vars()

            """ --- Build inner update graph for adapting the policy and sampling trajectories --- """
            # this graph is only used for adapting the policy and not computing the meta-updates
            self.adapted_policies_params, self.adapt_input_ph_dict = self._build_inner_adaption()

            """ ----- Build graph for the meta-update ----- """
            self.meta_op_phs_dict = OrderedDict()
            obs_phs, action_phs, adv_phs, dist_info_old_phs, all_phs_dict = self._make_input_placeholders('step0')
            self.meta_op_phs_dict.update(all_phs_dict)

            distribution_info_vars, current_policy_params = [], []
            all_surr_objs = []

        for i in range(self.meta_batch_size):
            dist_info_sym = self.policy.distribution_info_sym(obs_phs[i], params=None)
            distribution_info_vars.append(dist_info_sym)  # step 0
            current_policy_params.append(self.policy.policy_params) # set to real policy_params (tf.Variable)

        initial_distribution_info_vars = distribution_info_vars
        initial_action_phs = action_phs

        with tf.variable_scope(self.name):
            """ Inner updates"""
            for step_id in range(1, self.num_inner_grad_steps+1):
                surr_objs, adapted_policy_params, gradient_vectors = [], [], []

                # inner adaptation step for each task
                for i in range(self.meta_batch_size):
                    surr_loss = self._adapt_objective_sym(action_phs[i], adv_phs[i], dist_info_old_phs[i], distribution_info_vars[i])

                    adapted_params_var, gradient_vector = self._adapt_sym(surr_loss, current_policy_params[i])
                    gradient_vectors.append(gradient_vector)

                    adapted_policy_params.append(adapted_params_var)
                    surr_objs.append(surr_loss)

                all_surr_objs.append(surr_objs)
                # Create new placeholders for the next step
                obs_phs, action_phs, adv_phs, dist_info_old_phs, all_phs_dict = self._make_input_placeholders('step%i' % step_id)
                self.meta_op_phs_dict.update(all_phs_dict)

                # dist_info_vars_for_next_step
                distribution_info_vars = [self.policy.distribution_info_sym(obs_phs[i], params=adapted_policy_params[i])
                                          for i in range(self.meta_batch_size)]
                current_policy_params = adapted_policy_params
                self.gradients.append(gradient_vectors)

            """ Outer objective """
            surr_objs = []

            # meta-objective
            for i in range(self.meta_batch_size):
                log_likelihood = self.policy.distribution.log_likelihood_sym(action_phs[i], distribution_info_vars[i])
                surr_obj = - tf.reduce_mean(log_likelihood * adv_phs[i])
                surr_objs.append(surr_obj)

                if self.exploration:
                    log_likelihood_inital = self.policy.distribution.log_likelihood_sym(initial_action_phs[i],
                                                                                        initial_distribution_info_vars[i])
                    surr_obj += -tf.reduce_mean(adv_phs[i]) * tf.reduce_sum(log_likelihood_inital)

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

    def optimize_policy(self, all_samples_data, log=True):
        """
        Performs MAML outer step

        Args:
            all_samples_data (list) : list of lists of lists of samples (each is a dict) split by gradient update and
             meta task
            log (bool) : whether to log statistics

        Returns:
            None
        """
        meta_op_input_dict = self._extract_input_dict_meta_op(all_samples_data, self._optimization_keys)

        if log: logger.log("Optimizing")
        loss_before = self.optimizer.optimize(input_val_dict=meta_op_input_dict)

        if log: logger.log("Computing statistics")
        loss_after = self.optimizer.loss(input_val_dict=meta_op_input_dict)

        if log:
            logger.logkv('LossBefore', loss_before)
            logger.logkv('LossAfter', loss_after)

    def compute_gradients(self, all_samples_data, log=True):
        meta_op_input_dict = self._extract_input_dict_meta_op(all_samples_data, self._optimization_keys)
        feed_dict = utils.create_feed_dict(placeholder_dict=self.meta_op_phs_dict, value_dict=meta_op_input_dict)
        if log: logger.log("compute gradients")
        gradients_values = tf.get_default_session().run(self.gradients, feed_dict=feed_dict)
        return gradients_values

    def _build_inner_adaption(self):
        """
        Creates the symbolic graph for the one-step inner gradient update (It'll be called several times if
        more gradient steps are needed)

        Args:
            some placeholders

        Returns:
            adapted_policies_params (list): list of Ordered Dict containing the symbolic post-update parameters
            adapt_input_list_ph (list): list of placeholders

        """
        obs_phs, action_phs, adv_phs, dist_info_old_phs, adapt_input_ph_dict = self._make_input_placeholders('adapt')

        adapted_policies_params = []

        for i in range(self.meta_batch_size):
            with tf.variable_scope("adapt_task_%i" % i):
                with tf.variable_scope("adapt_objective"):
                    distribution_info_new = self.policy.distribution_info_sym(obs_phs[i],
                                                                              params=self.policy.policies_params_phs[i])

                    # inner surrogate objective
                    surr_obj_adapt = self._adapt_objective_sym(action_phs[i], adv_phs[i],
                                                               dist_info_old_phs[i], distribution_info_new)

                # get tf operation for adapted (post-update) policy
                with tf.variable_scope("adapt_step"):
                    adapted_policy_param, _ = self._adapt_sym(surr_obj_adapt, self.policy.policies_params_phs[i])
                adapted_policies_params.append(adapted_policy_param)

        return adapted_policies_params, adapt_input_ph_dict

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