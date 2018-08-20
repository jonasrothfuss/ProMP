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
        policy (Policy) : policy object
        inner_lr (float) : gradient step size used for inner step
        meta_batch_size (int): number of meta-tasks
        num_inner_grad_steps (int) : number of gradient updates taken per maml iteration
        learning_rate (float) : 
        max_epochs (int) :
        num_minibatches (int) : Currently not implemented
        clip_eps (float) :
        clip_outer (bool) : whether to use L^CLIP or L^KLPEN on outer gradient update
        target_outer_step (float) : target outer kl divergence, used only with L^KLPEN and when adaptive_outer_kl_penalty is true
        target_inner_step (float) : target inner kl divergence, used only when adaptive_inner_kl_penalty is true
        init_outer_kl_penalty (float) : initial penalty for outer kl, used only with L^KLPEN
        init_inner_kl_penalty (float) : initial penalty for inner kl
        adaptive_outer_kl_penalty (bool): whether to used a fixed or adaptive kl penalty on outer gradient update
        adaptive_inner_kl_penalty (bool): whether to used a fixed or adaptive kl penalty on inner gradient update
        anneal_factor (float) : multiplicative factor for clip_eps, updated every iteration
        entropy_bonus (float) : scaling factor for policy entropy
    """
    def __init__(
            self,
            learning_rate,
            inner_type,
            *args,
            trainable_inner_step_size=False,
            name="vpg_maml",
            **kwargs
            ):
        super(VPGMAML, self).__init__(*args, **kwargs)
        assert inner_type in ["log_likelihood", "likelihood_ratio", "dice"]

        self.optimizer = MAMLFirstOrderOptimizer(learning_rate=learning_rate)
        self.inner_type = inner_type
        self._optimization_keys = ['observations', 'actions', 'advantages', 'agent_infos']
        self.name = name
        self.trainable_inner_step_size = trainable_inner_step_size
        self.step_sizes = None

        self.build_graph()

    def adapt_objective_sym(self, action_sym, adv_sym, dist_info_old_sym, dist_info_new_sym):
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

        """ Create Variables """
        with tf.variable_scope(self.name):
            self.step_sizes = self._create_step_size_vars()

            """ --- Build inner update graph for adapting the policy and sampling trajectories --- """
            # this graph is only used for adapting the policy and not computing the meta-updates
            self.adapted_policies_params, self.adapt_input_ph_dict = self._build_inner_adaption()

            """ ----- Build graph for the meta-update ----- """
            self.meta_op_phs_dict = OrderedDict()
            obs_phs, action_phs, adv_phs, dist_info_old_phs, all_phs_dict = self.make_input_placeholders('step0')
            self.meta_op_phs_dict.update(all_phs_dict)

            distribution_info_vars, current_policy_params = [], []
            all_surr_objs = []

        for i in range(self.meta_batch_size):
            dist_info_sym = self.policy.distribution_info_sym(obs_phs[i], params=None)
            distribution_info_vars.append(dist_info_sym)  # step 0
            current_policy_params.append(self.policy.policy_params) # set to real policy_params (tf.Variable)

        with tf.variable_scope(self.name):
            """ Inner updates"""
            for step_id in range(1, self.num_inner_grad_steps+1):
                surr_objs, adapted_policy_params = [], []

                # inner adaptation step for each task
                for i in range(self.meta_batch_size):
                    surr_loss = self.adapt_objective_sym(action_phs[i], adv_phs[i], dist_info_old_phs[i], distribution_info_vars[i])

                    adapted_params_var = self.adapt_sym(surr_loss, current_policy_params[i])

                    adapted_policy_params.append(adapted_params_var)
                    surr_objs.append(surr_loss)

                all_surr_objs.append(surr_objs)
                # Create new placeholders for the next step
                obs_phs, action_phs, adv_phs, dist_info_old_phs, all_phs_dict = self.make_input_placeholders('step%i' % step_id)
                self.meta_op_phs_dict.update(all_phs_dict)

                # dist_info_vars_for_next_step
                distribution_info_vars = [self.policy.distribution_info_sym(obs_phs[i], params=adapted_policy_params[i])
                                          for i in range(self.meta_batch_size)]
                current_policy_params = adapted_policy_params

            """ Outer objective """
            surr_objs = []

            # meta-objective
            for i in range(self.meta_batch_size):
                likelihood_ratio = self.policy.distribution.log_likelihood_sym(action_phs[i], distribution_info_vars[i])
                surr_obj = - tf.reduce_mean(likelihood_ratio * adv_phs[i])
                surr_objs.append(surr_obj)

            """ Mean over meta tasks """
            meta_objective = tf.reduce_mean(tf.stack(surr_objs, 0))

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

    def _create_step_size_vars(self):
        # Step sizes
        with tf.variable_scope('inner_step_sizes'):
            step_sizes = dict()
            for key, param in self.policy.policy_params.items():
                shape = param.get_shape().as_list()
                init_stepsize = np.ones(shape, dtype=np.float32) * self.inner_lr
                step_sizes[key] = tf.Variable(initial_value=init_stepsize,
                                              name='%s_step_size' % key,
                                              dtype=tf.float32, trainable=self.trainable_inner_step_size)
        return step_sizes

