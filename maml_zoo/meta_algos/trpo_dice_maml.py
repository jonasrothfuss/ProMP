from maml_zoo.logger import logger
from maml_zoo.meta_algos.dice_maml import DICEMAML
from maml_zoo.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer
from maml_zoo import utils

import tensorflow as tf
import numpy as np
from collections import OrderedDict


class TRPO_DICEMAML(DICEMAML):
    """
    Algorithm for TRPO as meta-optimizer and DICE objective for inner step

    Args:
        max_path_length (int): maximum path length
        policy (Policy) : policy object
        name (str): tf variable scope
        step_size (float): ltrust region size for the meta policy optimization through TPRO
        inner_lr (float) : gradient step size used for inner step
        meta_batch_size (int): number of meta-learning tasks
        num_inner_grad_steps (int) : number of gradient updates taken per maml iteration
        trainable_inner_step_size (boolean): whether make the inner step size a trainable variable
    """
    def __init__(
            self,
            max_path_length,
            *args,
            name="trpo_dice_maml",
            step_size=0.01,
            **kwargs
            ):
        super(DICEMAML, self).__init__(*args, **kwargs)

        self.optimizer = ConjugateGradientOptimizer()
        self.step_size = step_size
        self.max_path_length = max_path_length
        self._optimization_keys = ['observations', 'actions', 'adjusted_rewards', 'advantages', 'mask', 'agent_infos']
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

        with tf.variable_scope(self.name):
            """ Inner updates"""
            for step_id in range(1, self.num_inner_grad_steps+1):
                with tf.variable_scope("inner_update_%i" % step_id):
                    surr_objs, adapted_policy_params = [], []

                    # inner adaptation step for each task
                    for i in range(self.meta_batch_size):
                        action_stacked = self._reshape_action_phs(action_phs[i])
                        surr_loss = self._adapt_objective_sym(action_stacked, adj_reward_phs[i], mask_phs[i], distribution_info_vars[i])

                        adapted_params_var = self._adapt_sym(surr_loss, current_policy_params[i])

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

            """ Outer (meta-)objective """

            # create additional placeholders for advantages
            adv_phs, all_phs_dict = self._make_adv_phs(prefix='step%i' % self.num_inner_grad_steps)
            self.meta_op_phs_dict.update(all_phs_dict)

            # meta-objective
            surr_objs, outer_kls = [], []
            for i in range(self.meta_batch_size):
                action_stacked = self._reshape_action_phs(action_phs[i])
                dist_info_old_stacked = self._reshape_dist_info(dist_info_old_phs[i])

                # surrogate objective
                likelihood_ratio = self.policy.distribution.likelihood_ratio_sym(action_stacked, dist_info_old_stacked,
                                                                                 distribution_info_vars[i])
                likelihood_ratio = tf.reshape(likelihood_ratio, shape=tf.shape(mask_phs[i]))
                surr_obj = - tf.reduce_mean(mask_phs[i] * likelihood_ratio * adv_phs[i])

                # KL-divergence for constaraint
                kl_sym = self.policy.distribution.kl_sym(dist_info_old_stacked, distribution_info_vars[i])
                kl_sym = tf.reshape(kl_sym, shape=tf.shape(mask_phs[i]))
                outer_kl = tf.reduce_mean(mask_phs[i] * kl_sym)

                surr_objs.append(surr_obj)
                outer_kls.append(outer_kl)

            mean_outer_kl = tf.reduce_mean(tf.stack(outer_kls))

            """ Mean over meta tasks """
            meta_objective = tf.reduce_mean(tf.stack(surr_objs, 0))

            self.optimizer.build_graph(
                loss=meta_objective,
                target=self.policy,
                input_ph_dict=self.meta_op_phs_dict,
                leq_constraint=(mean_outer_kl, self.step_size),
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
        logger.log("Computing KL before")
        mean_kl_before = self.optimizer.constraint_val(meta_op_input_dict)

        logger.log("Computing loss before")
        loss_before = self.optimizer.loss(meta_op_input_dict)
        logger.log("Optimizing")
        self.optimizer.optimize(meta_op_input_dict)
        logger.log("Computing loss after")
        loss_after = self.optimizer.loss(meta_op_input_dict)

        logger.log("Computing KL after")
        mean_kl = self.optimizer.constraint_val(meta_op_input_dict)
        if log:
            logger.logkv('MeanKLBefore', mean_kl_before)
            logger.logkv('MeanKL', mean_kl)

            logger.logkv('LossBefore', loss_before)
            logger.logkv('LossAfter', loss_after)
            logger.logkv('dLoss', loss_before - loss_after)


    def _make_adv_phs(self, prefix):
        adv_phs = []
        all_phs_dict = OrderedDict()
        for task_id in range(self.meta_batch_size):
            # adjusted reward ph
            ph = tf.placeholder(dtype=tf.float32, shape=[None, self.max_path_length],
                                name='advantages' + '_' + prefix + '_' + str(task_id))
            all_phs_dict['%s_task%i_%s' % (prefix, task_id, 'advantages')] = ph
            adv_phs.append(ph)
        return adv_phs, all_phs_dict

    def _reshape_dist_info(self, dist_info):
        for key, value in dist_info.items():
            dist_info[key] = tf.reshape(value, [-1, self.policy.action_dim])
        return dist_info