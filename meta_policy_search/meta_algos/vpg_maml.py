from meta_policy_search.utils import logger
from meta_policy_search.meta_algos.base import MAMLAlgo
from meta_policy_search.optimizers.maml_first_order_optimizer import MAMLFirstOrderOptimizer

import tensorflow as tf
from collections import OrderedDict


class VPGMAML(MAMLAlgo):
    """
    Algorithm for PPO MAML

    Args:
        policy (Policy): policy object
        name (str): tf variable scope
        learning_rate (float): learning rate for the meta-objective
        exploration (bool): use exploration / pre-update sampling term / E-MAML term
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
        if exploration: # add adjusted average rewards tp optimization keys
            self._optimization_keys.append('adj_avg_rewards')

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
        """

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
                surr_objs, adapted_policy_params = [], []

                # inner adaptation step for each task
                for i in range(self.meta_batch_size):
                    surr_loss = self._adapt_objective_sym(action_phs[i], adv_phs[i], dist_info_old_phs[i], distribution_info_vars[i])

                    adapted_params_var = self._adapt_sym(surr_loss, current_policy_params[i])

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

            """ Outer objective """
            surr_objs = []

            # meta-objective
            for i in range(self.meta_batch_size):
                log_likelihood = self.policy.distribution.log_likelihood_sym(action_phs[i], distribution_info_vars[i])
                surr_obj = - tf.reduce_mean(log_likelihood * adv_phs[i])

                if self.exploration:
                    # add adj_avg_reward placeholder
                    adj_avg_rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='adj_avg_rewards' + '_' + str(self.num_inner_grad_steps) + '_' + str(i))
                    self.meta_op_phs_dict['step%i_task%i_%s' % (self.num_inner_grad_steps, i, 'adj_avg_rewards')] = adj_avg_rewards

                    log_likelihood_inital = self.policy.distribution.log_likelihood_sym(initial_action_phs[i],
                                                                                        initial_distribution_info_vars[i])
                    surr_obj += - tf.reduce_mean(adj_avg_rewards) * tf.reduce_mean(log_likelihood_inital)

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

