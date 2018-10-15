from meta_policy_search.meta_algos.dice_maml import DICEMAML

import tensorflow as tf
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

        with meta_update_scope:
            """ Inner updates"""
            for step_id in range(1, self.num_inner_grad_steps+1):
                with tf.variable_scope("inner_update_%i"%step_id):
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
