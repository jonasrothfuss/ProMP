from meta_policy_search.utils import logger
from meta_policy_search.meta_algos.base import MAMLAlgo
from meta_policy_search.optimizers.maml_first_order_optimizer import MAMLPPOOptimizer

import tensorflow as tf
import numpy as np
from collections import OrderedDict

class ProMP(MAMLAlgo):
    """
    ProMP Algorithm

    Args:
        policy (Policy): policy object
        name (str): tf variable scope
        learning_rate (float): learning rate for optimizing the meta-objective
        num_ppo_steps (int): number of ProMP steps (without re-sampling)
        num_minibatches (int): number of minibatches for computing the ppo gradient steps
        clip_eps (float): PPO clip range
        target_inner_step (float) : target inner kl divergence, used only when adaptive_inner_kl_penalty is true
        init_inner_kl_penalty (float) : initial penalty for inner kl
        adaptive_inner_kl_penalty (bool): whether to used a fixed or adaptive kl penalty on inner gradient update
        anneal_factor (float) : multiplicative factor for annealing clip_eps. If anneal_factor < 1, clip_eps <- anneal_factor * clip_eps at each iteration
        inner_lr (float) : gradient step size used for inner step
        meta_batch_size (int): number of meta-learning tasks
        num_inner_grad_steps (int) : number of gradient updates taken per maml iteration
        trainable_inner_step_size (boolean): whether make the inner step size a trainable variable

    """
    def __init__(
            self,
            *args,
            name="ppo_maml",
            learning_rate=1e-3,
            num_ppo_steps=5,
            num_minibatches=1,
            clip_eps=0.2,
            target_inner_step=0.01,
            init_inner_kl_penalty=1e-2,
            adaptive_inner_kl_penalty=True,
            anneal_factor=1.0,
            **kwargs
            ):
        super(ProMP, self).__init__(*args, **kwargs)

        self.optimizer = MAMLPPOOptimizer(learning_rate=learning_rate, max_epochs=num_ppo_steps, num_minibatches=num_minibatches)
        self.clip_eps = clip_eps
        self.target_inner_step = target_inner_step
        self.adaptive_inner_kl_penalty = adaptive_inner_kl_penalty
        self.inner_kl_coeff = init_inner_kl_penalty * np.ones(self.num_inner_grad_steps)
        self.anneal_coeff = 1
        self.anneal_factor = anneal_factor
        self._optimization_keys = ['observations', 'actions', 'advantages', 'agent_infos']
        self.name = name
        self.kl_coeff = [init_inner_kl_penalty] * self.meta_batch_size * self.num_inner_grad_steps

        self.build_graph()

    def _adapt_objective_sym(self, action_sym, adv_sym, dist_info_old_sym, dist_info_new_sym):
        with tf.variable_scope("likelihood_ratio"):
            likelihood_ratio_adapt = self.policy.distribution.likelihood_ratio_sym(action_sym,
                                                                                   dist_info_old_sym, dist_info_new_sym)
        with tf.variable_scope("surrogate_loss"):
            surr_obj_adapt = -tf.reduce_mean(likelihood_ratio_adapt * adv_sym)
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
            all_surr_objs, all_inner_kls = [], []

        for i in range(self.meta_batch_size):
            dist_info_sym = self.policy.distribution_info_sym(obs_phs[i], params=None)
            distribution_info_vars.append(dist_info_sym)  # step 0
            current_policy_params.append(self.policy.policy_params) # set to real policy_params (tf.Variable)

        with tf.variable_scope(self.name):
            """ Inner updates"""
            for step_id in range(1, self.num_inner_grad_steps+1):
                surr_objs, kls, adapted_policy_params = [], [], []

                # inner adaptation step for each task
                for i in range(self.meta_batch_size):
                    surr_loss = self._adapt_objective_sym(action_phs[i], adv_phs[i], dist_info_old_phs[i], distribution_info_vars[i])
                    kl_loss = tf.reduce_mean(self.policy.distribution.kl_sym(dist_info_old_phs[i], distribution_info_vars[i]))

                    adapted_params_var = self._adapt_sym(surr_loss, current_policy_params[i])

                    adapted_policy_params.append(adapted_params_var)
                    kls.append(kl_loss)
                    surr_objs.append(surr_loss)

                all_surr_objs.append(surr_objs)
                all_inner_kls.append(kls)

                # Create new placeholders for the next step
                obs_phs, action_phs, adv_phs, dist_info_old_phs, all_phs_dict = self._make_input_placeholders('step%i' % step_id)
                self.meta_op_phs_dict.update(all_phs_dict)

                # dist_info_vars_for_next_step
                distribution_info_vars = [self.policy.distribution_info_sym(obs_phs[i], params=adapted_policy_params[i])
                                          for i in range(self.meta_batch_size)]
                current_policy_params = adapted_policy_params

            # per step: compute mean of kls over tasks
            mean_inner_kl_per_step = tf.stack([tf.reduce_mean(tf.stack(inner_kls)) for inner_kls in all_inner_kls])

            """ Outer objective """
            surr_objs, outer_kls = [], []

            # Create placeholders
            inner_kl_coeff = tf.placeholder(tf.float32, shape=[self.num_inner_grad_steps], name='inner_kl_coeff')
            self.meta_op_phs_dict['inner_kl_coeff'] = inner_kl_coeff

            clip_eps_ph = tf.placeholder(tf.float32, shape=[], name='clip_eps')
            self.meta_op_phs_dict['clip_eps'] = clip_eps_ph

            # meta-objective
            for i in range(self.meta_batch_size):
                likelihood_ratio = self.policy.distribution.likelihood_ratio_sym(action_phs[i], dist_info_old_phs[i],
                                                                                 distribution_info_vars[i])
                outer_kl = tf.reduce_mean(self.policy.distribution.kl_sym(dist_info_old_phs[i], distribution_info_vars[i]))

                # clipped likelihood ratio
                clipped_obj = tf.minimum(likelihood_ratio * adv_phs[i],
                                         tf.clip_by_value(likelihood_ratio,
                                                          1 - clip_eps_ph,
                                                          1 + clip_eps_ph) * adv_phs[i])
                surr_obj = - tf.reduce_mean(clipped_obj)


                surr_objs.append(surr_obj)
                outer_kls.append(outer_kl)

            mean_outer_kl = tf.reduce_mean(tf.stack(outer_kls))
            inner_kl_penalty = tf.reduce_mean(inner_kl_coeff * mean_inner_kl_per_step)

            """ Mean over meta tasks """
            meta_objective = tf.reduce_mean(tf.stack(surr_objs, 0)) + inner_kl_penalty

            self.optimizer.build_graph(
                loss=meta_objective,
                target=self.policy,
                input_ph_dict=self.meta_op_phs_dict,
                inner_kl=mean_inner_kl_per_step,
                outer_kl=mean_outer_kl,
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

        # add kl_coeffs / clip_eps to meta_op_input_dict
        meta_op_input_dict['inner_kl_coeff'] = self.inner_kl_coeff

        meta_op_input_dict['clip_eps'] = self.clip_eps

        if log: logger.log("Optimizing")
        loss_before = self.optimizer.optimize(input_val_dict=meta_op_input_dict)

        if log: logger.log("Computing statistics")
        loss_after, inner_kls, outer_kl = self.optimizer.compute_stats(input_val_dict=meta_op_input_dict)

        if self.adaptive_inner_kl_penalty:
            if log: logger.log("Updating inner KL loss coefficients")
            self.inner_kl_coeff = self.adapt_kl_coeff(self.inner_kl_coeff, inner_kls, self.target_inner_step)


        if log:
            logger.logkv('LossBefore', loss_before)
            logger.logkv('LossAfter', loss_after)
            logger.logkv('KLInner', np.mean(inner_kls))
            logger.logkv('KLCoeffInner', np.mean(self.inner_kl_coeff))

    def adapt_kl_coeff(self, kl_coeff, kl_values, kl_target):
        if hasattr(kl_values, '__iter__'):
            assert len(kl_coeff) == len(kl_values)
            return np.array([_adapt_kl_coeff(kl_coeff[i], kl, kl_target) for i, kl in enumerate(kl_values)])
        else:
            return _adapt_kl_coeff(kl_coeff, kl_values, kl_target)

def _adapt_kl_coeff(kl_coeff, kl, kl_target):
    if kl < kl_target / 1.5:
        kl_coeff /= 2

    elif kl > kl_target * 1.5:
        kl_coeff *= 2
    return kl_coeff