import tensorflow as tf
import numpy as np
from maml_zoo.logger import logger
from maml_zoo.meta_algos.base import MAMLAlgo
from maml_zoo.optimizers.maml_first_order_optimizer import MAMLPPOOptimizer
from collections import OrderedDict

class MAMLPPO(MAMLAlgo):
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
            max_epochs, #TODO: does this make sense?
            num_minibatches,
            *args,
            clip_eps=0.2, 
            clip_outer=True,
            target_outer_step=0.001,
            target_inner_step=0.01,
            init_outer_kl_penalty=1e-3,
            init_inner_kl_penalty=1e-2,
            adaptive_outer_kl_penalty=True,
            adaptive_inner_kl_penalty=True,
            anneal_factor=1.0,
            trainable_inner_step_size=False,
            name="ppo_maml",
            **kwargs
            ):
        super(MAMLPPO, self).__init__(*args, **kwargs)

        self.optimizer = MAMLPPOOptimizer(learning_rate=learning_rate, max_epochs=max_epochs, num_minibatches=num_minibatches)
        self.clip_eps = clip_eps
        self.clip_outer = clip_outer
        self.target_outer_step = target_outer_step
        self.target_inner_step = target_inner_step
        self.adaptive_outer_kl_penalty = adaptive_outer_kl_penalty
        self.adaptive_inner_kl_penalty = adaptive_inner_kl_penalty
        self.inner_kl_coeff = [init_inner_kl_penalty] * self.meta_batch_size * self.num_inner_grad_steps
        self.outer_kl_coeff = [init_outer_kl_penalty] * self.meta_batch_size
        self.anneal_coeff = 1
        self.anneal_factor = anneal_factor
        self._optimization_keys = ['observations', 'actions', 'advantages', 'agent_infos']
        self.name = name
        self.kl_coeff = [init_inner_kl_penalty] * self.meta_batch_size * self.num_inner_grad_steps
        self.trainable_inner_step_size = trainable_inner_step_size
        self.step_sizes = None

        self.build_graph()

    def adapt_objective_sym(self, action_sym, adv_sym, dist_info_old_sym, dist_info_new_sym):
        with tf.variable_scope("likelihood_ratio"):
            likelihood_ratio_adapt = self.policy._dist.likelihood_ratio_sym(action_sym,
                                                                            dist_info_old_sym, dist_info_new_sym)
        with tf.variable_scope("surrogate_loss"):
            surr_obj_adapt = -tf.reduce_mean(likelihood_ratio_adapt * adv_sym)
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
        all_surr_objs, all_inner_kls = [], []


        for i in range(self.meta_batch_size):
            dist_info_sym = self.policy.distribution_info_sym(obs_phs[i], params=None)
            distribution_info_vars.append(dist_info_sym) # step 0
            current_policy_params.append(self.policy.policy_params) # set to real policy_params (tf.Variable)

        with tf.variable_scope(self.name):
            """ Inner updates"""
            for step_id in range(1, self.num_inner_grad_steps+1):
                surr_objs, kls, adapted_policy_params = [], [], []

                # inner adaptation step for each task
                for i in range(self.meta_batch_size):
                    surr_loss = self.adapt_objective_sym(action_phs[i], adv_phs[i], dist_info_old_phs[i], distribution_info_vars[i])
                    kl_loss = tf.reduce_mean(self.policy.distribution.kl_sym(dist_info_old_phs[i], distribution_info_vars[i]))

                    adapted_params_var = self.adapt_sym(surr_loss, current_policy_params[i])

                    adapted_policy_params.append(adapted_params_var)
                    kls.append(kl_loss)
                    surr_objs.append(surr_loss)

                all_surr_objs.append(surr_objs)
                all_inner_kls.append(kls)

                # Create new placeholders for the next step
                obs_phs, action_phs, adv_phs, dist_info_phs, all_phs = self.make_input_placeholders('step%i'%(step_id))
                self.meta_op_phs_dict.update(all_phs_dict)

                # dist_info_vars_for_next_step
                distribution_info_vars = [self.policy.distribution_info_sym(obs_phs[i], params=adapted_policy_params[i])
                                          for i in range(self.meta_batch_size)]
                current_policy_params = adapted_policy_params

            # per step: compute mean of kls over tasks
            mean_kl_per_step = tf.stack([tf.reduce_mean(tf.stack(inner_kls)) for inner_kls in all_inner_kls])

            """ Outer objective """
            surr_objs, outer_kls = [], []

            # Create placeholders
            inner_kl_coeff = tf.placeholder(tf.float32, shape=[self.num_inner_grad_steps], name='inner_kl_coeff')
            self.meta_op_phs_dict['inner_kl_coeff'] = inner_kl_coeff

            if not self.clip_outer:
                clip_eps_ph = tf.placeholder(tf.float32, shape=[], name='clip_eps')
                self.meta_op_phs_dict['clip_eps'] = clip_eps_ph
            else:
                outer_kl_coeff = tf.placeholder(tf.float32, shape=[], name='outer_kl_coef')
                self.meta_op_phs_dict['outer_kl_coeff'] = outer_kl_coeff #TODO not sure if we should do that

            # meta-objective
            for i in range(self.meta_batch_size):
                likelihood_ratio = self.policy.distribution.likelihood_ratio_sym(action_phs[i], dist_info_phs[i],
                                                                                 distribution_info_vars[i])
                inner_kl_penalty = tf.reduce_mean(inner_kl_coeff * mean_kl_per_step)

                if self.clip_outer: # clipped likelihood ratio
                    clipped_obj = tf.minimum(likelihood_ratio * adv_phs[i],
                                             tf.clip_by_value(likelihood_ratio,
                                                              1 - clip_eps_ph,
                                                              1 + clip_eps_ph) * adv_phs[i])
                    surr_obj = - tf.reduce_mean(clipped_obj) + inner_kl_penalty

                else: # outer kl penalty
                    outer_kl = tf.reduce_mean(self.policy.distribution.kl_sym(dist_info_phs[i], distribution_info_vars[i]))
                    outer_kl_penalty = outer_kl_coeff * outer_kl
                    surr_obj = - tf.reduce_mean(likelihood_ratio * adv_phs[i]) + inner_kl_penalty + outer_kl_penalty

                surr_objs.append(surr_obj)
                outer_kls.append(outer_kl)

            """ Sum over meta tasks """
            meta_objective = tf.reduce_mean(tf.stack(surr_objs, 0))  # mean over meta_batch_size (the diff tasks)

            input_list = all_inputs #TODO how do tho his with the input list of the optimizer

            self.optimizer.build_graph(
                loss=meta_objective,
                target=self.policy,
                inputs=input_list,
                extra_inputs=extra_inputs,
                inner_kl=all_inner_kls,
                outer_kl=outer_kls,
            )
    def optimize_policy(self, all_samples_data, log=True):
        """
        Performs MAML outer step for each task

        Args:
            all_samples_data (list) : list of lists of lists of samples (each is a dict) split by gradient update and
             meta task
            log (bool) : whether to log statistics

        Returns:
            None
        """
        assert len(all_samples_data) == self.num_inner_grad_steps + 1  # we collected the rollouts to compute the grads and then the test!

        input_list = self._extract_input_list(all_samples_data, self._optimization_keys)

        extra_inputs = self.inner_kl_coeff
        if not self.clip_outer:
            extra_inputs += self.outer_kl_coeff

        extra_inputs += [self.anneal_coeff]
        self.anneal_coeff *= self.anneal_factor

        if log: logger.log("Computing loss before")
        loss_before = self.optimizer.loss(input_list, extra_inputs=extra_inputs)

        if log: logger.log("Optimizing")
        self.optimizer.optimize(input_list, extra_inputs=extra_inputs)

        if log: logger.log("Computing loss after")
        loss_after = self.optimizer.loss(input_list, extra_inputs=extra_inputs)

        inner_kls = self.optimizer.inner_kl(input_list, extra_inputs=extra_inputs)
        if self.adaptive_inner_kl_penalty:
            if log: logger.log("Updating inner KL loss coefficients")
            for step_inner_kl_coeff, step_inner_kls in zip(self.inner_kl_coeff, inner_kls):
                self._adapt_kl_coeff(step_inner_kl_coeff, step_inner_kls, self.target_inner_step)

        outer_kls = self.optimizer.outer_kl(input_list, extra_inputs=extra_inputs)
        if self.adaptive_outer_kl_penalty:
            if log: logger.log("Updating outer KL loss coefficients")
            self._adapt_kl_coeff(self.outer_kl_coeff, outer_kls, self.target_outer_step)

        if log:
            logger.logkv('LossBefore', loss_before)
            logger.logkv('LossAfter', loss_after)
            logger.logkv('dLoss', loss_before - loss_after)
            logger.logkv('klDiff', np.mean(inner_kls))
            logger.logkv('klCoeff', np.mean(self.inner_kl_coeff))
            if not self.clip_outer: logger.logkv('outerklDiff', np.mean(outer_kls))


    def _adapt_kl_coeff(self, kl_params, kl_values, kl_target):
        for i, kl in enumerate(kl_values):
            if kl < kl_target/1.5:
                kl_params[i] /= 2

            elif kl > kl_target * 1.5:
                kl_params[i] *= 2

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

