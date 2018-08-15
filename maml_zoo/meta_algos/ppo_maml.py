import tensorflow as tf
import numpy as np
from maml_zoo.logger import logger
from maml_zoo.meta_algos.base import MAMLAlgo
from maml_zoo.optimizers.maml_first_order_optimizer import MAMLPPOOptimizer

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
            max_epochs,
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
        self.step_sizes = None

        self.build_graph()

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
        dist = self.policy.distribution

        """ Create Variables """
        step_sizes, inner_kl_coeffs, anneal_ph, outer_kl_coeffs = self._create_opt_variables(scope=self.name)

        kl_coeffs = inner_kl_coeffs + [outer_kl_coeffs]
        self.step_sizes = step_sizes

        """ Inner update for test-time """
        obs_phs, action_phs, adv_phs, dist_info_phs = self.make_input_placeholders(prefix='init', scope=self.name)
        all_surr_objs, all_inputs, all_entropies, all_inner_kls = [], [], [], []
        surr_objs_test, surr_objs_train = [], []  # Used for computing fast gradient step
        distribution_info_vars, current_policy_params = [], []

        for i in range(self.meta_batch_size):
            # Train
            dist_info_var = self.policy.distribution_info_sym(obs_phs[i], params=None)
            distribution_info_vars.append(dist_info_var)
            current_policy_params.append(self.policy_params)

            # Test
            distribution_info_var_test = self.policy.distribution_info_sym(obs_phs[i], params=self.policies_params_ph[i])
            likelihood_ratio_test = dist.likelihood_ratio_sym(action_phs[i], dist_info_phs[i], distribution_info_var_test)
            surr_objs_test.append(-tf.reduce_mean(likelihood_ratio_test * adv_phs[i]))

        all_inputs += obs_phs + action_phs + adv_phs + sum(list(zip(*dist_info_phs)), []) # [obs_phs], [action_phs], [adv_phs], [dist_info1_ph], [dist_info2_ph], ...

        # For computing the fast update for sampling
        self.adapt_sym_test(all_inputs, surr_objs_test)

        """ Inner updates"""
        for j in range(self.num_inner_grad_steps):
            surr_objs, entropies, kls = [], [], []
            adapted_policy_params = []
            adapted_policy_dist_info_vars = []
            # Create objective for gradient step
            # TODO: Probably it'd be faster if we did this with a tf.map
            for i in range(self.meta_batch_size):
                output_inner_build = self._build_inner_objective(obs_phs[i], action_phs[i],
                                                                 adv_phs[i], dist_info_phs[i],
                                                                 distribution_info_vars[i], current_policy_params[i])

                kl_loss, entropy, surr_loss, dist_info_var, adapted_params_var = output_inner_build

                adapted_policy_dist_info_vars.append(dist_info_var)
                adapted_policy_params.append(adapted_params_var)
                entropies.append(entropy)
                kls.append(kl_loss)
                surr_objs.append(surr_loss)

            all_surr_objs.append(surr_objs)
            all_entropies.append(entropies)
            all_inner_kls.append(kls)

            # Update graph for next gradient step
            obs_phs, action_phs, adv_phs, dist_info_phs = self.make_input_placeholders(str(j), scope=self.name)

            current_policy_params = adapted_policy_params
            distribution_info_vars = adapted_policy_dist_info_vars

            all_inputs += obs_phs + action_phs + adv_phs + sum(list(zip(*dist_info_phs)), []) # [obs_phs], [action_phs], [adv_phs], [dist_info1_ph], [dist_info2_ph], ...

        """ Outer objective """
        surr_objs = []
        outer_kls = []

        for i in range(self.meta_batch_size):
            surr_obj, outer_kl = self._build_outer_objective(all_inner_kls[i], inner_kl_coeffs[i], all_entropies[i],
                                                             action_phs[i], adv_phs[i], dist_info_phs[i],
                                                             distribution_info_vars[i], anneal_ph, outer_kl_coeffs[i])
            surr_objs.append(surr_obj)
            outer_kls.append(outer_kl)

        """ Sum over meta tasks """
        meta_objective = tf.reduce_mean(tf.stack(surr_objs, 0))  # mean over meta_batch_size (the diff tasks)

        kl_coeffs = sum(kl_coeffs, [])
        extra_inputs = tuple(kl_coeffs) + (anneal_ph,)
        input_list = tuple(all_inputs)

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

        extra_inputs = tuple(self.inner_kl_coeff)
        if not self.clip_outer:
            extra_inputs += tuple(self.outer_kl_coeff)

        extra_inputs += (self.anneal_coeff,)
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
            self._adapt_kl_coeff(self.inner_kl_coeff, inner_kls, self.target_inner_step)

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

    def _build_inner_objective(self, obs_ph, action_ph, adv_ph, dist_info_ph, distribution_info_var,
                               current_policy_params):
        if self.entropy_bonus > 0:
            entropy = self.entropy_bonus * tf.reduce_mean(self.policy.dist.entropy_sym(distribution_info_var))
        else:  # Save a computation
            entropy = 0

        kl_loss = tf.reduce_mean(self.policy.distribution.kl_sym(dist_info_ph, distribution_info_var))
        likelihood_ratio = self.policy.distribution.likelihood_ratio_sym(action_ph, dist_info_ph, distribution_info_var)
        surr_loss = - tf.reduce_mean(likelihood_ratio * adv_ph)

        dist_info_var, adapted_params_var = self.adapt_sym(surr_loss, obs_ph, current_policy_params)

        return kl_loss, entropy, surr_loss, dist_info_var, adapted_params_var

    def _build_outer_objective(self, kl_penalties, inner_kl_coeffs, entropies_bonus, action_ph,
                               adv_ph, dist_info_ph, distribution_info_var, anneal_ph, outer_kl_coeffs):

        kl_penalty = sum(list(kl_penalties[j] * inner_kl_coeffs[j] for j in range(self.num_inner_grad_steps)))
        entropy_bonus = sum(list(entropies_bonus[j] for j in range(self.num_inner_grad_steps)))

        likelihood_ratio = self.policy.distribution.likelihood_ratio_sym(action_ph, dist_info_ph, distribution_info_var)
        if self.clip_outer:
            clipped_obj = tf.minimum(likelihood_ratio * adv_ph,
                                     tf.clip_by_value(likelihood_ratio,
                                                      1 - self.clip_eps * anneal_ph,
                                                      1 + self.clip_eps * anneal_ph) * adv_ph)
            surr_obj = - tf.reduce_mean(clipped_obj) - entropy_bonus + kl_penalty
            outer_kl = []

        else:
            outer_kl = tf.reduce_mean(self.policy.dist.kl_sym(dist_info_ph, distribution_info_var))
            outer_kl_penalty = outer_kl_coeffs[0] * outer_kl
            surr_obj = - tf.reduce_mean(likelihood_ratio * adv_ph) - entropy_bonus + \
                       kl_penalty + outer_kl_penalty

        return surr_obj, outer_kl

    def _adapt_kl_coeff(self, kl_params, kl_values, kl_target):
        for i, kl in enumerate(kl_values):
            if kl < kl_target/1.5:
                kl_params[i] /= 2

            elif kl > kl_target * 1.5:
                kl_params[i] *= 2

    def _create_opt_variables(self, scope):
        with tf.variable_scope(scope):
            # TODO: Make the option of having a trainable step size
            # Step sizes
            step_sizes = dict()
            for key, param in self.policy_params.items():
                shape = param.get_shape().as_list()
                init_stepsize = np.ones(shape, dtype=np.float32) * self.inner_lr
                step_sizes[key] = tf.Variable(initial_value=init_stepsize,
                                              name='%s_step_size' % key,
                                              dtype=tf.float32)

            # Inner KL coeffs
            inner_kl_coeffs = [list(tf.placeholder(tf.float32, shape=[], name='kl_%s_%s' % (j, i))
                                    for i in range(self.meta_batch_size))
                               for j in range(self.num_inner_grad_steps)]

            # Annealing ph
            anneal_ph = tf.placeholder(tf.float32, shape=[], name='clip_ph')

            # Outer KL coeffs
            outer_kl_coeffs = [list() for _ in range(self.meta_batch_size)]
            if not self.clip_outer:
                outer_kl_coeffs = [tf.placeholder(tf.float32, shape=[], name='kl_outer_%s' % i)
                                   for i in range(self.meta_batch_size)]

        return step_sizes, inner_kl_coeffs, anneal_ph, outer_kl_coeffs

