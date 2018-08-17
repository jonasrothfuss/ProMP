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

    def _build_inner_adaption(self, obs_phs, action_phs, adv_phs, dist_info_phs):
        """
        Creates the symbolic graph for the one-step inner gradient update (It'll be called several times if
        more gradient steps are needed)

        Args:
            some placeholders

        Returns:
            adapted_policies_params (list): list of Ordered Dict containing the symbolic post-update parameters
            adapt_input_list_ph (list): list of placeholders

        """
        adapted_policies_params = []

        self.lrs = [] #TODO remove
        self.advs = []
        self.grads = []
        self.params_vars = []

        for i in range(self.meta_batch_size):
            with tf.variable_scope("adapt_task_%i"%i):
                with tf.variable_scope("adapt_objective"):
                    # inner surrogate objective
                    with tf.variable_scope("likelihood_ratio"):
                        likelihood_ratio_adapt = self.policy.likelihood_ratio_sym(obs_phs[i], action_phs[i],
                                                                              dist_info_phs[i], self.policy.policies_params_phs[i])
                    with tf.variable_scope("surrogate_loss"):
                        surr_obj_adapt = -tf.reduce_mean(likelihood_ratio_adapt * adv_phs[i])

                self.lrs.append(likelihood_ratio_adapt) #TODO remove
                self.advs.append(adv_phs)
                self.params_vars.append(self.policy.policies_params_phs[i])

                # get tf operation for adapted (post-update) policy
                with tf.variable_scope("adapt_step"):
                    adapted_policy_param = self.adapt_sym(surr_obj_adapt, self.policy.policies_params_phs[i])
                adapted_policies_params.append(adapted_policy_param)

        return adapted_policies_params


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
        #TODO: check why we have an multiple inner kl coefs
        with tf.variable_scope(self.name):
            step_sizes, inner_kl_coeffs, anneal_ph, outer_kl_coeffs = self._create_opt_variables()

            kl_coeffs = inner_kl_coeffs + outer_kl_coeffs # TODO sth is wrong here
            self.step_sizes = step_sizes

            """ Prepare some stuff """
            obs_phs, action_phs, adv_phs, dist_info_phs, all_phs = self.make_input_placeholders('_adapt')
            all_surr_objs, all_inputs, all_entropies, all_inner_kls = [], [], [], []
            distribution_info_vars, current_policy_params = [], []

            """ --- Build inner update graph for adapting the policy and sampling trajectories --- """
            # this graph is only used for adapting the policy and not computing the meta-updates

            self.adapted_policies_params = self._build_inner_adaption(obs_phs, action_phs, adv_phs, dist_info_phs)
            self.adapt_input_list_ph = all_phs

            # """ ----- Build graph for the meta-update ----- """
            # all_inputs = all_phs  # [obs_phs, action_phs, adv_phs, dist_info1_ph, dist_info2_ph, ...]
            #
            # for i in range(self.meta_batch_size):
            #     # Train
            #     dist_info_var = self.policy.distribution_info_sym(obs_phs[i], params=None)
            #     distribution_info_vars.append(dist_info_var)
            #     current_policy_params.append(self.policy_params)
            #
            # """ Inner updates"""
            # for j in range(self.num_inner_grad_steps):
            #     surr_objs, entropies, kls = [], [], []
            #     adapted_policy_params = []
            #     adapted_policy_dist_info_vars = []
            #     # Create objective for gradient step
            #     for i in range(self.meta_batch_size):
            #         output_inner_build = self._build_inner_objective(obs_phs[i], action_phs[i],
            #                                                          adv_phs[i], dist_info_phs[i],
            #                                                          distribution_info_vars[i], current_policy_params[i])
            #
            #         kl_loss, entropy, surr_loss, dist_info_var, adapted_params_var = output_inner_build
            #
            #         adapted_policy_dist_info_vars.append(dist_info_var)
            #         adapted_policy_params.append(adapted_params_var)
            #         entropies.append(entropy)
            #         kls.append(kl_loss)
            #         surr_objs.append(surr_loss)
            #
            #     all_surr_objs.append(surr_objs)
            #     all_entropies.append(entropies)
            #     all_inner_kls.append(kls)
            #
            #     # Update graph for next gradient step
            #     obs_phs, action_phs, adv_phs, dist_info_phs, all_phs = self.make_input_placeholders(str(j), scope=self.name)
            #
            #     current_policy_params = adapted_policy_params
            #     distribution_info_vars = adapted_policy_dist_info_vars
            #
            #     all_inputs += all_phs # [obs_phs], [action_phs], [adv_phs], [dist_info1_ph], [dist_info2_ph], ...
            #
            # """ Outer objective """
            # surr_objs = []
            # outer_kls = []
            #
            # for i in range(self.meta_batch_size):
            #     all_inner_kls_i = [all_inner_kls[j][i] for j in range(self.num_inner_grad_steps)] # Todo: make this cleaner
            #     inner_kl_coeffs_i = [inner_kl_coeffs[j][i] for j in range(self.num_inner_grad_steps)]
            #     all_entropies_i = [all_entropies[j][i] for j in range(self.num_inner_grad_steps)]
            #     surr_obj, outer_kl = self._build_outer_objective(all_inner_kls_i, inner_kl_coeffs_i, all_entropies_i,
            #                                                      action_phs[i], adv_phs[i], dist_info_phs[i],
            #                                                      distribution_info_vars[i], anneal_ph, outer_kl_coeffs[i])
            #
            #     surr_objs.append(surr_obj)
            #     outer_kls.append(outer_kl)
            #
            # """ Sum over meta tasks """
            # meta_objective = tf.reduce_mean(tf.stack(surr_objs, 0))  # mean over meta_batch_size (the diff tasks)
            #
            # kl_coeffs = sum(kl_coeffs, [])
            # extra_inputs = kl_coeffs + [anneal_ph]
            # input_list = all_inputs
            #
            # self.optimizer.build_graph(
            #     loss=meta_objective,
            #     target=self.policy,
            #     inputs=input_list,
            #     extra_inputs=extra_inputs,
            #     inner_kl=all_inner_kls,
            #     outer_kl=outer_kls,
            # )

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

    def _build_inner_objective(self, obs_ph, action_ph, adv_ph, dist_info_ph, distribution_info_var,
                               current_policy_params):
        if self.entropy_bonus > 0:
            entropy = self.entropy_bonus * tf.reduce_mean(self.policy.distribution.entropy_sym(distribution_info_var))
        else:  # Save a computation
            entropy = 0

        kl_loss = tf.reduce_mean(self.policy.distribution.kl_sym(dist_info_ph, distribution_info_var))
        likelihood_ratio = self.policy.distribution.likelihood_ratio_sym(action_ph, dist_info_ph, distribution_info_var)
        surr_loss = - tf.reduce_mean(likelihood_ratio * adv_ph)

        adapted_params_var = self.adapt_sym(surr_loss, current_policy_params)
        dist_info_var = self.policy.distribution_info_sym(obs_ph, params=adapted_params_var)

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
            outer_kl = tf.reduce_mean(self.policy.distribution.kl_sym(dist_info_ph, distribution_info_var))
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

    def _create_opt_variables(self):
        # Step sizes
        with tf.variable_scope('inner_step_sizes'):
            step_sizes = dict()
            for key, param in self.policy.policy_params.items():
                shape = param.get_shape().as_list()
                init_stepsize = np.ones(shape, dtype=np.float32) * self.inner_lr
                step_sizes[key] = tf.Variable(initial_value=init_stepsize,
                                              name='%s_step_size' % key,
                                              dtype=tf.float32, trainable=self.trainable_inner_step_size)
        # Inner KL coeffs
        inner_kl_coeffs = [tf.placeholder(tf.float32, shape=[], name='kl_coef_%i'%j) for j in range(self.num_inner_grad_steps)]

        # Annealing ph
        anneal_ph = tf.placeholder(tf.float32, shape=[], name='clip_ph')

        # Outer KL coeffs
        outer_kl_coeffs = [list() for _ in range(self.meta_batch_size)]
        if not self.clip_outer:
            outer_kl_coeffs = [tf.placeholder(tf.float32, shape=[], name='kl_outer_%s' % i)
                               for i in range(self.meta_batch_size)]

        return step_sizes, inner_kl_coeffs, anneal_ph, outer_kl_coeffs

