from maml_zoo.meta_algos.base import MAMLAlgo

class MAMLPPO(MAMLAlgo):
    """
    Algorithm for TRPO MAML
    Args:
        optimizer (Optimizer) : Optimizer to use
        inner_lr (float) : gradient step size used for inner step
        clip_eps (float) :
        clip_outer (bool) : whether to use L^CLIP or L^KLPEN on outer gradient update
        target_outer_step (float) : target outer kl divergence, used only with L^KLPEN and when adaptive_outer_kl_penalty is true
        target_inner_step (float) : target inner kl divergence, used only when adaptive_inner_kl_penalty is true
        init_outer_kl_penalty (float) : initial penalty for outer kl, used only with L^KLPEN
        init_inner_kl_penalty (float) : initial penalty for inner kl
        adaptive_outer_kl_penalty (bool): whether to used a fixed or adaptive kl penalty on outer gradient update
        adaptive_inner_kl_penalty (bool): whether to used a fixed or adaptive kl penalty on inner gradient update
        anneal_factor (float) : multiplicative factor for clip_eps, updated every iteration
        multi_adam (bool) : whether to keep separate momentum separately for each PPO step
        num_inner_grad_steps (int) : number of gradient updates taken per maml iteration
        entropy_bonus (float) : scaling factor for policy entropy
    """
    def __init__(
            self,
            optimizer,
            inner_lr,
            clip_eps=0.2, 
            clip_outer=True,
            target_outer_step=0.001,
            target_inner_step=0.01,
            init_outer_kl_penalty=1e-3,
            init_inner_kl_penalty=1e-2,
            adaptive_outer_kl_penalty=True,
            adaptive_inner_kl_penalty=True,
            anneal_factor=1,
            multi_adam=False,
            num_inner_grad_steps=1,
            entropy_bonus=0,
            ):
        
        super(MAMLPPO, self).__init__(optimizer, inner_lr, num_inner_grad_steps, entropy_bonus)
        self.optimizer = optimizer
        self.clip_eps = clip_eps
        self.clip_outer = clip_outer
        self.target_outer_step  = target_outer_step
        self.target_inner_step = target_inner_step
        self.adaptive_outer_kl_penalty = adaptive_outer_kl_penalty
        self.adaptive_inner_kl_penalty = adaptive_inner_kl_penalty
        self.kl_coeff = [init_inner_kl_penalty] * self.meta_batch_size * self.num_inner_grad_steps
        self.outer_kl_coeff = [init_outer_kl_penalty] * self.meta_batch_size
        self.anneal_coeff = 1
        self.anneal_factor = anneal_factor
        self._optimization_keys = ['observations', 'actions', 'advantages', 'agent_infos']

    def build_graph(self, policy, meta_batch_size):
        """
        Creates computation graph
        Args:
            policy (Policy) : policy for this algorithm
            meta_batch_size (int) : number of metalearning tasks
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
        self.meta_batch_size = meta_batch_size
        self.policy = policy
        dist = policy.distribution

        all_surr_objs, input_list = [], []
        entropy_list = [] # Total entropy across all gradient steps
        kl_list = []

        obs_phs, action_phs, adv_phs, dist_info_phs = self.make_placeholders('init')
        _surr_objs_ph = [] # Used for computing fast gradient step
        dist_info_vars_list, new_params = [], []

        for i in range(self.meta_batch_size):    
            dist_info_vars, params = self.init_dist_sym(obs_phs[i], all_params=self.all_params)
            dist_info_vars_list.append(dist_info_vars)
            new_params.append(params)
            _dist_info_vars, _ = self.init_dist_sym(obs_phs[i], all_params=self.all_params_ph[i])
            _lr = dist.likelihood_ratio_sym(action_phs[i], dist_info_phs[i], _dist_info_vars)
            _surr_objs_ph.append(-tf.reduce_mean(_lr * adv_phs[i]))


        input_list += obs_phs + action_phs + adv_phs + sum(list(zip(*dist_info_phs)), []) # [obs_phs], [action_phs], [adv_phs], [dist_info1_ph], [dist_info2_ph], ...
        # For computing the fast update for sampling
        self.set_inner_obj(input_list, _surr_objs_ph)
        
        for j in range(self.num_inner_grad_steps):
            surr_objs = []
            entropies = []
            kls = []
            # Create objective for gradient step
            for i in range(self.meta_batch_size):
                if self.entropy_bonus > 0:
                    entropy = self.entropy_bonus * tf.reduce_mean(dist.entropy_sym(dist_info_vars_list[i]))
                else: # Save a computation
                    entropy = 0
                kls.append(tf.reduce_mean(dist.kl_sym(dist_info_phs[i], dist_info_vars_list[i])))
                entropies.append(entropy)

                lr = dist.likelihood_ratio_sym(action_phs[i], dist_info_phs[i], dist_info_vars_list[i])
                surr_objs.append(- tf.reduce_mean(lr * adv_phs[i]))
            
            all_surr_objs.append(surr_objs)
            entropy_list.append(entropies)
            
            # Update graph for next gradient step
            obs_phs, action_phs, adv_phs, dist_info_phs = self.make_placeholders(str(j))

            cur_params = new_params
            new_params = []  # if there are several grad_updates the new_params are overwritten

            for i in range(self.meta_batch_size):
                dist_info_vars, params = self.compute_updated_dist_sym(i, surr_objs[i], obs_phs[i],
                                                                            params_dict=cur_params[i])
                dist_info_vars_list[i] = dist_info_vars
                new_params.append(params)

            input_list += obs_phs + action_phs + adv_phs + sum(list(zip(*dist_info_phs)), []) # [obs_phs], [action_phs], [adv_phs], [dist_info1_ph], [dist_info2_ph], ...         
            
        """ TRPO-Objective """
        surr_objs = []
        kls = []

        kl_coeff_vars_list = list(list(tf.placeholder(tf.float32, shape=[], name='kl_%s_%s' % (j, i))
                                  for i in range(self.meta_batch_size)) for j in range(self.num_inner_grad_steps))
        outer_kl_list = []
        if not self.clip_outer:
            outer_kl_coeff_vars = [list(tf.placeholder(tf.float32, shape=[], name='kl_outer_%s' % i) for i in range(self.meta_batch_size))]
            kl_coeff_vars_list += outer_kl_coeff_vars
        
        anneal_ph = tf.placeholder(tf.float32, shape=[], name='clip_ph')

        for i in range(self.meta_batch_size):
            kl_penalty = sum(list(kl_list[j][i] * kl_coeff_vars_list[j][i] for j in range(self.num_inner_grad_steps)))
            entropy_bonus = sum(list(entropy_list[j][i] for j in range(self.num_inner_grad_steps)))

            lr = dist.likelihood_ratio_sym(action_phs[i], dist_info_phs[i], dist_info_vars_list[i])

            if self.clip_outer:
                clipped_obj = tf.minimum(lr * adv_vars[i], tf.clip_by_value(lr, 1 - self.clip_eps * anneal_ph, 1 + self.clip_eps * anneal_ph) * adv_vars[i])
                surr_objs.append(- tf.reduce_mean(clipped_obj) - entropy_bonus + kl_penalty)
            else:
                outer_kl = tf.reduce_mean(dist.kl_sym(old_dist_info_vars[i], dist_info_vars))
                outer_kl_penalty = outer_kl_coeff_vars[0][i] * outer_kl
                surr_objs.append(- tf.reduce_mean(lr * adv_vars[i]) - entropy_bonus + kl_penalty + outer_kl_penalty)
                outer_kl_list.append(outer_kl)

        """ Sum over meta tasks """        
        meta_obj = tf.reduce_mean(tf.stack(surr_objs, 0))  # mean over meta_batch_size (the diff tasks)

        kl_coeff_vars_list = sum(kl_coeff_vars_list, [])
        extra_inputs = tuple(kl_coeff_vars_list) + (anneal_ph,)
        self.optimizer.update_opt(
            loss=surr_obj,
            target=self.policy,
            inputs=input_list,
            extra_inputs=extra_inputs,
            inner_kl=kl_list,
            outer_kl=outer_kl_list,
            meta_batch_size=self.meta_batch_size,
            num_grad_updates=self.num_inner_grad_steps,
        )

    def optimize_policy(self, all_samples_data, log=True):
        """
        Performs MAML outer step for each task
        Args:
            all_samples_data (list) : list of lists of lists of samples (each is a dict) split by gradient update and meta task
            log (bool) : whether to log statistics
        Returns:
            None
        """
        assert len(all_samples_data) == self.num_inner_grad_steps + 1  # we collected the rollouts to compute the grads and then the test!

        input_list = []
        for step in range(len(all_samples_data)):  # these are the gradient steps
            obs_list, action_list, adv_list, dist_info_list = [], [], [], []
            for i in range(self.meta_batch_size):

                inputs = ext.extract(
                    all_samples_data[step][i], *self._optimization_keys
                )
                obs_list.append(inputs[0])
                action_list.append(inputs[1])
                adv_list.append(inputs[2])
                dist_info_list.append([inputs[3][k] for k in self.policy.distribution.dist_info_keys])

            input_list += obs_list + action_list + adv_list + sum(list(zip(*dist_info_list)), [])  # [ [obs_0], [act_0], [adv_0], [dist1_0], [dist2_0], [obs_1], ... ]

        extra_inputs = tuple(self.kl_coeff)
        if not self.clip_outer:
            extra_inputs += tuple(self.outer_kl_coeff)

        extra_inputs += tuple(self.anneal_coeff)
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
            for i, kl in enumerate(inner_kls):
                if kl < self.target_inner_step / 1.5:
                    self.kl_coeff[i] /= 2
                if kl > self.target_inner_step * 1.5:
                    self.kl_coeff[i] *= 2
            
        outer_kls = self.optimizer.outer_kl(input_list, extra_inputs=extra_inputs)
        if self.adaptive_outer_kl_penalty:
            if log: logger.log("Updating outer KL loss coefficients")
            for i, kl in enumerate(outer_kls):
                if kl < self.target_outer_step / 1.5:
                    self.outer_kl_coeff[i] /= 2
                if kl > self.target_outer_step * 1.5:
                    self.outer_kl_coeff[i] *= 2

        if log:
            logger.record_tabular('LossBefore', loss_before)
            logger.record_tabular('LossAfter', loss_after)
            logger.record_tabular('dLoss', loss_before - loss_after)
            logger.record_tabular('klDiff', np.mean(inner_kls))
            logger.record_tabular('klCoeff', np.mean(self.kl_coeff))
            if not self.clip_outer: logger.record_tabular('outerklDiff', np.mean(outer_kls))