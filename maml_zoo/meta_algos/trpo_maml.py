from maml_zoo.meta_algos.base import MAMLAlgo

class MAMLTRPO(MAMLAlgo):
	"""
	Algorithm for TRPO MAML
	Args:
		optimizer (Optimizer) : Optimizer to use
		inner_step_Size (float) : maximum kl divergence of inner step
		inner_type (str) : log_likelihood, likelihood_ratio, or dice
		num_inner_grad_steps (int) : number of gradient updates taken per maml iteration
	"""
    def __init__(
            self,
            optimizer,
            inner_lr,
            inner_step_size,
            inner_type,
            num_inner_grad_steps=1,
            ):
        
        assert inner_type in ["log_likelihood", "likelihood_ratio", "dice"]
        super(MAMLTRPO, self).__init__(optimizer, inner_lr, num_inner_grad_steps)
        self.inner_step_size = inner_step_size
        self.inner_type = inner_type
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
        new_params = None
        for j in range(self.num_inner_grad_steps):
            obs_phs, action_phs, adv_phs, dist_info_phs = self.make_placeholders(str(j))
            surr_objs = []

            cur_params = new_params
            new_params = []  # if there are several grad_updates the new_params are overwritten
            kls = []
            entropies = []
            _surr_objs_ph = [] # Used for computing fast gradient step

            for i in range(self.meta_batch_size):
                if j == 0:
                    dist_info_vars, params = self.init_dist_sym(obs_phs[i], all_params=self.all_params)
                else:
                    dist_info_vars, params = self.compute_updated_dist_sym(i, all_surr_objs[-1][i], obs_phs[i],
                                                                               params_dict=cur_params[i])
                if self.entropy_bonus > 0:
                    entropy = self.entropy_bonus * tf.reduce_mean(dist.entropy_sym(dist_info_vars))
                else: # Save a computation
                    entropy = 0
                entropies.append(entropy)

                new_params.append(params)
                
                if inner_type == 'log_likelihood':
                	logli = dist.log_likelihood_sym(action_phs[i], dist_info_vars)
                	surr_objs.append(- tf.reduce_mean(logli * adv_phs[i]))
                	if j == 0:
                    	_dist_info_vars, _ = self.init_dist_sym(obs_phs[i], all_params=self.all_params_ph[i])
                    	_logli = dist.log_likelihood_sym(action_phs[i], _dist_info_vars)
                    	_surr_objs_ph.append(-tf.reduce_mean(_logli * adv_phs[i]))

                elif inner_type == 'likelihood_ratio':
                	lr = dist.likelihood_ratio_sym(action_phs[i], dist_info_phs[i], dist_info_vars)
                	surr_objs.append(- tf.reduce_mean(lr * adv_phs[i]))
                	if j == 0:
                    	_dist_info_vars, _ = self.init_dist_sym(obs_phs[i], all_params=self.all_params_ph[i])
                    	_lr = dist.likelihood_ratio_sym(action_phs[i], dist_info_phs[i], _dist_info_vars)
                    	_surr_objs_ph.append(-tf.reduce_mean(_lr * adv_phs[i]))
                else:
                	raise NotImplementedError

            input_list += obs_phs + action_phs + adv_phs + sum(list(zip(*dist_info_phs)), []) # [obs_phs], [action_phs], [adv_phs], [dist_info1_ph], [dist_info2_ph], ...
            if j == 0:
                # For computing the fast update for sampling
                self.set_inner_obj(input_list, _surr_objs_ph)
                self._update_input_keys = self._optimization_keys
                init_input_list = input_list

            all_surr_objs.append(surr_objs)
            entropy_list.append(entropies)

        """ TRPO-Objective """
        obs_phs, action_phs, adv_phs, dist_info_phs = self.make_vars('test')
        surr_objs = []

        for i in range(self.meta_batch_size):
            dist_info_vars, _ = self.compute_updated_dist_sym(i, all_surr_objs[-1][i], obs_phs[i], params_dict=new_params[i])

            kl = dist.kl_sym(dist_info_phs[i], dist_info_vars)
            kls.append(kl)
            
            lr = dist.likelihood_ratio_sym(action_phs[i], dist_info_phs[i], dist_info_vars)
            surr_objs.append(- tf.reduce_mean(lr*adv_phs[i]
                             + sum(list(entropy_list[j][i] for j in range(self.num_inner_grad_steps)))))

        """ Sum over meta tasks """
        surr_obj = tf.reduce_mean(tf.stack(surr_objs, 0))  # mean over meta_batch_size (the diff tasks)
        input_list += obs_phs + action_phs + adv_phs + sum(list(zip(*dist_info_phs)), [])

        mean_kl = tf.reduce_mean(tf.concat(kls, 0))  ##CF shouldn't this have the option of self.kl_constrain_step == -1?
        max_kl = tf.reduce_max(tf.concat(kls, 0))

        self.optimizer.update_opt(
            loss=surr_obj,
            target=self.policy,
            leq_constraint=(mean_kl, self.inner_step_size),
            inputs=input_list,
            constraint_name="mean_kl"
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
        assert len(all_samples_data) == self.num_grad_updates + 1  # we collected the rollouts to compute the grads and then the test!

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

        logger.log("Computing KL before")
        mean_kl_before = self.optimizer.constraint_val(input_list)

        logger.log("Computing loss before")
        loss_before = self.optimizer.loss(input_list)
        logger.log("Optimizing")
        self.optimizer.optimize(input_list)
        logger.log("Computing loss after")
        loss_after = self.optimizer.loss(input_list)
        
        logger.log("Computing KL after")
        mean_kl = self.optimizer.constraint_val(input_list)
        logger.record_tabular('MeanKLBefore', mean_kl_before)  # this now won't be 0!
        logger.record_tabular('MeanKL', mean_kl)

        logger.record_tabular('LossBefore', loss_before)
        logger.record_tabular('LossAfter', loss_after)
        logger.record_tabular('dLoss', loss_before - loss_after)