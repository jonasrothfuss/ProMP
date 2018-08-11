from maml_zoo.meta_algos.base import Algo

class MAMLTRPO(Algo):
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

    def make_placeholders(self, prefix=''):
        """
        Args:
            prefix (str) : a string to prepend to the name of each variable
        Returns:
            (tuple) : a tuple containing lists of placeholders for each input type and meta task
        """
        obs_phs, action_phs, adv_phs, dist_info_phs = [], [], [], []

        dist_info_specs = self.policy.distribution.dist_info_specs

        for i in range(self.meta_batch_size):
        	obs_phs.append(tf.placeholder(
				dtype=tf.float32, 
				shape=[None, np.prod(self.env.observation_space.shape)], 
				name='obs' + prefix + '_' + str(i)
			))
            action_phs.append(tf.placeholder(
            	dtype=tf.float32,
                shape=[None, np.prod(self.env.action_space.shape)],
                name='action' + prefix + '_' + str(i),
            ))
            adv_phs.append(tf.placeholder(
            	dtype=tf.float32, 
            	shape=[None], 
            	name='advantage' + prefix + '_' + str(i),
            ))
            dist_info_phs.append([tf.placeholder(
            	dtype=tf.float32, 
            	shape=[None] + list(shape), name='%s%s_%i' % (k, prefix, i))
            	for k, shape in dist_info_specs
        	])
        return obs_phs, action_phs, adv_phs, dist_info_phs

    def init_dist_sym(self, obs_var, params_phs, is_training=False):
        """
        Creates the symbolic representation of the current tf policy
        Args:
            obs_var (list) : list of obs placeholders split by env
            params_ph (dict) : dict of placeholders for initial policy params
            is_training (bool) : used for batch norm # (Do we support this?)
        Returns:
            (tf_op) : symbolic representation the policy's output for each obs
        """
        # return_params = True
        # if all_params is None:
        #     return_params = False
        #     all_params = self.all_params

        return self.policy._forward(obs_var, all_params, is_training), all_params

    def compute_updated_dist_sym(self, surr_obj, obs_var, params_phs, is_training=False):
        """
        Creates the symbolic representation of the tf policy after one gradient step towards the surr_obj
        Args:
            surr_obj (tf_op) : tensorflow op for task specific (inner) objective
            obs_var (list) : list of obs placeholders split by env
            params_ph (dict) : dict of placeholders for current policy params
            is_training (bool) : used for batch norm # (Do we support this?)
        Returns:
            (tf_op) : symbolic representation the policy's output for each obs
        """
        update_param_keys = params_phs.keys()

        grads = tf.gradients(surr_obj, [params_phs[key] for key in update_param_keys])

        gradients = dict(zip(update_param_keys, grads))
        params_dict = dict(zip(update_param_keys, [
            old_params_dict[key] - tf.multiply(self.param_step_sizes[key + "_step_size"], gradients[key]) for key in
            update_param_keys]))

        return self.init_dist_sym(obs_var, all_params=params_dict, is_training=is_training)

    def compute_updated_dists(self, samples):
        """
        Performs MAML inner step for each task and performs an update with the resulting gradients
        Args:
            samples (list) : list of lists of samples (each is a dict) split by meta task
        Returns:
            None
        """
		sess = tf.get_default_session()
        num_tasks = len(samples)
        assert num_tasks == self.meta_batch_size
        input_list = list([] for _ in range(len(self.update_input_keys)))
        for i in range(num_tasks):
            inputs = ext.extract(
                all_samples_data[step][i], *self._optimization_keys
            )
            obs_list.append(inputs[0])
            action_list.append(inputs[1])
            adv_list.append(inputs[2])
            dist_info_list.append([inputs[3][k] for k in self.policy.distribution.dist_info_keys])

		input_list += obs_list + action_list + adv_list + sum(list(zip(*dist_info_list)))

        feed_dict_inputs = list(zip(self.input_list_for_grad, inputs))
        feed_dict_params = list((self.all_params_ph[i][key], self.all_param_vals[i][key])
                                for i in range(num_tasks) for key in self.all_params_ph[0].keys())
        feed_dict = dict(feed_dict_inputs + feed_dict_params)
        new_param_vals, gradients = sess.run([self.all_fast_params_tensor, self._all_param_gradients], feed_dict=feed_dict)
        self.policy.update_task_parameters(new_param_vals)

    def set_inner_obj(self, input_list, surr_objs_tensor):
        self.input_list_for_grad = input_list
    	self.surr_objs = surr_objs_tensor
    	update_param_keys = self.all_params.keys()
        with tf.variable_scope(self.name):
            # Create the symbolic graph for the one-step inner gradient update (It'll be called several times if
            # more gradient steps are needed
            for i in range(self.num_tasks):
                # compute gradients for a current task (symbolic)
                gradients = dict(zip(update_param_keys, tf.gradients(self.surr_objs[i],
                                                                     [self.all_params_ph[i][key] for key in update_param_keys]
                                                                     )))

                # gradient update for params of current task (symbolic)
                fast_params_tensor = OrderedDict(zip(update_param_keys,
                                                     [self.all_params_ph[i][key] - tf.multiply(
                                                         self.param_step_sizes[key + "_step_size"], gradients[key]) for
                                                      key in update_param_keys]))

                # tensors that represent the updated params for all of the tasks (symbolic)
                self.all_fast_params_tensor.append(fast_params_tensor)
                self._all_param_gradients.append(gradients)

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