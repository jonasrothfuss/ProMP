class Trainer(object):
	def __init__(
			self,
			algo,
			env,
			sampler,
			policy,
			n_itr,
            meta_batch_size,
            num_grad_updates=1,
			scope=None,
			load_policy=None,
			):
		"""
		Args:
			algo (Algo) :
			env (Env) : # Do we need this?
			sampler (Sampler) : 
			policy (Policy) : # Do we need this?
			n_itr (int) : Number of iterations to train for
			meta_batch_size (int) : Number of meta tasks
			num_grad_updates (int) : Number of inner steps per maml iteration
			scope (str) : Scope for identifying the algorithm. Must be specified if running multiple algorithms
			load_policy (Policy) : Policy to reload from
		"""
		raise NotImplementedError

	def train(self):
		"""
		Trains policy on env using algo
		Pseudocode:
			algo.init_opt()
			for itr in n_itr:
				for step in num_grad_updates:
					sampler.sample()
					algo.compute_updated_dists()
				algo.optimize_policy()
				sampler.update_goals()

		"""
		raise NotImplementedError

	def start_worker(self):
		""" 
		What is this for?
		"""
		pass

	def terminate(self):
		"""
		What is this for?
		"""
		pass


    def get_itr_snapshot(self, itr):
    	"""
		Gets the current policy and env for storage
    	"""
        return dict(
            itr=itr,
            policy=self.policy,
            env=self.env,
        )