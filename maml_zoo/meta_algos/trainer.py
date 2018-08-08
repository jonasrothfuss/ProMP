class Trainer(object):
	def __init__(
			self,
			algo,
			env,
			policy,
			n_itr=500,
            meta_batch_size=20,
            num_grad_updates=1,
			scope=None,
			load_policy=None,
			):
		"""
		Args:
			algo (Algo) :
			env (Env) : 
			Policy (Policy) :
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