class Algo(object):

    def __init__(
            self,
            optimizer,
            inner_loss,
            ):
        self.optimizer = optimizer
        self.policy = None
        self.meta_batch_size = None
        self.num_grad_steps = None
        # (?) self.current_policy_params = [None] * meta_batch_size
        pass

    def build_graph(self, policy, meta_batch_size, num_grad_steps=1):
        """
        Creates computation graph
        Pseudocode:
        for task in meta_batch_size:
            make_vars
            init_dist_info_sym
        for step in num_grad_steps:
            for task in meta_batch_size:
                make_vars
                update_dist_info_sym
        set objectives for optimizer
        """
        self.policy = policy
        self.meta_batch_size = meta_batch_size
        self.num_grad_steps = num_grad_steps
        raise NotImplementedError

    def make_vars(self, prefix=''):
        """
        Args:
            prefix (str) : a string to prepend to the name of each variable
        Returns:
            (tuple) : a tuple containing lists of placeholders for each input type and meta task
        """
        raise NotImplementedError

    def init_dist_sym(self, obs_var, params_var, is_training=False):
        """
        Creates the symbolic representation of the current tf policy
        Args:
            obs_var (list) : list of obs placeholders split by env
            params_ph (dict) : dict of placeholders for initial policy params
            is_training (bool) : used for batch norm # (Do we support this?)
        Returns:
            (tf_op) : symbolic representation the policy's output for each obs
        """
        raise NotImplementedError

    def compute_updated_dists_sym(self, surr_obj, obs_var, params_var, is_training=False):
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
        raise NotImplementedError

    def compute_updated_dists(self, samples):
        """
        Performs MAML inner step for each task and stores resulting gradients # (in the policy?)
        Args:
            samples (list) : list of lists of samples (each is a dict) split by meta task
        Returns:
            None
        """
        raise NotImplementedError

    def optimize_policy(self, all_samples_data, log=True):
        """
        Performs MAML outer step for each task
        Args:
            all_samples_data (list) : list of lists of lists of samples (each is a dict) split by gradient update and meta task
            log (bool) : whether to log statistics
        Returns:
            None
        """
        raise NotImplementedError