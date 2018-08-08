class Algo(object):

    def __init__(
            self,
            optimizer,
            meta_batch_size,
            ):
        self.optimizer = optimizer
        self.current_policy_params = [None] * meta_batch_size
        pass

    def make_vars(self, stepnum='0'):
        raise NotImplementedError

    def init_opt(self):
        raise NotImplementedError

    def init_dist_info_sym(self):
        """

        """

    def update_dist_info_sym(self):
        """

        """

    def compute_updated_dists(self, samples):
        """
        Performs MAML inner step for each task
        Args:
            samples (list) : tbd
        Returns:
            None
        """
        raise NotImplementedError

    def optimize_policy(self, all_samples_data, log=True):
        """
        Performs MAML outer step for each task
        Args:
            all_samples_data (list) : list of lists of samples split by meta task
            log (bool) : whether to log statistics
        Returns:
            None
        """
        raise NotImplementedError


    def get_itr_snapshot(self, itr):
        return dict(
            itr=itr,
            policy=self.policy,
            env=self.env,
        )