class Distribution(object):
    """ 
    General methods for a generic distribution
    """
    @property
    def dim(self):
        raise NotImplementedError

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        """
        Compute the symbolic KL divergence of two distributions
        Args:
            old_dist_info_vars (placeholder) :
            new_dist_info_vars (placeholder) :
        Returns:
            (placeholder) : Symbolic representation of kl divergence (tensorflow op)
        """
        raise NotImplementedError

    def kl(self, old_dist_info, new_dist_info):
        """
        Compute the KL divergence of two distributions
        Args: 
            old_dist_info (dist_info)
            new_dist_info (dist_info)
        Returns:
            (float): kl divergence of distributions
        """
        raise NotImplementedError

    def likelihood_ratio_sym(self, x_var, old_dist_info_vars, new_dist_info_vars):
        raise NotImplementedError

    def likelihood_ratio(self, x_var, old_dist_info, new_dist_info):
        raise NotImplementedError

    def entropy_sym(self, dist_info_vars):
        raise NotImplementedError

    def entropy(self, dist_info):
        raise NotImplementedError

    def log_likelihood_sym(self, x_var, dist_info_vars):
        raise NotImplementedError

    def log_likelihood(self, xs, dist_info):
        raise NotImplementedError

    def sample(self, dist_info):
        """
        Draws a sample from the distribution
        Args:
            dist_info (obj) : an instantiation of this distribution 
        Returns:
            (obj): a sample drawn from this instantiation
        """
        raise NotImplementedError

    @property
    def dist_info_specs(self):
        raise NotImplementedError

    @property
    def dist_info_keys(self):
        return [k for k, _ in self.dist_info_specs]
