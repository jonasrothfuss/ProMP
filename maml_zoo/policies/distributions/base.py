class Distribution(object):
    """ 
    General methods for a generic distribution
    """
    @property
    def dim(self):
        raise NotImplementedError

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        """
        Symbolic KL divergence of two distributions

        Args:
            old_dist_info_vars (dict) : dict of old distribution parameters as tf.Tensor
            new_dist_info_vars (dict) : dict of new distribution parameters as tf.Tensor

        Returns:
            (tf.Tensor) : Symbolic representation of kl divergence (tensorflow op)
        """
        raise NotImplementedError

    def kl(self, old_dist_info, new_dist_info):
        """
        Compute the KL divergence of two distributions

        Args: 
            old_dist_info (dict): dict of old distribution parameters as numpy array
            new_dist_info (dict): dict of new distribution parameters as numpy array

        Returns:
            (numpy array): kl divergence of distributions
        """
        raise NotImplementedError

    def likelihood_ratio_sym(self, x_var, old_dist_info_vars, new_dist_info_vars):
        """
        Symbolic likelihood ratio p_new(x)/p_old(x) of two distributions

        Args:
            x_var (tf.Tensor): variable where to evaluate the likelihood ratio p_new(x)/p_old(x)
            old_dist_info_vars (dict) : dict of old distribution parameters as tf.Tensor
            new_dist_info_vars (dict) : dict of new distribution parameters as tf.Tensor

        Returns:
          (tf.Tensor): likelihood ratio
        """
        raise NotImplementedError

    def likelihood_ratio(self, x_var, old_dist_info, new_dist_info):
        """
        Compute the likelihood ratio p_new(x)/p_old(x) of two distributions

        Args:
            x_var (numpy array): variable where to evaluate the likelihood ratio p_new(x)/p_old(x)
            old_dist_info_vars (dict) : dict of old distribution parameters as numpy array
            new_dist_info_vars (dict) : dict of new distribution parameters as numpy array

        Returns:
          (numpy array): likelihood ratio
        """
        raise NotImplementedError

    def entropy_sym(self, dist_info_vars):
        """
        Symbolic entropy of the distribution

        Args:
            dist_info (dict) : dict of distribution parameters as tf.Tensor

        Returns:
            (tf.Tensor): entropy
        """
        raise NotImplementedError

    def entropy(self, dist_info):
        """
        Compute the entropy of the distribution

        Args:
            dist_info (dict) : dict of distribution parameters as numpy array

        Returns:
          (numpy array): entropy
        """
        raise NotImplementedError

    def log_likelihood_sym(self, x_var, dist_info_vars):
        """
        Symbolic log likelihood log p(x) of the distribution

        Args:
            x_var (tf.Tensor): variable where to evaluate the log likelihood
            dist_info_vars (dict) : dict of distribution parameters as tf.Tensor

        Returns:
             (numpy array): log likelihood
        """
        raise NotImplementedError

    def log_likelihood(self, xs, dist_info):
        """
        Compute the log likelihood log p(x) of the distribution

        Args:
           x_var (numpy array): variable where to evaluate the log likelihood
           dist_info_vars (dict) : dict of distribution parameters as numpy array

        Returns:
            (numpy array): log likelihood
        """
        raise NotImplementedError

    def sample(self, dist_info):
        """
        Draws a sample from the distribution

        Args:
            dist_info (dict) : dict of distribution parameter instantiations as numpy array

        Returns:
            (obj): sample drawn from the corresponding instantiation
        """
        raise NotImplementedError

    @property
    def dist_info_specs(self):
        raise NotImplementedError

    @property
    def dist_info_keys(self):
        return [k for k, _ in self.dist_info_specs]
