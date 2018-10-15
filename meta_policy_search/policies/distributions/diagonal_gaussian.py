import tensorflow as tf
import numpy as np
from meta_policy_search.policies.distributions.base import Distribution

class DiagonalGaussian(Distribution):
    """
    General methods for a diagonal gaussian distribution of this size 
    """
    def __init__(self, dim):
        self._dim = dim

    @property
    def dim(self):
        return self._dim

    def kl_sym(self, old_dist_info_vars, new_dist_info_vars):
        """
        Computes the symbolic representation of the KL divergence of two multivariate 
        Gaussian distribution with diagonal covariance matrices

        Args:
            old_dist_info_vars (dict) : dict of old distribution parameters as tf.Tensor
            new_dist_info_vars (dict) : dict of new distribution parameters as tf.Tensor

        Returns:
            (tf.Tensor) : Symbolic representation of kl divergence (tensorflow op)
        """
        old_means = old_dist_info_vars["mean"]
        old_log_stds = old_dist_info_vars["log_std"]
        new_means = new_dist_info_vars["mean"]
        new_log_stds = new_dist_info_vars["log_std"]

        # assert ranks
        tf.assert_rank(old_means, 2), tf.assert_rank(old_log_stds, 2)
        tf.assert_rank(new_means, 2), tf.assert_rank(new_log_stds, 2)

        old_std = tf.exp(old_log_stds)
        new_std = tf.exp(new_log_stds)

        numerator = tf.square(old_means - new_means) + \
                    tf.square(old_std) - tf.square(new_std)
        denominator = 2 * tf.square(new_std) + 1e-8
        return tf.reduce_sum(
            numerator / denominator + new_log_stds - old_log_stds, reduction_indices=-1)

    def kl(self, old_dist_info, new_dist_info):
        """
        Compute the KL divergence of two multivariate Gaussian distribution with
        diagonal covariance matrices

       Args:
            old_dist_info (dict): dict of old distribution parameters as numpy array
            new_dist_info (dict): dict of new distribution parameters as numpy array

        Returns:
            (numpy array): kl divergence of distributions
        """
        old_means = old_dist_info["mean"]
        old_log_stds = old_dist_info["log_std"]
        new_means = new_dist_info["mean"]
        new_log_stds = new_dist_info["log_std"]

        old_std = np.exp(old_log_stds)
        new_std = np.exp(new_log_stds)
        numerator = np.square(old_means - new_means) + \
                    np.square(old_std) - np.square(new_std)
        denominator = 2 * np.square(new_std) + 1e-8
        return np.sum(
            numerator / denominator + new_log_stds - old_log_stds, axis=-1)

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
        with tf.variable_scope("log_li_new"):
            logli_new = self.log_likelihood_sym(x_var, new_dist_info_vars)
        with tf.variable_scope("log_li_old"):
            logli_old = self.log_likelihood_sym(x_var, old_dist_info_vars)
        return tf.exp(logli_new - logli_old)

    def log_likelihood_sym(self, x_var, dist_info_vars):
        """
        Symbolic log likelihood log p(x) of the distribution

        Args:
            x_var (tf.Tensor): variable where to evaluate the log likelihood
            dist_info_vars (dict) : dict of distribution parameters as tf.Tensor

        Returns:
             (numpy array): log likelihood
        """
        means = dist_info_vars["mean"]
        log_stds = dist_info_vars["log_std"]

        # assert ranks
        tf.assert_rank(x_var, 2), tf.assert_rank(means, 2), tf.assert_rank(log_stds, 2)

        zs = (x_var - means) / tf.exp(log_stds)
        return - tf.reduce_sum(log_stds, reduction_indices=-1) - \
               0.5 * tf.reduce_sum(tf.square(zs), reduction_indices=-1) - \
               0.5 * self.dim * np.log(2 * np.pi)

    def log_likelihood(self, xs, dist_info):
        """
        Compute the log likelihood log p(x) of the distribution

        Args:
           x_var (numpy array): variable where to evaluate the log likelihood
           dist_info_vars (dict) : dict of distribution parameters as numpy array

        Returns:
            (numpy array): log likelihood
        """
        means = dist_info["mean"]
        log_stds = dist_info["log_std"]
        zs = (xs - means) / np.exp(log_stds)
        return - np.sum(log_stds, axis=-1) - \
               0.5 * np.sum(np.square(zs), axis=-1) - \
               0.5 * self.dim * np.log(2 * np.pi)

    def entropy_sym(self, dist_info_vars):
        """
        Symbolic entropy of the distribution

        Args:
            dist_info (dict) : dict of distribution parameters as tf.Tensor

        Returns:
            (tf.Tensor): entropy
        """
        log_stds = dist_info_vars["log_std"]
        return tf.reduce_sum(log_stds + np.log(np.sqrt(2 * np.pi * np.e)), reduction_indices=-1)

    def entropy(self, dist_info):
        """
        Compute the entropy of the distribution

        Args:
            dist_info (dict) : dict of distribution parameters as numpy array

        Returns:
          (numpy array): entropy
        """
        log_stds = dist_info["log_std"]
        return np.sum(log_stds + np.log(np.sqrt(2 * np.pi * np.e)), axis=-1)

    def sample(self, dist_info):
        """
        Draws a sample from the distribution

        Args:
           dist_info (dict) : dict of distribution parameter instantiations as numpy array

        Returns:
           (obj): sample drawn from the corresponding instantiation
        """
        means = dist_info["mean"]
        log_stds = dist_info["log_std"]
        rnd = np.random.normal(size=means.shape)
        return rnd * np.exp(log_stds) + means

    @property
    def dist_info_specs(self):
        return [("mean", (self.dim,)), ("log_std", (self.dim,))]
