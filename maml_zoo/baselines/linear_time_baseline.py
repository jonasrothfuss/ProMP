from maml_zoo.baselines.base import Baseline
from maml_zoo.utils.serializable import Serializable
import numpy as np


class LinearTimeBaseline(Baseline):
    """
    Linear (polynomial) reward baseline model
    (see. Duan et al. 2016, "Benchmarking Deep Reinforcement Learning for Continuous Control", ICML)

    Fits the following linear model

    reward = b0 + b1*obs + b2*obs^2 + b3*t + b4*t^2+  b5*t^3

    Args:
        reg_coeff: list of paths

    """
    def __init__(self, reg_coeff=1e-5):
        super(LinearTimeBaseline, self).__init__()
        self._coeffs = None
        self._reg_coeff = reg_coeff

    def get_param_values(self, **tags):
        """
        Returns the parameter values of the baseline object

        Returns:
            numpy array of linear_regression coefficients

        """
        return self._coeffs

    def set_params(self, value, **tags):
        """
        Sets the parameter values of the baseline object

        Args:
            value: numpy array of linear_regression coefficients

        """
        self._coeffs = value

    def _features(self, path):
        # TODO: Should we add the option of normalizing the obs??
        path_length = len(path["discounted_rewards"])
        time_step = np.arange(path_length).reshape(-1, 1) / 100.0
        return np.concatenate([time_step, time_step ** 2, time_step ** 3, np.ones((path_length, 1))],
                              axis=1)

    def fit(self, paths):
        """
        Fits the linear baseline model with the provided paths via damped least squares

        Args:
            paths (list): list of paths

        """
        featmat = np.concatenate([self._features(path) for path in paths], axis=0)
        rewards = np.concatenate([path["discounted_rewards"] for path in paths], axis=0)
        reg_coeff = self._reg_coeff
        for _ in range(5):
            self._coeffs = np.linalg.lstsq(
                featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
                featmat.T.dot(rewards),
                rcond=None
            )[0]
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10

    def predict(self, path):
        """
        Predicts the linear reward baselines estimates for a provided trajectory / path.
        If the baseline is not fitted - returns zero baseline

        Args:
           path (dict): dict of lists/numpy array containing trajectory / path information
                 such as "observations", "rewards", ...

        Returns:
             (np.ndarray): numpy array of the same length as paths["observations"] specifying the reward baseline

        """
        if self._coeffs is None:
            return np.zeros(len(path["rewards"]))
        return self._features(path).dot(self._coeffs)