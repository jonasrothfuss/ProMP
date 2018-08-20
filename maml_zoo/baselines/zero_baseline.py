from maml_zoo.baselines.base import Baseline
import numpy as np


class ZeroBaseline(Baseline):
    """
    Dummy baseline
    """

    def __init__(self):
        super(ZeroBaseline, self).__init__()

    def get_param_values(self, **kwargs):
        """
        Returns the parameter values of the baseline object

        Returns:
            (None): coefficients of the baseline

        """
        return None

    def set_param_values(self, value, **kwargs):
        """
        Sets the parameter values of the baseline object

        Args:
            value (None): coefficients of the baseline

        """
        pass

    def fit(self, paths, **kwargs):
        """
        Improves the quality of zeroes output by baseline

        Args:
            paths: list of paths

        """
        pass

    def predict(self, path):
        """
        Produces some zeroes

        Args:
            path (dict): dict of lists/numpy array containing trajectory / path information
                such as "observations", "rewards", ...

        Returns:
             (np.ndarray): numpy array of the same length as paths["observations"] specifying the reward baseline
                
        """
        return np.zeros_like(path["rewards"])