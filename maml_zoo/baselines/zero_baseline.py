from maml_zoo.baselines.base import Baseline
import numpy as np

class ZeroBaseline(Baseline):
    """
    Dummy baseline
    """

    def __init__(self):
        pass

    @overrides
    def get_param_values(self, **kwargs):
        """
        Returns the parameter values of the baseline object

        Returns:
            numpy array of linear_regression coefficients

        """
        return None

    @overrides
    def set_param_values(self, val, **kwargs):
        """
        Sets the parameter values of the baseline object

        Args:
            value: numpy array of linear_regression coefficients

        """
        pass

    @overrides
    def fit(self, paths, **kwargs):
        """
        Improves the quality of zeroes output by baseline

        Args:
            paths: list of paths

        """
        pass

    @overrides
    def predict(self, path):
        """
        Produces some zeroes

        Args:
            path: dict of lists/numpy array containing trajectory / path information
                such as "observations", "rewards", ...
                
        """
        return np.zeros_like(path["rewards"])