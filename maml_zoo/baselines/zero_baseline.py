from maml_zoo.baselines.base import Baseline
import numpy as np

class ZeroBaseline(Baseline):

    def __init__(self, env_spec):
        pass

    @overrides
    def get_param_values(self, **kwargs):
        return None

    @overrides
    def set_param_values(self, val, **kwargs):
        pass

    @overrides
    def fit(self, paths, **kwargs):
        pass

    @overrides
    def predict(self, path):
        return np.zeros_like(path["rewards"])
