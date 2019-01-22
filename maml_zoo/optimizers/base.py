from maml_zoo import utils

class Optimizer(object):
    def __init__(self):
        self._input_ph_dict = None

    def build_graph(self, loss, target, input_ph_dict, *args, **kwargs):
        """
        Sets the objective function and target weights for the optimize function
        
        Args:
            loss (tf_op) : minimization objective
            target (Policy) : Policy whose values we are optimizing over
            input_ph_dict (dict) : dict containing the placeholders of the computation graph corresponding to loss
        """
        raise NotImplementedError

    def optimize(self, input_val_dict):
        """
        Carries out the optimization step

        Args:
            input_val_dict (dict): dict containing the values to be fed into the computation graph

        """
        raise NotImplementedError

    def loss(self, input_val_dict):
        """
        Computes the value of the loss for given inputs

        Args:
            input_val_dict (dict): dict containing the values to be fed into the computation graph

        Returns:
            (float): value of the loss

        """
        raise NotImplementedError

    def create_feed_dict(self, input_val_dict):
        return utils.create_feed_dict(placeholder_dict=self._input_ph_dict, value_dict=input_val_dict)
