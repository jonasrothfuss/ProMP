class Optimizer(object):
    def __init__(self):
        pass

    def build_graph(self, loss, target, inputs, extra_inputs=()):
        """
        Sets the objective function and target weights for the optimize function
        
        Args:
            loss (tf_op) : minimization objective
            target (Policy) : Policy whose values we are optimizing over
            inputs (tuple) : tuple of tf.placeholders for input data
            extra_inputs (tuple) : tuple of tf.placeholders for hyperparameters (e.g. learning rate, if annealed)
        """
        raise NotImplementedError

    def optimize(self, inputs, extra_inputs):
        """
        Carries out the optimization step

        Args:
            inputs (tuple): inputs for the optimization
            extra_inputs (tuple): extra inputs for the optimization

        """
        raise NotImplementedError

    def loss(self, inputs, extra_inputs):
        """
        Computes the value of the loss for given inputs

        Args:
            inputs (tuple): inputs needed to compute the loss function
            extra_inputs (tuple): additional inputs needed to compute the loss function

        Returns:
            (float): value of the loss

        """
        raise NotImplementedError
