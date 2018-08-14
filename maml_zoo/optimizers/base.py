class Optimizer(object):
    def __init__(self):
        pass

    def update_objective(self, loss, target, inputs, extra_inputs=()):
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

        """
        raise NotImplementedError

    def loss(self, inputs, extra_inputs):
        """

        """
        raise NotImplementedError
