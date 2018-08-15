from maml_zoo.logger import logger
from maml_zoo.optimizers.base import Optimizer
import tensorflow as tf


class MAMLFirstOrderOptimizer(Optimizer):
    """
    Optimizer for first order methods (SGD, Adam)

    """
    def __init__(
            self,
            tf_optimizer_cls=tf.train.AdamOptimizer,
            tf_optimizer_args=None,
            learning_rate=1e-3,
            max_epochs=1,
            tolerance=1e-6,
            num_minibatches=1,
            verbose=False
            ):
        """

        Args:
            tf_optimizer_cls (tf.train.optimizer): desired tensorflow optimzier for training
            tf_optimizer_args (dict or None): arguments for the optimizer
            learning_rate (float): learning rate
            max_epochs: number of maximum epochs for training
            tolerance (float): tolerance for early stopping. If the loss fucntion decreases less than the specified tolerance
            after an epoch, then the training stops.
            num_minibatches (int): number of mini-batches for performing the gradient step. The mini-batch size is
            batch size//num_minibatches.
            verbose (bool): Whether to log or not the optimization process
        """
        self._target = None
        if tf_optimizer_args is None:
            tf_optimizer_args = dict()
        tf_optimizer_args['learning_rate'] = learning_rate

        self._tf_optimizer = tf_optimizer_cls(**tf_optimizer_args)
        self._max_epochs = max_epochs
        self._tolerance = tolerance
        self._num_minibatches = num_minibatches # Unused
        self._verbose = verbose
        self._all_inputs = None
        self._train_op = None
        self._loss = None
        
    def build_graph(self, loss, target, inputs, extra_inputs=(), **kwargs):
        # TODO: Can we get rid of the extra_inputs?? And just have them for the auxiliary objectives
        # and not for the main loss
        """
        Sets the objective function and target weights for the optimize function

        Args:
            loss (tf_op) : minimization objective
            target (Policy) : Policy whose values we are optimizing over
            inputs (tuple) : tuple of tf.placeholders for input data 
            extra_inputs (tuple) : tuple of tf.placeholders for hyperparameters (e.g. learning rate, if annealed)
        """
        assert isinstance(inputs, tuple)
        assert isinstance(extra_inputs, tuple)

        self._target = target
        self._train_op = self._tf_optimizer.minimize(loss, var_list=target.get_params())
        self._all_inputs = inputs + extra_inputs
        self._loss = loss

    def loss(self, inputs, extra_inputs=()):
        """
        Computes the value of the loss for given inputs

        Args:
            inputs (tuple): inputs needed to compute the loss function
            extra_inputs (tuple): additional inputs needed to compute the loss function

        Returns:
            (float): value of the loss

        """
        assert isinstance(inputs, tuple)
        assert isinstance(extra_inputs, tuple)

        sess = tf.get_default_session()
        loss = sess.run(self._loss, feed_dict=dict(list(zip(self._all_inputs, inputs + extra_inputs))))
        return loss

    def optimize(self, inputs, extra_inputs=()):
        """
        Carries out the optimization step

        Args:
            inputs (tuple): inputs for the optimization
            extra_inputs (tuple): extra inputs for the optimization

        """
        assert isinstance(inputs, tuple)
        assert isinstance(extra_inputs, tuple)

        sess = tf.get_default_session()

        last_loss = sess.run(self._loss, feed_dict=dict(list(zip(self._all_inputs, inputs + extra_inputs))))

        # Overload self._batch size
        # dataset = MAMLBatchDataset(inputs, num_batches=self._batch_size, extra_inputs=extra_inputs, meta_batch_size=self.meta_batch_size, num_grad_updates=self.num_grad_updates)
        # Todo: reimplement minibatches
        all_inputs = inputs + extra_inputs

        for epoch in range(self._max_epochs):
            if self._verbose:
                logger.log("Epoch %d" % (epoch))

            sess.run(self._train_op, dict(list(zip(self._all_inputs, all_inputs))))
                
            new_loss = tf.get_default_session().run(self._loss, feed_dict=dict(list(zip(self._all_inputs, inputs + extra_inputs))))

            if self._verbose:
                logger.log("Epoch: %d | Loss: %f" % (epoch, new_loss))

            if abs(last_loss - new_loss) < self._tolerance:
                break
            last_loss = new_loss


class MAMLPPOOptimizer(MAMLFirstOrderOptimizer):
    """
    Adds inner and outer kl terms to first order optimizer  #TODO: (Do we really need this?)

    """
    def __init__(self, *args, **kwargs):
        # Todo: reimplement minibatches
        super(MAMLPPOOptimizer, self).__init__(*args, **kwargs)
        self._inner_kl = None
        self._outer_kl = None

    def build_graph(self, loss, target, inputs, inner_kl=None, outer_kl=None, extra_inputs=(), **kwargs):
        """
        Sets the objective function and target weights for the optimize function

        Args:
            loss (tf_op) : minimization objective
            target (Policy) : Policy whose values we are optimizing over
            inputs (tuple) : tuple of tf.placeholders for input data
            inner_kl (list): List with the inner kl loss for each task
            outer_kl (list): List with the outer kl loss for each task
            extra_inputs (tuple) : tuple of tf.placeholders for hyperparameters (e.g. learning rate, if annealed)
        """
        super(MAMLPPOOptimizer, self).build_graph(loss, target, inputs, extra_inputs)
        assert inner_kl is not None

        self._inner_kl = inner_kl
        self._outer_kl = outer_kl

    def inner_kl(self, inputs, extra_inputs=()):
        """
        Computes the value of the KL-divergence between pre-update policies for given inputs

        Args:
            inputs (tuple): inputs needed to compute the inner KL
            extra_inputs (tuple): additional inputs needed to compute the inner KL

        Returns:
            (float): value of the loss
        """
        assert isinstance(inputs, tuple)
        assert isinstance(extra_inputs, tuple)

        sess = tf.get_default_session()
        inner_kl = sess.run(self._inner_kl, feed_dict=dict(list(zip(self._all_inputs, inputs + extra_inputs))))
        return inner_kl

    def outer_kl(self, inputs, extra_inputs=()):
        """
        Computes the value of the KL-divergence between post-update policies for given inputs

        Args:
            inputs (tuple): inputs needed to compute the outer KL
            extra_inputs (tuple): additional inputs needed to compute the outer KL

        Returns:
            (float): value of the loss
        """
        assert isinstance(inputs, tuple)
        assert isinstance(extra_inputs, tuple)

        sess = tf.get_default_session()
        outer_kl = sess.run(self._outer_kl, feed_dict=dict(list(zip(self._all_inputs, inputs + extra_inputs))))
        return outer_kl

