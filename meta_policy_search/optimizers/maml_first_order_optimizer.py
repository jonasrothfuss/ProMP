from meta_policy_search.utils import logger
from meta_policy_search.optimizers.base import Optimizer
import tensorflow as tf

class MAMLFirstOrderOptimizer(Optimizer):
    """
    Optimizer for first order methods (SGD, Adam)

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
        self._input_ph_dict = None
        
    def build_graph(self, loss, target, input_ph_dict):
        """
        Sets the objective function and target weights for the optimize function

        Args:
            loss (tf_op) : minimization objective
            target (Policy) : Policy whose values we are optimizing over
            input_ph_dict (dict) : dict containing the placeholders of the computation graph corresponding to loss
        """
        assert isinstance(loss, tf.Tensor)
        assert hasattr(target, 'get_params')
        assert isinstance(input_ph_dict, dict)

        self._target = target
        self._input_ph_dict = input_ph_dict
        self._loss = loss
        self._train_op = self._tf_optimizer.minimize(loss, var_list=target.get_params())

    def loss(self, input_val_dict):
        """
        Computes the value of the loss for given inputs

        Args:
            input_val_dict (dict): dict containing the values to be fed into the computation graph

        Returns:
            (float): value of the loss

        """
        sess = tf.get_default_session()
        feed_dict = self.create_feed_dict(input_val_dict)
        loss = sess.run(self._loss, feed_dict=feed_dict)
        return loss

    def optimize(self, input_val_dict):
        """
        Carries out the optimization step

        Args:
            input_val_dict (dict): dict containing the values to be fed into the computation graph

        Returns:
            (float) loss before optimization

        """

        sess = tf.get_default_session()
        feed_dict = self.create_feed_dict(input_val_dict)

        # Overload self._batch size
        # dataset = MAMLBatchDataset(inputs, num_batches=self._batch_size, extra_inputs=extra_inputs, meta_batch_size=self.meta_batch_size, num_grad_updates=self.num_grad_updates)
        # Todo: reimplement minibatches

        loss_before_opt = None
        for epoch in range(self._max_epochs):
            if self._verbose:
                logger.log("Epoch %d" % epoch)

            loss, _ = sess.run([self._loss, self._train_op], feed_dict)
            if not loss_before_opt: loss_before_opt = loss

            # if self._verbose:
            #     logger.log("Epoch: %d | Loss: %f" % (epoch, new_loss))
            #
            # if abs(last_loss - new_loss) < self._tolerance:
            #     break
            # last_loss = new_loss
        return loss_before_opt


class MAMLPPOOptimizer(MAMLFirstOrderOptimizer):
    """
    Adds inner and outer kl terms to first order optimizer  #TODO: (Do we really need this?)

    """
    def __init__(self, *args, **kwargs):
        # Todo: reimplement minibatches
        super(MAMLPPOOptimizer, self).__init__(*args, **kwargs)
        self._inner_kl = None
        self._outer_kl = None

    def build_graph(self, loss, target, input_ph_dict, inner_kl=None, outer_kl=None):
        """
        Sets the objective function and target weights for the optimize function

        Args:
            loss (tf.Tensor) : minimization objective
            target (Policy) : Policy whose values we are optimizing over
            input_ph_dict (dict) : dict containing the placeholders of the computation graph corresponding to loss
            inner_kl (list): list with the inner kl loss for each task
            outer_kl (list): list with the outer kl loss for each task
        """
        super(MAMLPPOOptimizer, self).build_graph(loss, target, input_ph_dict)
        assert inner_kl is not None

        self._inner_kl = inner_kl
        self._outer_kl = outer_kl

    def compute_stats(self, input_val_dict):
        """
        Computes the value the loss, the outer KL and the inner KL-divergence between the current policy and the
        provided dist_info_data

        Args:
           inputs (list): inputs needed to compute the inner KL
           extra_inputs (list): additional inputs needed to compute the inner KL

        Returns:
           (float): value of the loss
           (ndarray): inner kls - numpy array of shape (num_inner_grad_steps,)
           (float): outer_kl
        """
        sess = tf.get_default_session()
        feed_dict = self.create_feed_dict(input_val_dict)
        loss, inner_kl, outer_kl = sess.run([self._loss, self._inner_kl, self._outer_kl], feed_dict=feed_dict)
        return loss, inner_kl, outer_kl



