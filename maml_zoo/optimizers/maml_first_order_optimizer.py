from maml_zoo.logger import logger
from maml_zoo.optimizers.base import Optimizer
import tensorflow as tf
import time

class MAMLFirstOrderOptimizer(Optimizer):
    def __init__(
            self,
            tf_optimizer_cls=None,
            tf_optimizer_args=None,
            learning_rate=1e-3,
            max_epochs=1,
            tolerance=1e-6,
            minibatch_splits=1,
            verbose=False
            ):
        """
        Args:
            learning_rate (float) : initial learning rate
        """
        self._opt_fun = None
        self._target = None
        if tf_optimizer_cls is None:
            tf_optimizer_cls = tf.train.AdamOptimizer
        if tf_optimizer_args is None:
            tf_optimizer_args = dict(learning_rate=1e-3)
        self._tf_optimizer = tf_optimizer_cls(**tf_optimizer_args)
        self._max_epochs = max_epochs
        self._tolerance = tolerance
        self._minibatch_splits = minibatch_splits
        self._verbose = verbose
        self._all_inputs = None
        self._train_op = None
        
    def update_opt(self, loss, target, inputs, extra_inputs=(), **kwargs):
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
        assert isinstance(inputs, tuple)
        assert isinstance(extra_inputs, tuple)
        return tf.get_default_session().run(self._loss, feed_dict=dict(list(zip(self._all_inputs, inputs + extra_inputs))))

    def optimize(self, inputs, extra_inputs=()):
        assert isinstance(inputs, tuple)
        assert isinstance(extra_inputs, tuple)

        last_loss = tf.get_default_session().run(self._loss, feed_dict=dict(list(zip(self._all_inputs, inputs + extra_inputs))))

        start_time = time.time()
        # Overload self._batch size
        # dataset = MAMLBatchDataset(inputs, num_batches=self._batch_size, extra_inputs=extra_inputs, meta_batch_size=self.meta_batch_size, num_grad_updates=self.num_grad_updates)
        # Todo: reimplement minibatches
        all_inputs = inputs + extra_inputs

        sess = tf.get_default_session()
        for epoch in range(self._max_epochs):
            if self._verbose:
                logger.log("Epoch %d" % (epoch))

            if hasattr(self, 'multi_adam') and self.multi_adam:
                sess.run(self._train_ops[epoch], dict(list(zip(self._all_inputs, all_inputs))))
            else:
                sess.run(self._train_op, dict(list(zip(self._all_inputs, all_inputs))))
                
            new_loss = tf.get_default_session().run(self._loss, feed_dict=dict(list(zip(self._all_inputs, inputs + extra_inputs))))

            if self._verbose:
                logger.log("Epoch: %d | Loss: %f" % (epoch, new_loss))

            if abs(last_loss - new_loss) < self._tolerance:
                break
            last_loss = new_loss

class MAMLPPOOptimizer(MAMLFirstOrderOptimizer):
    def __init__(
            self,
            tf_optimizer_cls=None,
            tf_optimizer_args=None,
            learning_rate=1e-3,
            max_epochs=1,
            tolerance=1e-6,
            verbose=False,
            multi_adam=False
        ):
        """
        multi_adam (bool) : whether to keep separate momentum separately for each PPO step
        """
        # Todo: reimplement minibatches
        super(MAMLPPOOptimizer, self).__init__(tf_optimizer_cls, tf_optimizer_args, learning_rate, max_epochs, tolerance, verbose)
        self.minibatch_splits = minibatch_splits
        self.multi_adam = multi_adam
        if self.multi_adam:
            if tf_optimizer_cls is None:
                tf_optimizer_cls = tf.train.AdamOptimizer
            if tf_optimizer_args is None:
                tf_optimizer_args = dict(learning_rate=learning_rate)
            self._tf_optimizers = [tf_optimizer_cls(**tf_optimizer_args) for _ in range(max_epochs)]

    def update_opt(self, loss, target, inputs, inner_kl, outer_kl=None, extra_inputs=(), **kwargs):
        """
        Sets the objective function and target weights for the optimize function
        Args:
            loss (tf_op) : minimization objective
            target (Policy) : Policy whose values we are optimizing over
            inputs (tuple) : tuple of tf.placeholders for input data 
            extra_inputs (tuple) : tuple of tf.placeholders for hyperparameters (e.g. learning rate, if annealed)
        """
        super().update_opt(loss, target, inputs, extra_inputs)
        self._inner_kl = inner_kl
        self._outer_kl = outer_kl

        if self.multi_adam:
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            if update_ops:
                # for batch norm
                updates = tf.group(*update_ops)
                with tf.control_dependencies([updates]):
                    self._train_ops = [optimizer.minimize(loss, var_list=target.get_params()) for optimizer in self._tf_optimizers]
            else:
                self._train_ops = [optimizer.minimize(loss, var_list=target.get_params()) for optimizer in self._tf_optimizers]

    def inner_kl(self, inputs, extra_inputs=()):
        assert isinstance(inputs, tuple)
        assert isinstance(extra_inputs, tuple)
        return tf.get_default_session().run(self._inner_kl, feed_dict=dict(list(zip(self._all_inputs, inputs + extra_inputs))))

    def outer_kl(self, inputs, extra_inputs=()):
        assert isinstance(inputs, tuple)
        assert isinstance(extra_inputs, tuple)
        return tf.get_default_session().run(self._outer_kl, feed_dict=dict(list(zip(self._all_inputs, inputs + extra_inputs))))
