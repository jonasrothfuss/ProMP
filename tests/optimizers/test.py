import unittest
import numpy as np
from maml_zoo.optimizers.maml_first_order_optimizer import MAMLFirstOrderOptimizer
from maml_zoo.optimizers.maml_first_order_optimizer import MAMLPPOOptimizer
import tensorflow as tf


def fc(x, scope, nh, *, init_scale=1.0, init_bias=0.0):
    with tf.variable_scope(scope):
        nin = x.get_shape()[1].value
        w = tf.get_variable("w", [nin, nh], initializer=tf.orthogonal_initializer(init_scale))
        b = tf.get_variable("b", [nh], initializer=tf.constant_initializer(init_bias))
        return tf.matmul(x, w)+b


class Mlp(object):
    def __init__(self, inputs, output_size, hidden_size=(32, 32), name='mlp'):
        activ = tf.tanh
        curr_output = inputs
        self.name = name
        with tf.variable_scope(self.name):
            for i, size in enumerate(hidden_size):
                curr_output = activ(fc(curr_output, str(i), nh=size, init_scale=np.sqrt(2)))
            self.output = fc(curr_output, 'y_pred', nh=output_size, init_scale=np.sqrt(2))
        self.params = tf.trainable_variables(scope=self.name)

    def get_params(self):
        return self.params


class CombinedMlp(object):
    def __init__(self, mlps):
        self.params = sum([mlp.params for mlp in mlps], [])
        self.output = [mlp.output for mlp in mlps]

    def get_params(self):
        return self.params


class TestOptimizer(unittest.TestCase):
    def setUp(self):
        self.foo = MAMLFirstOrderOptimizer()
        sess = tf.get_default_session()
        if sess is None:
            tf.InteractiveSession()

    def testSine(self):
        input_phs = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        target_phs = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        network = Mlp(input_phs, 1, hidden_size=(32,32), name='sin')
        loss = tf.reduce_mean(tf.square(network.output - target_phs))
        all_input_phs = (input_phs, target_phs)
        self.foo.update_opt(loss, network, all_input_phs)
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())

        for i in range(2000):
            xs = np.random.normal(0, 1, (1000, 1))
            ys = np.sin(xs)
            inputs = (xs, ys)
            self.foo.optimize(inputs)
            if i % 100 == 0:
                print(self.foo.loss(inputs))

        xs = np.random.normal(0, 1, (10, 1))
        ys = np.sin(xs) 
        y_pred = sess.run(network.output, feed_dict=dict(list(zip(all_input_phs, (xs, ys)))))
        self.assertTrue(np.allclose(ys, y_pred, rtol=1e-1, atol=1e-1))

    def testGauss(self):
        input_phs = tf.placeholder(dtype=tf.float32, shape=[None, 100])
        target_mean_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])
        target_std_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])

        mean_network = Mlp(input_phs, 1, hidden_size=(64,64), name='mean')
        std_network = Mlp(input_phs, 1, hidden_size=(64,64), name='std')
        
        target_std = tf.exp(target_std_ph)
        pred_std = tf.exp(std_network.output)

        numerator = tf.square(target_mean_ph - mean_network.output) + tf.square(target_std) - tf.square(pred_std)
        denominator = 2 * tf.square(pred_std) + 1e-8
        loss = tf.reduce_mean(tf.reduce_sum(numerator / denominator + std_network.output - target_std_ph, axis=-1))

        joined_network = CombinedMlp([mean_network, std_network])
        all_input_phs = (input_phs, target_mean_ph, target_std_ph)

        self.foo.update_opt(loss, joined_network, all_input_phs)
        
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())

        for i in range(2000):
            means = np.random.random(size=(1000))
            stds = np.random.random(size=(1000))
            inputs = np.vstack([np.random.normal(mean, np.exp(std), 100) for mean, std in zip(means, stds)])
            all_inputs = (inputs, means.reshape(-1, 1), stds.reshape(-1, 1))
            self.foo.optimize(all_inputs)
            if i % 100 == 0:
                print(self.foo.loss(all_inputs))

        means = np.random.random(size=(10))
        stds = np.random.random(size=(10))
        inputs = np.vstack([np.random.normal(mean, np.exp(std), 100) for mean, std in zip(means, stds)])
        mean_pred, std_pred = sess.run(joined_network.output, feed_dict=dict(list(zip(all_input_phs, (inputs, means.reshape(-1, 1), stds.reshape(-1, 1))))))

        self.assertTrue(np.mean(np.square(mean_pred - means)) < 0.2)
        self.assertTrue(np.mean(np.square(std_pred - stds)) < 0.2)


if __name__ == '__main__':
    unittest.main()