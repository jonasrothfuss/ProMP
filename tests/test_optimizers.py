import unittest
import numpy as np
from meta_policy_search.optimizers.maml_first_order_optimizer import MAMLFirstOrderOptimizer
from collections import OrderedDict
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


class TestOptimizer(unittest.TestCase): #TODO add test for ConjugateGradientOptimizer

    def testSine(self):
        np.random.seed(65)
        for optimizer in [MAMLFirstOrderOptimizer()]:
            tf.reset_default_graph()
            with tf.Session():
                input_phs = tf.placeholder(dtype=tf.float32, shape=[None, 1])
                target_phs = tf.placeholder(dtype=tf.float32, shape=[None, 1])
                network = Mlp(input_phs, 1, hidden_size=(32,32), name='sin')
                loss = tf.reduce_mean(tf.square(network.output - target_phs))
                input_ph_dict = OrderedDict({'x': input_phs, 'y': target_phs})
                optimizer.build_graph(loss, network, input_ph_dict)
                sess = tf.get_default_session()
                sess.run(tf.global_variables_initializer())

                for i in range(5000):
                    xs = np.random.normal(0, 3, (1000, 1))
                    ys = np.sin(xs)
                    inputs = {'x': xs, 'y': ys}
                    optimizer.optimize(inputs)
                    if i % 100 == 0:
                        print(optimizer.loss(inputs))

                xs = np.random.normal(0, 3, (100, 1))
                ys = np.sin(xs)
                y_pred = sess.run(network.output, feed_dict=dict(list(zip(input_ph_dict.values(), (xs, ys)))))
                self.assertLessEqual(np.mean((ys-y_pred)**2), 0.02)

    def testGauss(self):
        np.random.seed(65)
        for optimizer in [MAMLFirstOrderOptimizer()]:
            tf.reset_default_graph()
            with tf.Session():
                input_phs = tf.placeholder(dtype=tf.float32, shape=[None, 100])
                target_mean_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])
                target_std_ph = tf.placeholder(dtype=tf.float32, shape=[None, 1])

                mean_network = Mlp(input_phs, 1, hidden_size=(8,8), name='mean')
                std_network = Mlp(input_phs, 1, hidden_size=(8,8), name='std')

                target_std = tf.exp(target_std_ph)
                pred_std = tf.exp(std_network.output)

                numerator = tf.square(target_mean_ph - mean_network.output) + tf.square(target_std) - tf.square(pred_std)
                denominator = 2 * tf.square(pred_std) + 1e-8
                loss = tf.reduce_mean(tf.reduce_sum(numerator / denominator + std_network.output - target_std_ph, axis=-1))

                joined_network = CombinedMlp([mean_network, std_network])
                input_ph_dict = OrderedDict({'x': input_phs, 'y_mean': target_mean_ph, 'y_std': target_std_ph})

                optimizer.build_graph(loss, joined_network, input_ph_dict)

                sess = tf.get_default_session()
                sess.run(tf.global_variables_initializer())

                for i in range(2000):
                    means = np.random.random(size=(1000))
                    stds = np.random.random(size=(1000))
                    inputs = np.vstack([np.random.normal(mean, np.exp(std), 100) for mean, std in zip(means, stds)])
                    all_inputs = {'x': inputs, 'y_mean': means.reshape(-1, 1), 'y_std': stds.reshape(-1, 1)}
                    optimizer.optimize(all_inputs)
                    if i % 100 == 0:
                        print(optimizer.loss(all_inputs))

                means = np.random.random(size=(20))
                stds = np.random.random(size=(20))

                inputs = np.stack([np.random.normal(mean, np.exp(std), 100) for mean, std in zip(means, stds)], axis=0)
                values_dict = OrderedDict({'x': inputs, 'y_mean': means.reshape(-1, 1), 'y_std': stds.reshape(-1, 1)})

                mean_pred, std_pred = sess.run(joined_network.output, feed_dict=dict(list(zip(input_ph_dict.values(),
                                                                                              values_dict.values()))))

                self.assertTrue(np.mean(np.square(mean_pred - means)) < 0.2)
                self.assertTrue(np.mean(np.square(std_pred - stds)) < 0.2)


if __name__ == '__main__':
    unittest.main()