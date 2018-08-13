import unittest
from maml_zoo.policies.gaussian_mlp_policy import GaussianMLPPolicy
import numpy as np
import tensorflow as tf
import pickle

class DummySpace(object):
    def __init__(self, dim):
        self._dim = dim

    @property
    def flat_dim(self):
        return self._dim


class DummyEnvSpec(object):
    def __init__(self, obs_dim, act_dim):
        self._observation_space = DummySpace(obs_dim)
        self._action_space = DummySpace(act_dim)

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    def get_obs(self, n=None):
        if n is None:
            return np.random.uniform(0, 1, size=(self.observation_space.flat_dim,))
        else:
            return np.random.uniform(0, 1, size=(n, self.observation_space.flat_dim))


class TestPolicy(unittest.TestCase):

    def setUp(self):
        self.policy = GaussianMLPPolicy(name='test_policy',
                                        hidden_sizes=(64, 64))
        self.env_spec = DummyEnvSpec(23, 7)
        self.policy.build_graph(self.env_spec)
        sess = tf.get_default_session()
        if sess is None:
            tf.InteractiveSession()

    def test_output_sym(self):
        obs_ph_1 = tf.placeholder(dtype=tf.float32, name="obs_ph_1",
                                   shape=(None, self.env_spec.observation_space.flat_dim))
        output_sym_1 = self.policy.output_sym(obs_ph_1, {})

        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())

        n_obs = self.env_spec.get_obs(n=100)
        action, agent_infos = self.policy.get_actions(n_obs)
        agent_infos_output_sym = sess.run(output_sym_1, feed_dict={obs_ph_1: n_obs})

        self.assertEqual(agent_infos, agent_infos_output_sym)

    def test_get_action(self):
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())

        obs = self.env_spec.get_obs()
        action, agent_infos = self.policy.get_action(obs)
        actions, agents_infos = self.policy.get_action(np.expand_dims(obs, 0))
        self.assertEquals(actions[0], action)
        self.assertEquals(agent_infos, dict([(k, v[0]) for k, v in agent_infos.items()]))

    def testSerialize(self):
        sess = tf.get_default_session()
        sess.run(tf.global_variables_initializer())
        all_param_values = self.policy.get_param_values()
        for var in all_param_values:
            var += 1
        self.policy.set_params(all_param_values)

        obs = self.env_spec.get_obs()
        pre_action, pre_agent_infos = self.policy.get_action(obs)
        pkl = pickle.dumps(self.policy)
        self.policy = pickle.loads(self.policy)
        post_action, post_agent_infos = self.policy.get_action(obs)
        self.assertEquals(pre_action, post_action)
        for key in pre_agent_infos.keys():
            self.assertEquals(pre_agent_infos[key], post_agent_infos[key])

if __name__ == '__main__':
    unittest.main()
