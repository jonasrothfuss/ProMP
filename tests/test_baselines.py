import unittest
import numpy as np
import pickle
from maml_zoo.utils import utils
from maml_zoo.policies.base import Policy
from maml_zoo.baselines.linear_baseline import LinearFeatureBaseline, LinearTimeBaseline
from maml_zoo.samplers.maml_sampler import MAMLSampler
from gym import Env


class RandomEnv(Env):
    def __init__(self):
        self.state = np.zeros(1)
        self.goal = 0

    def sample_tasks(self, n_tasks):
        """ 
        Args:
            n_tasks (int) : number of different meta-tasks needed
        Returns:
            tasks (list) : an (n_tasks) length list of reset args
        """
        return np.random.choice(100, n_tasks, replace=False) # Ensure every env has a different goal

    def set_task(self, task):
        """
        Args:
            task: task of the meta-learning environment
        """
        self.goal = task

    def get_task(self):
        """
        Returns:
            task: task of the meta-learning environment
        """
        return self.goal

    def step(self, action):
        self.state += (self.goal - action) * np.random.random()
        return self.state * 100 + self.goal, (self.goal - action)[0], 0, {}

    def reset(self):
        self.state = np.zeros(1)
        return self.state

    def env_spec(self):
        return None


class RandomPolicy(Policy):
    def get_actions(self, observations):
        return [[np.random.random() + obs / 100 for obs in task] for task in observations], None


class TestLinearFeatureBaseline(unittest.TestCase):
    def setUp(self):
        self.random_env = RandomEnv()
        self.random_policy = RandomPolicy(1, 1)
        self.meta_batch_size = 2
        self.batch_size = 10
        self.path_length = 100
        self.linear = LinearFeatureBaseline()
        self.sampler = MAMLSampler(self.random_env, self.random_policy, self.batch_size,
                                   self.meta_batch_size, self.path_length, parallel=True)

    def testFit(self):
        paths = self.sampler.obtain_samples()
        for task in paths.values():
            unfit_error = 0
            for path in task:
                path["returns"] = utils.discount_cumsum(path["rewards"], 0.99)
                unfit_pred = self.linear.predict(path)
                unfit_error += sum([np.square(pred - actual) for pred, actual in zip(unfit_pred, path['returns'])])
            self.linear.fit(task)
            fit_error = 0
            for path in task:
                fit_pred = self.linear.predict(path)
                fit_error += sum([np.square(pred - actual) for pred, actual in zip(fit_pred, path['returns'])])
            self.assertTrue(fit_error < unfit_error)

    def testSerialize(self):
        paths = self.sampler.obtain_samples()
        for task in paths.values():
            for path in task:
                path["returns"] = utils.discount_cumsum(path["rewards"], 0.99)
            self.linear.fit(task)
            fit_error_pre = 0
            for path in task:
                fit_pred = self.linear.predict(path)
                fit_error_pre += sum([np.square(pred - actual) for pred, actual in zip(fit_pred, path['returns'])])
            pkl = pickle.dumps(self.linear)
            self.linear = pickle.loads(pkl)
            fit_error_post = 0
            for path in task:
                fit_pred = self.linear.predict(path)
                fit_error_post += sum([np.square(pred - actual) for pred, actual in zip(fit_pred, path['returns'])])
            self.assertEqual(fit_error_pre, fit_error_post)


class TestLinearFeatureBaseline(unittest.TestCase):
    def setUp(self):
        self.random_env = RandomEnv()
        self.random_policy = RandomPolicy(1, 1)
        self.meta_batch_size = 2
        self.batch_size = 10
        self.path_length = 100
        self.linear = LinearTimeBaseline()
        self.sampler = MAMLSampler(self.random_env, self.random_policy, self.batch_size,
                                   self.meta_batch_size, self.path_length, parallel=True)

    def testFit(self):
        base_path = np.arange(-4.0, 22.0, step=.6)
        task1 = [{'discounted_rewards': base_path + np.random.normal(scale=2, size=base_path.shape),
                  'observations': base_path} for i in range(10)]
        task2 = [{'discounted_rewards': base_path**3 + np.random.normal(scale=2, size=base_path.shape),
                  'observations': base_path} for i in range(10)]


        for task in [task1, task2]:
            unfit_error = np.sum([np.sum(path['discounted_rewards']**2) for path in task])
            print('unfit_error', unfit_error)
            self.linear.fit(task, target_key='discounted_rewards')
            fit_error = 0
            for path in task:
                fit_pred = self.linear.predict(path)
                fit_error += sum([np.square(pred - actual) for pred, actual in zip(fit_pred, path['discounted_rewards'])])
            print('fit_error', fit_error)
            self.assertTrue(2*fit_error < unfit_error)

    def testSerialize(self):
        base_path = np.arange(-4.0, 22.0, step=.6)
        task1 = [{'discounted_rewards': base_path + np.random.normal(scale=2, size=base_path.shape),
                  'observations': base_path} for i in range(10)]
        task2 = [{'discounted_rewards': base_path**3 + np.random.normal(scale=2, size=base_path.shape),
                  'observations': base_path} for i in range(10)]

        for task in [task1, task2]:
            self.linear.fit(task, target_key='discounted_rewards')
            fit_error_pre = 0
            for path in task:
                fit_pred = self.linear.predict(path)
                fit_error_pre += sum([np.square(pred - actual) for pred, actual in zip(fit_pred, path['discounted_rewards'])])
            pkl = pickle.dumps(self.linear)
            self.linear = pickle.loads(pkl)
            fit_error_post = 0
            for path in task:
                fit_pred = self.linear.predict(path)
                fit_error_post += sum([np.square(pred - actual) for pred, actual in zip(fit_pred, path['discounted_rewards'])])
            self.assertEqual(fit_error_pre, fit_error_post)

if __name__ == '__main__':
    unittest.main()
