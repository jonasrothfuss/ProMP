import unittest
import numpy as np
import pickle
from maml_zoo.utils import utils
from maml_zoo.envs.base import MetaEnv
from maml_zoo.policies.base import Policy
from maml_zoo.baselines.linear_feature_baseline import LinearFeatureBaseline
from maml_zoo.samplers.maml_sampler import MAMLSampler

class RandomEnv(MetaEnv):
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

class TestBaseline(unittest.TestCase):
    def setUp(self):
        self.random_env = RandomEnv()
        self.random_policy = RandomPolicy()
        self.meta_batch_size = 2
        self.batch_size = 10
        self.path_length = 100
        self.linear = LinearFeatureBaseline()
        self.sampler = MAMLSampler(self.batch_size, self.path_length, parallel=True)
        self.sampler.build_sampler(self.random_env, self.random_policy, self.meta_batch_size, )

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

if __name__ == '__main__':
    unittest.main()