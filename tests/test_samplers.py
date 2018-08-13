import unittest
import numpy as np
from maml_zoo.envs.base import MetaEnv
from maml_zoo.policies.base import Policy
from maml_zoo.samplers.iterative_env_executor import MAMLIterativeEnvExecutor
from maml_zoo.samplers.parallel_env_executor import MAMLParallelEnvExecutor
from maml_zoo.samplers.maml_sampler import MAMLSampler
from maml_zoo.samplers.maml_sample_processor import MAMLSampleProcessor
from maml_zoo.baselines.linear_feature_baseline import LinearFeatureBaseline

class TestEnv(MetaEnv):
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
        self.state += self.goal - action
        return self.state * 100 + self.goal, (self.goal - action)[0], 0, {'e':self.state}

    def reset(self):
        self.state = np.zeros(1)
        return self.state

    def env_spec(self):
        return None

class RandomEnv(TestEnv):
    def step(self, action):
        self.state += (self.goal - action) * np.random.random()
        return self.state * 100 + self.goal, (self.goal - action)[0], 0, {'e':self.state}

class TestPolicy(Policy):
    def get_actions(self, observations):
        return [[np.ones(1) for batch in task] for task in observations], None

class ReturnPolicy(Policy):
    def get_actions(self, observations):
        return [[batch / 100 for batch in task] for task in observations], None

class RandomPolicy(Policy):
    def get_actions(self, observations):
        return [[np.random.random() * batch for batch in task] for task in observations], [[{'a':1, 'b':2} for batch in task] for task in observations]

class TestSampler(unittest.TestCase):
    def setUp(self):
        self.test_env = TestEnv()
        self.random_env = RandomEnv()
        self.test_policy = TestPolicy()
        self.return_policy = ReturnPolicy()
        self.random_policy = RandomPolicy()
        self.meta_batch_size = 3
        self.batch_size = 4
        self.path_length = 5
        self.it_sampler = MAMLSampler(self.batch_size, self.path_length, parallel=False)
        self.par_sampler = MAMLSampler(self.batch_size, self.path_length, parallel=True)
        self.sample_processor = MAMLSampleProcessor(LinearFeatureBaseline())

    def testSingle(self):
        for sampler in [self.par_sampler]:
            sampler.build_sampler(self.test_env, self.test_policy, self.meta_batch_size)
            paths = sampler.obtain_samples()
            self.assertEqual(len(paths), self.meta_batch_size)
            for task in paths.values():
                self.assertEqual(len(task), self.batch_size)
                for path in task:
                    self.assertEqual(len(path), self.path_length)
                    for act in path['actions']:
                        self.assertEqual(act, 1)
                    path_state = 0
                    for obs in path['observations']:
                        self.assertEqual(obs, path_state)
                        path_state += -100

    def testGoalSet(self):
        for sampler in [self.it_sampler, self.par_sampler]:
            sampler.build_sampler(self.test_env, self.return_policy)
            sampler.update_tasks()
            paths = sampler.obtain_samples()
            self.assertEqual(len(paths), self.meta_batch_size)

            for task in paths.values(): # All paths in task are equal
                for j in range(self.path_length): # batch size
                    curr_obs = task[0]["observations"][j]
                    for path in task:
                        self.assertEqual(path["observations"][j], curr_obs)
            for j in range(1, self.path_length): # All paths in different tasks are different
                for i in range(self.batch_size):
                    curr_obs = paths[0][i]['observations'][j]
                    for h in range(1, self.meta_batch_size):
                        self.assertNotEqual(paths[h][i]['observations'][j], curr_obs)

    def testRandomSeeds(self):
        for sampler in [self.it_sampler, self.par_sampler]:
            sampler.build_sampler(self.random_env, self.test_policy)
            sampler.update_tasks()
            paths = sampler.obtain_samples()
            self.assertEqual(len(paths), self.meta_batch_size)

            for task in paths.values(): # All rewards in task are equal, but obs are not
                for j in range(1, self.path_length): # batch size
                    curr_obs = task[0]["observations"][j]
                    curr_rew = task[0]['rewards'][j]
                    for h in range(1, self.batch_size):
                        self.assertNotEqual(task[h]["observations"][j], curr_obs)
                        self.assertEqual(task[h]['rewards'][j], curr_rew)

    def testInfoDicts(self):
        for sampler in [self.it_sampler, self.par_sampler]:
            sampler.build_sampler(self.random_env, self.random_policy)
            sampler.update_tasks()
            paths = sampler.obtain_samples()
            self.assertEqual(len(paths), self.meta_batch_size)

            for task in paths.values(): # All rewards in task are equal, but obs are not
                for h in range(1, self.batch_size): # batch size
                    curr_agent_infos = task[h]["agent_infos"]
                    curr_env_infos = task[h]['env_infos']
                    self.assertEqual(type(curr_agent_infos), dict)
                    self.assertEqual(type(curr_env_infos), dict)
                    self.assertEqual(len(curr_agent_infos.keys()), 2)
                    self.assertEqual(len(curr_env_infos.keys()), 1)

    def testProcessor(self):
        for sampler in [self.it_sampler, self.par_sampler]:
            sampler.build_sampler(self.random_env, self.random_policy)
            sampler.update_tasks()
            paths = sampler.obtain_samples()
            for path in paths.values():
                samples_data = self.sample_processor.process_samples(path)
                self.assertEqual(len(samples_data.keys()), 7)
                for value in samples_data.values():
                    if type(value) == dict:
                        for sub_value in value.values():
                            self.assertEqual(len(sub_value), self.batch_size * self.path_length)
                    else:
                        self.assertEqual(len(value), self.batch_size * self.path_length)

if __name__ == '__main__':
    unittest.main()