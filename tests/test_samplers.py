import unittest
import numpy as np
from maml_zoo.policies.base import Policy
from maml_zoo.samplers import MAMLSampler
from maml_zoo.samplers import MAMLSampleProcessor
from maml_zoo.samplers import SampleProcessor
from maml_zoo.samplers import DiceSampleProcessor
from maml_zoo.samplers import DiceMAMLSampleProcessor
from maml_zoo.baselines.linear_baseline import LinearFeatureBaseline, LinearTimeBaseline
from maml_zoo.baselines.zero_baseline import ZeroBaseline


class TestEnv():
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
        self.test_policy = TestPolicy(obs_dim=3, action_dim=4)
        self.return_policy = ReturnPolicy(obs_dim=3, action_dim=4)
        self.random_policy = RandomPolicy(obs_dim=3, action_dim=4)
        self.meta_batch_size = 3
        self.batch_size = 4
        self.path_length = 5
        self.it_sampler = MAMLSampler(self.test_env, self.test_policy, self.batch_size, self.meta_batch_size, self.path_length, parallel=False)
        self.par_sampler = MAMLSampler(self.test_env, self.test_policy, self.batch_size, self.meta_batch_size, self.path_length, parallel=True)
        self.sample_processor = SampleProcessor(baseline=LinearFeatureBaseline())
        self.maml_sample_processor = MAMLSampleProcessor(baseline=LinearFeatureBaseline())

    def testSingle(self):
        for sampler in [self.par_sampler]:
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

    def testRandomSeeds1(self):
        for sampler_parallel in [True, False]:
            np.random.seed(22)
            sampler = MAMLSampler(self.random_env, self.random_policy, self.batch_size, self.meta_batch_size,
                                     self.path_length, parallel=sampler_parallel)
            sampler.update_tasks()
            paths1 = sampler.obtain_samples()

            np.random.seed(22)
            sampler = MAMLSampler(self.random_env, self.random_policy, self.batch_size, self.meta_batch_size,
                                  self.path_length, parallel=sampler_parallel)
            sampler.update_tasks()
            paths2 = sampler.obtain_samples()

            for task1, task2 in zip(paths1.values(), paths2.values()): # All rewards in task are equal, but obs are not
                for j in range(self.batch_size):
                    for k in range(self.path_length):
                        self.assertEqual(task1[j]["observations"][k], task2[j]["observations"][k])

    def testRandomSeeds2(self):
        for sampler_parallel in [True, False]:
            np.random.seed(22)
            sampler = MAMLSampler(self.random_env, self.test_policy, self.batch_size, self.meta_batch_size,
                                  self.path_length, parallel=sampler_parallel)
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
        it_sampler = MAMLSampler(self.random_env, self.random_policy, self.batch_size, self.meta_batch_size,
                                      self.path_length, parallel=False)
        par_sampler = MAMLSampler(self.random_env, self.random_policy, self.batch_size, self.meta_batch_size,
                                       self.path_length, parallel=True)

        for sampler in [it_sampler, par_sampler]:
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

    def testMAMLSampleProcessor(self):
        for sampler in [self.it_sampler, self.par_sampler]:
            sampler.update_tasks()
            paths_meta_batch = sampler.obtain_samples()
            samples_data_meta_batch = self.maml_sample_processor.process_samples(paths_meta_batch)
            self.assertEqual(len(samples_data_meta_batch), self.meta_batch_size)
            for samples_data in samples_data_meta_batch:
                self.assertEqual(len(samples_data.keys()), 8)
                self.assertEqual(samples_data['advantages'].size, self.path_length*self.batch_size)

    def testSampleProcessor(self):
        for sampler in [self.it_sampler, self.par_sampler]:
            sampler.update_tasks()
            paths_meta_batch = sampler.obtain_samples()
            for paths in paths_meta_batch.values():
                samples_data = self.sample_processor.process_samples(paths)
                self.assertEqual(len(samples_data.keys()), 7)
                self.assertEqual(samples_data['advantages'].size, self.path_length*self.batch_size)

class TestDiceSampleProcessor(unittest.TestCase):

    def setUp(self):
        self.test_env = TestEnv()
        self.random_env = RandomEnv()
        self.test_policy = TestPolicy(obs_dim=3, action_dim=4)
        self.return_policy = ReturnPolicy(obs_dim=3, action_dim=4)
        self.random_policy = RandomPolicy(obs_dim=3, action_dim=4)
        self.meta_batch_size = 3
        self.batch_size = 10
        self.path_length = 5
        self.it_sampler = MAMLSampler(self.test_env, self.test_policy, self.batch_size, self.meta_batch_size,
                                      self.path_length, parallel=False)

        self.paths = self.it_sampler.obtain_samples()

        self.it_sampler_rand = MAMLSampler(self.random_env, self.random_policy, self.batch_size, self.meta_batch_size,
                                      self.path_length, parallel=False)

        self.paths_rand = self.it_sampler_rand.obtain_samples()


        self.baseline = LinearTimeBaseline()
        self.dics_sample_proc = DiceSampleProcessor(self.baseline, max_path_length=6)

    def test_discounted_reward(self):
        samples_data, paths = self.dics_sample_proc._compute_samples_data(self.paths[0])
        self.assertAlmostEqual(paths[0]['discounted_rewards'][0], -1)
        self.assertAlmostEqual(paths[0]['discounted_rewards'][3], -0.99**3)

    def test_adjusted_reward(self):
        paths = self.paths[0]
        paths[0]['rewards'][0] = 0
        paths[0]['rewards'][3] = -2
        samples_data, paths = self.dics_sample_proc._compute_samples_data(paths)
        self.assertGreaterEqual(paths[0]['adjusted_rewards'][0], 0.3)
        self.assertLessEqual(paths[0]['adjusted_rewards'][3], -0.3)

    def test_process_samples(self):
        samples_data = self.dics_sample_proc.process_samples(self.paths[0])
        self.assertAlmostEqual(samples_data['observations'].shape, (self.batch_size, 6, 1))
        self.assertAlmostEqual(samples_data['actions'].ndim, 3)
        self.assertAlmostEqual(samples_data['rewards'].ndim, 2)
        self.assertAlmostEqual(samples_data['mask'].ndim, 2)
        self.assertAlmostEqual(samples_data['mask'][2][5], 0)
        self.assertAlmostEqual(samples_data['mask'][2][2], 1)
        self.assertAlmostEqual(samples_data['env_infos']['e'][0][5], 0)
        self.assertAlmostEqual(samples_data['env_infos']['e'][2][0], -5)

    def test_dice_maml_processor(self):
        maml_sample_processor = DiceMAMLSampleProcessor(self.baseline, max_path_length=6)
        maml_samples_data = maml_sample_processor.process_samples(self.paths)
        for samples_data in maml_samples_data:
            self.assertAlmostEqual(samples_data['observations'].shape, (self.batch_size, 6, 1))
            self.assertAlmostEqual(samples_data['actions'].ndim, 3)
            self.assertAlmostEqual(samples_data['rewards'].ndim, 2)
            self.assertAlmostEqual(samples_data['mask'].ndim, 2)
            self.assertAlmostEqual(samples_data['mask'][2][5], 0)
            self.assertAlmostEqual(samples_data['mask'][2][2], 1)
            self.assertAlmostEqual(samples_data['env_infos']['e'][0][5], 0)
            self.assertAlmostEqual(samples_data['env_infos']['e'][2][0], -5)

    def test_process_samples_advantages1(self):
        return_baseline = LinearFeatureBaseline()
        sample_processor = DiceSampleProcessor(self.baseline, max_path_length=6, return_baseline=return_baseline)
        samples_data = sample_processor.process_samples(self.paths[0])
        self.assertAlmostEqual(samples_data['advantages'].shape, (self.batch_size, 6))
        self.assertAlmostEqual(samples_data['advantages'].ndim, 2)

    def test_process_samples_advantages2(self):
        for normalize_adv in [True, False]:
            for paths in [self.paths, self.paths_rand]:
                return_baseline = LinearFeatureBaseline()
                dice_sample_processor = DiceSampleProcessor(self.baseline, max_path_length=6, gae_lambda=1.0,
                                                            discount=0.97, normalize_adv=normalize_adv, return_baseline=return_baseline)
                dice_samples_data = dice_sample_processor.process_samples(paths[0])
                mask = dice_samples_data['mask']

                # reshape data and filter out masked items:

                sample_processor = SampleProcessor(return_baseline, gae_lambda=1.0, discount=0.97, normalize_adv=normalize_adv)
                samples_data = sample_processor.process_samples(paths[0])

                self.assertAlmostEqual(np.sum(mask[:,:, None]*dice_samples_data['observations']), np.sum(samples_data['observations']))
                self.assertAlmostEqual(np.sum(mask[:, :, None] * dice_samples_data['actions']),
                                       np.sum(samples_data['actions']))
                self.assertAlmostEqual(np.sum(mask * dice_samples_data['advantages']),
                                      np.sum(samples_data['advantages']), places=2)
                self.assertAlmostEqual(np.sum(mask * dice_samples_data['rewards']),
                                      np.sum(samples_data['rewards']))


class PointEnv(TestEnv):
    def __init__(self):
        self.reset()
        self.goal = np.array([0,0])

    def sample_tasks(self, n_tasks):
        return [np.array([0,0]) for _ in range(n_tasks)]

    def step(self, action):
        self.state += np.clip(action, -0.1, 0.1)
        goal_distance = np.linalg.norm(self.goal - self.state)
        done = goal_distance < 0.1
        return self.state, - goal_distance, done, {'e': self.state}

    def reset(self):
        self.state = np.random.uniform(-2, 2, size=(2,))
        return self.state

class PointEnvPolicy:
    def __init__(self):
        pass

    def get_actions(self, observations):
        return [-np.clip(obs, -0.1, 0.1) + np.random.normal(0, scale=0.03, size=2) for obs in observations], None

class SampleProcConsistency(unittest.TestCase):

    def setUp(self):
        self.baseline = ZeroBaseline()
        env = PointEnv()
        policy = PointEnvPolicy()

        self.meta_batch_size = 3
        self.batch_size = 20
        self.path_length = 25
        self.it_sampler = MAMLSampler(env, policy, self.batch_size, self.meta_batch_size,
                                      self.path_length, parallel=False)

        self.dics_sample_proc = DiceSampleProcessor(self.baseline, max_path_length=self.path_length, discount=1.0,
                                                    gae_lambda=1., normalize_adv=False)
        self.sample_proc = SampleProcessor(self.baseline, discount=1.0, gae_lambda=1., normalize_adv=False)


    def testAdvantagesMatchAdjustedRewards1(self):
        tasks = self.it_sampler.obtain_samples()

        for task in tasks.values():
            # adds advantages and adjusted_rewards to paths
            _ = self.sample_proc.process_samples(task)
            _ = self.dics_sample_proc.process_samples(task)

            for path in task:
                self.assertAlmostEqual(path['adjusted_rewards'][0], path['rewards'][0], places=3)

                advs = path['advantages']
                adjusted_rewards = path['adjusted_rewards']
                path_length = len(path['advantages'])
                for step in range(path_length):
                    adv_from_adj_rew = np.sum(adjusted_rewards[step:path_length])
                    self.assertAlmostEqual(advs[step], adv_from_adj_rew, places=3)

    def testAdvantagesMatchAdjustedRewards2(self):
        tasks = self.it_sampler.obtain_samples()

        for task in tasks.values():
            # adds advantages and adjusted_rewards to paths
            sample_data = self.sample_proc.process_samples(task)
            dice_sample_data = self.dics_sample_proc.process_samples(task)

            step = 0
            for path_id in range(dice_sample_data['mask'].shape[0]):
                mask = dice_sample_data['mask'][path_id]
                adj_rewards = np.multiply(dice_sample_data['adjusted_rewards'][path_id], dice_sample_data['mask'][path_id])
                for i, mask_element in enumerate(mask):
                    if mask_element > 0:
                        adv = sample_data['advantages'][step]
                        adv_from_adj_rew = np.sum(adj_rewards[i:])
                        self.assertAlmostEqual(adv, adv_from_adj_rew)
                        step += 1

if __name__ == '__main__':
    unittest.main()