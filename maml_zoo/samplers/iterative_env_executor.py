import numpy as np
import pickle as pickle
from sandbox_maml.rocky.tf.misc import tensor_utils

class MAMLIterativeEnvExecutor(object):
    def __init__(self, env, meta_batch_size, envs_per_task, max_path_length):
        env_pkl = pickle.dumps(env)
        self.envs = [[pickle.load(env_pkl) for _ in range(envs_per_task)] for _ in range(meta_batch_size)]
        self.ts = np.zeros((meta_batch_size, envs_per_task), dtype='int')
        self.max_path_length = max_path_length

    def step(self, actions):
        """
        Executes actions on each env
        Args:
            actions (list) : a list of lists of actions, of length meta_batch_size x envs_per_task
        Returns
            (tuple) : a length 4 tuple of lists, containing obs (np.array), rewards (float), dones (bool), env_infos (dict)
                      each list is of length meta_batch_size x envs_per_task (assumes that every task has same number of envs)
        """
        all_results = [env.step(a) for (a, env) in zip(action_n, self.envs)]
        obs, rewards, dones, env_infos = list(map(list, list(zip(*all_results))))
        dones = np.asarray(dones)
        self.ts += 1
        dones[self.ts >= self.max_path_length] = True
        for i in np.where(dones)[0]:
            obs[i] = self.envs[i].reset()
            self.ts[i] = 0
        return obs, rewards, dones, env_infos

    def reset(self):
        results = [[env.reset() for env in env_list] for env_list in self.envs]
        self.ts[:] = 0
        return results