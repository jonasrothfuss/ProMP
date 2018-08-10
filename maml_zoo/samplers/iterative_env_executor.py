import numpy as np
import pickle as pickle

class MAMLIterativeEnvExecutor(object):
    def __init__(self, env, meta_batch_size, envs_per_task, max_path_length):
        env_pkl = pickle.dumps(env)
        self.envs = np.asarray([pickle.loads(env_pkl) for _ in range(meta_batch_size * envs_per_task)])
        self.ts = np.zeros(len(self.envs), dtype='int')
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
        actions = sum(actions, [])
        all_results = [env.step(a) for (a, env) in zip(actions, self.envs)] 
        obs, rewards, dones, env_infos = list(map(list, list(zip(*all_results))))
        dones = np.asarray(dones)
        self.ts += 1
        dones[self.ts >= self.max_path_length] = True
        for i in np.where(dones)[0]:
            obs[i] = self.envs[i].reset()
            self.ts[i] = 0
        return obs, rewards, dones, env_infos

    def set_tasks(self, tasks):
        envs_per_task = np.split(self.envs, len(tasks))
        for task, envs in zip(tasks, envs_per_task):
            for env in envs:
                env.set_task(task)

    def reset(self):
        results = [env.reset() for env in self.envs]
        self.ts[:] = 0
        return results
        
    @property
    def num_envs(self):
        return len(self.envs)