from maml_zoo.samplers.base import Sampler
from maml_zoo.samplers.iterative_env_executor import MAMLIterativeEnvExecutor
from maml_zoo.samplers.parallel_env_executor import MAMLParallelEnvExecutor
from maml_zoo.utils.progbar import ProgBarCounter
from maml_zoo.logger import logger
from maml_zoo.utils import utils
import numpy as np
import time
import itertools

class MAMLSampler(Sampler):
    def __init__(
            self,
            batch_size,
            max_path_length,
            envs_per_task=None,
            parallel=True
            ):
        """
        Args:
            env (Env) : 
            policy (Policy) : 
            batch_size (int) : number of trajectories per task
            meta_batch_size (int) : number of meta tasks
            max_path_length (int) : max number of steps per trajectory
            envs_per_task (int) : number of envs to run for each task
        """
        super(MAMLSampler, self).__init__(batch_size, max_path_length)
        if envs_per_task is None:
            self.envs_per_task = batch_size
        else:
            self.envs_per_task = envs_per_task
        self.parallel = parallel

    def build_sampler(self, env, policy, meta_batch_size):
        super(MAMLSampler, self).build_sampler(env, policy)
        self.meta_batch_size = meta_batch_size
        self.total_samples = meta_batch_size * self.batch_size * self.max_path_length
        if self.parallel:
            self.vec_env = MAMLParallelEnvExecutor(env, self.meta_batch_size, self.envs_per_task, self.max_path_length)
        else:
            self.vec_env = MAMLIterativeEnvExecutor(env, self.meta_batch_size, self.envs_per_task, self.max_path_length)

    def update_tasks(self):
        """
        Samples a new goal for each meta task
        """
        tasks = self.env.sample_tasks(self.meta_batch_size)
        assert len(tasks) == self.meta_batch_size
        self.vec_env.set_tasks(tasks)

    def obtain_samples(self, log_prefix=''):
        """
        Collect batch_size trajectories from each task
        Args:
            log_prefix (str) : prefix for logger
        Returns: 
            (dict) : A dict of paths of size [meta_batch_size] x (batch_size) x [5] x (max_path_length)
        """
        paths = {}
        for i in range(self.meta_batch_size):
            paths[i] = []

        n_samples = 0
        obses = self.vec_env.reset()
        running_paths = [None] * self.vec_env.num_envs

        pbar = ProgBarCounter(self.total_samples)
        policy_time = 0
        env_time = 0
        process_time = 0

        policy = self.policy
        
        while n_samples < self.total_samples:
            t = time.time()

            obs_per_task = np.split(np.asarray(obses), self.meta_batch_size)
            actions, agent_infos = policy.get_actions(obs_per_task)

            policy_time += time.time() - t
            t = time.time()
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions)
            env_time += time.time() - t

            t = time.time()
            
            # agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
            if not env_infos:
                env_infos = [dict() for _ in range(self.vec_env.num_envs)]
            if not agent_infos:
                agent_infos = [dict() for _ in range(self.vec_env.num_envs)]
            else:
                assert len(agent_infos) == self.meta_batch_size
                assert len(agent_infos[0]) == self.envs_per_task
                agent_infos = sum(agent_infos, [])
            actions = sum(actions, [])

            for idx, observation, action, reward, env_info, agent_info, done in zip(itertools.count(), obses, actions,
                                                                                    rewards, env_infos, agent_infos,
                                                                                    dones):
                if running_paths[idx] is None:
                    running_paths[idx] = dict(
                        observations=[],
                        actions=[],
                        rewards=[],
                        env_infos=[],
                        agent_infos=[],
                    )
                running_paths[idx]["observations"].append(observation)
                running_paths[idx]["actions"].append(action)
                running_paths[idx]["rewards"].append(reward)
                running_paths[idx]["env_infos"].append(env_info)
                running_paths[idx]["agent_infos"].append(agent_info)
                if done:
                    try:
                        paths[idx // self.envs_per_task].append(dict(
                            observations=np.asarray(running_paths[idx]["observations"]),
                            actions=np.asarray(running_paths[idx]["actions"]),
                            rewards=np.asarray(running_paths[idx]["rewards"]),
                            env_infos=utils.stack_tensor_dict_list(running_paths[idx]["env_infos"]),
                            agent_infos=utils.stack_tensor_dict_list(running_paths[idx]["agent_infos"]),
                        ))
                    except:
                        import pdb; pdb.set_trace()
                    n_samples += len(running_paths[idx]["rewards"])
                    running_paths[idx] = None
            process_time += time.time() - t
            pbar.inc(len(obses))
            obses = next_obses
        pbar.stop()

        logger.logkv(log_prefix+"PolicyExecTime", policy_time)
        logger.logkv(log_prefix+"EnvExecTime", env_time)
        logger.logkv(log_prefix+"ProcessExecTime", process_time)

        return paths