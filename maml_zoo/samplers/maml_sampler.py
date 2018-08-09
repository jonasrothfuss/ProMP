from maml_zoo.samplers.base import Sampler
from maml_zoo.samplers.iterative_env_executor import MAMLIterativeEnvExecutor
from maml_zoo.samplers.parallel_env_executor import MAMLParallelEnvExecutor
from maml_zoo.utils.progbar import ProgBarCounter
import time

class MAMLSampler(Sampler):
    def __init__(
            self,
            env,
            policy,
            batch_size,
            meta_batch_size,
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
        super(self, MAMLSampler).__init__(env, policy, batch_size, max_path_length, n_envs)
        self.meta_batch_size = meta_batch_size
        if envs_per_task is None:
            self.envs_per_task = batch_size
        else:
            self.envs_per_task = envs_per_task
        self.parallel = parallel

    def build_sampler(self, env, policy):
        super(self, MAMLSampler).build_sampler(env, policy)
        if self.parallel:
            self.vec_env = MAMLParallelEnvExecutor(env, self.meta_batch_size, self.envs_per_task, self.max_path_length)
        else:
            self.vec_env = MAMLIterativeEnvExecutor(env, self.meta_batch_size, self.envs_per_task, self.max_path_length)

    def set_tasks(self, tasks):
        """
        Sets the parameters for all environments corresponding to each task
        Args:
            tasks (list) : a list of length meta_batch_size, specifying reset args for each task
        """
        assert len(tasks) == self.meta_batch_size
        self.vec_env.set_task(tasks)

    def obtain_samples(self, log_prefix=''):
        """
        Collect batch_size trajectories from each task
        Args:
            log_prefix (str) : prefix for logger
        Returns: 
            (dict) : A dict of paths of size [meta_batch_size] x (batch_size) x [5] x (max_path_length)
        """
        paths = {}
        for i in range(self.n_tasks):
            paths[i] = []

        envs_per_task = self.vec_env.envs_per_task

        n_samples = 0
        obses = self.vec_env.reset()
        running_paths = [None] * self.vec_env.num_envs

        pbar = ProgBarCounter(self.algo.batch_size)
        policy_time = 0
        env_time = 0
        process_time = 0

        policy = self.algo.policy
        
        while n_samples < self.algo.batch_size:
            t = time.time()

            obs_per_task = np.split(np.asarray(obses), self.n_tasks)
            actions, agent_infos = policy.get_actions_batch(obs_per_task)

            policy_time += time.time() - t
            t = time.time()
            next_obses, rewards, dones, env_infos = self.vec_env.step(actions, reset_args)
            env_time += time.time() - t

            t = time.time()

            agent_infos = tensor_utils.split_tensor_dict_list(agent_infos)
            if env_infos is None:
                env_infos = [dict() for _ in range(self.vec_env.num_envs)]
            if agent_infos is None:
                agent_infos = [dict() for _ in range(self.vec_env.num_envs)]
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
                    paths[idx // n_envs_per_task].append(dict(
                        observations=self.env_spec.observation_space.flatten_n(running_paths[idx]["observations"]),
                        actions=self.env_spec.action_space.flatten_n(running_paths[idx]["actions"]),
                        rewards=np.asarray(running_paths[idx]["rewards"]),
                        env_infos=np.asarray(running_paths[idx]["env_infos"]),
                        agent_infos=np.asarray(running_paths[idx]["agent_infos"]),
                    ))
                    n_samples += len(running_paths[idx]["rewards"])
                    running_paths[idx] = None
            process_time += time.time() - t
            pbar.inc(len(obses))
            obses = next_obses
        pbar.stop()

        logger.record_tabular(log_prefix+"PolicyExecTime", policy_time)
        logger.record_tabular(log_prefix+"EnvExecTime", env_time)
        logger.record_tabular(log_prefix+"ProcessExecTime", process_time)

        return paths