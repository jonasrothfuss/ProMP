import numpy as np
from rllab_maml.misc import special
from rllab_maml.misc import tensor_utils
from rllab_maml.algos import util
import rllab_maml.misc.logger as logger


class Sampler(object):
    def __init__(
            self,
            env,
            policy,
            baseline,
            batch_size,
            max_path_length,
            n_envs=None,
            discount=0.99,
            gae_lambda=1,
            center_adv=False,
            positive_adv=False,
            ):
        """
        # Do we care about non-meta samplers?
        # Make PostProcessor a separate class?
        Args:
            env (Env) : 
            policy (Policy) : 
            baseline (Baseline) : 
            batch_size (int) : number of trajectories per task
            max_path_length (int) : max number of steps per trajectory # Should this go here or in env?
            n_envs (int) : number of envs to run in parallel, multiple of n_tasks # Divisor of batch_size?
            discount (float) :
            gae_lambda (float) : 
            center_adv (bool) : 
            positive_adv (bool) : 
        """
        pass

    def obtain_samples(self):
        """
        Collect batch_size trajectories
        Returns: 
            (list) : A list of paths.
        """
        raise NotImplementedError

    def process_samples(self, paths):
        """
        Return processed sample data (typically a dictionary of concatenated tensors) based on the collected paths.
        Args:
            paths (list): A list of collected paths.
        Returns:
            (list) : Processed sample data.
        """
        raise NotImplementedError

    def start_worker(self):
        """
        Is this necessary?
        """
        raise NotImplementedError

    def shutdown_worker(self):
        """
        Is this necessary?
        """
        raise NotImplementedError


class MAMLSampler(Sampler):
    def __init__(
            self, 
            env,
            policy,
            baseline,
            meta_batch_size,
            batch_size,
            max_path_length,
            n_envs=None,
            center_adv=False,
            positive_adv=False,
            ):
        """
        Args:
            env (Env) : 
            policy (Policy) : 
            baseline (Baseline) : 
            meta_batch_size (int) : number of meta tasks
            batch_size (int) : number of trajectories per task
            max_path_length (int) : max number of steps per trajectory # Should this go here or in env?
            n_envs (int) : number of envs to run in parallel, multiple of n_tasks # Divisor of batch_size?
            center_adv (bool) : 
            positive_adv (bool) : 
        """
        pass

    def obtain_samples(self, reset_args=None):
        """
        Collect batch_size trajectories for each task
        Args:
            reset_args (reset_args) : 
        Returns: 
            (list) : A list of paths.
        """
        raise NotImplementedError
