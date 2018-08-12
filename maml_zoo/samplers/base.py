import numpy as np

class Sampler(object):
    def __init__(
            self,
            batch_size,
            max_path_length,
            ):
        """
        Args:
            env (Env) : 
            policy (Policy) : 
            batch_size (int) : number of trajectories per task
            max_path_length (int) : max number of steps per trajectory
        """
        self.env = None
        self.policy = None
        self.batch_size = batch_size
        self.max_path_length = max_path_length

    def build_sampler(self, env, policy):
        self.env = env
        self.env_spec = env.env_spec
        self.policy = policy

    def obtain_samples(self):
        """
        Collect batch_size trajectories
        Returns: 
            (list) : A list of paths.
        """
        raise NotImplementedError

class SampleProcessor(object):
    def __init__(
            self, 
            discount=0.99,
            gae_lambda=1,
            center_adv=False,
            positive_adv=False,
            ):
        """ 
        Args:
            discount (float) :
            gae_lambda (float) : 
            center_adv (bool) : 
            positive_adv (bool) : 
        """
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.center_adv = center_adv
        self.positive_adv = positive_adv
        self.baseline = None

    def build_sample_processor(self, baseline):
        """
        Args:
            baseline (Baseline) : 
        """
        self.baseline = baseline

    def process_samples(self, paths):
        """
        Return processed sample data (typically a dictionary of concatenated tensors) based on the collected paths.
        Args:
            paths (list): A list of collected paths.
        Returns:
            (list) : Processed sample data.
        """
        raise NotImplementedError