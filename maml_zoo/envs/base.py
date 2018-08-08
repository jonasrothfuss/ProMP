from gym.envs.mujoco.mujoco_env import MujocoEnv


class MetaEnv(MujocoEnv):
    def sample_tasks(self, n_tasks):
        """ 
        Args:
            n_tasks (int) : number of different meta-tasks needed
        Returns:
            tasks (list) : an (n_tasks) length list of reset args
        """
        raise NotImplementedError

    def set_task(self, task):
        """
        Args:
            task: task of the meta-learning environment

        """
        raise NotImplementedError

    def get_task(self):
        """
        Returns:
            task: task of the meta-learning environment

        """
        raise NotImplementedError
