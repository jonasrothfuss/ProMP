from maml_zoo.samplers.base import SampleProcessor
from maml_zoo.util import utils

class MAMLSampleProcessor(Sampler):
    def process_samples(self, paths, log=True, log_prefix=''):
        """
        Return processed sample data (typically a dictionary of concatenated tensors) based on the collected paths.
        Args:
            paths (list): A list of paths of size (batch_size) x [5] x (max_path_length)
        Returns:
            (dict) : Processed sample data of size [7] x (batch_size x max_path_length)
        """
        baselines = []
        returns = []

        for idx, path in enumerate(paths):
            path["returns"] = utils.discount_cumsum(path["rewards"], self.discount)
        
        if log: logger.log("fitting baseline...")
        self.baseline.fit(paths, log=log)
        if log: logger.log("fitted")

        all_path_baselines = [self.baseline.predict(path) for path in paths]

        for idx, path in enumerate(paths):
            path_baselines = np.append(all_path_baselines[idx], 0)
            deltas = path["rewards"] + \
                     self.discount * path_baselines[1:] - \
                     path_baselines[:-1]
            path["advantages"] = special.discount_cumsum(
                deltas, self.discount * self.gae_lambda)
            baselines.append(path_baselines[:-1])
            returns.append(path["returns"])

        observations = np.concatenate([path["observations"] for path in paths])
        actions = np.concatenate([path["actions"] for path in paths])
        rewards = np.concatenate([path["rewards"] for path in paths])
        returns = np.concatenate([path["returns"] for path in paths])
        advantages = np.concatenate([path["advantages"] for path in paths])
        env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])
        agent_infos = tensor_utils.concat_tensor_dict_list([path["agent_infos"] for path in paths])

        if self.center_adv:
            advantages = utils.center_advantages(advantages)

        if self.positive_adv:
            advantages = utils.shift_advantages_to_positive(advantages)

        average_discounted_return = \
            np.mean([path["returns"][0] for path in paths])

        undiscounted_returns = [sum(path["rewards"]) for path in paths]

        samples_data = dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            returns=returns,
            advantages=advantages,
            env_infos=env_infos,
            agent_infos=agent_infos,
            paths=paths,
        )

        # ent = np.mean(self.policy.distribution.entropy(agent_infos)) # Todo: give access to policy?

        ev = utils.explained_variance_1d(
            np.concatenate(baselines),
            returns
        )

        # Todo: Log for all paths
        if log == 'reward':
            logger.record_tabular(log_prefix + 'AverageReturn', np.mean(undiscounted_returns))
        elif log == 'all' or log is True:
            logger.record_tabular(log_prefix + 'AverageDiscountedReturn',
                                  average_discounted_return)
            logger.record_tabular(log_prefix + 'AverageReturn', np.mean(undiscounted_returns))
            logger.record_tabular(log_prefix + 'ExplainedVariance', ev)
            logger.record_tabular(log_prefix + 'NumTrajs', len(paths))
            logger.record_tabular(log_prefix + 'Entropy', ent)
            logger.record_tabular(log_prefix + 'Perplexity', np.exp(ent))
            logger.record_tabular(log_prefix + 'StdReturn', np.std(undiscounted_returns))
            logger.record_tabular(log_prefix + 'MaxReturn', np.max(undiscounted_returns))
            logger.record_tabular(log_prefix + 'MinReturn', np.min(undiscounted_returns))

        return samples_data