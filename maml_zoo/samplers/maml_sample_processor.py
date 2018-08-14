from maml_zoo.samplers.base import SampleProcessor
from maml_zoo.utils import utils
from maml_zoo.logger import logger
import numpy as np

class MAMLSampleProcessor(SampleProcessor):

    def process_samples(self, paths_meta_batch, log=False, log_prefix=''):
        """
        Processes sampled paths. This involves:
            - computing discounted rewards (returns)
            - fitting baseline estimator using the path returns and predicting the return baselines
            - estimating the advantages using GAE (+ advantage normalization id desired)
            - stacking the path data
            - logging statistics of the paths

        Args:
            paths (dict): A list of dict of lists, size: [meta_batch_size] x (batch_size) x [5] x (max_path_length)
            log (boolean): indicates whether to log
            log_prefix (str): prefix for the logging keys

        Returns:
            (dict of dicts) : Processed sample data among the meta-batch; size: [meta_batch_size] x [7] x (batch_size x max_path_length)
        """
        assert type(paths_meta_batch) == dict, 'paths must be a list'
        assert self.baseline, 'baseline must be specified - use self.build_sample_processor(baseline_obj)'

        samples_data_meta_batch = {}
        all_paths = []
        for meta_task, paths in paths_meta_batch.items():
            # 1) compute discounted rewards (returns)
            for idx, path in enumerate(paths):
                path["returns"] = utils.discount_cumsum(path["rewards"], self.discount)

            # 2) fit baseline estimator using the path returns and predict the return baselines
            self.baseline.fit(paths)
            all_path_baselines = [self.baseline.predict(path) for path in paths]

            # 3) compute advantages
            paths = self._compute_advantages(paths, all_path_baselines)

            # 4) stack path data
            observations, actions, rewards, returns, advantages, env_infos, agent_infos = self._stack_path_data(paths)

            # 5) if desired normalize / shift advantages
            if self.normalize_adv:
                advantages = utils.normalize_advantages(advantages)
            if self.positive_adv:
                advantages = utils.shift_advantages_to_positive(advantages)

            # 6) create samples_data object
            samples_data = dict(
                observations=observations,
                actions=actions,
                rewards=rewards,
                returns=returns,
                advantages=advantages,
                env_infos=env_infos,
                agent_infos=agent_infos,
            )
            samples_data_meta_batch[meta_task] = samples_data
            all_paths.extend(paths)


        # 7) log statistics if desired
        self._log_path_stats(all_paths, log=log, log_prefix='')

        assert all([set(samples_data.keys()) >= set(('observations', 'actions', 'rewards', 'advantages', 'returns'))
                for samples_data in samples_data_meta_batch.values()])
        return samples_data_meta_batch