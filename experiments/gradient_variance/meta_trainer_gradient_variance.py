import tensorflow as tf
import numpy as np
import time
from maml_zoo.logger import logger
from scipy.spatial.distance import cdist


class TrainerGradientStd(object):
    """
    Performs steps for MAML

    Args:
        algo (Algo) :
        env (Env) :
        sampler (Sampler) : 
        sample_processor (SampleProcessor) : 
        baseline (Baseline) : 
        policy (Policy) : 
        n_itr (int) : Number of iterations to train for
        start_itr (int) : Number of iterations policy has already trained for, if reloading
        num_inner_grad_steps (int) : Number of inner steps per maml iteration
        sess (tf.Session) : current tf session (if we loaded policy, for example)
        num_sapling_rounds
    """
    def __init__(
            self,
            algo,
            env,
            sampler,
            sample_processor,
            policy,
            n_itr,
            start_itr=0,
            num_inner_grad_steps=1,
            sess=None,
            num_sapling_rounds=10
            ):
        self.algo = algo
        self.env = env
        self.sampler = sampler
        self.sample_processor = sample_processor
        self.baseline = sample_processor.baseline
        self.policy = policy
        self.n_itr = n_itr
        self.start_itr = start_itr
        self.num_inner_grad_steps = num_inner_grad_steps
        self.num_sapling_rounds = num_sapling_rounds
        if sess is None:
            sess = tf.Session()
        self.sess = sess

    def train(self):
        """
        Trains policy on env using algo

        Pseudocode:
            for itr in n_itr:
                for step in num_inner_grad_steps:
                    sampler.sample()
                    algo.compute_updated_dists()
                algo.optimize_policy()
                sampler.update_goals()
        """
        with self.sess.as_default() as sess:

            # initialize uninitialized vars  (only initialize vars that were not loaded)
            uninit_vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
            sess.run(tf.variables_initializer(uninit_vars))
            n_timesteps = 0

            start_time = time.time()
            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                logger.log("\n ---------------- Iteration %d ----------------" % itr)

                gradients = []
                for i in range(self.num_sapling_rounds):
                    logger.log("\n ----- Sampling Round %d ---" % i)

                    dry = i < self.num_sapling_rounds-1

                    if not dry: self.sampler.update_tasks()
                    self.policy.switch_to_pre_update()  # Switch to pre-update policy

                    all_samples_data, all_paths = [], []

                    for step in range(self.num_inner_grad_steps+1):
                        logger.log('** Step ' + str(step) + ' **')

                        logger.log("Obtaining samples...")
                        paths = self.sampler.obtain_samples(log=True, log_prefix='Step_%d-' % step)
                        all_paths.append(paths)

                        logger.log("Processing samples...")
                        samples_data = self.sample_processor.process_samples(paths, log='all', log_prefix='Step_%d-' % step)
                        all_samples_data.append(samples_data)

                        if not dry: self.log_diagnostics(sum(list(paths.values()), []), prefix='Step_%d-' % step)

                        if step < self.num_inner_grad_steps:
                            logger.log("Computing inner policy updates...")
                            self.algo._adapt(samples_data)

                    """ compute gradients """
                    gradients.append(self.algo.compute_gradients(all_samples_data))

                    if not dry:
                        """ ------------ Compute and log gradient variance ------------"""
                        # compute variance of adaptation gradients
                        for step_id in range(self.num_inner_grad_steps):
                            meta_batch_size = len(gradients[0][0])
                            grad_std, grad_rstd = [], []
                            for task_id in range(meta_batch_size):
                                stacked_grads = np.stack([gradients[round_id][step_id][task_id]
                                                          for round_id in range(self.num_sapling_rounds)], axis=1)
                                std = np.std(stacked_grads, axis=1)
                                mean = np.abs(np.mean(stacked_grads, axis=1))
                                grad_std.append(np.mean(std))
                                grad_rstd.append(np.mean(std/mean))

                            logger.logkv('Step_%i-GradientMean', np.mean(mean))
                            logger.logkv('Step_%i-GradientStd'%step_id, np.mean(grad_std))
                            logger.logkv('Step_%i-GradientRStd' % step_id, np.mean(grad_rstd))

                        # compute variance of meta gradients
                        stacked_grads = np.stack([gradients[round_id][self.num_inner_grad_steps]
                                                  for round_id in range(self.num_sapling_rounds)], axis=1)
                        std = np.std(stacked_grads, axis=1)
                        mean = np.abs(np.mean(stacked_grads, axis=1))

                        meta_grad_std = np.mean(std)
                        meta_grad_rstd = np.mean(std/(mean + 1e-8))
                        meta_grad_rvar = np.mean(std**2/(mean + 1e-8))

                        logger.logkv('Meta-GradientMean', np.mean(mean))
                        logger.logkv('Meta-GradientStd', meta_grad_std)
                        logger.logkv('Meta-GradientRStd', meta_grad_rstd)
                        logger.logkv('Meta-GradientRVariance', meta_grad_rvar)

                        # compute cosine dists
                        cosine_dists = cdist(np.transpose(stacked_grads), np.transpose(np.mean(stacked_grads, axis=1).reshape((-1, 1))),
                                             metric='cosine')
                        mean_abs_cos_dist = np.mean(np.abs(cosine_dists))
                        mean_squared_cosine_dists = np.mean(cosine_dists**2)
                        mean_squared_cosine_dists_sqrt = np.sqrt(mean_squared_cosine_dists)

                        logger.logkv('Meta-GradientCosAbs', mean_abs_cos_dist)
                        logger.logkv('Meta-GradientCosVar', mean_squared_cosine_dists)
                        logger.logkv('Meta-GradientCosStd', mean_squared_cosine_dists_sqrt)


                        """ ------------------ Outer Policy Update ---------------------"""

                        logger.log("Optimizing policy...")
                        # This needs to take all samples_data so that it can construct graph for meta-optimization.
                        self.algo.optimize_policy(all_samples_data)

                        """ ------------------- Logging Stuff --------------------------"""
                        n_timesteps += (self.num_inner_grad_steps+1) * self.sampler.total_samples
                        logger.logkv('n_timesteps', n_timesteps)

                        logger.log("Saving snapshot...")
                        params = self.get_itr_snapshot(itr)  # , **kwargs)
                        logger.save_itr_params(itr, params)
                        logger.log("Saved")

                        logger.logkv('Itr', itr)
                        logger.logkv('Time', time.time() - start_time)
                        logger.logkv('ItrTime', time.time() - itr_start_time)

                logger.dumpkvs()

        logger.log("Training finished")
        self.sess.close()        

    def get_itr_snapshot(self, itr):
        """
        Gets the current policy and env for storage
        """
        return dict(itr=itr, policy=self.policy, env=self.env, baseline=self.baseline)

    def log_diagnostics(self, paths, prefix):
        # TODO: we aren't using it so far
        self.env.log_diagnostics(paths, prefix)
        self.policy.log_diagnostics(paths, prefix)
        self.baseline.log_diagnostics(paths, prefix)
