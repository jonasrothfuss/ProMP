import tensorflow as tf
import numpy as np
import time
import maml_zoo.logger as logger
# import maml_zoo.utils.plotter as plotter

class Trainer(object):
    def __init__(
            self,
            algo,
            env,
            sampler,
            baseline,
            policy,
            n_itr,
            meta_batch_size,
            num_grad_updates=1,
            scope=None,
            load_policy=None,
            ):
        """
        Args:
            algo (Algo) :
            env (Env) :
            sampler (Sampler) : 
            baseline (Baseline) : 
            policy (Policy) : 
            n_itr (int) : Number of iterations to train for
            meta_batch_size (int) : Number of meta tasks
            num_grad_updates (int) : Number of inner steps per maml iteration
            scope (str) : Scope for identifying the algorithm. Must be specified if running multiple algorithms
            load_policy (Policy) : Policy to reload from
        """
        self.algo = algo
        self.sampler = sampler
        self.sample_processor = None
        # env.build_env()
        self.n_itr = n_itr
        self.meta_batch_size = meta_batch_size
        self.num_grad_updates = num_grad_updates
        self.scope = scope
        with tf.Session() as sess:
            # Code for loading a previous policy. Somewhat hacky because needs to be in sess.
            if load_policy is not None:
                import joblib
                policy = joblib.load(load_policy)['policy']
            
            policy.build_graph(env)
            sampler.build_sampler(env, policy, meta_batch_size)
            # Sample processor here?
            algo.build_graph(policy, meta_batch_size, num_grad_updates)
            
            # initialize uninitialized vars  (only initialize vars that were not loaded)
            uninit_vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
            sess.run(tf.variables_initializer(uninit_vars))

    def train(self):
        """
        Trains policy on env using algo
        Pseudocode:
            for itr in n_itr:
                for step in num_grad_updates:
                    sampler.sample()
                    algo.compute_updated_dists()
                algo.optimize_policy()
                sampler.update_goals()
        """
        with tf.Session() as sess:
            start_time = time.time()
            for itr in range(self.start_itr, self.n_itr):
                itr_start_time = time.time()
                with logger.prefix('itr #%d | ' % itr):
                    logger.log("Sampling set of tasks/goals for this meta-batch...")

                    self.sampler.update_tasks()
                    self.policy.switch_to_pre_update()  # Switch to pre-update policy

                    all_samples_data, all_paths = [], []
                    list_sampling_time, list_inner_step_time, list_outer_step_time, list_proc_samples_time = [], [], [], []
                    start_total_inner_time = time.time()
                    for step in range(self.num_grad_updates+1):
                        logger.log('** Step ' + str(step) + ' **')

                        """ -------------------- Sampling --------------------------"""

                        logger.log("Obtaining samples...")
                        time_env_sampling_start = time.time()
                        paths = self.sampler.obtain_samples(log_prefix=str(step))
                        list_sampling_time.append(time.time() - time_env_sampling_start)
                        all_paths.append(paths)

                        """ ----------------- Processing Samples ---------------------"""

                        logger.log("Processing samples...")
                        time_proc_samples_start = time.time()
                        samples_data = {}
                        for key in paths.keys(): # Keys are tasks
                            samples_data[key] = self.sampler_processor.process_samples(paths[key], log=False)
                        all_samples_data.append(samples_data)
                        list_proc_samples_time.append(time.time() - time_proc_samples_start)

                        # for logging purposes
                        self.sample_processor.process_samples(flatten_list(paths.values()), prefix=str(step), log=True)
                        logger.log("Logging diagnostics...")
                        self.log_diagnostics(flatten_list(paths.values()), prefix=str(step))

                        """ ------------------- Inner Policy Update --------------------"""

                        time_inner_step_start = time.time()
                        if step < self.num_grad_updates:
                            logger.log("Computing inner policy updates...")
                            self.algo.compute_updated_dists(samples_data)
                        list_inner_step_time.append(time.time() - time_inner_step_start)
                    total_inner_time = time.time() - start_total_inner_time

                    time_maml_opt_start = time.time()
                    """ ------------------ Outer Policy Update ---------------------"""

                    logger.log("Optimizing policy...")
                    # This needs to take all samples_data so that it can construct graph for meta-optimization.
                    time_outer_step_start = time.time()
                    self.algo.optimize_policy(all_samples_data)

                    """ ------------------- Logging Stuff --------------------------"""

                    logger.logkv('Time-OuterStep', time.time() - time_outer_step_start)
                    logger.logkv('Time-TotalInner', total_inner_time)
                    logger.logkv('Time-InnerStep', np.sum(list_inner_step_time))
                    logger.logkv('Time-SampleProc', np.sum(list_proc_samples_time))
                    logger.logkv('Time-Sampling', np.sum(list_sampling_time))

                    logger.log("Saving snapshot...")
                    params = self.get_itr_snapshot(itr)  # , **kwargs)
                    logger.save_itr_params(itr, params)
                    logger.log("Saved")

                    logger.logkv('Time', time.time() - start_time)
                    logger.logkv('ItrTime', time.time() - itr_start_time)
                    logger.logkv('Time-MAMLSteps', time.time() - time_maml_opt_start)

                    logger.dumpkvs()

    def get_itr_snapshot(self, itr):
        """
        Gets the current policy and env for storage
        """
        return dict(itr=itr, policy=self.policy, env=self.env)
