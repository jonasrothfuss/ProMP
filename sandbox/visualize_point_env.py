import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import sys
import joblib
import tensorflow as tf
import pickle
from maml_zoo.logger import logger
from maml_zoo.envs.normalized_env import normalize
from maml_zoo.envs.point_env_2d_corner import MetaPointEnvCorner
from maml_zoo.envs.point_env_2d_walls import MetaPointEnvWalls
from maml_zoo.envs.point_env_2d_momentum import MetaPointEnvMomentum
from maml_zoo.samplers.vectorized_env_executor import MAMLIterativeEnvExecutor
from maml_zoo.baselines.linear_baseline import LinearFeatureBaseline
from maml_zoo.samplers.maml_sampler import MAMLSampler
from maml_zoo.samplers.maml_sample_processor import MAMLSampleProcessor
from maml_zoo.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from maml_zoo.meta_algos.vpg_maml import VPGMAML

def save_plots(policy, algo, sampler, sample_processor, dir='.', log=False):
    fig = plt.figure()
    
    tasks = env.sample_tasks(40)
    sampler.vec_env.set_tasks(tasks)
    
    axes = []
    for i in range(6):
        ax = fig.add_subplot(231 + i)
        axes.append(ax)
        ax.set(adjustable='box-forced', aspect='equal')
        plt.xlim(-2.5, 2.5)
        plt.ylim(-2.5, 2.5)
        if args.env == 'walls':
            ax.scatter(tasks[i]['goal'][0], tasks[i]['goal'][1], c='g', s=100**2)
            circle1 = plt.Circle((0, 0), 1, color='y', fill=False, zorder=-2)
            circle2 = plt.Circle((0, 0), 2, color='y', fill=False, zorder=-2)
            ax.add_artist(circle1)
            ax.add_artist(circle2)
            gap1 = plt.Circle(tasks[i]['gap_1'], 1, color='w', fill=True, zorder=-1)
            gap2 = plt.Circle(tasks[i]['gap_2'], 1, color='w', fill=True, zorder=-1)
            ax.add_artist(gap1)
            ax.add_artist(gap2)
        else:
            ax.scatter(tasks[i][0], tasks[i][1], c='g', s=100**2)

    all_samples_data = []
    paths = sampler.obtain_samples(log=False)
    samples_data = sample_processor.process_samples(paths, log=True, log_prefix="0_")
    all_samples_data.append(samples_data)
    for i in range(6): # plot 6 meta_tasks
        ax = axes[i]
        for j in range(5): # plot 5 trajectories per task
            obses = paths[i][j]['observations']
            colors = np.arange(len(obses))
            ax.scatter(obses[:,0], obses[:,1], c=colors, cmap='Blues')

    color_list = ['Purples', 'Reds', 'Oranges', 'YlGn', 'Greens']
    for it, color in enumerate(color_list):
        # Update:
        algo._adapt(samples_data)

        # Postupdate:
        paths = sampler.obtain_samples(log=False)
        samples_data = sample_processor.process_samples(paths, log=True, log_prefix="%i_" % (it + 1))
        all_samples_data.append(samples_data)

        for i in range(6): # plot 6 meta_tasks
            ax = axes[i]
            for j in range(5): # plot 5 trajectories per task
                obses = paths[i][j]['observations']
                colors = np.arange(len(obses))
                ax.scatter(obses[:,0], obses[:,1], c=colors, cmap=color)

        if it == algo.num_inner_grad_steps - 1:
            break

    if log:
        logger.dumpkvs()
        with open('sandbox/ll_exp.pkl', 'wb') as pickle_file:
            pickle.dump(all_samples_data, pickle_file)
    # plt.savefig(dir + '/ll_mom_exp.png')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--policy', type=str, default=None,
                        help='policy to load')
    parser.add_argument('--env', type=str, default='corner',
                        help='which point env to use')
    args = parser.parse_args(sys.argv[1:])

    logger.configure(format_strs=['stdout'])

    sess = tf.InteractiveSession()

    policy = joblib.load(args.policy)['policy']
    policy.switch_to_pre_update()

    baseline = LinearFeatureBaseline()

    if args.env == 'corner':
        env = MetaPointEnvCorner('sparse')
    elif args.env == 'momentum':
        env = MetaPointEnvMomentum('sparse')
    elif args.env == 'walls':
        env = MetaPointEnvWalls('dense')
    else:
        raise NotImplementedError()
    env = normalize(env)

    env.set_task(env.sample_tasks(1)[0])

    sampler = MAMLSampler(
        env=env,
        policy=policy,
        rollouts_per_meta_task=20,
        meta_batch_size=40,
        max_path_length=100,
        parallel=True,
    )

    sample_processor = MAMLSampleProcessor(
        baseline=baseline,
        discount=0.99,
        gae_lambda=1,
        normalize_adv=True,
        positive_adv=False,
    )

    algo = VPGMAML(
        policy=policy,
        inner_lr=0.1,
        meta_batch_size=40,
        num_inner_grad_steps=3,
        inner_type='log_likelihood'
    )

    uninit_vars = [var for var in tf.global_variables() if not sess.run(tf.is_variable_initialized(var))]
    sess.run(tf.variables_initializer(uninit_vars))

    save_plots(policy, algo, sampler, sample_processor, log=True)