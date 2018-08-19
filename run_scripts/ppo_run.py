from maml_zoo.baselines.linear_feature_baseline import LinearFeatureBaseline
from maml_zoo.envs.point_env_2d import MetaPointEnv
from maml_zoo.envs.half_cheetah_rand_direc import HalfCheetahRandDirecEnv
from maml_zoo.meta_algos.ppo_maml import MAMLPPO
from maml_zoo.meta_trainer import Trainer
from maml_zoo.samplers.maml_sampler import MAMLSampler
from maml_zoo.samplers.maml_sample_processor import MAMLSampleProcessor
from maml_zoo.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
from maml_zoo.logger import logger

import numpy as np
import tensorflow as tf
baseline = LinearFeatureBaseline()

# env = MetaPointEnv() # Wrappers? normalization?
logger.configure()
env = HalfCheetahRandDirecEnv()

policy = MetaGaussianMLPPolicy(
        name="meta-policy",
        obs_dim=np.prod(env.observation_space.shape),
        action_dim=np.prod(env.action_space.shape),
        meta_batch_size=20,
        hidden_sizes=(64, 64),
        learn_std=True,
        hidden_nonlinearity=tf.tanh,
        output_nonlinearity=None,
    )

sampler = MAMLSampler(
    env=env,
    policy=policy,
    batch_size=5,  # This batch_size is confusing
    meta_batch_size=20,
    max_path_length=100,
    parallel=False,
)

sample_processor = MAMLSampleProcessor(
    baseline=baseline,
    discount=0.99,
    gae_lambda=1.0,
    normalize_adv=True,
    positive_adv=False,
)

algo = MAMLPPO(
    policy=policy,
    inner_lr=0.1,
    meta_batch_size=20,
    num_inner_grad_steps=1,
    learning_rate=1e-3,
    max_epochs=5,
    num_minibatches=1,
    clip_eps=0.5,
    clip_outer=True,
    target_outer_step=0,
    target_inner_step=2e-2,
    init_outer_kl_penalty=0,
    init_inner_kl_penalty=1e-3,
    adaptive_outer_kl_penalty=False,
    adaptive_inner_kl_penalty=True,
    anneal_factor=1.0,
    entropy_bonus=0.0,
)

trainer = Trainer(
    algo=algo,
    policy=policy,
    env=env,
    sampler=sampler,
    sample_processor=sample_processor,
    n_itr=300,
    num_inner_grad_steps=1,  # This is repeated in MAMLPPO, it's confusing
)

trainer.train()