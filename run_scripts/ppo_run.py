from maml_zoo.baselines.linear_feature_baseline import LinearFeatureBaseline
from maml_zoo.envs.half_cheetah_rand_direc import HalfCheetahRandDirec # Not yet implemented
from maml_zoo.meta_algos.ppo_maml import MAMLPPO
from maml_zoo.meta_trainer import Trainer
from maml_zoo.samplers.maml_sampler import MAMLSampler
from maml_zoo.samplers.maml_sample_processor import MAMLSampleProcessor
from maml_zoo.policies. # todo
from maml_zoo.logger import logger


baseline = LinearFeatureBaseline()

env = HalfCheetahRandDirec() # Wrappers? normalization?

sampler = MAMLSampler(


	)

sample_processor = MAMLSampleProcessor(
		discount=0.99,
        gae_lambda=1,
        center_adv=False,
        positive_adv=False,
	)

policy = MAMLGaussianMLPPolicy(

	)


algo = MAMLPPO(
		optimizer=None, # Todo: how to define optimizer?
        inner_lr=0.1,
        clip_eps=0.5, 
        clip_outer=True,
        target_outer_step=0.001,
        target_inner_step=0.01,
        init_outer_kl_penalty=1e-3,
        init_inner_kl_penalty=1e-2,
        adaptive_outer_kl_penalty=True,
        adaptive_inner_kl_penalty=True,
        anneal_factor=1,
        multi_adam=False,
        num_inner_grad_steps=1,
	)

trainer = Trainer(
        algo=algo,
        env=env,
        sampler=sampler,
        baseline=baseline,
        policy=policy,
        n_itr=100,
        meta_batch_size=40,
        num_grad_updates=1,
        scope=None,
        load_policy=None,
	)

trainer.train()

# variant generator, run_experiment code goes here