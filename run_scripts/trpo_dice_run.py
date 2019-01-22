from maml_zoo.envs.point_envs.point_env_2d import MetaPointEnv
from maml_zoo.envs.mujoco_envs.half_cheetah_rand_direc import HalfCheetahRandDirecEnv
from maml_zoo.envs.normalized_env import normalize
from maml_zoo.meta_algos.trpo_dice_maml import TRPO_DICEMAML
from maml_zoo.meta_trainer import Trainer
from maml_zoo.samplers import MAMLSampler
from maml_zoo.samplers import DiceMAMLSampleProcessor
from maml_zoo.policies.meta_gaussian_mlp_policy import MetaGaussianMLPPolicy
import os
from maml_zoo.logger import logger
import json
import numpy as np


maml_zoo_path = '/'.join(os.path.realpath(os.path.dirname(__file__)).split('/')[:-1])


def main(config):
    reward_baseline = LinearTimeBaseline()
    return_baseline = LinearFeatureBaseline()
    env = normalize(HalfCheetahRandDirecEnv())

    policy = MetaGaussianMLPPolicy(
            name="meta-policy",
            obs_dim=np.prod(env.observation_space.shape),
            action_dim=np.prod(env.action_space.shape),
            meta_batch_size=config['meta_batch_size'],
            hidden_sizes=config['hidden_sizes'],
        )

    sampler = MAMLSampler(
        env=env,
        policy=policy,
        rollouts_per_meta_task=config['rollouts_per_meta_task'],  # This batch_size is confusing
        meta_batch_size=config['meta_batch_size'],
        max_path_length=config['max_path_length'],
        parallel=config['parallel'],
    )

    sample_processor = DiceMAMLSampleProcessor(
        baseline=reward_baseline,
        max_path_length=config['max_path_length'],
        discount=config['discount'],
        normalize_adv=config['normalize_adv'],
        positive_adv=config['positive_adv'],
        return_baseline=return_baseline

    )

    algo = TRPO_DICEMAML(
        policy=policy,
        max_path_length=config['max_path_length'],
        meta_batch_size=config['meta_batch_size'],
        num_inner_grad_steps=config['num_inner_grad_steps'],
        inner_lr=config['inner_lr'],
        step_size=config['step_size']
    )

    trainer = Trainer(
        algo=algo,
        policy=policy,
        env=env,
        sampler=sampler,
        sample_processor=sample_processor,
        n_itr=config['n_itr'],
        num_inner_grad_steps=config['num_inner_grad_steps'],  # This is repeated in MAMLPPO, it's confusing
    )
    trainer.train()


if __name__=="__main__":
    idx = np.random.randint(0, 1000)
    logger.configure(dir=maml_zoo_path + '/data/vpg/test_%d' % idx, format_strs=['stdout', 'log', 'csv'],
                     snapshot_mode='last_gap')
    config = json.load(open(maml_zoo_path + "/configs/trpo_dice_maml_config.json", 'r'))
    json.dump(config, open(maml_zoo_path + '/data/vpg/test_%d/params.json' % idx, 'w'))
    main(config)
