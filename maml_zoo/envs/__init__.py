from maml_zoo.envs.base import MetaEnv
from gym.envs.registration import register

register(
    id='HalfCheetahRandDirec-v0',
    entry_point='maml_zoo.envs.half_cheetah_rand_direc:HalfCheetahRandDirecEnv',
)