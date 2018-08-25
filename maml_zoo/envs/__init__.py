from maml_zoo.envs.base import MetaEnv
from gym.envs.registration import register

register(
    id='HalfCheetahRandDirec-v0',
    entry_point='maml_zoo.envs.half_cheetah_rand_direc:HalfCheetahRandDirecEnv',
)

register(
    id='HalfCheetahRandVel-v0',
    entry_point='maml_zoo.envs.half_cheetah_rand_vel:HalfCheetahRandVelEnv',
)

register(
    id='AntRandDirec-v0',
    entry_point='maml_zoo.envs.ant_rand_direc:AntRandDirecEnv',
)

register(
    id='AntRandVel-v0',
    entry_point='maml_zoo.envs.ant_rand_vel:AntRandVelEnv',
)

register(
    id='SwimmerRandVel-v0',
    entry_point='maml_zoo.envs.swimmer_rand_vel:SwimmerRandVelEnv',
)