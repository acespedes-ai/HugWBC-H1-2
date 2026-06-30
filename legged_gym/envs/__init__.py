from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR

from .h1.h1 import H1Robot
from legged_gym.envs.h1.h1interrupt import H1InterruptRobot

from legged_gym.envs.h1.h1_config import H1Cfg, H1CfgPPO
from legged_gym.envs.h1.h1interrupt_config import H1InterruptCfg, H1InterruptCfgPPO

from legged_gym.envs.h1_2.h1_2 import H1_2Robot
from legged_gym.envs.h1_2.h1_2_config import H1_2Cfg, H1_2CfgPPO
from legged_gym.envs.h1_2.h1_2interrupt import H1_2InterruptRobot
from legged_gym.envs.h1_2.h1_2interrupt_config import H1_2InterruptCfg, H1_2InterruptCfgPPO
from legged_gym.envs.h1_2.h1_2_walk import H1_2WalkRobot
from legged_gym.envs.h1_2.h1_2_walk_config import H1_2WalkCfg, H1_2WalkCfgPPO

from legged_gym.utils.task_registry import task_registry

task_registry.register( "h1int",    H1InterruptRobot,   H1InterruptCfg(),   H1InterruptCfgPPO())
task_registry.register( "h1_2",     H1_2Robot,          H1_2Cfg(),          H1_2CfgPPO())
task_registry.register( "h1_2int",  H1_2InterruptRobot, H1_2InterruptCfg(), H1_2InterruptCfgPPO())
task_registry.register( "h1_2walk", H1_2WalkRobot,      H1_2WalkCfg(),      H1_2WalkCfgPPO())

