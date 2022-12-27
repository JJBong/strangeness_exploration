from functools import partial
from smac.env import StarCraft2Env
from envs.multiagentenv import MultiAgentEnv
from envs.payoff_matrix import PayOffMatrix
from envs.pettingzoo import PettingZoo
from envs.pressureplate import PressurePlateWrapper
from envs.rwarehouse import RWarehouse
import sys
import os

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["payoff_matrix"] = partial(env_fn, env=PayOffMatrix)
REGISTRY["pettingzoo"] = partial(env_fn, env=PettingZoo)
REGISTRY["pressureplate"] = partial(env_fn, env=PressurePlateWrapper)
REGISTRY["rwarehouse"] = partial(env_fn, env=RWarehouse)

# from multiagent_mujoco.multiagent_mujoco import MujocoMulti
# REGISTRY["ma_mujoco"] = partial(env_fn, env=MujocoMulti)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH",
                          os.path.join(os.getcwd(), "3rdparty", "StarCraftII"))
