from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

"""Example of running StarCraft2 with RLlib PPO.

In this setup, each agent will be controlled by an independent PPO policy.
However the policies share weights.

Increase the level of parallelism by changing --num-workers.
"""

import argparse

from ray.tune import run_experiments, register_env
from ray.rllib.models import ModelCatalog

from smac.examples.rllib.env import RLlibStarCraft2Env
from smac.examples.rllib.model import MaskedActionsModel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-iters", type=int, default=100)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--map-name", type=str, default="8m")
    args = parser.parse_args()

    ray.init()

    register_env("smac", lambda smac_args: RLlibStarCraft2Env(**smac_args))
    ModelCatalog.register_custom_model("mask_model", MaskedActionsModel)

    run_experiments({
        "ppo_sc2": {
            "run": "PPO",
            "env": "smac",
            "stop": {
                "training_iteration": args.NUM_ITERS,
            },
            "config": {
                "num_workers": args.NUM_WORKERS,
                "observation_filter": "NoFilter",  # breaks the action mask
                "vf_share_layers": True,  # don't create a separate value model
                "env_config": {
                    "map_name": args.MAP_NAME,
                },
                "model": {
                    "custom_model": "mask_model",
                },
            },
        },
     })
