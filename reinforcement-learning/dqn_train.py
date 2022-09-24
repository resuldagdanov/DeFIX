#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
DQN Algorithm. Tested with CARLA.
You can visualize experiment results in ~/ray_results using TensorBoard.
"""

from __future__ import print_function

import argparse
import os
import yaml

import torch
import ray

from datetime import datetime
from ray.rllib.agents.dqn import DQNTrainer
from ray.tune.logger import pretty_print

from rllib_integration.carla_env import CarlaEnv
from rllib_integration.carla_core import kill_all_servers
from rllib_integration.helper import launch_tensorboard, get_checkpoint

from dqn_example.dqn_experiment import DQNExperiment
from dqn_example.dqn_callbacks import DQNCallbacks
from dqn_example.dqn_trainer import CustomDQNTrainer

from ray.rllib.models import ModelCatalog
from custom_networks.dqn_network import DQNNetwork

# Set the experiment to EXPERIMENT_CLASS so that it is passed to the configuration
EXPERIMENT_CLASS = DQNExperiment

ModelCatalog.register_custom_model("custom_dqn_network", DQNNetwork)


def run(args):
    try:
        ray.init(address= "auto" if args.auto else None)

        num_of_iterations = args.n_iters
        checkpoint_save_freq = 25

        # ray.tune.run(
        #     CustomDQNTrainer,
        #     name=args.name,
        #     local_dir=args.directory,
        #     stop={
        #         # "perf/ram_util_percent": 85.0},
        #         "training_iteration": num_of_iterations
        #     },
        #     checkpoint_freq=checkpoint_save_freq,
        #     checkpoint_at_end=True,
        #     restore=get_checkpoint(args.name, args.directory, args.restore, args.overwrite),
        #     config=args.config,
        #     queue_trials=True
        #     )

        trainer = DQNTrainer(config=args.config, env=CarlaEnv)

        for _iter in range(num_of_iterations):
            result = trainer.train()
            print("[Info] -> Result: ", pretty_print(result))

            if _iter % checkpoint_save_freq == 0:

                # saves checkpoint
                # checkpoint = trainer.save()
                # print("\n[Info] -> Checkpoint Saved @:", checkpoint)

                # default policy
                policy = trainer.get_policy()
                model = policy.model

                epsilon = policy.exploration.get_info()['cur_epsilon']

                # saving torch model
                torch.save(model.fc_layers.state_dict(), os.path.join(args.directory, "checkpoint_iter_" + str(_iter) + ".pth"))
                print("[Info] -> Model Checkpoint is Saved ! when exploration epsilon value is :", epsilon)
        
    finally:
        print("\n[Info]: Shut Down!")
        kill_all_servers()
        ray.shutdown()


def parse_config(args):
    """
    Parses the .yaml configuration file into a readable dictionary
    """
    with open(args.configuration_file) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        config["env"] = CarlaEnv
        config["env_config"]["experiment"]["type"] = EXPERIMENT_CLASS
        config["callbacks"] = DQNCallbacks
        config["evaluation_config"]["env_config"] = config["env_config"]

    config = update_routes_and_scenarios_files(config)

    return config


def update_routes_and_scenarios_files(config):
    # current working directory
    cwd = os.getcwd()
    path = cwd + "/custom_scenario_runner/"

    if not os.path.exists(path):
        raise Exception(path + " does not exist!")

    config["env_config"]["experiment"]["hero"]["routes"] = path + config["env_config"]["experiment"]["hero"]["routes"]
    config["env_config"]["experiment"]["hero"]["scenarios"] = path + config["env_config"]["experiment"]["hero"]["scenarios"]

    if not os.path.exists(config["env_config"]["experiment"]["hero"]["routes"]):
        raise Exception(config["env_config"]["experiment"]["hero"]["routes"] + " does not exist!")

    if not os.path.exists(config["env_config"]["experiment"]["hero"]["scenarios"]):
        raise Exception(config["env_config"]["experiment"]["hero"]["scenarios"] + " does not exist!")

    return config


def main():
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument("configuration_file",
                           help="Configuration file (*.yaml)")
    argparser.add_argument("-d", "--directory",
                           metavar='D',
                           default=os.path.expanduser("~") + "/ray_results/carla_rllib",
                           help="Specified directory to save results (default: ~/ray_results/carla_rllib")
    argparser.add_argument("-n", "--name",
                           metavar="N",
                           default="dqn_example",
                           help="Name of the experiment (default: dqn_example)")
    argparser.add_argument("--restore",
                           action="store_true",
                           default=False,
                           help="Flag to restore from the specified directory")
    argparser.add_argument("--overwrite",
                           action="store_true",
                           default=False,
                           help="Flag to overwrite a specific directory (warning: all content of the folder will be lost.)")
    argparser.add_argument("--tboff",
                           action="store_true",
                           default=False,
                           help="Flag to deactivate Tensorboard")
    argparser.add_argument("--auto",
                           action="store_true",
                           default=False,
                           help="Flag to use auto address")
    argparser.add_argument("-i", "--n_iters",
                           metavar="I",
                           default=200_000,
                           help="Total number of training iterations")

    args = argparser.parse_args()
    args.config = parse_config(args)

    today = datetime.today() # month - date - year
    now = datetime.now() # hours - minutes - seconds

    current_date = str(today.strftime("%b_%d_%Y"))
    current_time = str(now.strftime("%H_%M_%S"))

    # month_date_year-hour_minute_second
    time_info = current_date + "-" + current_time + "/"

    # directory for saving trained model checkpoints
    checkpoint_path = os.path.join(os.path.join(os.environ.get('DeFIX_PATH'), "checkpoint/models/reinforcement/"), time_info)
    os.mkdir(checkpoint_path)
    args.directory = checkpoint_path

    if not args.tboff:
        launch_tensorboard(logdir=os.path.join(args.directory, args.name), host="0.0.0.0" if args.auto else "localhost")

    run(args)


if __name__ == '__main__':

    try:
        main()

    except KeyboardInterrupt:
        pass
    
    finally:
        print('\n chuff chuff is FINISHED !')
