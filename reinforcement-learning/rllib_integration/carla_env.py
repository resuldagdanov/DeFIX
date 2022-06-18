#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


from __future__ import print_function

import gym

from rllib_integration.carla_core import CarlaCore
from srunner.scenariomanager.carla_data_provider import *


class CarlaEnv(gym.Env):
    """
    This is a carla environment, responsible of handling all the CARLA related steps of the training.
    """
    def __init__(self, config):
        """
        Initializes the environment
        """
        self.config = config

        self.experiment = self.config["experiment"]["type"](self.config["experiment"])
        self.action_space = self.experiment.get_action_space()
        self.observation_space = self.experiment.get_observation_space()

        self.core = CarlaCore(self.config['carla'])
        self.core.setup_experiment(self.experiment.config)

        self.reset()

    def reset(self):
        # Reset sensors hero and experiment
        self.hero = self.core.reset_hero(self.experiment.config["hero"])
        self.experiment.reset()

        # set route planner into experiment
        self.experiment.route_planner = self.core.route_planner
        self.experiment.command_planner = self.core.command_planner

        # Tick once and get the observations
        sensor_data = self.core.tick(None)
        observation, _ = self.experiment.get_observation(sensor_data, self.hero)

        return observation

    def step(self, action):
        """
        Computes one tick of the environment in order to return the new observation,
        as well as the rewards
        """
        control = self.experiment.compute_action(core=self.core, action_value=action)
        sensor_data = self.core.tick(control)

        observation, info = self.experiment.get_observation(sensor_data, self.hero)
        done = self.experiment.get_done_status(sensor_data, self.core)
        reward = self.experiment.compute_reward(sensor_data, self.core)

        print(f"action {action} control {control} reward {reward} done {done}\n")

        if done:
            print("*******End of the episode!*******")
        return observation, reward, done, info
