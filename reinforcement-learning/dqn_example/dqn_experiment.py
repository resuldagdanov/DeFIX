#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import os
import math
import numpy as np
import torch
from torchvision import models
from gym.spaces import Box, Discrete

import carla

from rllib_integration.base_experiment import BaseExperiment
from rllib_integration.helper import post_process_image, process_image_for_dnn, get_position, get_speed, calculate_high_level_action, traffic_data
from rllib_integration.helper import PIDController

IS_STUCK_VEHICLE = False


class DQNExperiment(BaseExperiment):
    def __init__(self, config={}):
        super().__init__(config)  # Creates a self.config with the experiment configuration

        self.ResNetShape = (1, 1000)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.frame_stack = self.config["others"]["framestack"]
        self.max_time_idle = self.config["others"]["max_time_idle"]
        self.max_time_episode = self.config["others"]["max_time_episode"]
        
        self.allowed_types = [carla.LaneType.Driving, carla.LaneType.Parking]
        self.last_heading_deviation = 0
        self.last_action = None

        self.turn_controller = PIDController(K_P=1.25, K_I=0.75, K_D=0.3, n=40)
        self.speed_controller = PIDController(K_P=5.0, K_I=0.5, K_D=1.0, n=40)

        self.route_planner = None
        self.command_planner = None

        self.ego_gps = None
        self.previous_ego_gps = None
        self.near_node = None
        self.compass = 0.0
        self.speed_ms = 0.0

        # load pretrained ResNet
        self.resnet_model = models.resnet50(pretrained=False)

        # NOTE: comment out the following two lines if resnet model is not pre-save
        resnet_model_path = os.path.join(os.path.join(os.environ.get("DeFIX_PATH"), "checkpoint/models/"), "resnet50")
        self.resnet_model.load_state_dict(torch.load(resnet_model_path))

        # freeze weights
        for param in self.resnet_model.parameters():
            param.requires_grad = False

        self.resnet_model.eval()
        self.resnet_model.to(self.device)

    def reset(self):
        """
        Called at the beginning and each time the simulation is reset
        """
        # Ending variables
        self.time_idle = 0
        self.time_episode = 0
        self.done_time_idle = False
        self.done_falling = False
        self.done_time_episode = False
        self.done_collision = False

        self.count_steps = 0

        # hero variables
        self.last_location = None
        self.last_velocity = 0

        # sensor stack
        self.prev_image_0 = None
        self.prev_image_1 = None
        self.prev_image_2 = None

        self.last_heading_deviation = 0

    def get_action_space(self):
        """
        Returns the action space, in this case, a discrete space
        """
        return Discrete(4)

    def get_observation_space(self):
        num_of_channels = 3
        image_space = Box(
            low=-np.inf, # 0.0
            high=np.inf, # 255.0
            shape=self.ResNetShape,
            # shape=(
            #     self.config["hero"]["sensors"]["birdview"]["size"],
            #     self.config["hero"]["sensors"]["birdview"]["size"],
            #     num_of_channels * self.frame_stack,
            # ),
            dtype=np.float32,
        )
        return image_space

    def compute_action(self, core, action_value):
        """
        Given the action, returns a carla.VehicleControl() which will be applied to the hero
        """

        throttle, steer, brake = calculate_high_level_action(world=core.world,
                                                             turn_controller=self.turn_controller,
                                                             speed_controller=self.speed_controller,
                                                             high_level_action=action_value,
                                                             gps=self.ego_gps,
                                                             initial_gps=self.initial_ego_gps,
                                                             theta=self.compass,
                                                             initial_theta=self.initial_compass,
                                                             speed=self.speed_ms,
                                                             near_node=self.near_node,
                                                             far_target=self.far_node)

        action = carla.VehicleControl()
        action.throttle = float(throttle)
        action.steer = float(steer)
        action.brake = float(brake)
        action.reverse = False
        action.hand_brake = False

        self.last_action = action
        return action

    def get_observation(self, sensor_data, hero):
        """
        Function to do all the post processing of observations (sensor data).

        :param sensor_data: dictionary {sensor_name: sensor_data}

        Should return a tuple or list with two items, the processed observations,
        as well as a variable with additional information about such observation.
        The information variable can be empty
        """
        stack_images = False # TODO: remove this option

        # get orientation and position of the ego vehicle from the simulator
        ego_transform = hero.get_transform()
        
        # yaw angle of the ego vehicle in degrees
        ego_yaw_deg = ego_transform.rotation.yaw
        # simulator position of the ego vehicle (x, y, z)
        ego_location = ego_transform.location

        # NOTE: IMU sensor data is not accurate and lags, so ground truth simulator is used
        # self.compass = sensor_data['imu'][1][-1] # radians

        # map coordinate frame is rotated 90 degrees
        self.compass = np.math.radians(ego_yaw_deg + 90)
        
        # ignore IMU sensor nan values
        if np.isnan(self.compass) or self.compass is None:
            print("[Error]: compass is nan ->", self.compass)
            self.compass = 0.0

        # NOTE: GPS sensor data is not accurate and lags, so ground truth simulator is used
        # self.ego_gps = get_position(gps=sensor_data['gps'][1][:2], route_planner=self.route_planner)

        # directly convert ego vehicle position obtained from simulator to map coordinate frame (90 degrees)
        self.ego_gps = np.array([-ego_location.y, ego_location.x])
        
        if self.count_steps % 10 == 0:
            self.previous_ego_gps = self.ego_gps

        self.count_steps += 1
        
        if self.count_steps == 1:
            self.initial_compass = self.compass
            self.initial_ego_gps = self.ego_gps
        
        speed_kmph = get_speed(hero)
        self.speed_ms = speed_kmph / 3.6

        self.near_node, near_command = self.route_planner.run_step(gps=self.ego_gps)
        self.far_node, far_command = self.command_planner.run_step(gps=self.ego_gps)

        if stack_images:
            # image = post_process_image(sensor_data['birdview'][1], normalized = False, grayscale = False)
            image = post_process_image(sensor_data['front_camera'][1], normalized=False, grayscale=False) # TODO: give normalized input to the network

            if self.prev_image_0 is None:
                self.prev_image_0 = image
                self.prev_image_1 = self.prev_image_0
                self.prev_image_2 = self.prev_image_1

            images = image

            if self.frame_stack >= 2:
                images = np.concatenate([self.prev_image_0, images], axis=2)
            if self.frame_stack >= 3 and images is not None:
                images = np.concatenate([self.prev_image_1, images], axis=2)
            if self.frame_stack >= 4 and images is not None:
                images = np.concatenate([self.prev_image_2, images], axis=2)

            self.prev_image_2 = self.prev_image_1
            self.prev_image_1 = self.prev_image_0
            self.prev_image_0 = image

            return images, {}

        else:
            # pre-process image format
            processed_image = process_image_for_dnn(image=sensor_data['front_camera'][1], normalized=True, torch_normalize=True)
            processed_image = processed_image.to(self.device)

            # apply freezed pre-trained resnet model onto the image
            with torch.no_grad():
                image_features_torch = self.resnet_model(processed_image)
                image_features = image_features_torch.cpu().detach().numpy()[0]
                image_features = image_features.reshape(self.ResNetShape)

            return image_features, {}

    def get_done_status(self, sensor_data, core):
        """
        Returns whether or not the experiment has to end
        """
        hero = core.hero
        speed_kmph = get_speed(hero)

        self.done_time_idle = self.max_time_idle < self.time_idle

        if speed_kmph > 1.0:
            self.time_idle = 0
        else:
            self.time_idle += 1

        self.time_episode += 1
        self.done_time_episode = self.max_time_episode < self.time_episode
        self.done_falling = hero.get_location().z < -0.5

        if 'collision' in sensor_data:
            self.done_collision = True

        return self.done_time_idle or self.done_falling or self.done_time_episode or self.done_collision

    def compute_reward(self, sensor_data, core):
        hero = core.hero
        world = core.world

        reward = 0.0
        throttle = self.last_action.throttle

        is_light, is_walker, is_vehicle, _ = traffic_data(hero, world)

        # TODO: check if correctly applicable
        if IS_STUCK_VEHICLE:
            objects_of_concern = []
        else:
            print("[Traffic]: traffic light-", is_light, " walker-", is_walker, " vehicle-", is_vehicle)
            objects_of_concern = [is_light, is_walker, is_vehicle]

        if any(x is not None for x in objects_of_concern):

            # fix step counter while waiting traffic light or front vehicle, and pedestrian
            self.time_episode -= 1

            # accelerating while it should brake
            if throttle < 0.1:
                print("[Reward]: correctly braking !")
                # braking is desired
                reward += 50
            else:
                print("[Penalty]: not braking !")
                reward -= 50

            reward -= self.speed_ms
        else:
            reward += self.speed_ms
        
        # distance from starting position
        reward += np.linalg.norm(self.ego_gps - self.previous_ego_gps)

        # negative reward for collision
        if 'collision' in sensor_data:
            print("[Penalty]: collision !")
            reward = -1000

        print("[Info]: Step-", self.count_steps, " speed m/s-", self.speed_ms, " throttle-", throttle, " ego_gps-", self.ego_gps)
        
        return reward