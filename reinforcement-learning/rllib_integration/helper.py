#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.


import collections.abc
import os
import shutil

import torch
import math
import cv2
import numpy as np
np.random.seed(0)

from tensorboard import program
from collections import deque

import carla
from srunner.scenariomanager.carla_data_provider import * 

DEBUG_MODE = False


def process_image_for_dnn(image, normalized=True, torch_normalize=True):
    """
    Convert image to format required for ResNet network and normalize between -1 and 1 if required
    """
    image = image[:, :, :3]
    image = image[:, :, ::-1]
    image = image[:, :, ::-1]

    # convert width height channel to channel width height
    image = np.array(image.transpose((2, 0, 1)), np.float32)

    # BGRA to BGR
    image = image[:3, :, :]

    # BGR to RGB
    image = image[::-1, :, :]

    if normalized:
        # normalize to 0 - 1
        image = image.astype(np.float32) / 255
    
    # convert image to torch tensor
    image = torch.from_numpy(image.copy()).unsqueeze(0)

    # normalize input image (using default torch normalization technique)
    if torch_normalize:
        image = normalize_rgb(image)

    return image


def normalize_rgb(image):
    """
    Default pytorch image normalization calculations
    """
    x = image.clone()
    x[:, 0] = (x[:, 0] - 0.485) / 0.229
    x[:, 1] = (x[:, 1] - 0.456) / 0.224
    x[:, 2] = (x[:, 2] - 0.406) / 0.225
    return x


def post_process_image(image, normalized=True, grayscale=True):
    """
    Convert image to gray scale and normalize between -1 and 1 if required
    :param image:
    :param normalized:
    :param grayscale
    :return: image
    """
    if isinstance(image, list):
        image = image[0]
    if grayscale:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = image[:, :, np.newaxis]

    if normalized:
        return (image.astype(np.float32) - 128) / 128
    else:
        return image.astype(np.uint8)


def join_dicts(d, u):
    """
    Recursively updates a dictionary
    """
    result = d.copy()

    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            result[k] = join_dicts(d.get(k, {}), v)
        else:
            result[k] = v
    return result


def find_latest_checkpoint(directory):
    """
    Finds the latest checkpoint, based on how RLLib creates and names them.
    """
    start = directory
    max_checkpoint_int = -1
    checkpoint_path = ""

    # 1st layer: Check for the different run folders
    for f in os.listdir(start):
        if os.path.isdir(start + "/" + f):
            temp = start + "/" + f

            # 2nd layer: Check all the checkpoint folders
            for c in os.listdir(temp):
                if "checkpoint_" in c:

                    # 3rd layer: Get the most recent checkpoint
                    checkpoint_int = int(''.join([n for n in c
                                                  if n.isdigit()]))
                    if checkpoint_int > max_checkpoint_int:
                        max_checkpoint_int = checkpoint_int
                        checkpoint_path = temp + "/" + c + "/" + c.replace(
                            "_", "-")

    if not checkpoint_path:
        raise FileNotFoundError(
            "Could not find any checkpoint, make sure that you have selected the correct folder path"
        )

    return checkpoint_path


def get_checkpoint(name, directory, restore=False, overwrite=False):
    training_directory = os.path.join(directory, name)

    if overwrite and restore:
        raise RuntimeError(
            "Both 'overwrite' and 'restore' cannot be True at the same time")

    if overwrite:
        if os.path.isdir(training_directory):
            shutil.rmtree(training_directory)
            print("Removing all contents inside '" + training_directory + "'")
        return None


    if restore:
        return find_latest_checkpoint(training_directory)

    if os.path.isdir(training_directory) and len(os.listdir(training_directory)) != 0:
        raise RuntimeError(
            "The directory where you are trying to train (" +
            training_directory + ") is not empty. "
            "To start a new training instance, make sure this folder is either empty, non-existing "
            "or use the '--overwrite' argument to remove all the contents inside"
        )

    return None


def launch_tensorboard(logdir, host="localhost", port="6006"):
    tb = program.TensorBoard()
    tb.configure(argv=[None, "--logdir", logdir, "--host", host, "--port", port])
    url = tb.launch()


def get_position(gps, route_planner):
    # gets global latitude and longitude coordinates
    converted_gps = (gps - route_planner.mean) * route_planner.scale
    return converted_gps


# TODO: Resetting ego-vehicle! RuntimeError: trying to operate on a destroyed actor; an actor's function was called, but the actor is already destroyed.
def get_speed(hero):
    # computes the speed of the hero vehicle in km/h
    vel = hero.get_velocity()
    return 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)


def get_angle_to(pos, theta, target):
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
        ])

    aim = R.T.dot(target - pos)
    angle = -np.degrees(np.arctan2(-aim[1], aim[0]))
    angle = 0.0 if np.isnan(angle) else angle 

    return angle


def get_control(turn_controller, speed_controller, target, far_target, gps, theta, speed):
    # steering
    angle_unnorm = get_angle_to(pos=gps, theta=theta, target=target)
    angle = angle_unnorm / 90

    steer = turn_controller.step(angle)
    steer = np.clip(steer, -1.0, 1.0)
    steer = round(steer, 3)

    # acceleration
    angle_far_unnorm = get_angle_to(pos=gps, theta=theta, target=far_target)
    should_slow = abs(angle_far_unnorm) > 45.0 or abs(angle_unnorm) > 5.0
    target_speed = 4.0 if should_slow else 7.0
    
    delta = np.clip(target_speed - speed, 0.0, 0.25)
    throttle = speed_controller.step(delta)
    throttle = np.clip(throttle, 0.0, 0.75)

    return steer, throttle, angle


def shift_point(ego_compass, ego_gps, near_node, offset_amount):
    # rotation matrix
    R = np.array([
        [np.cos(np.pi / 2 + ego_compass), -np.sin(np.pi / 2 + ego_compass)],
        [np.sin(np.pi / 2 + ego_compass), np.cos(np.pi / 2 + ego_compass)]
    ])

    # transpose of rotation matrix
    trans_R = R.T

    local_command_point = np.array([near_node[0] - ego_gps[0], near_node[1] - ego_gps[1]])
    local_command_point = trans_R.dot(local_command_point)

    # positive offset shifts near node to right; negative offset shifts near node to left
    local_command_point[0] += offset_amount
    local_command_point[1] += 0

    new_near_node = np.linalg.inv(trans_R).dot(local_command_point)

    new_near_node[0] += ego_gps[0]
    new_near_node[1] += ego_gps[1]

    return new_near_node


def calculate_high_level_action(world, turn_controller, speed_controller, high_level_action, gps, initial_gps, theta, initial_theta, speed, near_node, far_target):
    """
    0 -> brake
    1 -> no brake - go to left lane of next_waypoint
    2 -> no brake - keep lane (stay at next_waypoint's lane)
    3 -> no brake - go to right lane of next_waypoint
    """

    # left
    if high_level_action == 1:
        offset = -3.5
        new_near_node = shift_point(ego_compass=initial_theta, ego_gps=initial_gps, near_node=near_node, offset_amount=offset)
    
    # right
    elif high_level_action == 3:
        offset = 3.5
        new_near_node = shift_point(ego_compass=initial_theta, ego_gps=initial_gps, near_node=near_node, offset_amount=offset)
    
    # keep lane
    else:
        offset = 0.0
        new_near_node = near_node
    
    if DEBUG_MODE and (high_level_action == 1 or high_level_action == 3):
        world.debug.draw_point(carla.Location(new_near_node[1], -new_near_node[0], 1.0), size=0.1, color=carla.Color(0, 0, 255), life_time=-1.0)

    # get auto-pilot actions
    steer, throttle, angle = get_control(turn_controller, speed_controller, new_near_node, far_target, gps, theta, speed)

    # brake
    if high_level_action == 0:
        throttle = 0.0
        brake = 1.0
    # no brake
    else:
        throttle = throttle
        brake = 0.0

    return throttle, steer, brake


def traffic_data(hero_vehicle, world):
    all_actors = world.get_actors()

    lights_list = all_actors.filter('*traffic_light*')
    walkers_list = all_actors.filter('*walker*')
    vehicle_list = all_actors.filter('*vehicle*')
    stop_list = all_actors.filter('*stop*')

    traffic_lights = get_nearby_lights(hero_vehicle, lights_list)
    stops = get_nearby_lights(hero_vehicle, stop_list)

    if len(stops) == 0:
        stop = None
    else:
        stop = stops

    light = is_light_red(traffic_lights)
    walker = is_walker_hazard(hero_vehicle, walkers_list)
    vehicle = is_vehicle_hazard(hero_vehicle, vehicle_list)
    
    # TODO: should be used with autopilot
    # stop = is_stop_sign(stop)

    return light, walker, vehicle, stop


def get_nearby_lights(vehicle, lights, pixels_per_meter=5.5, size=512, radius=5):
    result = list()

    transform = vehicle.get_transform()
    pos = transform.location
    theta = np.radians(90 + transform.rotation.yaw)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)],
        ])

    for light in lights:
        delta = light.get_transform().location - pos

        target = R.T.dot([delta.x, delta.y])
        target *= pixels_per_meter
        target += size // 2

        if min(target) < 0 or max(target) >= size:
            continue

        trigger = light.trigger_volume
        light.get_transform().transform(trigger.location)
        dist = trigger.location.distance(vehicle.get_location())
        a = np.sqrt(
                trigger.extent.x ** 2 +
                trigger.extent.y ** 2 +
                trigger.extent.z ** 2)
        b = np.sqrt(
                vehicle.bounding_box.extent.x ** 2 +
                vehicle.bounding_box.extent.y ** 2 +
                vehicle.bounding_box.extent.z ** 2)

        total = a + b
        threshold = 1.0

        if dist + threshold > total:
            continue

        result.append(light)

    return result


def is_light_red(traffic_lights):
    for light in traffic_lights:
        
        if light.get_state() == carla.TrafficLightState.Red:
            return True
        
        elif light.get_state() == carla.TrafficLightState.Yellow:
            return True
    
    return None


def is_walker_hazard(hero_vehicle, walkers_list):
    p1 = _numpy(hero_vehicle.get_location())
    v1 = 10.0 * _orientation(hero_vehicle.get_transform().rotation.yaw)
    
    for walker in walkers_list:
        v2_hat = _orientation(walker.get_transform().rotation.yaw)
        s2 = np.linalg.norm(_numpy(walker.get_velocity()))
        
        if s2 < 0.05:
            v2_hat *= s2
        
        p2 = -3.0 * v2_hat + _numpy(walker.get_location())
        v2 = 8.0 * v2_hat
        
        collides, collision_point = get_collision(p1, v1, p2, v2)
        
        if collides:
            return walker
    
    return None

# TODO: not used yet
def is_stop_sign(is_stop, stop_step, not_brake_step):
    if stop_step < 200 and is_stop is not None:
        stop_step += 1
        not_brake_step = 0
        return True
    
    else:
        if not_brake_step < 300:
            not_brake_step += 1 
        else:
            stop_step = 0
        return None


def _numpy(carla_vector, normalize=False):
    result = np.float32([carla_vector.x, carla_vector.y])

    if normalize:
        return result / (np.linalg.norm(result) + 1e-4)

    return result


def _orientation(yaw):
    return np.float32([np.cos(np.radians(yaw)), np.sin(np.radians(yaw))])


def get_collision(p1, v1, p2, v2):
    A = np.stack([v1, -v2], 1)
    b = p2 - p1

    if abs(np.linalg.det(A)) < 1e-3:
        return False, None

    x = np.linalg.solve(A, b)

    # how many seconds until collision
    collides = all(x >= 0) and all(x <= 1)

    return collides, p1 + x[0] * v1


def is_vehicle_hazard(hero_vehicle, vehicle_list):
    o1 = _orientation(hero_vehicle.get_transform().rotation.yaw)
    p1 = _numpy(hero_vehicle.get_location())

    # increases the threshold distance
    s1 = max(10, 3.0 * np.linalg.norm(_numpy(hero_vehicle.get_velocity())))

    v1_hat = o1
    v1 = s1 * v1_hat
    
    for target_vehicle in vehicle_list:
        if target_vehicle.id == hero_vehicle.id:
            continue
        
        o2 = _orientation(target_vehicle.get_transform().rotation.yaw)
        p2 = _numpy(target_vehicle.get_location())
        
        s2 = max(5.0, 2.0 * np.linalg.norm(_numpy(target_vehicle.get_velocity())))
        
        v2_hat = o2
        v2 = s2 * v2_hat
        p2_p1 = p2 - p1
        
        distance = np.linalg.norm(p2_p1)
        
        p2_p1_hat = p2_p1 / (distance + 1e-4)
        
        angle_to_car = np.degrees(np.arccos(v1_hat.dot(p2_p1_hat)))
        angle_between_heading = np.degrees(np.arccos(o1.dot(o2)))
        
        angle_to_car = min(angle_to_car, 360.0 - angle_to_car)
        angle_between_heading = min(angle_between_heading, 360.0 - angle_between_heading)
        
        if angle_between_heading > 60.0 and not (angle_to_car < 15 and distance < s1):
            continue
        elif angle_to_car > 30.0:
            continue
        elif distance > s1 and distance < s2:
            target_vehicle_speed = target_vehicle.get_velocity()
            continue
        elif distance > s1:
            continue
        
        return target_vehicle
    
    return None


class RoutePlanner(object):
    def __init__(self, min_distance, max_distance, debug_size=256):
        self.route = deque()
        self.min_distance = min_distance
        self.max_distance = max_distance

        # self.mean = np.array([49.0, 8.0]) # for carla 9.9
        # self.scale = np.array([111324.60662786, 73032.1570362]) # for carla 9.9
        # self.mean = np.array([0.0, 0.0]) # for carla 9.10
        # self.scale = np.array([111324.60662786, 111319.490945]) # for carla 9.10
        self.mean = np.array([0.0, 0.0]) # for carla 9.11
        self.scale = np.array([111324.60662786, 111324.60662786]) # for carla 9.11

    def set_route(self, global_plan, gps=False):
        self.route.clear()

        for pos, cmd in global_plan:
            if gps:
                pos = np.array([pos['lat'], pos['lon']])
                pos -= self.mean
                pos *= self.scale
            else:
                pos = np.array([pos.location.x, pos.location.y])
                pos -= self.mean

            self.route.append((pos, cmd))

    def run_step(self, gps):
        if len(self.route) == 1:
            return self.route[0]

        to_pop = 0
        farthest_in_range = -np.inf
        cumulative_distance = 0.0

        for i in range(1, len(self.route)):
            if cumulative_distance > self.max_distance:
                break

            cumulative_distance += np.linalg.norm(self.route[i][0] - self.route[i-1][0])
            distance = np.linalg.norm(self.route[i][0] - gps)

            if distance <= self.min_distance and distance > farthest_in_range:
                farthest_in_range = distance
                to_pop = i

        for _ in range(to_pop):
            if len(self.route) > 2:
                self.route.popleft()

        return self.route[1]


class PIDController(object):
    def __init__(self, K_P=1.0, K_I=0.0, K_D=0.0, n=20):
        self._K_P = K_P
        self._K_I = K_I
        self._K_D = K_D

        self._window = deque([0 for _ in range(n)], maxlen=n)
        self._max = 0.0
        self._min = 0.0

    def step(self, error):
        self._window.append(error)
        self._max = max(self._max, abs(error))
        self._min = -abs(self._max)

        if len(self._window) >= 2:
            integral = np.mean(self._window)
            derivative = (self._window[-1] - self._window[-2])
        else:
            integral = 0.0
            derivative = 0.0

        return self._K_P * error + self._K_I * integral + self._K_D * derivative