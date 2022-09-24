#!/usr/bin/env python

# Copyright (c) 2021 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import os
import random
import signal
import subprocess
import time
import psutil

import carla

from rllib_integration.sensors.sensor_interface import SensorInterface
from rllib_integration.sensors.factory import SensorFactory
from rllib_integration.helper import join_dicts

from srunner.scenariomanager.carla_data_provider import *
from leaderboard.utils.route_indexer import RouteIndexer

from custom_scenario_runner.custom_route_scenario import CustomRouteScenario
from custom_scenario_runner.custom_scenario_manager import CustomScenarioManager

from rllib_integration.helper import RoutePlanner


BASE_CORE_CONFIG = {
    "host": "localhost",  # Client host
    "timeout": 10.0,  # Timeout of the client
    "timestep": 0.05,  # Time step of the simulation
    "retries_on_error": 10,  # Number of tries to connect to the client
    "resolution_x": 600,  # Width of the server spectator camera
    "resolution_y": 400,  # Height of the server spectator camera
    "quality_level": "Epic",  # Quality level of the simulation. Can be 'Low', 'High', 'Epic'
    "enable_map_assets": False,  # enable / disable all town assets except for the road
    "enable_rendering": True,  # enable / disable camera images
    "show_display": False  # Whether or not the server will be displayed
}

# if False: only clear weather will be active
IS_WEATHER_CHANGE = True

WEATHERS = {
        'ClearNoon': carla.WeatherParameters.ClearNoon,
        'ClearSunset': carla.WeatherParameters.ClearSunset,

        'CloudyNoon': carla.WeatherParameters.CloudyNoon,
        'CloudySunset': carla.WeatherParameters.CloudySunset,

        'WetNoon': carla.WeatherParameters.WetNoon,
        'WetSunset': carla.WeatherParameters.WetSunset,

        'MidRainyNoon': carla.WeatherParameters.MidRainyNoon,
        'MidRainSunset': carla.WeatherParameters.MidRainSunset,

        'WetCloudyNoon': carla.WeatherParameters.WetCloudyNoon,
        'WetCloudySunset': carla.WeatherParameters.WetCloudySunset,

        'HardRainNoon': carla.WeatherParameters.HardRainNoon,
        'HardRainSunset': carla.WeatherParameters.HardRainSunset,

        'SoftRainNoon': carla.WeatherParameters.SoftRainNoon,
        'SoftRainSunset': carla.WeatherParameters.SoftRainSunset,
}
WEATHERS_IDS = list(WEATHERS)

def is_used(port):
    """
    Checks whether or not a port is used
    """
    return port in [conn.laddr.port for conn in psutil.net_connections()]


def kill_all_servers():
    """
    Kill all PIDs that start with Carla
    """
    processes = [p for p in psutil.process_iter() if "carla" in p.name().lower()]
    for process in processes:
        os.kill(process.pid, signal.SIGKILL)


class CarlaCore:
    """
    Class responsible of handling all the different CARLA functionalities, such as server-client connecting,
    actor spawning and getting the sensors data.
    """
    def __init__(self, config={}):
        """
        Initialize the server and client
        """
        self.client = None
        self.world = None
        self.map = None
        self.hero = None
        self.config = join_dicts(BASE_CORE_CONFIG, config)
        self.sensor_interface = SensorInterface()

        self.is_scenario_and_hero_initialized = False

        self.weather_id = WEATHERS_IDS[0]

        self.init_server()
        self.connect_client()

    def init_server(self):
        """
        Start a server on a random port
        """
        self.server_port = random.randint(15000, 32000)

        # Ray tends to start all processes simultaneously. Use random delays to avoid problems
        time.sleep(random.uniform(0, 1))

        uses_server_port = is_used(self.server_port)
        uses_stream_port = is_used(self.server_port + 1)
        while uses_server_port and uses_stream_port:
            if uses_server_port:
                print("Is using the server port: " + str(self.server_port))
            if uses_stream_port:
                print("Is using the streaming port: " + str(self.server_port+1))
            self.server_port += 2
            uses_server_port = is_used(self.server_port)
            uses_stream_port = is_used(self.server_port+1)

        if self.config["show_display"]:
            server_command = [
                "{}/CarlaUE4.sh".format(os.environ["CARLA_ROOT"]),
                "-windowed",
                "-ResX={}".format(self.config["resolution_x"]),
                "-ResY={}".format(self.config["resolution_y"]),
                "-prefernvidia",
                "-vulkan"
            ]
        else:
            server_command = [
                "DISPLAY= ",
                "{}/CarlaUE4.sh".format(os.environ["CARLA_ROOT"]),
                "-opengl"  # no-display isn't supported for Unreal 4.24 with vulkan
            ]

        server_command += [
            "--carla-rpc-port={}".format(self.server_port),
            "-quality-level={}".format(self.config["quality_level"])
        ]

        server_command_text = " ".join(map(str, server_command))
        print(server_command_text)
        server_process = subprocess.Popen(
            server_command_text,
            shell=True,
            preexec_fn=os.setsid,
            stdout=open(os.devnull, "w"),
        )

    def connect_client(self):
        """
        Connect to the client
        """
        for i in range(self.config["retries_on_error"]):
            try:
                self.client = carla.Client(self.config["host"], self.server_port)
                self.client.set_timeout(self.config["timeout"])
                self.world = self.client.get_world()

                settings = self.world.get_settings()
                settings.no_rendering_mode = not self.config["enable_rendering"]
                settings.synchronous_mode = True
                settings.fixed_delta_seconds = self.config["timestep"]
                self.world.apply_settings(settings)

                self.world.tick()

                return

            except Exception as e:
                print(" Waiting for server to be ready: {}, attempt {} of {}".format(e, i + 1, self.config["retries_on_error"]))
                time.sleep(3)

        raise Exception("Cannot connect to server. Try increasing 'timeout' or 'retries_on_error' at the carla configuration")

    def setup_experiment(self, experiment_config):
        """
        Initialize the hero and sensors
        """
        self.world = self.client.load_world(
            map_name = experiment_config["town"],
            reset_settings = False,
            map_layers = carla.MapLayer.All if self.config["enable_map_assets"] else carla.MapLayer.NONE)

        self.map = self.world.get_map()

        # Choose the weather of the simulation
        # weather = getattr(carla.WeatherParameters, experiment_config["weather"])
        # self.world.set_weather(weather)

        self.tm_port = self.server_port // 10 + self.server_port % 10
        while is_used(self.tm_port):
            print("Traffic manager's port " + str(self.tm_port) + " is already being used. Checking the next one")
            tm_port += 1
        print("Traffic manager connected to port " + str(self.tm_port))

        self.traffic_manager = self.client.get_trafficmanager(self.tm_port)
        self.traffic_manager.set_hybrid_physics_mode(experiment_config["background_activity"]["tm_hybrid_mode"])
        seed = experiment_config["background_activity"]["seed"]
        if seed is not None:
            self.traffic_manager.set_random_device_seed(seed)

        self.world.reset_all_traffic_lights()

        self.world.tick()

    def scenario_cleanup(self):
        self.manager.stop_scenario()
        self.scenario.remove_all_actors()
        if self.manager:
            self.manager.cleanup()
        CarlaDataProvider.cleanup()
        print("Scenario cleaning up")

    def scenario_and_hero_init(self, hero_config):
        CarlaDataProvider.set_client(self.client)
        CarlaDataProvider.set_world(self.world)
        CarlaDataProvider.set_traffic_manager_port(self.tm_port)

        self.manager = CustomScenarioManager(self.config["timeout"], self.config["debug_mode"] > 1)
        route_indexer = RouteIndexer(hero_config['routes'], hero_config['scenarios'], hero_config['repetitions'])

        # only one route is allowed
        route_indexer.peek()
        route_indexer_config = route_indexer.next()

        # create and load scenario
        self.scenario = CustomRouteScenario(world=self.world, config=route_indexer_config, ego_vehicle_type=hero_config['ego_vehicle_type'], debug_mode=hero_config['debug_mode'])
        self.manager.load_scenario(self.scenario, route_indexer_config.repetition_index)
        
        self.hero = self.scenario.ego_vehicle

        if not self.is_scenario_and_hero_initialized:
            self.is_scenario_and_hero_initialized = True

    def reset_hero(self, hero_config):
        """
        This function resets / spawns the hero vehicle and its sensors
        """
        # Part 1: destroy all sensors (if necessary)
        self.sensor_interface.destroy()

        self.tick_counter = 0
        self.world.tick()

        self.hero_blueprints = self.world.get_blueprint_library().find(hero_config['ego_vehicle_type'])
        self.hero_blueprints.set_attribute("role_name", "hero")

        # If already spawned, destroy it
        if self.hero is not None:
            self.hero.destroy()
            self.hero = None
        
        print(f"self.is_scenario_and_hero_initialized {self.is_scenario_and_hero_initialized}")
        if self.is_scenario_and_hero_initialized:
            self.scenario_cleanup()

        self.scenario_and_hero_init(hero_config)

        # make and set route planner
        self.route_planner = RoutePlanner(min_distance=4.0, max_distance=50.0)
        self.command_planner = RoutePlanner(min_distance=7.5, max_distance=25.0, debug_size=257)
        self.route_planner.set_route(global_plan=self.scenario.gps_route, gps=True)
        self.command_planner.set_route(global_plan=self.scenario.global_plan, gps=True)

        # Part 3: Spawn the new sensors
        for name, attributes in hero_config["sensors"].items():
            sensor = SensorFactory.spawn(name, attributes, self.sensor_interface, self.hero)
            print(f"name {name} sensor {sensor} spawned!")

        self.manager.init_tick_scenario()
        
        # sync state
        self.world.tick()

        return self.hero

    def change_weather(self):
        index = random.choice(range(len(WEATHERS)))

        self.weather_id = WEATHERS_IDS[index]
        self.world.set_weather(WEATHERS[WEATHERS_IDS[index]])

    def tick(self, control):
        """
        Performs one tick of the simulation, moving all actors, and getting the sensor data
        """
        self.tick_counter += 1
        
        if self.tick_counter % 20 == 0 and IS_WEATHER_CHANGE:
            self.change_weather()

        # Move hero vehicle and scenario vehicles
        if control is not None:
            timestamp = None
            world = CarlaDataProvider.get_world()
            if world:
                snapshot = world.get_snapshot()
                if snapshot:
                    timestamp = snapshot.timestamp
            self.manager._tick_scenario(timestamp, control)

        # Tick once the simulation
        self.world.tick()

        # Move the spectator
        if self.config["enable_rendering"]:
            self.set_spectator_camera_view()

        # Return the new sensor data
        return self.get_sensor_data()

    def set_spectator_camera_view(self):
        """
        This positions the spectator as a 3rd person view of the hero vehicle
        """
        transform = self.hero.get_transform()

        # Get the camera position
        server_view_x = transform.location.x - 5 * transform.get_forward_vector().x
        server_view_y = transform.location.y - 5 * transform.get_forward_vector().y
        server_view_z = transform.location.z + 3

        # Get the camera orientation
        server_view_roll = transform.rotation.roll
        server_view_yaw = transform.rotation.yaw
        server_view_pitch = transform.rotation.pitch

        # Get the spectator and place it on the desired position
        self.spectator = self.world.get_spectator()
        self.spectator.set_transform(
            carla.Transform(
                carla.Location(x=server_view_x, y=server_view_y, z=server_view_z),
                carla.Rotation(pitch=server_view_pitch,yaw=server_view_yaw,roll=server_view_roll),
            )
        )

    def get_sensor_data(self):
        """
        Returns the data sent by the different sensors at this tick
        """
        sensor_data = self.sensor_interface.get_data()
        # print("---------")
        # world_frame = self.world.get_snapshot().frame
        # print("World frame: {}".format(world_frame))
        # for name, data in sensor_data.items():
        #     print("{}: {}".format(name, data[0]))
        return sensor_data