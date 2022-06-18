Original CARLA RLLib Integration Repository: [rllib-integration](https://github.com/carla-simulator/rllib-integration)

## Installation Steps

* Download [Carla 0.9.11](https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.11.tar.gz)
* unzip folder to PATH_TO_CARLA_ROOT_SH (user defined path)

```sh
conda create -n carla python=3.7

conda activate carla

pip install -r requirements.txt
```

## Prepare Necessary Directory Exports to Bashrc

```sh
gedit ~/.bashrc

export DeFIX_PATH=PATH_TO_MAIN_DeFIX_REPO
export CARLA_ROOT=PATH_TO_CARLA_ROOT_SH

export SCENARIO_RUNNER_ROOT="${DeFIX_PATH}/scenario_runner"
export LEADERBOARD_ROOT="${DeFIX_PATH}/leaderboard"
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":"${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.11-py3.7-linux-x86_64.egg":${PYTHONPATH}

source ~/.bashrc
```

## Provision Steps

* move 'resnet50' file to directory: <DeFIX_PATH/checkpoint/models/>

## Training

```python
python3 dqn_train.py dqn_example/dqn_config.yaml --overwrite
```

## Solved Problems

* ModuleNotFoundError: No module named 'aiohttp.frozenlist' #pip install aiohttp==3.7.4

* AttributeError: module 'aioredis' has no attribute 'create_redis_pool' #pip install aioredis==1.3.1

* dqn_trainer.py # add import os

* dqn_config.yaml blueprint: "vehicle.lincoln.mkz_2017" -> blueprint: "vehicle.lincoln.mkz2017"

* numpy randint problem #pip install gym==0.21.0