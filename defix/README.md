# DeFIX
Detecting and Fixing Failure Scenarios with Reinforcement Learning in Imitation Learning Based Autonomous Driving

## List of CARLA Scenarios
<div align="center">

| Scenario ID | Scenario Name |
| :-: | :-: |
| 0 | Dynamic Vehicle Collision |
| 1 | Emerging Pedestrian Collision |
| 2 | Stuck Vehicle & Static Objects |
| 3 | Vehicle Running Red Light |
| 4 | Crossing Signalized Traffic Intersections |
| 5 | Crossing Un-signalized Intersections |

</div>

## Provision Steps
* move 'resnet50.zip' file to directory: <DeFIX_PATH/checkpoint/models/>
* specify which RL agents to evaluate in <DeFIX_PATH/defix/evaluate.py> script
* imitation learning models should be inside "DeFIX_PATH/checkpoint/models/imitation/" folder
* all reinforcement learning models should be inside "DeFIX_PATH/checkpoint/models/reinforcement/" folder
* policy classifier model should be inside "DeFIX_PATH/checkpoint/models/policy_classifier/" folder

## Prepare Necessary Directory Exports to Bashrc
```sh
gedit ~/.bashrc

export DeFIX_PATH=PATH_TO_MAIN_DeFIX_REPO
export CARLA_ROOT=PATH_TO_CARLA_ROOT_SH

export SCENARIO_RUNNER_ROOT="${DeFIX_PATH}/scenario_runner"
export LEADERBOARD_ROOT="${DeFIX_PATH}/leaderboard"
export PYTHONPATH="${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":"${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg":${PYTHONPATH}

source ~/.bashrc
```

## Run Carla Server on GPU
```sh
cd $CARLA_ROOT
./CarlaUE4.sh -prefernvidia
```

## Run Evaluation of DeFIX Methodology
```sh
cd $DeFIX_PATH/defix
. run_evaluation.sh
```
