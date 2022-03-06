#!/bin/bash

pkill -9 python
pkill -9 python
pkill -9 python

export xml_file="original_routes/routes_town05_tiny.xml"
export json_file="all_towns_traffic_scenarios_WOR.json"

now=`date +"%Y_%m_%d_%H_%M_%S"`
json=".json"
basename "${xml_file}"
f="$(basename -- ${xml_file})"
route_name=${f::-4}
CHECKPOINT_ENDPOINT_FILE_NAME="${now}_${route_name}${json}"

export CARLA_SERVER=${CARLA_ROOT}/CarlaUE4.sh
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI
export PYTHONPATH=$PYTHONPATH:${CARLA_ROOT}/PythonAPI/carla
export PYTHONPATH=$PYTHONPATH:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg
export PYTHONPATH=$PYTHONPATH:${DeFIX_PATH}
export PYTHONPATH=$PYTHONPATH:${DeFIX_PATH}/leaderboard:leaderboard
export PYTHONPATH=$PYTHONPATH:${DeFIX_PATH}/scenario_runner:scenario_runner
export PYTHONPATH=$PYTHONPATH:${DeFIX_PATH}/imitation-learning:imitation-learning
export PYTHONPATH=$PYTHONPATH:${DeFIX_PATH}/reinforcement-learning:reinforcement-learning

export TEAM_CONFIG=${DeFIX_PATH}
export LEADERBOARD_ROOT=${DeFIX_PATH}/leaderboard

export ROUTES=${DeFIX_PATH}/data/routes/${xml_file}
export SCENARIOS=${DeFIX_PATH}/data/scenarios/${json_file}

export CHECKPOINT_PATH=${DeFIX_PATH}/checkpoint/
export CHECKPOINT_ENDPOINT=${DeFIX_PATH}/results/${CHECKPOINT_ENDPOINT_FILE_NAME}
export SAVE_DATASET_NAME=${now}
export CHALLENGE_TRACK_CODENAME=SENSORS
export DEBUG_CHALLENGE=0
export RESUME=False

export TEAM_AGENT=${DeFIX_PATH}/defix/evaluate_defix.py

[ -d ${DeFIX_PATH}/results ] || mkdir ${DeFIX_PATH}/results

python3 ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
            --scenarios=${SCENARIOS}  \
            --routes=${ROUTES} \
            --repetitions=1 \
            --track=${CHALLENGE_TRACK_CODENAME} \
            --checkpoint=${CHECKPOINT_ENDPOINT} \
            --agent=${TEAM_AGENT} \
            --agent-config=${TEAM_CONFIG} \
            --debug=${DEBUG_CHALLENGE} \
            --record=${RECORD_PATH} \
            --resume=${RESUME}