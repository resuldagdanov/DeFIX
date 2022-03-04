# DeFIX
Detecting and Fixing Failure Scenarios with Reinforcement Learning in Imitation Learning Based Autonomous Driving

## Abstract
Safely navigating through an urban environment without violating any traffic rules is a crucial performance and safety target for reliable autonomous driving. In this paper, we present a Reinforcement Learning (RL) based methodology to DEtect and FIX (DeFIX) failures of an Imitation Learning (IL) agent by extracting infraction spots and re-constructing mini-scenarios on these infraction areas to train an RL agent for fixing the shortcomings of the IL approach. DeFIX is a continuous learning framework, where extraction of failure scenarios and training of RL agents are executed in an infinite loop. After each new policy is trained and added to the library of policies, a policy classifier method effectively decides on which policy to activate at each step during the evaluation. It is demonstrated that even with only one RL agent trained on failure scenario of an IL agent, DeFIX method is either competitive or does outperform state-of-the-art IL and RL based autonomous urban driving benchmarks. We trained and validated our approach on the most challenging map (Town05) of CARLA simulator which involves complex, realistic, and adversarial driving scenarios.

## Method Overview
<p align="center">
    <img src="figures/method_overview.png" width="1000px"/>
</p>

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

## Adversarial Urban Scenarios
<p align="center">
    <img src="figures/adversarial_scenarios.png" width="1000px"/>
</p>

## Network Architectures
Pre-trained ResNet-50 model is used as a backbone to all proposed networks. The last layer of a ResNet-50 is unfrozen and fine-tuned during the training of brake and policy classifiers. Speed, orientation, location, and front camera data are obtained from sensors while sequential target locations are given priorly. Target locations are converted to the relative local coordinate frame by using vehicle position and orientation obtained from IMU and GPS sensors, respectively. In the supervised learning network, a 704-dimensional feature vector represents the fused information of sensor inputs, from there it is processed through fully connected layers. IL agent only decides when to apply a brake action while policy classifier decides which trained agent to activate during evaluation. ResNet-50 model is completely frozen during DQN trainings. The state-space for the DQN agent is a 1000-dimensional vector of ResNet backbone output. DQN agents output high-level action commands (lane keeping, right and left lane changing, stopping). Low-level steering and throttle actions are determined with lateral and longitudinal PID controllers, respectively.

<p align="center">
    <img src="figures/network_architectures.png" width="1500px"/>
</p>