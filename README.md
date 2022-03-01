# DeFIX
DeFIX: Detecting and Fixing Failure Scenarios with Reinforcement Learning in Imitation Learning Based Autonomous Driving

## Abstract
Safely navigating through an urban environment without violating any traffic rules is a crucial performance and safety target for reliable autonomous driving. In this paper, we present a Reinforcement Learning (RL) based methodology to DEtect and FIX (DeFIX) failures of an Imitation Learning (IL) agent by extracting infraction spots and re-constructing mini-scenarios on these infraction areas to train an RL agent for fixing the shortcomings of the IL approach. DeFIX is a continuous learning framework, where extraction of failure scenarios and training of RL agents are executed in an infinite loop. After each new policy is trained and added to the library of policies, a policy classifier method effectively decides on which policy to activate at each step during the evaluation. It is demonstrated that even with only one RL agent trained on failure scenario of an IL agent, DeFIX method is either competitive or does outperform state-of-the-art IL and RL based autonomous urban driving benchmarks. We trained and validated our approach on the most challenging map (Town05) of CARLA simulator which involves complex, realistic, and adversarial driving scenarios.

## Method Overview
<p align="center">
    <img src="images/method_overview.png" width="550px"/>
</p>
