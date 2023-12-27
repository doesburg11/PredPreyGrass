
### Predator, Prey, Grass PettingZoo environment
 
 This model simulates a predator-prey relationship in a grid (intially bounded) environment. The population consists of wolf packs (predators) and sheep herds (prey). The predators gain energy from consuming prey, and the sheep gain energy from consuming grass (a primary producer). The environment is initially inspired by Netlogo's PredatorPreyGame (https://ccl.northwestern.edu/netlogo/models/PredatorPreyGameHubNet) and implemented in Python 3.11.5 using the PettingZoo verzon 1.24.2 Multi Agent Reinforcement Learning (MARL) library. Full rquirements used can be found in the Wiki pages.

 The PredPreyGrass envrionment has been initially modified from PettingZoo's the (SISL) Pursuit_v4 environment (https://pettingzoo.farama.org/environments/sisl/pursuit/):
 1. The envrionment is added with an addtional Predator learning agent. 
 2. The Pursuers have been converted to a Prey learning agent,
 3. Evaders have been converted to grass and are permanently 'freezed' and do not move.

 Similar to the PettingZoo' Pursuit envrionment, grass agent are left out of the 'self.agents' array. Including resulted into signifcant computing efficency without obvious advantages.
 
### Learning algorithm 
The Multi Agent Reinforcement Learning algorithm to control the PredPreyGrass environment is PPO from stable baselines3.



