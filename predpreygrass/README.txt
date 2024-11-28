Structure of the respository

predpreygrass/
├── multi_objective/                                # UNDER CONSTRUCTION
└── single_objective/                               # Directory for single-objective environments and related scripts
    ├── agents/
    │   ├── discrete_agent.py                       # Implementation of discrete agent for single-objective environment
    ├── config/
    │   ├── config_predpreygrass.py                 # Configuration file for single-objective environment
    │   └── README.md                               # Documentation for the configuration settings
    ├── envs/
    │   ├── base_env/                               # Base environment implementation details 
    │   ├── predpreygrass_aec_v0.py                 # Implementation of agent-environment cycle (AEC) version
    │   ├── predpreygrass_parallel_v0.py            # Parallel environment implementation
    │   └── README.md                               # Documentation for environments
    ├── eval/
    │   ├── evaluate_ppo_from_file.py               # Script for evaluating trained PPO models
    │   ├── evaluate_random_policy.py               # Script for evaluating random policies
    │   └── utils/                                  # Utility scripts for evaluation (details not provided)
    ├── train/
    │   ├── README.md                               # Documentation for training scripts
    │   ├── train_ppo.py                            # Training script using PPO for single-objective environment
    │   └── utils/                                  # Utility scripts for training (details not provided)
    ├── tune/
    │   ├── so_predpreygrass_v0_ppo_tuning_2.py     # UNDER CONSTRUCTION
    │   └── so_predpreygrass_v0_ppo_tuning.py       # UNDER CONSTRUCTION
    └── utils/                                      
        ├── env.py                                  # General utility functions for the environment
        └── renderer.py                             # Utility script for rendering the environment

