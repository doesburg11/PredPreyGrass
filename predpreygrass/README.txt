Structure of the respository

predpreygrass/
└── single_objective/                                                           # Directory for single-objective environments and related scripts
    ├── agents/
    │   ├── discrete_agent.py                                                   # Implementation of discrete agent for single-objective environment
    ├── config/
    │   ├── config_predpreygrass.py                                             # Configuration file for single-objective environment
    │   └── README.md                                                           # Documentation for the configuration settings
    ├── envs/
    │   ├── pettingzoo/                                                         # Base environment implementation details 
    │   ├── rllib/                                                              # RLlib new API stack experimentation     
    │   ├── predpreygrass_aec_v0.py                                             # Implementation of agent-environment cycle (AEC) version
    │   └── README.md                                                           # Documentation for environments
    ├── eval/
    │   ├── evaluate_ppo_from_file_aec_env.py                                   # Script for evaluating trained PPO models
    │   ├── evaluate_random_policy_to_file_aec_env.py                           # Script for evaluating random policies
    │   ├── evaluate_random_policy_aec_env.py                                   # Script for evaluating random policies
    │   ├── parameter_variation_train_wrapped_to_parallel_and_evaluate_aec.py   # Script for evaluating random policies
    │   └── utils/                                                              # Utility scripts for evaluation (details not provided)
    ├── train/
    │   ├── README.md                                                           # Documentation for training scripts
    │   ├── train_sb3_ppo_parallel_wrapped_aec_env.py                                                        # Training script using PPO for single-objective environment
    │   └── utils/                                                              # Utility scripts for training (details not provided)
    └── utils/                                      
        └── renderer.py                                                         # Utility script for rendering the environment

