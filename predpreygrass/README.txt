Structure of the respository

├── global_config.py
├── pettingzoo
│   ├── agents
│   │   └── discrete_agent.py
│   ├── config
│   │   └── config_predpreygrass.py
│   ├── envs
│   │   ├── predpreygrass_aec.py
│   │   ├── predpreygrass_aec_v0.py
│   │   └── predpreygrass_base.py
│   ├── eval
│   │   ├── evaluate_ppo_from_file_aec_env.py
│   │   ├── evaluate_random_policy.py
│   │   ├── evaluate_random_policy_step_wise.py
│   │   ├── evaluate_random_policy_to_file_aec_env.py
│   │   ├── parameter_variation_train_wrapped_to_parallel_and_evaluate_aec.py
│   │   └── utils
│   │       ├── evaluator.py
│   │       ├── population_plotter.py
│   └── train
│       ├── train_sb3_ppo_parallel_wrapped_aec_env.py
│       └── utils
│           ├── config_saver.py
│           ├── logger.py
│           └── trainer.py
├── rllib
│   ├── config_env.py
│   ├── evaluate.py
│   ├── predpreygrass_rllib_env.py
│   ├── random_policy.py
│   ├── random_policy_step_wise.py
│   ├── time_test_random_policy .py
│   └── train.py
└── utils
    └── renderer.py