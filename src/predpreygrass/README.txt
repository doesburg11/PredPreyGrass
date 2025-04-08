Structure of the respository

├── assets
│   └── images
├── CODE_OF_CONDUCT.md
├── LICENSE
├── PredPreyGrass.code-workspace
├── predpreygrass.egg-info
│   ├── dependency_links.txt
│   ├── PKG-INFO
│   ├── requires.txt
│   ├── SOURCES.txt
│   └── top_level.txt
├── predpreygrass.ipynb
├── pyproject.toml
├── README.md
├── requirements_lock.txt
├── requirements.txt
├── setup.py
└── src
    ├── predpreygrass
    │   ├── global_config.py
    │   ├── __init__.py
    │   ├── pettingzoo
    │   │   ├── agents
    │   │   │   ├── 
    │   │   │   └── discrete_agent.py
    │   │   ├── config
    │   │   │   ├── config_predpreygrass.py
    │   │   │   └── README.md
    │   │   ├── envs
    │   │   │   ├── __init__.py
    │   │   │   ├── predpreygrass_aec.py
    │   │   │   ├── predpreygrass_aec_v0.py
    │   │   │   ├── predpreygrass_base.py
    │   │   │   └── README.md
    │   │   ├── eval
    │   │   │   ├── evaluate_ppo_from_file_aec_env.py
    │   │   │   ├── evaluate_random_policy.py
    │   │   │   ├── evaluate_random_policy_step_wise.py
    │   │   │   ├── evaluate_random_policy_to_file_aec_env.py
    │   │   │   ├── parameter_variation_train_wrapped_to_parallel_and_evaluate_aec.py
    │   │   │   ├── README.md
    │   │   │   └── utils
    │   │   │       ├── evaluator.py
    │   │   │       └── population_plotter.py
    │   │   ├── README.md
    │   │   └── train
    │   │       ├── README.md
    │   │       ├── train_sb3_ppo_parallel_wrapped_aec_env.py
    │   │       └── utils
    │   │           ├── config_saver.py
    │   │           ├── _continue_training_saved_model.py
    │   │           ├── logger.py
    │   │           └── trainer.py
    │   ├── README.txt
    │   ├── rllib
    │   │   ├── readme.md
    │   │   ├── utils
    │   │   │   ├── create_video_from_checkpoint.py
    │   │   │   └── create_video_from_checkpoint_speed.py
    │   │   ├── v0_neumann
    │   │   │   ├── config_env.py
    │   │   │   ├── evaluate_ppo_from_checkpoint.py
    │   │   │   ├── __init__.py
    │   │   │   ├── predpreygrass_rllib_env.py
    │   │   │   ├── random_policy.py
    │   │   │   ├── random_policy_step_wise.py
    │   │   │   ├── random_policy_time_test.py
    │   │   │   └── train_rllib_ppo_multiagent_env.py
    │   │   ├── v1_moore
    │   │   │   ├── config_env.py
    │   │   │   ├── evaluate_ppo_from_checkpoint.py
    │   │   │   ├── predpreygrass_rllib_env.py
    │   │   │   ├── random_policy.py
    │   │   │   ├── random_policy_time_test.py
    │   │   │   └── train_rllib_ppo_multiagent_env.py
    │   │   ├── v2_speed
    │   │   │   ├── config_env.py
    │   │   │   ├── evaluate_ppo_from_checkpoint.py
    │   │   │   ├── __init__.py
    │   │   │   ├── predpreygrass_rllib_env.py
    │   │   │   ├── random_policy.py
    │   │   │   ├── random_policy_time_test.py
    │   │   │   └── train_rllib_ppo_multiagent_env.py
    │   │   ├── v3_age
    │   │   │   ├── config_env.py
    │   │   │   ├── evaluate_ppo_from_checkpoint_old.py
    │   │   │   ├── evaluate_ppo_from_checkpoint.py
    │   │   │   ├── evaluate_random_policy.py
    │   │   │   ├── __init__.py
    │   │   │   ├── predpreygrass_rllib_env.py
    │   │   │   ├── random_policy.py
    │   │   │   ├── random_policy_ time_test.py
    │   │   │   └── train_rllib_ppo_multiagentenv.py
    │   │   ├── v4_select_coef
    │   │   │   ├── config_env.py
    │   │   │   ├── evaluate_ppo_from_checkpoint copy.py
    │   │   │   ├── evaluate_ppo_from_checkpoint.py
    │   │   │   ├── evaluate_random_policy.py
    │   │   │   ├── __init__.py
    │   │   │   ├── predpreygrass_rllib_env.py
    │   │   │   ├── random_policy.py
    │   │   │   ├── random_policy_ time_test.py
    │   │   │   └── train_rllib_ppo_multiagentenv.py
    │   │   └── v4_select_coef_HBP
    │   │       ├── config_env.py
    │   │       ├── evaluate_ppo_from_checkpoint.py
    │   │       ├── evaluate_random_policy.py
    │   │       ├── __init__.py
    │   │       ├── predpreygrass_rllib_env.py
    │   │       ├── random_policy.py
    │   │       ├── random_policy_ time_test.py
    │   │       └── train_rllib_ppo_multiagentenv.py
    │   └── utils
    │       └── renderer.py
    └── predpreygrass.egg-info
        ├── dependency_links.txt
        ├── PKG-INFO
        ├── requires.txt
        ├── SOURCES.txt
        └── top_level.txt

