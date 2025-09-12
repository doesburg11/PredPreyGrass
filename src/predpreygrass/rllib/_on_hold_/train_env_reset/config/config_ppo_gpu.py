config_ppo = {
    # Training
    "train_batch_size": 1024,
    "gamma": 0.99,
    "lr": 0.0003,
    # Learners
    "num_gpus_per_learner": 1,
    "num_learners": 1,
    # Env runners
    "num_env_runners": 8,
    "num_envs_per_env_runner": 3,
    "rollout_fragment_length": "auto",
    "sample_timeout_s": 600,
    "num_cpus_per_env_runner": 3,
    # Resources
    "num_cpus_for_main_process": 4,
}
