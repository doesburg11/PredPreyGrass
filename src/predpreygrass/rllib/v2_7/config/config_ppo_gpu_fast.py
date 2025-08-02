config_ppo = {
    "max_iters": 1000,
    # training
    "train_batch_size_per_learner": 1024,
    "gamma": 0.99,
    "lr": 0.0003,
    "minibatch_size": 128,
    "num_epochs": 5,
    # Learners
    "num_gpus_per_learner": 1,
    "num_learners": 1,
    # Environment Runners
    "num_env_runners": 8,
    "num_envs_per_env_runner": 3,
    "rollout_fragment_length": "auto",
    "sample_timeout_s": 600,
    "num_cpus_per_env_runner": 3,
    # Resources
    "num_cpus_for_main_process": 4,
}
