config_impala = {
    "max_iters": 1000,
    # Core learning
    "lr": 0.0003,
    "gamma": 0.99,
    "train_batch_size_per_learner": 4096,
    "entropy_coeff": 0.01,
    "vf_loss_coeff": 0.5,
    # Resources
    "num_learners": 1,
    "num_env_runners": 4,
    "num_envs_per_env_runner": 2,
    "num_gpus_per_learner": 1,
    "num_cpus_for_main_process": 4,
    "num_cpus_per_env_runner": 4,
    "sample_timeout_s": 600,
    "rollout_fragment_length": 100,
}
