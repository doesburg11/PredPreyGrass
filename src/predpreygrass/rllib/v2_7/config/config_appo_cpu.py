config_ppo = {
    # Training
    "train_batch_size": 8000,  # Smaller than GPU setup, but large enough
    "gamma": 0.99,
    "lr": 1e-4,  # Lower than default for async stability
    "kl_coeff": 0.2,
    "kl_target": 0.01,
    "clip_param": 0.3,
    "use_kl_loss": True,
    "entropy_coeff": 0.01,
    # Learners
    "num_gpus_per_learner": 0,
    "num_learners": 1,
    # Env runners
    "num_env_runners": 2,  # Reduced to fit 8 cores
    "num_envs_per_env_runner": 1,
    "rollout_fragment_length": 400,
    "sample_timeout_s": 600,
    "num_cpus_per_env_runner": 2,
    # Resources
    "num_cpus_for_main_process": 2,
}
