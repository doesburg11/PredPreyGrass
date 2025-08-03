config_ppo = {
    # Training
    "train_batch_size": 32000,  # More stable updates
    "gamma": 0.99,
    "lr": 5e-5,  # Lower learning rate for async stability
    # PPO-like constraints (added manually in training config)
    "kl_coeff": 0.2,
    "kl_target": 0.01,
    "clip_param": 0.3,
    "use_kl_loss": True,
    "entropy_coeff": 0.01,  # Encourage exploration
    # Learners
    "num_gpus_per_learner": 1,
    "num_learners": 1,
    # Env runners
    "num_env_runners": 4,
    "num_envs_per_env_runner": 8,
    "rollout_fragment_length": 1000,  # Longer rollouts = more stable V-trace
    "sample_timeout_s": 600,
    "num_cpus_per_env_runner": 2,
    # Resources
    "num_cpus_for_main_process": 4,
}
