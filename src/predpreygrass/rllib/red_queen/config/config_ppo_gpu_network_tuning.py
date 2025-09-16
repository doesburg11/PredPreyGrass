# identical to epoch archive epoch = 10

config_ppo = {
    "max_iters": 1000,
    # Core learning
    "lr": 0.0003,
    "gamma": 0.99,
    "lambda_": 1.0,
    # Batch size sampling
    "train_batch_size_per_learner": 1024, # train_batch_size deprecated in new stack
    "minibatch_size": 128,
    "num_epochs": 30,
    "entropy_coeff": 0.0,
    "vf_loss_coeff": 1.0,
    "clip_param": 0.3,
    # Resources (32 CPUs total; reserve ~2 for Ray/main)
    # Uses ~30 CPUs for sampling, 1–2 for overhead.
    "num_learners": 1,
    "num_gpus_per_learner": 1,
    "num_env_runners": 8,
    "num_envs_per_env_runner": 3,
    "num_cpus_per_env_runner": 3,
    "num_cpus_for_main_process": 4,
    # Sampling
    "sample_timeout_s": 600,
    "rollout_fragment_length": "auto",
    # KL / exploration
    "kl_coeff": 0.2,
    "kl_target": 0.01,
}
