config_ppo = {
    "max_iters": 250,
    # Core learning
    "lr": 5e-5,
    "gamma": 0.99,
    "lambda_": 0.95,
    "train_batch_size_per_learner": 1024,
    "minibatch_size": 128,
    "num_epochs": 6,
    "entropy_coeff": 0.01,
    "vf_loss_coeff": 1.0,
    "clip_param": 0.3,
    # Resources
    "num_learners": 1,
    "num_env_runners": 5,
    "num_envs_per_env_runner": 1,
    "num_gpus_per_learner": 0,
    "num_cpus_for_main_process": 1,
    "num_cpus_per_env_runner": 1,
    "sample_timeout_s": 600,
    "rollout_fragment_length": "auto",
    # KL / exploration
    "kl_coeff": 0.2,
    "kl_target": 0.01,
}
