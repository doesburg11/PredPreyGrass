config_ppo = {
    "max_iters": 9,
    # Core learning
    "lr": 1e-4,
    "gamma": 0.99,
    "lambda_": 0.95,
    "train_batch_size_per_learner": 1024,
    "minibatch_size": 128,
    "num_epochs": 10,
    "entropy_coeff": 0.01,
    "vf_loss_coeff": 1.0,
    "clip_param": 0.3,
    # Resources
    "num_cpus_for_main_process": 1,
    "num_learners": 1,
    "num_gpus_per_learner": 0,
    "num_env_runners": 1,
    "num_envs_per_env_runner": 4,
    "num_cpus_per_env_runner": 1,
    "sample_timeout_s": 600,
    "rollout_fragment_length": "auto",
    # KL / explorationclear
    "kl_coeff": 0.2,
    "kl_target": 0.01,
    # pbt parameters
    "pbt_num_samples": 4,
    "perturbation_interval": 3,
}
