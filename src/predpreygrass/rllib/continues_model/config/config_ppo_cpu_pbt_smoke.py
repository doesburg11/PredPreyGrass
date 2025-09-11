config_ppo = {
    "max_iters": 3,
    # Core learning
    "lr": 1e-4,
    "gamma": 0.99,
    "lambda_": 0.95,
    "train_batch_size_per_learner": 256,
    "minibatch_size": 64,
    "num_epochs": 1,
    "entropy_coeff": 0.01,
    "vf_loss_coeff": 1.0,
    "clip_param": 0.3,
    # Resources
    "num_cpus_for_main_process": 1,
    "num_learners": 1,
    "num_gpus_per_learner": 0,
    "num_env_runners": 1,
    "num_envs_per_env_runner": 1,
    "num_cpus_per_env_runner": 1,
    "sample_timeout_s": 60,
    "rollout_fragment_length": 32,
    # KL / explorationclear
    "kl_coeff": 0.2,
    "kl_target": 0.01,
    # pbt parameters
    "pbt_num_samples": 3,
    "perturbation_interval": 1,
    "resample_probability": 1.0,
    "quantile_fraction": 0.5,
    # PBT mutation ranges/choices
    "pbt_lr_choices": [5e-4, 1e-4],
    "pbt_clip_range": (0.1, 0.3),
    "pbt_entropy_choices": [0.0, 0.001],
    "pbt_num_epochs_range": (1, 3),
    "pbt_minibatch_choices": [64, 128],
    "pbt_train_batch_size_choices": [256, 512],
}
