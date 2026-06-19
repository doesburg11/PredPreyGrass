config_ppo = {
    "max_iters": 100,
    # Core learning
    "lr": 0.0001,
    "gamma": 0.99,
    "lambda_": 1.0,
    "train_batch_size_per_learner": 1024,
    "minibatch_size": 128,
    "num_epochs": 30,
    "entropy_coeff": 0.0,
    "vf_loss_coeff": 1.0,
    "clip_param": 0.3,
    # Resources
    "num_learners": 1,
    "num_env_runners": 4,
    "num_envs_per_env_runner": 1,
    "num_gpus_per_learner": 0,
    "num_cpus_for_main_process": 1,
    "num_cpus_per_env_runner": 1,
    "sample_timeout_s": 600,
    "rollout_fragment_length": "auto",
    # KL / exploration
    "kl_coeff": 0.2,
    "kl_target": 0.01,
    # pbt parameters
    "pbt_num_samples": 6,
    "perturbation_interval": 3,
    "resample_probability": 0.25,
    "quantile_fraction": 0.25,
    # PBT mutation ranges/choices
    "pbt_lr_choices": [1e-4, 2e-4, 3e-4],
    # "pbt_clip_range": [0.2, 0.3],              
    # "pbt_entropy_choices": [0.0, 0.0001],
    "pbt_num_epochs_range": [25, 30],          
    "pbt_minibatch_choices": [128, 256, 512],
    "pbt_train_batch_size_choices": [1024, 2048, 4096],

}
