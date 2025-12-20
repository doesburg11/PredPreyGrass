config_appo = {
    "max_iters": 1000,
    # Core learning
    "lr": 3e-4,
    "gamma": 0.99,
    "train_batch_size_per_learner": 512,
    "entropy_coeff": 0.01,
    "vf_loss_coeff": 0.5,
    "clip_param": 0.3,
    "kl_coeff": 0.2,
    "kl_target": 0.01,
    "use_kl_loss": True,
    "vtrace": True,
    "grad_clip": 40.0,
    # Resources
    "num_learners": 1,
    "num_env_runners": 28,
    "num_envs_per_env_runner": 1,
    "num_gpus_per_learner": 1,
    "num_cpus_for_main_process": 4,
    "num_cpus_per_env_runner": 1,
    "sample_timeout_s": 600,
    "rollout_fragment_length": 50,
}
