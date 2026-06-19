config_appo = {
    "max_iters": 1000,
    # Core learning
    "lr": 0.0003,
    "gamma": 0.99,
    "train_batch_size_per_learner": 2048,
    "entropy_coeff": 0.0,
    "vf_loss_coeff": 0.5,
    "clip_param": 0.3,
    "kl_coeff": 0.2,
    "kl_target": 0.01,
    "use_kl_loss": True,
    "vtrace": True,
    "grad_clip": 40.0,
    # Resources
    # Keep one environment instance so mu-update semantics remain global.
    "num_learners": 1,
    "num_env_runners": 1,
    "num_envs_per_env_runner": 1,
    "num_gpus_per_learner": 1,
    "num_cpus_for_main_process": 4,
    "num_cpus_per_env_runner": 1,
    "sample_timeout_s": 600,
    "rollout_fragment_length": "auto",
}

config_ppo = config_appo
