config_ppo = {
    "max_iters": 100,
    # Core learning
    "lr": 5e-5,
    # Note: this is the *training algorithm's* gamma (RLlib's PPO discounting
    # over the rollout), separate from config_env's "gamma" entry, which is
    # carried there for the opponent-shaping training loop (Section 7/8 of
    # the module README) and isn't read by this script. Kept equal by
    # convention, not by any code-level link between the two.
    "gamma": 0.95,
    "lambda_": 0.95,
    "train_batch_size_per_learner": 2048,
    "minibatch_size": 256,
    "num_epochs": 6,
    "entropy_coeff": 0.01,
    "vf_loss_coeff": 1.0,
    "clip_param": 0.3,
    # Resources -- this environment is tiny (13-dim flat observation, no
    # images), so this trains fine on CPU with a handful of env runners.
    "num_learners": 1,
    "num_env_runners": 4,
    "num_envs_per_env_runner": 1,
    "num_gpus_per_learner": 0,
    "num_cpus_for_main_process": 1,
    "num_cpus_per_env_runner": 1,
    "sample_timeout_s": 300,
    "rollout_fragment_length": "auto",
    # KL / exploration
    "kl_coeff": 0.2,
    "kl_target": 0.01,
}
