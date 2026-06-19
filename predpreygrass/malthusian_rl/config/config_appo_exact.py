"""
Locked APPO settings for strict Malthusian reproduction runs.

This file is intentionally separate from exploratory CPU/GPU configs. The exact
trainer imports only this module and validates these values before launching.
"""

config_appo_exact = {
    # Run control
    "max_iters": 1000,
    "checkpoint_every": 10,
    # Core learning
    # The paper reports RMSProp learning rate sampled log-uniformly from
    # [0.0001, 0.005]. This fixed value is in-range and keeps exact runs
    # reproducible until the original per-run samples are known.
    "lr": 0.0003,
    "gamma": 0.99,
    # Paper batch size is 32 complete trajectories; RLlib consumes timesteps.
    # With the cited unroll length 20, this is the closest direct translation.
    "train_batch_size_per_learner": 640,
    "paper_batch_size_trajectories": 32,
    # Geometric midpoint of the cited log-uniform entropy range [0.00005, 0.05].
    # The paper did not publish the sampled value for each plotted run.
    "entropy_coeff": 0.0015811388300841897,
    "vf_loss_coeff": 0.5,
    "clip_param": 0.3,
    "opt_type": "rmsprop",
    "decay": 0.99,
    "momentum": 0.0,
    "epsilon": 0.0001,
    # Exact parity locks for APPO/V-trace.
    "vtrace": True,
    "vtrace_clip_rho_threshold": 1.0,
    "vtrace_clip_pg_rho_threshold": 1.0,
    "use_kl_loss": False,
    "kl_coeff": 0.0,
    "kl_target": 0.01,
    "grad_clip": 40.0,
    # Recurrent module locks.
    "use_lstm": True,
    "max_seq_len": 20,
    "lstm_cell_size": 64,
    "paper_network_architecture": True,
    # Resources. Keep a single environment instance so mu-update semantics remain global.
    "num_learners": 1,
    "num_env_runners": 1,
    "num_envs_per_env_runner": 1,
    "num_gpus_per_learner": 0,
    "num_cpus_for_main_process": 1,
    "num_cpus_per_env_runner": 1,
    "sample_timeout_s": 600,
    "rollout_fragment_length": 20,
    "use_circular_buffer": False,
    "simple_queue_size": 32,
}

paper_learner_citation_map = {
    "vtrace": {
        "value": True,
        "paper": "Section 2.4 states V-trace is used for species policy updates.",
        "status": "cited",
    },
    "vtrace_clip_rho_threshold": {
        "value": 1.0,
        "paper": "Section 2.4 says V-trace truncation levels are set to 1.",
        "status": "cited",
    },
    "vtrace_clip_pg_rho_threshold": {
        "value": 1.0,
        "paper": "Section 2.4 says V-trace truncation levels are set to 1.",
        "status": "cited",
    },
    "max_seq_len": {
        "value": 20,
        "paper": "RL Agent table reports LSTM unroll length 20.",
        "status": "cited",
    },
    "lstm_cell_size": {
        "value": 64,
        "paper": "Function approximation paragraph reports an LSTM of size 64.",
        "status": "cited",
    },
    "vf_loss_coeff": {
        "value": 0.5,
        "paper": "RL Agent table reports baseline loss scaling 0.5.",
        "status": "cited",
    },
    "gamma": {
        "value": 0.99,
        "paper": "RL Agent table reports discount 0.99.",
        "status": "cited",
    },
    "opt_type": {
        "value": "rmsprop",
        "paper": "Optimization table reports RMSProp.",
        "status": "cited",
    },
    "decay": {
        "value": 0.99,
        "paper": "Optimization table reports RMSProp decay 0.99.",
        "status": "cited",
    },
    "epsilon": {
        "value": 0.0001,
        "paper": "Optimization table reports RMSProp epsilon 0.0001.",
        "status": "cited",
    },
    "paper_batch_size_trajectories": {
        "value": 32,
        "paper": "Optimization table reports batch size 32 trajectories.",
        "status": "cited",
    },
    "train_batch_size_per_learner": {
        "value": 640,
        "paper": "RLlib translation of 32 trajectories x LSTM unroll length 20.",
        "status": "derived",
    },
    "entropy_coeff": {
        "value": config_appo_exact["entropy_coeff"],
        "paper": "RL Agent table reports a log-uniform range [0.00005, 0.05], not per-run samples.",
        "status": "derived_midpoint",
    },
    "lr": {
        "value": config_appo_exact["lr"],
        "paper": "Optimization table reports a log-uniform range [0.0001, 0.005], not per-run samples.",
        "status": "in_range_fixed_value",
    },
}

# Backward-compatible alias for local scripts that expect the older variable name.
config_appo = config_appo_exact
config_ppo = config_appo_exact
