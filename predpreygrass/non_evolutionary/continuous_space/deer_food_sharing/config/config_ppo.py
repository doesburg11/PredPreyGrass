"""PPO hyperparameters, resources, and run settings. Edit before training.

The values are those of stag_hunt_continuous's run of RESULTS.md section 8
there (1000 iterations, 14 env runners)."""

from pathlib import Path

# This module's runs: runs/ inside the module (gitignored).
RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"

# {run_name} becomes <module>_<Amsterdam timestamp> when training starts.
# Set an output directory to None to disable that output.
config_ppo = {
    "iterations": 1000,  # Number of PPO training iterations.
    # The batch size, and so learning, does not depend on the number of
    # workers; more workers only collect it faster.
    "num_env_runners": 14,
    "num_learners": 0,
    "num_gpus_per_learner": 1,  # Set to 0 for CPU-only training.
    "train_batch_size": 4800,
    "minibatch_size": 1024,
    "num_epochs": 10,
    "lr": 3e-4,
    "gae_lambda": 0.95,
    "gamma": 0.99,  # Per decision.
    "entropy_coeff": 0.01,
    "vf_share_layers": False,
    "grad_clip": 0.5,  # None disables gradient clipping.
    "vf_clip_param": 1000.0,
    "tensorboard_dir": str(RUNS_DIR / "{run_name}" / "tensorboard"),
    "checkpoint_dir": str(RUNS_DIR / "{run_name}" / "checkpoint"),
    "checkpoint_every": 10,  # Requires checkpoint_dir; 0 saves only at the end.
}
