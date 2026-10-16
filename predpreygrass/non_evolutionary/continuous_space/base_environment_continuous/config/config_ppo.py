"""PPO hyperparameters, resources, and run settings. Edit before training."""

from pathlib import Path

# This module's runs: runs/ inside the module (gitignored).
RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"

# {run_name} becomes <module>_<Amsterdam timestamp> when training starts.
# Set an output directory to None to disable that output.
config_ppo = {
    "mode": "ps",  # "ps": policy per species; "il": policy per individual.
    "iterations": 3000,  # Number of PPO training iterations.
    # 28 of the 32 CPU cores: the batch size, and so learning, is unchanged;
    # more workers only collect it faster (earlier runs used 16).
    "num_env_runners": 28,
    "num_learners": 0,
    "num_gpus_per_learner": 1,  # Set to 0 for CPU-only training.
    "train_batch_size": 4800,
    "minibatch_size": 1024,
    "num_epochs": 10,
    "lr": 3e-4,
    "gae_lambda": 0.95,
    "gamma": 0.99,  # Per decision; also used for environment reward shaping.
    "entropy_coeff": 0.01,
    "vf_share_layers": False,
    "grad_clip": 0.5,  # None disables gradient clipping.
    "vf_clip_param": 1000.0,  # Squared value-error cap for scaled death penalties.
    "tensorboard_dir": str(RUNS_DIR / "{run_name}" / "tensorboard"),
    "checkpoint_dir": str(RUNS_DIR / "{run_name}" / "checkpoint"),
    "checkpoint_every": 10,  # Requires checkpoint_dir; 0 saves only at the end.
    # Start both policies from a saved checkpoint (a run's checkpoint/ or
    # checkpoint/iter_<n> directory) instead of from scratch; None: scratch.
    # Only the network weights are loaded: optimizer state, iteration count
    # and output directories start fresh, as a new run.
    "init_checkpoint": None,
    # Policies kept as loaded and not trained, e.g. ["prey_policy"] to train
    # predators against fixed prey. Requires init_checkpoint and mode "ps".
    "frozen_policies": [],
}
