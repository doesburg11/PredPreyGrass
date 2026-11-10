"""PPO hyperparameters, resources, and run settings (deer_stag_hunt's), and
the arm. Edit before training, or set them from launchers/replicate.py."""

from pathlib import Path

# This module's runs: runs/ inside the module (gitignored).
RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"
# deer_stag_hunt's runs, whose deer the deer use.
DEER_RUNS_DIR = Path(__file__).resolve().parents[4] / "non_evolutionary" / "continuous_space" / "deer_stag_hunt" / "runs"

# {run_name} becomes <module>_<Amsterdam timestamp> when training starts.
# Set an output directory to None to disable that output.
config_ppo = {
    # "L": learning only (genes inherited, not expressed); "B": learning plus
    # the genes' innate bias (beta = genes.BETA0). The E arm is evolve.py.
    "arm": "B",
    # Section 2: a deer_stag_hunt run whose deer are replayed in training
    # order (deer_schedule.py). Section 1 instead set deer_checkpoint, one
    # checkpoint's deer, frozen; exactly one of the two is set.
    "deer_schedule": str(DEER_RUNS_DIR / "touch_1"),
    "deer_checkpoint": None,
    "seed": None,  # RLlib's seed (network initialisation, first episodes)
    "iterations": 1000,  # Number of PPO training iterations.
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
