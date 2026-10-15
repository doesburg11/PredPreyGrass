"""Evaluation settings. Edit this dictionary before running eval.py."""

from pathlib import Path

# This module's runs: runs/ inside the module (gitignored).
RUNS_DIR = Path(__file__).resolve().parents[1] / "runs"

config_eval = {
    # What to run: "checkpoint" (a trained run, below) or "random" (every
    # agent acts uniformly at random, as a baseline).
    "source": "random",
    # A run's checkpoint/ directory (its final policies) or one of its
    # checkpoint/iter_<n> directories; founders come from its gene pool.
    "checkpoint": str(RUNS_DIR / "<run>" / "checkpoint"),
    "episodes": 1,
    "seed": 0,  # Seed of the first episode; episode k uses seed + k - 1.
    "stochastic": True,  # Sample actions as in training; False takes the argmax.
    # Settings that replace the checkpoint's (or, for "random", config_env's)
    # values, e.g. {"max_time_steps": 2000}. Keys as in config_env.py.
    "env_overrides": {},
    # Display: "window" (live pygame window), "video" (also saves an mp4 per
    # episode in out_dir), or "none" (no rendering, fastest). On a machine
    # without a display, set SDL_VIDEODRIVER=dummy for "video".
    "render": "window",
    # "fit" scales the arena to fit the screen, with the population chart
    # beside it; an int is the arena's side in pixels; None draws 1:1.
    "window_size": "fit",
    # Window frame rate, in decisions per second; None: as fast as possible.
    # With action_repeat 8, 8 shows the simulation at about real time.
    "window_fps": 8,
    "out_dir": "videos",  # Where "video" saves its mp4s.
    "fps": 30,  # Frame rate of saved mp4s (one frame per decision).
    # The view cones of one predator and one prey.
    "draw_view_cones": True,
    # Write episode, decision, physics step, predators, prey and cumulative
    # births per decision to this CSV; None writes nothing.
    "population_csv": None,
}
