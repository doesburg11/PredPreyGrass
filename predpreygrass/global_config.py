import os
from pathlib import Path

# put in here your own directory to the output folder
PROJECT_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RESULTS_DIR = os.path.join(PROJECT_PATH, "output")

# Ray Tune / ERL training output roots, shared across all modules.
SIMULATION_RESULTS_DIR = Path.home() / "simulation_results"
RAY_RESULTS_DIR = SIMULATION_RESULTS_DIR / "ray_results"
ERL_RESULTS_DIR = SIMULATION_RESULTS_DIR / "erl_results"
