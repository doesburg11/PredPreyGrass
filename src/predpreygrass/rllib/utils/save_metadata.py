import os
import json
from datetime import datetime

def save_run_metadata_from_trial(config_env: dict, ppo_config: dict, model_arch: dict, trial):
    """
    Saves training metadata to the current Ray RLlib algorithm's log directory.

    Args:
        config_env (dict): The environment configuration dictionary.
        ppo_config (dict): PPO configuration parameters (learning, rollout, etc.).
        model_arch (dict): Model architecture settings (e.g. conv filters, FC layers).
        trial: The RLlib Algorithm instance (used to access .log_dir).
    """
    trial_dir = getattr(trial, "log_dir", None)
    if not trial_dir:
        print("[✘] Could not detect log_dir on the algorithm object. Metadata not saved.")
        return

    metadata = {
        "timestamp": datetime.now().isoformat(),
        "config_env": config_env,
        "ppo_config": ppo_config,
        "model_architecture": model_arch,
        "note": "Auto-saved via save_run_metadata_from_trial() at training init.",
    }

    try:
        os.makedirs(trial_dir, exist_ok=True)
        metadata_path = os.path.join(trial_dir, "run_metadata.json")
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=4)
        print(f"[✔] Saved metadata to {metadata_path}")
    except Exception as e:
        print(f"[✘] Failed to save metadata: {e}")
