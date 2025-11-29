"""Debug script to track limited-intake dynamics over time.

Usage (from project root):

    python -m predpreygrass.rllib.rllib_termination.debug_carcass_activity \
      --steps 500 --seed 42

This runs the rllib_termination PredPreyGrass environment and prints simple
aggregate stats about prey and grass energy per step. It is intended to
sanity-check the limited-intake (capped bite) logic without using carcass
channels.
"""

from __future__ import annotations

import argparse

from .config.config_env_rllib_termination import config_env
from .predpreygrass_rllib_env import PredPreyGrass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug limited-intake activity in rllib_termination env")
    parser.add_argument("--steps", type=int, default=500, help="Number of env steps to run")
    parser.add_argument("--seed", type=int, default=42, help="Seed for the env RNG")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = dict(config_env)
    cfg["seed"] = args.seed

    env = PredPreyGrass(cfg)
    obs, _ = env.reset(seed=args.seed)

    # Pick a sample prey to track over time (if any present)
    sample_prey_id = next((aid for aid in env.agents if "prey" in aid), None)

    header_cols = [
        "step",
        "total_prey_energy",
        "num_prey",
        "num_high_energy_prey",
        "total_grass_energy",
        "num_grass",
    ]
    if sample_prey_id is not None:
        header_cols.append(f"energy_{sample_prey_id}")
    print(",".join(header_cols))

    high_prey_threshold = cfg.get("prey_creation_energy_threshold", 0.0)

    for step in range(args.steps):
        action_dict = {agent_id: 0 for agent_id in env.agents}
        obs, rewards, terms, truncs, infos = env.step(action_dict)

        # Aggregate prey energy
        prey_energies = [float(env.agent_energies[a]) for a in env.agent_energies.keys() if "prey" in a]
        total_prey_energy = sum(prey_energies)
        num_prey = len(prey_energies)
        num_high_energy_prey = sum(e > high_prey_threshold for e in prey_energies)

        # Aggregate grass energy
        grass_energies = list(getattr(env, "grass_energies", {}).values())
        total_grass_energy = float(sum(grass_energies))
        num_grass = len(grass_energies)

        row = [
            str(step),
            f"{total_prey_energy:.3f}",
            str(num_prey),
            str(num_high_energy_prey),
            f"{total_grass_energy:.3f}",
            str(num_grass),
        ]
        if sample_prey_id is not None:
            energy_sample = float(env.agent_energies.get(sample_prey_id, 0.0))
            row.append(f"{energy_sample:.3f}")

        print(",".join(row))

        if terms.get("__all__") or truncs.get("__all__"):
            break


if __name__ == "__main__":
    main()
