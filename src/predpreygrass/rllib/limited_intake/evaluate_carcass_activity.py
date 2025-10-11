"""
Quick evaluator: runs the limited_intake environment for N steps with random actions
and records carcass activity metrics (energy and counts) over time. Produces a plot
and writes a CSV to the output folder.

Usage:
  python -m predpreygrass.rllib.limited_intake.evaluate_carcass_activity --steps 500 --seed 42 --out output/carcass_eval
"""
import argparse
import os
import csv
from datetime import datetime

import matplotlib.pyplot as plt

from .predpreygrass_rllib_env import PredPreyGrass


def build_default_config(seed: int = 0):
    # Minimal viable config; adjust as your project defaults require
    return {
        # toggles
        "debug_mode": False,
        "verbose_movement": False,
        "verbose_decay": False,
        "verbose_reproduction": False,
        "verbose_engagement": False,
        # episode steps
        "max_steps": 10_000,
        "seed": seed,
        # rewards
        "reward_predator_catch_prey": {"type_1_predator": 1.0, "type_2_predator": 1.0},
        "reward_prey_eat_grass": {"type_1_prey": 0.2, "type_2_prey": 0.2},
        "reward_predator_step": {"type_1_predator": 0.0, "type_2_predator": 0.0},
        "reward_prey_step": {"type_1_prey": 0.0, "type_2_prey": 0.0},
        "penalty_prey_caught": {"type_1_prey": -1.0, "type_2_prey": -1.0},
        "reproduction_reward_predator": {"type_1_predator": 0.0, "type_2_predator": 0.0},
        "reproduction_reward_prey": {"type_1_prey": 0.0, "type_2_prey": 0.0},
        # energy
        "energy_loss_per_step_predator": 0.1,
        "energy_loss_per_step_prey": 0.05,
        "predator_creation_energy_threshold": 8.0,
        "prey_creation_energy_threshold": 6.0,
        "max_energy_predator": 20.0,
        "max_energy_prey": 15.0,
        # multi-step eating caps
    # Intake caps are set in config_env_limited_intake.py: max_eating_predator and max_eating_prey
        # carcass decay/lifetime
        "carcass_decay_per_step": 0.0,
        "carcass_max_lifetime": None,
        # initial energy
        "initial_energy_predator": 6.0,
        "initial_energy_prey": 5.0,
        # population capacities
        "n_possible_type_1_predators": 4,
        "n_possible_type_2_predators": 0,
        "n_possible_type_1_prey": 6,
        "n_possible_type_2_prey": 0,
        "n_initial_active_type_1_predator": 2,
        "n_initial_active_type_2_predator": 0,
        "n_initial_active_type_1_prey": 3,
        "n_initial_active_type_2_prey": 0,
        # grid/obs
        "grid_size": 15,
        "num_obs_channels": 4,
        "predator_obs_range": 7,
        "prey_obs_range": 7,
        "include_visibility_channel": False,
        "respect_los_for_movement": False,
        "mask_observation_with_visibility": False,
        # grass
        "initial_num_grass": 20,
        "initial_energy_grass": 3.0,
        "energy_gain_per_step_grass": 0.5,
        "max_energy_grass": 5.0,
        # walls
        "num_walls": 10,
        "wall_placement_mode": "random",
        "manual_wall_positions": [],
        # mutation
        "mutation_rate_predator": 0.0,
        "mutation_rate_prey": 0.0,
        # actions
        "type_1_action_range": 3,
        "type_2_action_range": 3,
        # energy transfer
        "energy_transfer_efficiency": 1.0,
        "reproduction_energy_efficiency": 1.0,
        # reproduction
        "reproduction_cooldown_steps": 20,
        "reproduction_chance_predator": 0.0,
        "reproduction_chance_prey": 0.0,
        # move energy
        "move_energy_cost_factor": 0.0,
    }


def run_eval(steps: int, out_dir: str, seed: int, obs_mode: str = "random"):
    os.makedirs(out_dir, exist_ok=True)
    cfg = build_default_config(seed=seed)
    env = PredPreyGrass(cfg)
    obs, _ = env.reset(seed=seed)

    # Prepare CSV logging
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(out_dir, f"carcass_activity_{ts}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["step", "active_carcasses", "total_carcass_energy", "created", "created_energy", "consumed_energy", "decayed_energy", "expired", "removed"])

        active_counts = []
        total_energy_series = []

        for t in range(steps):
            # random actions
            actions = {a: env.action_spaces[a].sample() for a in env.agents}
            obs, rew, term, trunc, info = env.step(actions)

            metrics = env.get_carcass_metrics()
            active_counts.append(metrics["active_carcasses"])
            total_energy_series.append(metrics["total_carcass_energy"])
            step_m = metrics["step"]
            writer.writerow([
                env.current_step,
                metrics["active_carcasses"],
                f"{metrics['total_carcass_energy']:.6f}",
                step_m["created_count"],
                f"{step_m['created_energy']:.6f}",
                f"{step_m['consumed_energy']:.6f}",
                f"{step_m['decayed_energy']:.6f}",
                step_m["expired_count"],
                step_m["removed_count"],
            ])

    # Plot
    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.plot(total_energy_series, label="Total carcass energy", color="tab:red")
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Energy", color="tab:red")
    ax2 = ax1.twinx()
    ax2.plot(active_counts, label="Active carcasses", color="tab:blue", alpha=0.7)
    ax2.set_ylabel("Count", color="tab:blue")
    fig.tight_layout()
    png_path = os.path.join(out_dir, f"carcass_activity_{ts}.png")
    plt.savefig(png_path, dpi=150)
    print(f"Saved CSV to {csv_path}\nSaved plot to {png_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default=os.path.join("output", "carcass_eval"))
    args = parser.parse_args()
    run_eval(steps=args.steps, out_dir=args.out, seed=args.seed)


if __name__ == "__main__":
    main()
