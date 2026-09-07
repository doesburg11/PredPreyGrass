"""
Run one headless self-play episode from a single checkpoint (both predator_policy
and prey_policy from the SAME checkpoint -- unlike master_tournament_matrix.py,
which mixes checkpoints from different iterations) and plot predator/prey/grass
population counts over the episode, in the same style as the Lotka-Volterra-like
population chart in this directory's README.

No rendering (no PyGame) -- this is a headless population-count logger for
generating that chart from a specific trained checkpoint, not for interactive
debugging (see evaluate_ppo_from_checkpoint_debug.py for that).
"""
from predpreygrass.non_evolutionary.base_environment.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.non_evolutionary.base_environment.config_env import config_env
from predpreygrass.non_evolutionary.base_environment.evaluate_ppo_from_checkpoint_debug import policy_pi
from predpreygrass.non_evolutionary.base_environment.master_tournament_matrix import policy_mapping_fn

# --- External libraries ---
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from ray.rllib.core.rl_module.rl_module import RLModule


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-path", type=str, required=True, help="A single checkpoint_XXXXXX directory.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int, default=None, help="Override config_env's max_steps.")
    parser.add_argument("--output", type=str, default="population_dynamics.png")
    parser.add_argument("--title", type=str, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    checkpoint_path = Path(args.checkpoint_path).expanduser()

    rl_modules = {
        pid: RLModule.from_checkpoint(checkpoint_path / "learner_group" / "learner" / "rl_module" / pid)
        for pid in ("predator_policy", "prey_policy")
    }

    env_config = dict(config_env)
    if args.max_steps is not None:
        env_config["max_steps"] = args.max_steps
    env = PredPreyGrass(env_config)

    observations, _ = env.reset(seed=args.seed)
    active_agents = list(observations.keys())

    steps, predator_counts, prey_counts, grass_counts = [], [], [], []
    steps.append(env.current_step)
    predator_counts.append(env.current_num_predators)
    prey_counts.append(env.current_num_prey)
    grass_counts.append(env.current_num_grass)

    while True:
        action_dict = {
            agent_id: policy_pi(observations[agent_id], rl_modules[policy_mapping_fn(agent_id)], deterministic=True)
            for agent_id in active_agents
        }
        observations, rewards, terminations, truncations, _ = env.step(action_dict)

        steps.append(env.current_step)
        predator_counts.append(env.current_num_predators)
        prey_counts.append(env.current_num_prey)
        grass_counts.append(env.current_num_grass)

        active_agents = [
            agent_id for agent_id in observations
            if not terminations.get(agent_id, False) and not truncations.get(agent_id, False)
        ]
        if terminations.get("__all__", False) or truncations.get("__all__", False):
            break

    print(f"Episode finished at step {env.current_step}: "
          f"predators={env.current_num_predators}, prey={env.current_num_prey}, grass={env.current_num_grass}")

    plt.figure(figsize=(9, 5.4))
    plt.plot(steps, grass_counts, label="Grass", color="green")
    plt.plot(steps, prey_counts, label="Prey", color="blue")
    plt.plot(steps, predator_counts, label="Predators", color="red")
    plt.xlabel("Time step")
    plt.ylabel("Population size")
    plt.title(args.title or f"Population dynamics ({checkpoint_path.name})")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Saved plot to {args.output}")
