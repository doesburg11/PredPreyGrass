# === fitness_tracker.py ===
from collections import defaultdict
import json
import random


def get_offspring_totals(unique_agent_stats):
    """
    Aggregates total offspring per policy group (e.g., speed_1_prey).
    """
    offspring_totals = defaultdict(int)
    for stat in unique_agent_stats.values():
        key = stat.get("policy_group")
        offspring_totals[key] += stat.get("offspring_count", 0)
    return dict(offspring_totals)


def mutate_reward_value_gaussian(base_value: float, std_dev: float = 0.5) -> float:
    """
    Apply Gaussian mutation (mean=0, std=std_dev) to a reward value.
    Ensures mutated value remains >= 0.0.
    """
    return max(0.0, base_value + random.gauss(0, std_dev))


def select_winner_and_mutate(offspring_totals, reward_config, target_key="reward_prey_eat_grass"):
    """
    Compare two competing reward groups. If one wins, apply Gaussian mutation
    to the target reward of the losing type (copied from the winner).
    Returns a new config dict (copy).
    """
    if len(offspring_totals) < 2:
        print("[META-SELECTION] Skipped (not enough competing types).")
        return reward_config

    sorted_types = sorted(offspring_totals.items(), key=lambda x: x[1], reverse=True)
    winner, loser = sorted_types[0][0], sorted_types[1][0]

    print(f"[META-SELECTION] Winner: {winner} ({offspring_totals[winner]}), Loser: {loser} ({offspring_totals[loser]})")

    new_config = reward_config.copy()
    if loser in reward_config:
        new_config[loser] = reward_config[winner].copy()
        current_value = new_config[loser].get(target_key, 0)
        new_config[loser][target_key] = mutate_reward_value_gaussian(current_value)
        print(f"[MUTATION] Updated {loser} reward '{target_key}' from {current_value:.3f} to {new_config[loser][target_key]:.3f}")
    return new_config


def save_mutated_reward_config(config_dict, out_path):
    """
    Save updated reward configuration to disk.
    """
    with open(out_path, "w") as f:
        json.dump(config_dict, f, indent=2)
    print(f"[META-SELECTION] Mutated reward config saved to: {out_path}")


def mutate_env_reward_config_inplace(env, target_key="reward_prey_eat_grass", save_path=None):
    """
    Apply reward mutation to the environment in-place. Optionally save the updated config.
    """
    if not hasattr(env, "unique_agent_stats"):
        print("[META-SELECTION] Skipped mutation: env does not track unique_agent_stats")
        return

    offspring_totals = get_offspring_totals(env.unique_agent_stats)
    reward_config = env.config.get("reward_config", {})
    new_reward_config = select_winner_and_mutate(offspring_totals, reward_config, target_key)
    env.config["reward_config"] = new_reward_config

    if save_path:
        save_mutated_reward_config(new_reward_config, save_path)

    print("[META-SELECTION] Updated env.config['reward_config'] during training.")


def mutate_reward_config_from_stats(reward_config, offspring_totals, target_key="reward_prey_eat_grass", save_path=None):
    """
    Mutate the reward config dict based on offspring stats and save it if requested.
    """
    if not reward_config or not offspring_totals:
        print("[MUTATION] Skipped: Missing reward_config or offspring_totals")
        return

    # Example mutation logic: reward += 0.1 for winners, -= 0.1 for losers
    sorted_prey = sorted([(k, v) for k, v in offspring_totals.items() if "prey" in k], key=lambda x: x[1], reverse=True)

    if len(sorted_prey) < 2:
        print("[MUTATION] Skipped: Not enough prey types for mutation.")
        return

    winner, loser = sorted_prey[0][0], sorted_prey[1][0]

    for agent_key in reward_config:
        if "prey" in agent_key:
            delta = 0
            if agent_key == winner:
                delta = 0.1
            elif agent_key == loser:
                delta = -0.1

            current_value = reward_config[agent_key].get(target_key, 0.0)
            new_value = max(0.0, current_value + delta)
            reward_config[agent_key][target_key] = new_value

    if save_path:
        with open(save_path, "w") as f:
            json.dump(reward_config, f, indent=2)
        print(f"[MUTATION] Saved mutated reward config to {save_path}")
