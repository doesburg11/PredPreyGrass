# Script to verify kin survival rewards for prey agents
# Usage: Run after an evaluation to analyze agent_fitness_stats.json
import json

# Path to agent_fitness_stats.json (update as needed)
fitness_stats_path = "agent_fitness_stats.json"

with open(fitness_stats_path, "r") as f:
    stats = json.load(f)

kin_reward = 0.1  # Should match your env config

print("Agent ID           | Offspring | Kin Reward | Cumulative Reward | Match? | Offspring IDs")
print("-"*90)
for agent_id, record in stats.items():
    if "prey" not in agent_id:
        continue
    offspring_ids = record.get("offspring_ids", [])
    # Kin reward is kin_reward * number of steps each living offspring survived while parent was alive
    # This script checks if cumulative_reward matches expected kin reward (within float tolerance)
    expected_kin_reward = 0.0
    # For each offspring, estimate their lifetime overlap with parent
    for child_id in offspring_ids:
        child = stats.get(child_id)
        if not child:
            continue
        # Parent must be alive for kin reward to be given
        parent_birth = record.get("birth_step", 0)
        parent_death = record.get("death_step")
        child_birth = child.get("birth_step", 0)
        child_death = child.get("death_step")
        # Overlap: from child's birth to min(child's death, parent's death)
        overlap_start = max(parent_birth, child_birth)
        overlap_end = min(child_death or 1e9, parent_death or 1e9)
        overlap_steps = max(0, overlap_end - overlap_start)
        expected_kin_reward += kin_reward * overlap_steps
    actual = record.get("cumulative_reward", 0.0)
    match = abs(actual - expected_kin_reward) < 1e-3
    print(f"{agent_id:18} | {len(offspring_ids):9} | {expected_kin_reward:9.2f} | {actual:16.2f} | {str(match):5} | {offspring_ids}")
