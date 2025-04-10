from predpreygrass.rllib.v4_select_coef_HBP.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.utils.renderer import MatPlotLibRenderer, CombinedEvolutionVisualizer

# external libraries
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
import torch
import time
import os

verbose_grid = False
verbose_actions = False
seed = None # 42 # for random intialization of the environment

# Initialize Ray
ray.init(ignore_reinit_error=True)

def env_creator(config):
    return PredPreyGrass(config)

register_env("PredPreyGrass", lambda config: env_creator(config))

# Policy mapping function
def policy_mapping_fn(agent_id, *args, **kwargs):
    if "speed_1_predator" in agent_id:
        return "speed_1_predator"
    elif "speed_2_predator" in agent_id:
        return "speed_2_predator"
    elif "speed_1_prey" in agent_id:
        return "speed_1_prey"
    elif "speed_2_prey" in agent_id:
        return "speed_2_prey"
    else:
        return None
checkpoint_root = '/home/doesburg/Dropbox/02_marl_results/predpreygrass_results/ray_results/'

chechpoint_file = 'PPO_PredPreyGrass_d39e3_00000_0_2025-04-08_23-31-26/checkpoint_000039'
checkpoint_path = f"file://{os.path.abspath(checkpoint_root+chechpoint_file)}"

checkpoint_path = "/home/doesburg/Dropbox/02_marl_results/predpreygrass_results/ray_results/400/PPO_PredPreyGrass_d39e3_00000_0_2025-04-08_23-31-26/checkpoint_000039"
# Load RLlib Algorithm from checkpoint
trained_algo = Algorithm.from_checkpoint(checkpoint_path)
print("Checkpoint loaded successfully!")

rl_modules = trained_algo.config.get("rl_module_spec").build()

# Initialize the environment
env = env_creator({}) # PredPreyGrass()

# Reset environment and get initial observations
obs, _ = env.reset(seed=seed)


# intitialize matplot lib renderer
grid_size = (env.grid_size, env.grid_size)
all_agents = env.possible_agents + env.grass_agents
grid_visualizer = MatPlotLibRenderer(grid_size, all_agents, trace_length=5, show_gridlines=False, scale=2)
combined_evolution_visualizer = CombinedEvolutionVisualizer(destination_path=checkpoint_path)

step = 0
done = False
total_reward = 0

while not done:
    action_dict = {}

    for agent_id in env.agents:
        policy_id = policy_mapping_fn(agent_id) # Determine policy for each agent
        # Get the RLModule (policy model) from the Learner Group
        policy_module = rl_modules[policy_id]

        obs_tensor = torch.tensor(obs[agent_id]).float().unsqueeze(0)
        with torch.no_grad():
            action_out = policy_module._forward_inference({"obs": obs_tensor})
        action = torch.argmax(action_out["action_dist_inputs"], dim=-1).item()
        action_dict[agent_id] = action

    if verbose_actions:
        print(f"[Step {step}] Actions: {action_dict}")

    obs, rewards, terminations, truncations, _ = env.step(action_dict)
    combined_evolution_visualizer.record(
        agent_ids=env.agents,
        internal_ids=env.agent_internal_ids,
        agent_ages=env.agent_ages
    )

    if verbose_grid:
        print(f"--- Step {step} ---")
        env._print_grid_from_positions()
        env._print_grid_from_state()

    merged_positions = {**env.agent_positions, **env.grass_positions}
    grid_visualizer.update(merged_positions, step)

    step += 1
    total_reward += sum(rewards.values())
    done = terminations.get("__all__", False) or truncations.get("__all__", False)

print(f"\n✅ Evaluation complete! Total reward: {total_reward:.2f}")

# === REWARD SUMMARY ===
reward_sums = {
    "speed_1_predator": 0,
    "speed_2_predator": 0,
    "speed_1_prey": 0,
    "speed_2_prey": 0,
}

print("\n--- Reward per Agent ---")
for agent_id, reward in env.cumulative_rewards.items():
    print(f"{agent_id:20} → {reward:.2f}")
    for k in reward_sums:
        if k in agent_id:
            reward_sums[k] += reward

print("\n--- Aggregated Rewards ---")
print(f"Total steps: {step - 1}")
for k, v in reward_sums.items():
    print(f"{k:25}: {v:.2f}")

combined_evolution_visualizer.plot()
ray.shutdown()
