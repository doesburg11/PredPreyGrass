from predpreygrass.rllib.v4_select_coef_HBP.predpreygrass_rllib_env import PredPreyGrass  # Import the custom environment
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
    
checkpoint_root = '/home/doesburg/Dropbox/02_marl_results/predpreygrass_results/ray_results'
#checkpoint_root = '/home/doesburg/ray_results'
# /home/doesburg/Dropbox/02_marl_results/predpreygrass_results/ray_results/400/PPO_PredPreyGrass_d39e3_00000_0_2025-04-08_23-31-26/checkpoint_000039
chechpoint_file = '/PPO_2025-04-11_17-15-57/PPO_PredPreyGrass_de94b_00000_0_2025-04-11_17-15-57/checkpoint_000000'
# /home/doesburg/Dropbox/02_marl_results/predpreygrass_results/ray_results/PPO_2025-04-11_17-15-57/PPO_PredPreyGrass_de94b_00000_0_2025-04-11_17-15-57/checkpoint_000001
checkpoint_path = f"file://{os.path.abspath(checkpoint_root+chechpoint_file)}"
# Load RLlib Algorithm from checkpoint
trained_algo = Algorithm.from_checkpoint(checkpoint_path)
print("Checkpoint loaded successfully!")


# Access RLModules from learner_group
rl_modules = trained_algo.learner_group._learner.module

# Initialize the environment
env = env_creator({}) # PredPreyGrass()

# Reset environment and get initial observations
obs, _ = env.reset(seed=seed)


# intitialize matplot lib renderer
grid_size = (env.grid_size, env.grid_size)
all_agents = env.possible_agents + env.grass_agents
grid_visualizer = MatPlotLibRenderer(grid_size, all_agents, trace_length=5, show_gridlines=False, scale=2)
combined_evolution_visualizer = CombinedEvolutionVisualizer(destination_path=checkpoint_root)


step=0
done = False
total_reward = 0

# Run one evaluation episode
while not done:
    action_dict = {}

    for agent_id in env.agents:
        policy_id = policy_mapping_fn(agent_id)  # Determine policy for each agent
        # Get the RLModule (policy model) from the Learner Group
        policy_module = rl_modules[policy_id]
        # Convert observation to tensor format required for _forward_inference()
        obs_tensor = torch.tensor(obs[agent_id]).float().unsqueeze(0)  # Convert obs to tensor
        # Use _forward_inference() to compute the next action
        with torch.no_grad():
            action_output = policy_module._forward_inference({"obs": obs_tensor})
        # Extract the action correctly
        if "action_dist_inputs" in action_output:
            action = torch.argmax(action_output["action_dist_inputs"], dim=-1).item()
        else:
            raise KeyError(f"Unexpected output structure: {action_output}")
        # Store the computed action
        action_dict[agent_id] = action
    if verbose_actions:
        print("----------------------------------------------------------------------------------")
        print("Step:", step)
        print("----------------------------------------------------------------------------------")
        print("Actions:", action_dict)
        print("----------------------------------------------------------------------------------")

    # Step the environment with computed actions
    obs, rewards, terminations, truncations, _ = env.step(action_dict)
    combined_evolution_visualizer.record(
        agent_ids=env.agents,
        internal_ids=env.agent_internal_ids,
        agent_ages=env.agent_ages
    )

    if verbose_grid:
        print(f"Step {step}:")
        print("-----------------------------------------")
        env._print_grid_from_positions()
        env._print_grid_from_state()
        print("-----------------------------------------")

    # Merge agent and grass positions for rendering
    merged_positions = {**env.agent_positions, **env.grass_positions}
    grid_visualizer.update(merged_positions, step)
    step += 1
    total_reward += sum(rewards.values())

    # Check if episode is done
    done = terminations.get("__all__", False) or truncations.get("__all__", False)
    #time.sleep(0.1)

print(f"Evaluation complete! Total Reward: {total_reward}")
# --- REWARD SUMMARY ---
speed_1_predator_rewards = []
speed_1_prey_rewards = []
speed_2_predator_rewards = []
speed_2_prey_rewards = []


print("\n--- Reward Breakdown per Agent ---")
for agent_id, reward in env.cumulative_rewards.items():
    print(f"{agent_id:15}: {reward:.2f}")
    if "speed_1_predator" in agent_id:
        speed_1_predator_rewards.append(reward)
    elif "speed_2_predator" in agent_id:
        speed_2_predator_rewards.append(reward)
    elif "speed_1_prey" in agent_id:
        speed_1_prey_rewards.append(reward)
    elif "speed_2_prey" in agent_id:
        speed_2_prey_rewards.append(reward)


total_speed_1_predator_reward = sum(speed_1_predator_rewards)
total_speed_1_prey_reward = sum(speed_1_prey_rewards)
total_reward_all_speed_1 = total_speed_1_predator_reward + total_speed_1_prey_reward
total_speed_2_predator_reward = sum(speed_2_predator_rewards)
total_speed_2_prey_reward = sum(speed_2_prey_rewards)
total_reward_all_speed_2 = total_speed_2_predator_reward + total_speed_2_prey_reward


print("\n--- Aggregated Rewards ---")
print(f"Total number of steps            : {step-1}")
print(f"Total Low-Speed Predator Reward  : {total_speed_1_predator_reward:.2f}")
print(f"Total Low-Speed Prey Reward      : {total_speed_1_prey_reward:.2f}")
print(f"Total Low-Speed Agent Reward     : {total_reward_all_speed_1:.2f}")
print(f"Total High-Speed Predator Reward : {total_speed_2_predator_reward:.2f}")
print(f"Total High-Speed Prey Reward     : {total_speed_2_prey_reward:.2f}")
print(f"Total High-Speed Agent Reward    : {total_reward_all_speed_2:.2f}")


combined_evolution_visualizer.plot()
# Shutdown Ray after evaluation
ray.shutdown()
