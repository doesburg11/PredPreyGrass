# discretionary libararies
from predpreygrass.rllib.v1_moore.predpreygrass_rllib_env import PredPreyGrass  # Import the custom environment
from predpreygrass.utils.renderer import MatPlotLibRenderer

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

# Define environment registration
def env_creator(config):
    return PredPreyGrass(config)

register_env("PredPreyGrass", lambda config: env_creator(config))

# Policy mapping function
def policy_mapping_fn(agent_id, *args, **kwargs):
    if "predator" in agent_id:
        return "predator_policy"
    elif "prey" in agent_id:
        return "prey_policy"
    return None

# Load trained model from checkpoint
checkpoint_path = f"file://{os.path.abspath('./predpreygrass/rllib/v1/trained_model/PPO_PredPreyGrass_437b7_00000_0_2025-03-24_19-17-36/checkpoint_000029')}"
trained_algo = Algorithm.from_checkpoint(checkpoint_path)
print("Checkpoint loaded successfully!")


# Access RLModules from learner_group
rl_modules = trained_algo.learner_group._learner.module  # Retrieves policy modules

# Initialize the environment
env = env_creator({}) # PredPreyGrass()

# Reset environment and get initial observations
obs, _ = env.reset(seed=seed)


# intitialize matplot lib renderer
grid_size = (env.grid_size, env.grid_size)
all_agents = env.possible_agents + env.grass_agents
visualizer = MatPlotLibRenderer(grid_size, all_agents, trace_length=5)
step=0

done = False
total_reward = 0
predator_counts = []
prey_counts = []
time_steps = []

# Run one evaluation episode
while not done:
    action_dict = {}

    for agent_id in env.agents:
        policy_id = policy_mapping_fn(agent_id)  # Determine policy for each agent
        # Get the RLModule (policy model) from the Learner Group
        policy_module = rl_modules[policy_id]
        # Convert observation to tensor format required for _forward_inference()
        obs_tensor = torch.tensor(obs[agent_id]).float().unsqueeze(0)  # Add batch dimension
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
        print("Step:",step)
        print("----------------------------------------------------------------------------------")
        print("Actions:", action_dict)
        print("----------------------------------------------------------------------------------")

    # Step the environment with computed actions
    obs, rewards, terminations, truncations, _ = env.step(action_dict)
    if verbose_grid:
        print(f"Step {step}:")
        print("-----------------------------------------")
        #print(f"Actions: {action_dict}")
        env._print_grid_from_positions()
        env._print_grid_from_state()
        print("-----------------------------------------")

    # Print termination status for debugging
    #print(f"Terminations: {terminations}")
    merged_positions = {**env.agent_positions, **env.grass_positions}
    visualizer.update(merged_positions, step)
    step+=1
    # Count current number of agents
    num_predators = sum(1 for agent in env.agents if "predator" in agent)
    num_prey = sum(1 for agent in env.agents if "prey" in agent)

    # Store counts
    time_steps.append(step)
    predator_counts.append(num_predators)
    prey_counts.append(num_prey)

    # Sum rewards
    total_reward += sum(rewards.values())

    # Check if episode is done
    done = terminations.get("__all__", False) or truncations.get("__all__", False)
    time.sleep(0.1)

    # Print active agents after step
    #print(f"Active Agents After Step: {env.agents}")  # Debugging

print(f"Evaluation complete! Total Reward: {total_reward}")
# --- REWARD SUMMARY ---
predator_rewards = []
prey_rewards = []

print("\n--- Reward Breakdown per Agent ---")
for agent_id, reward in env.cumulative_rewards.items():
    print(f"{agent_id:15}: {reward:.2f}")
    if "predator" in agent_id:
        predator_rewards.append(reward)
    elif "prey" in agent_id:
        prey_rewards.append(reward)

total_predator_reward = sum(predator_rewards)
total_prey_reward = sum(prey_rewards)
total_reward_all = total_predator_reward + total_prey_reward

print("\n--- Aggregated Rewards ---")
print(f"Total number of steps: {step-1}")
print(f"Total Predator Reward: {total_predator_reward:.2f}")
print(f"Total Prey Reward:     {total_prey_reward:.2f}")
print(f"Total All-Agent Reward:{total_reward_all:.2f}")

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(time_steps, predator_counts, label='Predators', color='red')
plt.plot(time_steps, prey_counts, label='Prey', color='blue')
plt.xlabel('Time Step')
plt.ylabel('Number of Agents')
plt.title('Agent Population Over Time')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Shutdown Ray after evaluation
ray.shutdown()
