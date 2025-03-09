import ray
import torch
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from predpreygrass_15 import PredPreyGrass  # Import the custom environment
from predpreygrass.single_objective.utils.renderer import MatPlotLibRenderer
import time


verbose_grid = False
verbose_actions = True
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
checkpoint_path = "/home/doesburg/ray_results/PPO_2025-03-09_00-10-13/PPO_PredPreyGrass_7d8fb_00000_0_2025-03-09_00-10-13/checkpoint_000009"  # Update as needed

# Load RLlib Algorithm from checkpoint
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

    # Sum rewards
    total_reward += sum(rewards.values())

    # Check if episode is done
    done = terminations.get("__all__", False) or truncations.get("__all__", False)
    time.sleep(0.1)

    # Print active agents after step
    #print(f"Active Agents After Step: {env.agents}")  # Debugging

print(f"Evaluation complete! Total Reward: {total_reward}")

# Shutdown Ray after evaluation
ray.shutdown()
