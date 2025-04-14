from predpreygrass.rllib.v4_gpu.predpreygrass_rllib_env import PredPreyGrass  # Import the custom environment
from predpreygrass.utils.renderer import MatPlotLibRenderer

# external libraries
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
import torch
import time
import cv2
import os


verbose_grid = False
verbose_actions = False
seed = None  # 42 

# Initialize Ray
ray.init(ignore_reinit_error=True)

# Define environment registration
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
    
#checkpoint_root = '/home/doesburg/ray_results/'
checkpoint_root = '/home/doesburg/Dropbox/02_marl_results/predpreygrass_results/rllib/'
chechpoint_file = 'PPO_2025-04-04_17-20-08/PPO_PredPreyGrass_4b829_00000_0_2025-04-04_17-20-08/checkpoint_000024'
checkpoint_path = f"file://{os.path.abspath(checkpoint_root+chechpoint_file)}"
# Load RLlib Algorithm from checkpoint
trained_algo = Algorithm.from_checkpoint(checkpoint_path)
print("Checkpoint loaded successfully!")


# Access RLModules from learner_group
rl_modules = trained_algo.learner_group._learner.module  # Retrieves policy modules

# Initialize the environment
env = env_creator({})

# Reset environment and get initial observations
obs, _ = env.reset(seed=seed)

# Initialize MatPlotLib Renderer
grid_size = (env.grid_size, env.grid_size)
all_agents = env.possible_agents + env.grass_agents
grid_visualizer = MatPlotLibRenderer(grid_size, all_agents, trace_length=5, show_gridlines=False, scale=2)

# Directory for saving frames
frame_dir = "simulation_frames"
os.makedirs(frame_dir, exist_ok=True)
frame_paths = []

step = 0
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

    if verbose_grid:
        print(f"Step {step}:")
        print("-----------------------------------------")
        env._print_grid_from_positions()
        env._print_grid_from_state()
        print("-----------------------------------------")

    # Merge agent and grass positions for rendering
    merged_positions = {**env.agent_positions, **env.grass_positions}
    grid_visualizer.update(merged_positions, step)

    # Save frame from Matplotlib Renderer
    frame_filename = os.path.join(frame_dir, f"frame_{step:04d}.png")
    grid_visualizer.fig.savefig(frame_filename, bbox_inches='tight')
    frame_paths.append(frame_filename)

    step += 1
    total_reward += sum(rewards.values())

    # Check if episode is done
    done = terminations.get("__all__", False) or truncations.get("__all__", False)
    #time.sleep(0.1)

print(f"Evaluation complete! Total Reward: {total_reward}")

# Convert saved images into a video
video_path = "simulation.mp4"

frame_array = []
for frame_file in frame_paths:
    img = cv2.imread(frame_file)
    if img is not None:
        frame_array.append(img)

if frame_array:
    height, width, layers = frame_array[0].shape
    out = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), 10, (width, height))
    for frame in frame_array:
        out.write(frame)
    out.release()
    print(f"Video saved as {video_path}")
else:
    print("No frames captured, video not saved.")

# Shutdown Ray after evaluation
ray.shutdown()
