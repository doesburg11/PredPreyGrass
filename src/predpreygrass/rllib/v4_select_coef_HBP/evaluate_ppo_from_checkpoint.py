from predpreygrass.rllib.v4_select_coef_HBP.predpreygrass_rllib_env import PredPreyGrass  # Import the custom environment
from predpreygrass.rllib.v4_select_coef_HBP.config_env import config_env
from predpreygrass.utils.renderer import MatPlotLibRenderer, CombinedEvolutionVisualizer, PreyDeathCauseVisualizer

# external libraries
import ray
from ray.rllib.core.rl_module.rl_module import RLModule
from ray.tune.registry import register_env
import torch
from datetime import datetime
import os
import json

verbose_grid = False
verbose_actions = False
seed = None # 42 # Optional: set to integer for reproducibility

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

# === Set checkpoint paths ===
ray_results_dir = '/home/doesburg/Dropbox/02_marl_results/predpreygrass_results/ray_results'
checkpoint_root = '/PPO_2025-04-12_00-49-18/PPO_PredPreyGrass_33a72_00000_0_2025-04-12_00-49-18/'
checkpoint_dir = 'checkpoint_000099'
checkpoint_path = os.path.abspath(ray_results_dir + checkpoint_root+ checkpoint_dir)

# === Get training directory and prepare eval output dir ===
training_dir = os.path.dirname(os.path.dirname(checkpoint_path))
now = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
eval_output_dir = os.path.join(training_dir, f"eval_{checkpoint_dir}_{now}")
os.makedirs(eval_output_dir, exist_ok=True)

# === Save config_env.json ===
with open(os.path.join(eval_output_dir, "config_env.json"), "w") as f:
    json.dump(config_env, f, indent=4)

# Load RLModules directly from subfolders
module_paths = {
    pid: os.path.join(checkpoint_path, 'learner_group', 'learner', 'rl_module', pid)
    for pid in ["speed_1_predator", "speed_2_predator", "speed_1_prey", "speed_2_prey"]
}
rl_modules = {pid: RLModule.from_checkpoint(path) for pid, path in module_paths.items()}

# Initialize the environment
env = env_creator({}) # PredPreyGrass()

# Reset environment and get initial observations
obs, _ = env.reset(seed=seed)


# intitialize matplot lib renderer
grid_size = (env.grid_size, env.grid_size)
all_agents = env.possible_agents + env.grass_agents
grid_visualizer = MatPlotLibRenderer(
    grid_size, 
    all_agents, 
    trace_length=5, 
    show_gridlines=False, 
    scale=2,
    destination_path=None, # save to: eval_output_dir
)
combined_evolution_visualizer = CombinedEvolutionVisualizer(
    destination_path=eval_output_dir
)
prey_death_cause_visualizer = PreyDeathCauseVisualizer(
    destination_path=eval_output_dir
)


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

    # Step the environment with computed actions
    obs, rewards, terminations, truncations, _ = env.step(action_dict)
    combined_evolution_visualizer.record(
        agent_ids=env.agents,
        internal_ids=env.agent_internal_ids,
        agent_ages=env.agent_ages
    )
    prey_death_cause_visualizer.record(env.death_cause_prey)

    # Merge agent and grass positions for rendering
    merged_positions = {**env.agent_positions, **env.grass_positions}
    grid_visualizer.update(merged_positions, step)
    grid_visualizer.save_frame(step) 

    step += 1
    total_reward += sum(rewards.values())

    # Check if episode is done
    done = terminations.get("__all__", False) or truncations.get("__all__", False)
    #time.sleep(0.1)

print(f"Evaluation complete! Total Reward: {total_reward}")
# --- PREY DEATH CAUSE SUMMARY ---
death_log_path = os.path.join(eval_output_dir, "prey_death_causes.txt")
death_stats = {"eaten": 0, "starved": 0}

with open(death_log_path, "w") as f:
    f.write("--- Prey Death Causes ---\n")
    for internal_id, cause in env.death_cause_prey.items():
        f.write(f"Prey internal_id {internal_id:4d}: {cause}\n")
        if cause in death_stats:
            death_stats[cause] += 1
    f.write("\n--- Summary ---\n")
    f.write(f"Total prey eaten   : {death_stats['eaten']}\n")
    f.write(f"Total prey starved : {death_stats['starved']}\n")

print(f"Prey death summary written to: {death_log_path}")

# --- REWARD SUMMARY ---
reward_log_path = os.path.join(eval_output_dir, "reward_summary.txt")
with open(reward_log_path, "w") as f:
    f.write(f"Total Reward: {total_reward}\n")
    f.write("\n--- Reward Breakdown per Agent ---\n")
    speed_1_predator_rewards = []
    speed_2_predator_rewards = []
    speed_1_prey_rewards = []
    speed_2_prey_rewards = []
    for agent_id, reward in env.cumulative_rewards.items():
        f.write(f"{agent_id:15}: {reward:.2f}\n")
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

    f.write("\n--- Aggregated Rewards ---\n")
    f.write(f"Total number of steps            : {step-1}\n")
    f.write(f"Total Low-Speed Predator Reward  : {total_speed_1_predator_reward:.2f}\n")
    f.write(f"Total Low-Speed Prey Reward      : {total_speed_1_prey_reward:.2f}\n")
    f.write(f"Total Low-Speed Agent Reward     : {total_speed_1_predator_reward + total_speed_1_prey_reward:.2f}\n")
    f.write(f"Total High-Speed Predator Reward : {total_speed_2_predator_reward:.2f}\n")
    f.write(f"Total High-Speed Prey Reward     : {total_speed_2_prey_reward:.2f}\n")
    f.write(f"Total High-Speed Agent Reward    : {total_speed_2_predator_reward + total_speed_2_prey_reward:.2f}\n")


    print("\n--- Aggregated Rewards ---")
    print(f"Total number of steps            : {step-1}")
    print(f"Total Low-Speed Predator Reward  : {total_speed_1_predator_reward:.2f}")
    print(f"Total Low-Speed Prey Reward      : {total_speed_1_prey_reward:.2f}")
    print(f"Total Low-Speed Agent Reward     : {total_reward_all_speed_1:.2f}")
    print(f"Total High-Speed Predator Reward : {total_speed_2_predator_reward:.2f}")
    print(f"Total High-Speed Prey Reward     : {total_speed_2_prey_reward:.2f}")
    print(f"Total High-Speed Agent Reward    : {total_reward_all_speed_2:.2f}")


combined_evolution_visualizer.plot()
prey_death_cause_visualizer.plot()

# Shutdown Ray after evaluation
ray.shutdown()
