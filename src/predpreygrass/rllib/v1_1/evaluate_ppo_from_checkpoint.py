# discretionary libraries
from predpreygrass.rllib.v1_1.predpreygrass_rllib_env import PredPreyGrass  # Import the custom environment
from predpreygrass.utils.pygame_renderer import PyGameRenderer, ViewerControlHelper, LoopControlHelper

# external libraries
import ray
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
import torch
import os
import matplotlib.pyplot as plt
import pygame

verbose_grid = False
verbose_actions = False
seed = 42  # for random initialization of the environment

# Initialize Ray
ray.init(ignore_reinit_error=True)


# Define environment registration
def env_creator(config):
    return PredPreyGrass(config)


# Policy mapping function
def policy_mapping_fn(agent_id, *args, **kwargs):
    if "predator" in agent_id:
        return "predator_policy"
    elif "prey" in agent_id:
        return "prey_policy"
    return None


register_env("PredPreyGrass", lambda config: env_creator(config))

# Load trained model from checkpoint
root = "./src/predpreygrass/rllib/"
path = "v1/trained_policy/PPO_2025-06-06_21-34-53/PPO_PredPreyGrass_52139_00000_0_2025-06-06_21-34-53/checkpoint_000030"
checkpoint_path = f"file://{os.path.abspath(root+path)}"
trained_algo = Algorithm.from_checkpoint(checkpoint_path)
print("Checkpoint loaded successfully!")

# Access RLModules from learner_group
rl_modules = trained_algo.learner_group._learner.module  # Retrieves policy modules

# Initialize the environment
env = env_creator({})

# Reset environment and get initial observations
obs, _ = env.reset(seed=seed)

# Initialize PyGameRenderer
grid_size = (env.grid_size, env.grid_size)
visualizer = PyGameRenderer(grid_size)

# Initialize viewer control + loop helper
control = ViewerControlHelper()
loop_helper = LoopControlHelper()

# Optional: frame rate control
clock = pygame.time.Clock()
target_fps = 10  # Adjust as desired

total_reward = 0
predator_counts = []
prey_counts = []
time_steps = []

# --- Setup snapshots for stepping backwards ---
snapshots = []
max_snapshots = 100  # Keep last 100 steps
# Save initial snapshot
snapshots.append(env.get_state_snapshot())

# Run one evaluation episode
while not loop_helper.simulation_terminated:
    control.handle_events()

    # --- Backward step handling ---
    if control.step_backward:
        if len(snapshots) > 1:
            snapshots.pop()  # Discard current step
            env.restore_state_snapshot(snapshots[-1])
            print(f"[ViewerControl] Step Backward → Step {env.current_step}")

            # --- REGENERATE obs to match restored state ---
            obs = {agent: env._get_observation(agent) for agent in env.agents}

            # --- Also rewind history lists ---
            if len(time_steps) > 0:
                time_steps.pop()
                predator_counts.pop()
                prey_counts.pop()

            visualizer.update(
                agent_positions=env.agent_positions,
                grass_positions=env.grass_positions,
                agent_energies=env.agent_energies,
                grass_energies=env.grass_energies,
                step=env.current_step,
                agents_just_ate=env.agents_just_ate
            )
            pygame.time.wait(100)
        control.step_backward = False

    # Normal step forward
    if loop_helper.should_step(control):
        # Build action dict from PPO policy
        action_dict = {}
        for agent_id in env.agents:
            policy_id = policy_mapping_fn(agent_id)
            policy_module = rl_modules[policy_id]

            obs_tensor = torch.tensor(obs[agent_id]).float().unsqueeze(0)
            with torch.no_grad():
                action_output = policy_module._forward_inference({"obs": obs_tensor})

            if "action_dist_inputs" in action_output:
                action = torch.argmax(action_output["action_dist_inputs"], dim=-1).item()
            else:
                raise KeyError(f"Unexpected output structure: {action_output}")

            action_dict[agent_id] = action

        # Step env
        obs, rewards, terminations, truncations, _ = env.step(action_dict)

        # Save snapshot AFTER step
        snapshots.append(env.get_state_snapshot())
        if len(snapshots) > max_snapshots:
            snapshots.pop(0)

        # Update viewer
        visualizer.update(
            agent_positions=env.agent_positions,
            grass_positions=env.grass_positions,
            agent_energies=env.agent_energies,
            grass_energies=env.grass_energies,
            step=env.current_step,
            agents_just_ate=env.agents_just_ate
        )

        # Update loop control termination flag
        loop_helper.update_simulation_terminated(terminations, truncations)

        # Reset step_once
        control.step_once = False

        # Frame rate control
        clock.tick(target_fps)

        # Track stats
        num_predators = sum(1 for agent in env.agents if "predator" in agent)
        num_prey = sum(1 for agent in env.agents if "prey" in agent)

        time_steps.append(env.current_step)
        predator_counts.append(num_predators)
        prey_counts.append(num_prey)
        total_reward += sum(rewards.values())

    else:
        # While paused → update viewer so tooltips work
        visualizer.update(
            agent_positions=env.agent_positions,
            grass_positions=env.grass_positions,
            agent_energies=env.agent_energies,
            grass_energies=env.grass_energies,
            step=env.current_step,
            agents_just_ate=env.agents_just_ate
        )
        pygame.time.wait(50)

# --- End of main loop ---

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
print(f"Total number of steps: {env.current_step-1}")
print(f"Total Predator Reward: {total_predator_reward:.2f}")
print(f"Total Prey Reward:     {total_prey_reward:.2f}")
print(f"Total All-Agent Reward:{total_reward_all:.2f}")

# --- Plot ---
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

# Shutdown
ray.shutdown()
visualizer.close()
