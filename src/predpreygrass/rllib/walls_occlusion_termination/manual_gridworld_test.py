import numpy as np
import matplotlib.pyplot as plt
from predpreygrass.rllib.walls_occlusion_termination.predpreygrass_rllib_env_termination import PredPreyGrass

# --- Manual test config ---
manual_config = {
    "debug_mode": True,
    "verbose_movement": True,
    "verbose_decay": True,
    "verbose_reproduction": True,
    "verbose_engagement": True,
    "max_steps": 20,
    "reward_predator_catch_prey": 10.0,
    "reward_prey_eat_grass": 2.0,
    "reward_predator_step": 0.0,
    "reward_prey_step": 0.0,
    "penalty_prey_caught": -5.0,
    "reproduction_reward_predator": 0.0,
    "reproduction_reward_prey": 0.0,
    "energy_loss_per_step_predator": 0.1,
    "energy_loss_per_step_prey": 0.05,
    "predator_creation_energy_threshold": 5.0,
    "prey_creation_energy_threshold": 3.0,
    "move_energy_cost_factor": 0.0,
    "move_energy_cost_predator": 0.0,
    "move_energy_cost_prey": 0.0,
    "n_possible_type_1_predators": 2,
    "n_possible_type_2_predators": 0,
    "n_possible_type_1_prey": 2,
    "n_possible_type_2_prey": 0,
    "n_initial_active_type_1_predator": 1,
    "n_initial_active_type_2_predator": 0,
    "n_initial_active_type_1_prey": 1,
    "n_initial_active_type_2_prey": 0,
    "initial_energy_predator": 10.0,
    "initial_energy_prey": 5.0,
    "grid_size": 5,
    "num_obs_channels": 4,
    "predator_obs_range": 3,
    "prey_obs_range": 3,
    "include_visibility_channel": False,
    "respect_los_for_movement": False,
    "mask_observation_with_visibility": False,
    "initial_num_grass": 1,
    "initial_energy_grass": 3.0,
    "energy_gain_per_step_grass": 0.0,
    "max_energy_grass": 3.0,
    "manual_wall_positions": [(2,2)],
    "wall_positions": set(),
    "mutation_rate_predator": 0.0,
    "mutation_rate_prey": 0.0,
    "type_1_action_range": 3,
    "type_2_action_range": 3,
    "seed": 42,
    "reproduction_chance_predator": 0.0,
    "reproduction_chance_prey": 0.0,
    "reproduction_cooldown_steps": 100,
    "max_energy_gain_per_prey": 10.0,
    "energy_transfer_efficiency": 1.0,
    "max_energy_predator": 20.0,
}

def plot_grid(env):
    grid = np.zeros((env.grid_size, env.grid_size, 3))
    # Walls: red
    for (x, y) in env.wall_positions:
        grid[x, y, 0] = 1.0
    # Predators: blue
    for agent, pos in env.predator_positions.items():
        grid[pos[0], pos[1], 2] = 1.0
    # Prey: green
    for agent, pos in env.prey_positions.items():
        grid[pos[0], pos[1], 1] = 1.0
    # Grass: yellow
    for grass, pos in env.grass_positions.items():
        grid[pos[0], pos[1], 0] = 1.0
        grid[pos[0], pos[1], 1] = 1.0
    plt.imshow(grid)
    plt.title(f"Step {env.current_step}")
    plt.show()

def main():
    env = PredPreyGrass(manual_config)
    obs, _ = env.reset()
    plot_grid(env)
    # Example: manually step the environment
    # Move predator right, prey down
    actions = {
        list(env.predator_positions.keys())[0]: 5,  # action index for right
        list(env.prey_positions.keys())[0]: 7,      # action index for down
    }
    for i in range(5):
        obs, rewards, term, trunc, info = env.step(actions)
        plot_grid(env)
        print(f"Step {i+1} rewards: {rewards}")
        if term.get("__all__", False):
            print("Episode ended.")
            break

if __name__ == "__main__":
    main()
