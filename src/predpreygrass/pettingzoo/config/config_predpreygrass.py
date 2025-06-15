from predpreygrass.global_config import RESULTS_DIR

local_output_root = RESULTS_DIR

x_grid_size, y_grid_size = 25, 25
training_steps_string = "1_638_400"
# training_steps_string="11_468_800"

env_kwargs = dict(
    # reward parameters
    step_reward_predator=0.0,
    step_reward_prey=0.0,
    step_reward_grass=0.0,
    catch_reward_grass=0.0,
    catch_reward_prey=0.0,
    death_reward_prey=0.0,
    death_reward_predator=0.0,
    reproduction_reward_prey=10.0,
    reproduction_reward_predator=10.0,
    # agent parameters
    n_possible_predator=40,  # 60,  # maximum number of predators during runtime
    n_possible_prey=60,  # #80,
    n_possible_grass=25,
    n_initial_active_predator=6,
    n_initial_active_prey=8,
    # observation parameters
    max_observation_range=9,  # must be odd and not smaller than any obs_range
    obs_range_predator=7,
    obs_range_prey=9,
    # action parameters
    is_von_neumann_neighborhood=True,
    action_range=3,  # obsolete when is_von_neumann_neighborhood is True: action_range automatically set to 3
    # energy parameters
    energy_gain_per_step_predator=-0.15,  # -0.15,  # -0.15 # default
    energy_gain_per_step_prey=-0.05,  # -0.05,  # -0.05 # default
    energy_gain_per_step_grass=0.2,
    initial_energy_predator=5.0,
    initial_energy_prey=5.0,
    initial_energy_grass=3.0,
    max_energy_level_grass=4.0,
    motion_energy_per_distance_unit=-0.00,  # -0.01
    # create agents parameters
    prey_creation_energy_threshold=8,
    predator_creation_energy_threshold=12,
    # visualization parameters
    cell_scale=40,  # 40
    x_pygame_window=0,
    y_pygame_window=0,
    has_energy_chart=True,
    # evaluation parameters
    num_episodes=10,
    watch_grid_model=True,
    write_evaluation_to_file=True,
    # training parameters
    max_cycles=10000,
    # environment parameters
    x_grid_size=x_grid_size,
    y_grid_size=y_grid_size,
    is_torus=True,
    spawning_area_predator=dict(
        {
            "x_begin": 0,
            "y_begin": 0,
            "x_end": x_grid_size - 1,
            "y_end": y_grid_size - 1,
        }
    ),
    spawning_area_prey=dict(
        {
            "x_begin": 0,
            "y_begin": 0,
            "x_end": x_grid_size - 1,
            "y_end": y_grid_size - 1,
        }
    ),
    spawning_area_grass=dict(
        {
            "x_begin": 0,
            "y_begin": 0,
            "x_end": x_grid_size - 1,
            "y_end": y_grid_size - 1,
        }
    ),
)
