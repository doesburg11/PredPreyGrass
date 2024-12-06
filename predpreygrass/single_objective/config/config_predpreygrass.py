import os.path as osp
from predpreygrass.global_config import RESULTS_DIR

local_output_root = RESULTS_DIR

x_grid_size, y_grid_size = 25, 25

env_kwargs = dict(
    # reward parameters
    step_reward_predator=0.0,
    step_reward_prey=0.0,
    step_reward_grass=0.0,
    catch_reward_grass=0.0,
    catch_reward_prey=0.0,
    death_reward_prey=-15.0,
    death_reward_predator=0.0,
    reproduction_reward_prey=10.0,
    reproduction_reward_predator=10.0,
    # agent parameters
    n_possible_predator=40,  # maximum number of predators during runtime
    n_possible_prey=60,
    n_possible_grass=25,
    n_initial_active_predator=6,
    n_initial_active_prey=8,
    # observation parameters
    max_observation_range=9,  # must be odd and not smaller than any obs_range
    obs_range_predator=7,
    obs_range_prey=9,
    # energy parameters
    energy_gain_per_step_predator=-0.15,  # -0.15 # default
    energy_gain_per_step_prey=-0.05,  # -0.05 # default
    energy_gain_per_step_grass=0.2,
    initial_energy_predator=5.0,
    initial_energy_prey=5.0,
    initial_energy_grass=3.0,
    max_energy_level_grass=4.0,
    # create agents parameters
    create_prey=True,
    create_predator=True,
    regrow_grass=True,
    prey_creation_energy_threshold=8,
    predator_creation_energy_threshold=12,
    # visualization parameters
    cell_scale=40,
    x_pygame_window=0,
    y_pygame_window=0,
    show_energy_chart=True,
    # evaluation parameters
    num_episodes=100,
    watch_grid_model=False,
    # training parameters
    max_cycles=5000,
    #training_steps_string="688_128",
    training_steps_string="1_376_256",
    #training_steps_string="2064384",
    #training_steps_string="2752512",
    #training_steps_string="3440640",
    #training_steps_string="4128768",
    #training_steps_string="4816896",
    #training_steps_string="5505024",
    #training_steps_string="6193152",
    #training_steps_string="6881280",
    #training_steps_string="7569408",
    #training_steps_string="8257536",
    #training_steps_string="8945664",
    #training_steps_string="10321920",
    #training_steps_string="10_000_000",
    # environment parameters
    x_grid_size=x_grid_size,
    y_grid_size=y_grid_size,
    torus=True,
    is_parallel_environment=True,
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
