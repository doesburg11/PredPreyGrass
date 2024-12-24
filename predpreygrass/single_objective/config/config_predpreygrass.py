from predpreygrass.global_config import RESULTS_DIR
import numpy as np

local_output_root = RESULTS_DIR

x_grid_size, y_grid_size = 25, 25
training_steps_string= "10_000_000"
#training_steps_string="2_293_760"
#training_steps_string="4_587_520"
#training_steps_string="6_881_280"
#training_steps_string="9_175_040"
#training_steps_string="11_468_800"
#training_steps_string="13_762_560"
#training_steps_string="16_056_320"
#training_steps_string="18_350_080"
#training_steps_string="20_643_840"


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
    n_possible_predator=18, #60,  # maximum number of predators during runtime
    n_possible_prey=24, # #80,
    n_possible_grass=25,
    n_initial_active_predator=6,
    n_initial_active_prey=8,
    # observation parameters
    max_observation_range=9,  # must be odd and not smaller than any obs_range
    obs_range_predator=7,
    obs_range_prey=9,
    # action parameters
    motion_range = [
        #[-2,-2],  # move left left up up
        #[-2,-1],  # move left left up
        #[-2, 0],  # move left left
        #[-2, 1],  # move left left down
        #[-2, 2],  # move left left down down 
        #[-1,-2],  # move left up up
        [-1,-1],  # move left up
        [-1, 0],  # move left
        [-1, 1],  # move left down
        #[-1, 2],  # move left down down
        #[ 0,-2],  # move up up
        [ 0,-1],  # move up
        [ 0, 0],  # stay
        [ 0, 1],  # move down
        #[ 0, 2],  # move down down
        #[ 1,-2],  # move right up up
        [ 1,-1],  # move right up
        [ 1, 0],  # move right
        [ 1, 1],  # move right down
        #[ 1, 2],  # move right down down
        #[ 2,-2],  # move right right up up
        #[ 2,-1],  # move right right up
        #[ 2, 0],  # move right right
        #[ 2, 1],  # move right right down
        #[ 2, 2],  # move rihgt right down down
    ],
    # energy parameters
    energy_gain_per_step_predator= -0.11, #-0.15,  # -0.15 # default
    energy_gain_per_step_prey= -0.05, #-0.05,  # -0.05 # default
    energy_gain_per_step_grass=0.2,
    initial_energy_predator=5.0,
    initial_energy_prey=5.0,
    initial_energy_grass=3.0,
    max_energy_level_grass=4.0,
    has_motion_energy = True,
    motion_energy_per_distance_unit = -0.01, #-0.0001,
    # create agents parameters
    prey_creation_energy_threshold=8,
    predator_creation_energy_threshold=12,
    # visualization parameters
    cell_scale=40, # 40
    x_pygame_window=0,
    y_pygame_window=0,
    has_energy_chart=True,
    # evaluation parameters
    num_episodes=100,
    watch_grid_model=False,
    write_evaluation_to_file=True,
    # training parameters
    max_cycles=10000,
    # environment parameters
    x_grid_size=x_grid_size,
    y_grid_size=y_grid_size,
    is_torus=False,
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
