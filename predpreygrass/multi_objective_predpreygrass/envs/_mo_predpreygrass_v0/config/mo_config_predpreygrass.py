# put in here your own directory to the output folder
local_output_root = "/home/doesburg/Dropbox/02_marl_results/predpreygrass_results/"

training_steps_string = "100_000"

x_grid_size = 25
y_grid_size = 25

env_kwargs = dict(
    max_cycles=10000,
    # reward parameters
    reproduction_reward_prey=10.0,
    reproduction_reward_predator=10.0,
    # environment parameters
    is_parallel_wrapped=True,
    x_grid_size=x_grid_size,
    y_grid_size=y_grid_size,
    # agent parameters
    n_possible_predator=18,  # maximum number of predators during runtime
    n_possible_prey=24,
    n_possible_grass=30,  
    n_initial_active_predator=6,
    n_initial_active_prey=8,
    # observation parameters
    max_observation_range=9,  # must be odd and not smaller than any obs_range
    obs_range_predator=7,
    obs_range_prey=9,
    # energy parameters
    energy_gain_per_step_predator=-0.15,  
    energy_gain_per_step_prey=-0.05,
    energy_gain_per_step_grass=0.2,
    catch_prey_energy=0.0, #TODO: remove?
    catch_grass_energy=0.0, #TODO: remove?
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
    #vizualization parameters
    cell_scale=40,
    x_pygame_window=0,
    y_pygame_window=0,
    show_energy_chart=True,
    # evaluation parameters
    num_episodes=100,
    watch_grid_model=False,   
)
