# this file is used to set the benchmark parameters for the PredPreyGrass environment
# the benchamrk is used to test if it is still performing as expected when adjusted
local_output_directory = "/home/doesburg/Dropbox/02_marl_results/predpreygras_results/"

training_steps_string = "10_000_000"

env_kwargs = dict(
    # environment parameters
    max_cycles=1000, 
    x_grid_size=16,
    y_grid_size=16, 
    # agent parameters
    n_possible_predator=6,
    n_possible_prey=8,
    n_possible_grass=30,
    n_initial_active_predator=6,
    n_initial_active_prey=8,
    
    max_observation_range=7, # must be odd and not smaller than any obs_range
    obs_range_predator=5, # must be odd    
    obs_range_prey=7, # must be odd
    # energy parameters
    energy_gain_per_step_predator = -0.1,
    energy_gain_per_step_prey = -0.05,
    energy_gain_per_step_grass = 0.2,     
    initial_energy_predator = 5.0,
    initial_energy_prey = 5.0, 
    initial_energy_grass = 3.0,
    # reward parameters
    catch_grass_reward = 3.0,
    catch_prey_reward = 5.0,  
    death_reward_prey = 0,
    death_reward_predator = 0,
    # create agents parameters
    regrow_grass=False,
    create_prey = False,
    create_predator = False, 
    prey_creation_energy_threshold = 10.0,
    predator_creation_energy_threshold = 10.0,
    
    # visualization parameters
    cell_scale=40,
    x_pygame_window=0,
    y_pygame_window=0,

)