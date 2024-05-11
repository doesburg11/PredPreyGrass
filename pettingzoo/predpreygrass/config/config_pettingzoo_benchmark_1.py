"""
The benchmark configuration config_pettingzoo_benchmark_1.py is, somewhat arbitrarily, to 
test new developments in the environment. It is the same configuration used in an early 
stage of the environment some time ago and is displayed on the front page of the repository. 
It is used to test if it is still performing as expected when devoloping new features, such 
as for example creating new learning agents (Predator and Prey) or regrowing Grass agents.
"""

# put in here your own directory to the output folder
local_output_directory = "/home/doesburg/Dropbox/02_marl_results/predpreygras_results/"

training_steps_string = "10_000_000"

env_kwargs = dict(
    # environment parameters
    max_cycles=10000, 
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
    energy_gain_per_step_grass = 0,  
    catch_prey_energy = 5.0,
    catch_grass_energy = 3.0,   
    initial_energy_predator = 5.0,
    initial_energy_prey = 5.0, 
    initial_energy_grass = 0.0,
    # reward parameters
    step_reward_predator = 0.0,
    step_reward_prey = 0.0,
    catch_reward_prey = 5.0, # for predator 
    catch_reward_grass = 3.0, # for prey
    death_reward_prey = 0.0,
    death_reward_predator = 0.0,
    reproduction_reward_prey = 0.0,
    reproduction_reward_predator = 0.0,
    # create agents parameters
    regrow_grass=False,
    create_prey = False,
    create_predator = False, 
    prey_creation_energy_threshold = 0.0,
    predator_creation_energy_threshold = 0.0,
    # visualization parameters
    cell_scale=40,
    x_pygame_window=0,
    y_pygame_window=0,
    show_energy_chart=True,

)