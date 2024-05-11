# tuned example to run the predpreygrass environment with the maximum average episode length
# this benchmark has the maximum number of possible Predators (18) and Prey (24)
# which can still be displayed with the agent energy chart 

# put in here your own directory to the local output folder
local_output_directory = "/home/doesburg/Dropbox/02_marl_results/predpreygras_results/"

training_steps_string = "10_000_000"

env_kwargs = dict(
    # environment parameters
    max_cycles=10000, 
    x_grid_size=16,
    y_grid_size=16, 
    # agent parameters
    n_possible_predator=18,
    n_possible_prey=24,
    n_possible_grass=30,
    n_initial_active_predator=6,
    n_initial_active_prey=8,
    
    max_observation_range=7, # must be odd and not smaller than any obs_range
    obs_range_predator=5, # must be odd    
    obs_range_prey=7, # must be odd
    # energy parameters
    energy_gain_per_step_predator = -0.3, # tuned
    energy_gain_per_step_prey = -0.05,
    energy_gain_per_step_grass = 0.2, # tuned
    catch_prey_energy = 5.0,
    catch_grass_energy = 3.0,   
    initial_energy_predator = 5.0,
    initial_energy_prey = 5.0, 
    initial_energy_grass = 3.0,
    # reward parameters
    step_reward_predator = 0.0,
    step_reward_prey = 0.0,
    catch_reward_grass = 0.0,
    catch_reward_prey = 0.0,  
    death_reward_prey = -5.0, # tuned, however 0.0 is a viable option and leads to the same emergent behaviors
    death_reward_predator = 0.0,
    reproduction_reward_prey = 10.0,
    reproduction_reward_predator = 10.0,
    # create agents parameters
    regrow_grass=True,
    create_prey = True,
    create_predator = True, 
    prey_creation_energy_threshold = 8.0,
    predator_creation_energy_threshold = 10.0,
    
    # visualization parameters
    cell_scale=40,
    x_pygame_window=0,
    y_pygame_window=0,
    show_energy_chart=True,

)