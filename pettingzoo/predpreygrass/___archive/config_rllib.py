training_steps_string = "5_000_000"

config = dict(
    max_cycles=1000, 
    x_grid_size=16,
    y_grid_size=16, 
    n_initial_predator=3,
    n_initial_prey=3,
    n_initial_grass=30,
    max_observation_range=7, # must be odd and not smaller than any obs_range
    obs_range_predator=5, # must be odd    
    obs_range_prey=7, # must be odd
    energy_loss_per_step_predator = -0.1,
    energy_loss_per_step_prey = -0.05,     
    initial_energy_predator = 5.0,
    initial_energy_prey = 5.0, 
    catch_grass_reward = 3.0,
    catch_prey_reward = 3.0,      
    # visualization parameters
    cell_scale=40,
    x_pygame_window=0,
    y_pygame_window=0,
)