training_steps_string = "10_000_000"

env_kwargs = dict(
    max_cycles=500, 
    x_grid_size=16,
    y_grid_size=16, 
    n_possible_agent_list = [0, 10, 10, 30],  # 0: wall, 1: predator, 2: prey, 3: grass
    n_initial_agent_list =  [0, 6, 8, 30],      
    max_observation_range= 7, # must be odd and not smaller than any obs_range
    obs_range_agent_list = [0, 5, 7, 0], # wall, predator, prey, grass
    energy_loss_per_step_agent_list =   [0, -0.1, -0.1, 0],
    initial_energy_list =  [0, 5.0, 3.0, 2.0], 
    # visualization parameters
    cell_scale=40,
    x_pygame_window=0,
    y_pygame_window=0,
)