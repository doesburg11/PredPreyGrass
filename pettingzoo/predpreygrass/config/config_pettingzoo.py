# put in here your own directory to the output folder
local_output_directory = "/home/doesburg/Dropbox/02_marl_results/predpreygras_results/"

training_steps_string = "30_000_000"

env_kwargs = dict(
    # environment parameters
    max_cycles=5000, 
    x_grid_size=25,
    y_grid_size=25, 
    # agent parameters
    n_possible_predator=18,  # maximum number of predators during runtime
    n_possible_prey=24,
    n_possible_grass=25,  # optimized by parameter variation [10,12,14,16,18,20,...,26,28,40]
    n_initial_active_predator=6,
    n_initial_active_prey=8,
    # observation parameters
    max_observation_range=9, # must be odd and not smaller than any obs_range
    obs_range_predator=7,  
    obs_range_prey=9, 
    # energy parameters
    energy_gain_per_step_predator = -0.15, # optimized by parameter variation [-0.3,-0.25,-0.20,-0.15,-0.10]
    energy_gain_per_step_prey = -0.05,  
    energy_gain_per_step_grass = 0.2,  
    catch_prey_energy = 0.0,
    catch_grass_energy = 0.0,   
    initial_energy_predator = 5.0,
    initial_energy_prey = 5.0, 
    initial_energy_grass = 3.0,
    max_energy_level_grass = 4.0,
    # reward parameters
    step_reward_predator = 0.0,
    step_reward_prey = 0.0,
    step_reward_grass = 0.0,
    catch_reward_grass = 0.0,
    catch_reward_prey = 0.0,  
    death_reward_prey = -15.0,  # this results in a zero-sum element between Predator and Prey rewards
    death_reward_predator = 0.0,
    reproduction_reward_prey = 10.0,  
    reproduction_reward_predator = 10.0, 
    # create agents parameters
    create_prey = True,  # only effect on and applicable to fixed energy transfer environments
    create_predator = True,  # only effect on and applicable to fixed energy transfer environments
    regrow_grass = True,  # only effect on and applicable to fixed energy transfer environments
    prey_creation_energy_threshold = 8,  
    predator_creation_energy_threshold = 12, # optimized by parameter variation [2,4, 6,..., 22, 24]
    # visualization parameters
    cell_scale = 40,
    x_pygame_window = 0,
    y_pygame_window = 0,
    show_energy_chart = True,
    spawning_area_predator = dict({
            "x_begin": 0,
            "y_begin": 0,
            "x_end": 24,
            "y_end": 24,
        }),
    spawning_area_prey = dict({
            "x_begin": 0,
            "y_begin": 0,
            "x_end": 24,
            "y_end": 24,
        }), 
    spawning_area_grass = dict({
            "x_begin": 12,
            "y_begin": 0,
            "x_end": 13,
            "y_end": 24,
        }),



)