import sys
import os
import importlib

# Ensure parent directory is in sys.path for relative import
test_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(test_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

PredPreyGrass = importlib.import_module("predpreygrass_rllib_env").PredPreyGrass

def make_minimal_config():
    # Minimal config for a 2x2 grid, 1 predator, 1 prey, 1 grass, no walls, short episode
    return {
        'debug_mode': False,
        'verbose_movement': False,
        'verbose_decay': False,
        'verbose_reproduction': False,
        'verbose_engagement': False,
        'max_steps': 3,
        'reward_predator_catch_prey': 10.0,
        'reward_prey_eat_grass': 5.0,
        'reward_predator_step': 0.1,
        'reward_prey_step': 0.2,
        'penalty_prey_caught': -2.0,
        'reproduction_reward_predator': 1.0,
        'reproduction_reward_prey': 1.0,
        'energy_loss_per_step_predator': 0.0,
        'energy_loss_per_step_prey': 0.0,
        'predator_creation_energy_threshold': 100.0,
        'prey_creation_energy_threshold': 100.0,
        'max_energy_grass': 10.0,
        'n_possible_type_1_predators': 1,
        'n_possible_type_2_predators': 0,
        'n_possible_type_1_prey': 1,
        'n_possible_type_2_prey': 0,
        'n_initial_active_type_1_predator': 1,
        'n_initial_active_type_2_predator': 0,
        'n_initial_active_type_1_prey': 1,
        'n_initial_active_type_2_prey': 0,
        'initial_energy_predator': 10.0,
        'initial_energy_prey': 10.0,
        'grid_size': 2,
        'num_obs_channels': 4,
        'predator_obs_range': 2,
        'prey_obs_range': 2,
        'include_visibility_channel': False,
        'respect_los_for_movement': False,
        'mask_observation_with_visibility': False,
        'initial_num_grass': 1,
        'initial_energy_grass': 10.0,
        'energy_gain_per_step_grass': 0.0,
        'manual_wall_positions': [],
        'mutation_rate_predator': 0.0,
        'mutation_rate_prey': 0.0,
        'type_1_action_range': 1,
        'type_2_action_range': 1,
        'max_energy_gain_per_prey': 10.0,
        'max_energy_gain_per_grass': 10.0,
        'reproduction_energy_efficiency': 1.0,
        'seed': 42,
        'reproduction_cooldown_steps': 0,
    }

def test_cumulative_reward_tracking():
    config = make_minimal_config()
    env = PredPreyGrass(config)
    obs, _ = env.reset()
    predator = [a for a in env.agents if 'predator' in a][0]
    prey = [a for a in env.agents if 'prey' in a][0]
    # Place predator and prey on same cell to force catch
    env.agent_positions[predator] = (0, 0)
    env.agent_positions[prey] = (0, 0)
    env.predator_positions[predator] = (0, 0)
    env.prey_positions[prey] = (0, 0)
    # Step: predator catches prey
    actions = {predator: 0, prey: 0}
    obs, rewards, term, trunc, info = env.step(actions)
    # Check rewards
    assert rewards[predator] == config['reward_predator_catch_prey']
    assert rewards[prey] == config['penalty_prey_caught']
    # Check cumulative_reward in agent_stats_completed for prey
    prey_record = env.agent_stats_completed[prey]
    assert prey_record['cumulative_reward'] == config['penalty_prey_caught']
    # Predator should still be live
    pred_record = env.agent_stats_live[predator]
    assert pred_record['cumulative_reward'] == config['reward_predator_catch_prey']
    # Step again: predator gets step reward
    obs, rewards, term, trunc, info = env.step({predator: 0})
    pred_record = env.agent_stats_live[predator]
    assert pred_record['cumulative_reward'] == config['reward_predator_catch_prey'] + config['reward_predator_step']
    # Step to end
    for _ in range(2):
        obs, rewards, term, trunc, info = env.step({predator: 0})
    # After episode, predator should be in completed
    pred_record = env.agent_stats_completed[predator]
    # Should have accumulated all rewards
    expected = config['reward_predator_catch_prey'] + config['reward_predator_step'] * 3
    assert abs(pred_record['cumulative_reward'] - expected) < 1e-6
