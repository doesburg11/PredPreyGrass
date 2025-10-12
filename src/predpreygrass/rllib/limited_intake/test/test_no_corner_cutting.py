"""
Test: No corner cutting (diagonal movement blocked by orthogonal walls)
"""
from predpreygrass.rllib.walls_occlusion.predpreygrass_rllib_env import PredPreyGrass
import numpy as np

def test_no_corner_cutting():
    config = {
        'grid_size': 3,
        'manual_wall_positions': [(1,0), (0,1)],
        'num_walls': 2,
        'wall_placement_mode': 'manual',
        'respect_los_for_movement': True,
        'num_predators': 0,
        'num_prey': 1,
    'num_grass': 0,
    'initial_num_grass': 0,
        'prey_obs_range': 2,
        'predator_obs_range': 2,
        # Explicitly zero all agent types except one prey
        'n_possible_type_1_predators': 0,
        'n_possible_type_2_predators': 0,
        'n_possible_type_1_prey': 1,
        'n_possible_type_2_prey': 0,
        'n_initial_active_type_1_predator': 0,
        'n_initial_active_type_2_predator': 0,
        'n_initial_active_type_1_prey': 1,
        'n_initial_active_type_2_prey': 0,
    }
    env = PredPreyGrass(config)
    obs, _ = env.reset()
    prey_id = [aid for aid in env.agent_positions if 'prey' in aid][0]
    env.agent_positions[prey_id] = (0,0)
    # Try to move prey diagonally into (1,1)
    move_action = 3  # Down-right; adjust if needed for your action mapping
    new_pos = env._get_move(prey_id, move_action)
    print(f"Prey at (0,0) tries to move to (1,1): new_pos={new_pos}")
    assert new_pos == (0,0), "Diagonal move into (1,1) should be blocked by corner walls!"
    print("Test passed: Diagonal move blocked as expected.")

if __name__ == "__main__":
    test_no_corner_cutting()
