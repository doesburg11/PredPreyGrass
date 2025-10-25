import pytest
import numpy as np
from src.predpreygrass.rllib.numpy_vectorization.np_vec_env import PredPreyGrassEnv

def test_reproduction_reward():
    # Small env, 1 predator, 1 prey, both above reproduction threshold
    env = PredPreyGrassEnv(
        grid_shape=(5, 5),
        num_possible_predators=2,
        num_possible_prey=2,
        initial_num_predators=1,
        initial_num_prey=1,
        initial_num_grass=0,
        initial_energy_predator=20.0,  # well above threshold
        initial_energy_prey=20.0,      # well above threshold
        predator_creation_energy_threshold=10.0,
        prey_creation_energy_threshold=10.0,
        max_episode_steps=5,
        seed=42,
    )
    obs, infos = env.reset()
    # Step: both agents should reproduce
    actions = {aid: 0 for aid in env.agents}  # noop
    obs, rewards, term, trunc, infos = env.step(actions)
    # Check that both original agents got +10.0 reward
    pred_id = [aid for aid in rewards if 'predator' in aid and aid in rewards][0]
    prey_id = [aid for aid in rewards if 'prey' in aid and aid in rewards][0]
    assert rewards[pred_id] == pytest.approx(10.0, abs=1e-6), f"Predator did not get reproduction reward: {rewards[pred_id]}"
    assert rewards[prey_id] == pytest.approx(10.0, abs=1e-6), f"Prey did not get reproduction reward: {rewards[prey_id]}"
    # Check that new agents exist
    assert any(aid != pred_id and 'predator' in aid for aid in env.agents), "No new predator spawned"
    assert any(aid != prey_id and 'prey' in aid for aid in env.agents), "No new prey spawned"
