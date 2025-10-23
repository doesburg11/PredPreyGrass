from predpreygrass.rllib.numpy_vectorization.np_vec_env import PredPreyGrassEnv

if __name__ == "__main__":
    # 1 predator, 1 prey, both start in the same cell, so predation is guaranteed
    env = PredPreyGrassEnv(num_predators=1, num_prey=1, num_possible_predators=1, num_possible_prey=1,
                           predator_energy_init=5.0, prey_energy_init=5.0, grass_count=0)
    obs, infos = env.reset()
    # Force both to the same cell for deterministic predation
    ix_pred = env._id_to_ix['predator_0']
    ix_prey = env._id_to_ix['prey_0']
    env.pos[ix_pred] = [0, 0]
    env.pos[ix_prey] = [0, 0]
    print(f"Step 0: agents={env.agents}, active={env.active.tolist()}")
    print(f"  obs keys: {list(obs.keys())}")
    print(f"  predator_0 energy: {env.energy[ix_pred]}, prey_0 energy: {env.energy[ix_prey]}")
    # Step 1: predation should occur
    obs, rewards, terminations, truncations, infos = env.step({'predator_0': 0, 'prey_0': 0})
    print(f"\nStep 1: agents={env.agents}, active={env.active.tolist()}")
    print(f"  obs keys: {list(obs.keys())}")
    print(f"  rewards: {rewards}")
    print(f"  terminations: {terminations}")
    print(f"  predator_0 energy: {env.energy[ix_pred]}, prey_0 energy: {env.energy[ix_prey]}")
    # Step 2: only predator should be present
    obs, rewards, terminations, truncations, infos = env.step({'predator_0': 0})
    print(f"\nStep 2: agents={env.agents}, active={env.active.tolist()}")
    print(f"  obs keys: {list(obs.keys())}")
    print(f"  rewards: {rewards}")
    print(f"  terminations: {terminations}")
    print(f"  predator_0 energy: {env.energy[ix_pred]}, prey_0 energy: {env.energy[ix_prey]}")