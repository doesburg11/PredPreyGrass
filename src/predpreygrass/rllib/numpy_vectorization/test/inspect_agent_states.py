from predpreygrass.rllib.numpy_vectorization.np_vec_env import PredPreyGrassEnv

if __name__ == "__main__":
    env = PredPreyGrassEnv(num_predators=2, num_prey=2, predator_energy_init=0.2, prey_energy_init=0.2)
    obs, infos = env.reset()
    print("Initial agent states:")
    for aid in env.agents:
        ix = env._id_to_ix[aid]
        print(f"{aid}: energy={env.energy[ix]}, age={env.age[ix]}, active={env.active[ix]}")
    print()
    for step in range(5):
        action_dict = {aid: 0 for aid in env.agents}  # all noop
        obs, rewards, terminations, truncations, infos = env.step(action_dict)
        print(f"Step {step+1} agent states:")
        for aid in env.possible_agents:
            ix = env._id_to_ix[aid]
            print(f"{aid}: energy={env.energy[ix]}, age={env.age[ix]}, active={env.active[ix]}, terminated={terminations[aid]}")
        print(f"Terminations: {terminations}")
        print(f"Active agents: {env.agents}")
        print()
        if terminations.get("__all__", False):
            print(f"All agents terminated at step {step+1}")
            break
