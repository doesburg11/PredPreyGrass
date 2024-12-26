import numpy as np
from pettingzoo.utils.conversions import parallel_wrapper_fn
from predpreygrass.single_objective.envs.base_env.predpreygrass_base import PredPreyGrassAECEnv, PredPreyGrassParallelEnv


# Initialize environments
seed = 42  # Control randomness
np.random.seed(seed)

aec_env = PredPreyGrassAECEnv()
parallel_env = PredPreyGrassParallelEnv()
wrapped_aec_env = parallel_wrapper_fn(PredPreyGrassAECEnv)()

# Seed environments
aec_env.seed(seed)
parallel_env.seed(seed)
wrapped_aec_env.seed(seed)

# Reset environments
aec_obs = aec_env.reset()
parallel_obs = parallel_env.reset()
wrapped_aec_obs = wrapped_aec_env.reset()

def compare_environments(aec_env, parallel_env, wrapped_aec_env):
    actions = {agent: np.random.choice([0, 1, 2, 3]) for agent in parallel_env.agents}  # Example random actions
    aec_env.step(actions)
    parallel_env.step(actions)
    wrapped_aec_env.step(actions)

    # Compare observations
    assert aec_env.observe() == parallel_env.observe(), "Observation mismatch!"
    assert aec_env.observe() == wrapped_aec_env.observe(), "Observation mismatch with wrapped AEC!"
    if aec_env.observe() != parallel_env.observe():
        print("Observation mismatch detected!")
        print(f"AEC Observation: {aec_env.observe()}")
        print(f"Parallel Observation: {parallel_env.observe()}")

    # Compare rewards
    assert aec_env.rewards == parallel_env.rewards, "Reward mismatch!"
    if aec_env.rewards != parallel_env.rewards:
        print("Reward mismatch detected!")
        print(f"AEC Rewards: {aec_env.rewards}")
        print(f"Parallel Rewards: {parallel_env.rewards}")
    assert aec_env.rewards == wrapped_aec_env.rewards, "Reward mismatch with wrapped AEC!"

    # Compare done flags
    assert aec_env.dones == parallel_env.dones, "Done flag mismatch!"
    assert aec_env.dones == wrapped_aec_env.dones, "Done flag mismatch with wrapped AEC!"

for episode in range(10):  # Test 10 episodes
    obs_aec = aec_env.reset()
    obs_parallel = parallel_env.reset()
    obs_wrapped_aec = wrapped_aec_env.reset()

    for step in range(100):  # Simulate 100 steps per episode
        actions = {agent: np.random.choice([0, 1, 2, 3]) for agent in parallel_env.agents}

        obs_aec, rewards_aec, dones_aec, _ = aec_env.step(actions)
        obs_parallel, rewards_parallel, dones_parallel, _ = parallel_env.step(actions)
        obs_wrapped_aec, rewards_wrapped_aec, dones_wrapped_aec, _ = wrapped_aec_env.step(actions)

        if obs_aec != obs_parallel or obs_aec != obs_wrapped_aec:
            print(f"Step {step}: Observation mismatch!")
            print(f"AEC: {obs_aec}, Parallel: {obs_parallel}, Wrapped AEC: {obs_wrapped_aec}")

        if rewards_aec != rewards_parallel or rewards_aec != rewards_wrapped_aec:
            print(f"Step {step}: Reward mismatch!")
            print(f"AEC: {rewards_aec}, Parallel: {rewards_parallel}, Wrapped AEC: {rewards_wrapped_aec}")

        if dones_aec != dones_parallel or dones_aec != dones_wrapped_aec:
            print(f"Step {step}: Done flag mismatch!")
            print(f"AEC: {dones_aec}, Parallel: {dones_parallel}, Wrapped AEC: {dones_wrapped_aec}")

        if all(dones_aec.values()):
            break  # End episode if all agents are done

