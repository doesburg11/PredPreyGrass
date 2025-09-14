from predpreygrass.rllib.mutating_agents.predpreygrass_rllib_env import PredPreyGrass
from predpreygrass.rllib.mutating_agents.config.config_env_random import config_env

# external libraries
import time

n_steps = 1000
seed_value = 42  # Set seed for reproducibility

if __name__ == "__main__":
    env = PredPreyGrass(config=config_env)

    observations, _ = env.reset(seed=seed_value)

    # Seed the action spaces
    for agent in env.agents:
        env.action_spaces[agent].seed(seed_value)

    step_times = []
    total_start_time = time.time()  # Start timing the entire execution

    for step in range(n_steps):
        step_start = time.time()  # Start timing this step

        action_dict = {agent: env.action_spaces[agent].sample() for agent in env.agents}
        observations, rewards, terminations, truncations, info = env.step(action_dict)

        step_end = time.time()  # End timing this step
        step_times.append(step_end - step_start)

        if terminations["__all__"]:
            print("Environment terminated by termination.")
            break
        if truncations["__all__"]:
            print("Environment terminated by truncation.")
            break

    total_end_time = time.time()  # End timing the entire execution
    total_execution_time = total_end_time - total_start_time

    if step_times:
        average_step_time = sum(step_times) / len(step_times)
        print(f"Steps executed: {len(step_times)}")
        print(f"Average step execution time: {average_step_time:.6f} seconds")

    print(f"Total execution time: {total_execution_time:.4f} seconds")
