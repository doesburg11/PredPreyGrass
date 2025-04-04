from predpreygrass.rllib.v4_select_coef.predpreygrass_rllib_env import PredPreyGrass 
from predpreygrass.rllib.v4_select_coef.config_env import config_env


from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
import os

# --- Register environment ---
def env_creator(config):
    return PredPreyGrass(config or config_env)
register_env("PredPreyGrass", env_creator)


# Restore trained PPO algorithm
checkpoint_root = '/home/doesburg/ray_results/'
checkpoint_file = 'PPO_2025-04-03_21-48-39/PPO_PredPreyGrass_a3e02_00000_0_2025-04-03_21-48-39/checkpoint_000006'
checkpoint_path = f"file://{os.path.abspath(checkpoint_root+checkpoint_file)}"
algo = Algorithm.from_checkpoint(checkpoint_path)

# --- Recreate environment for evaluation ---
env_config = algo.config.env_config
env = PredPreyGrass(env_config)
obs, _ = env.reset()

# --- Main evaluation loop ---
done = False
total_reward = {agent_id: 0.0 for agent_id in obs}


# --- Main evaluation loop ---
done = False
while not done:
     # Format obs as expected: {agent_id: {"obs": obs_value}}
    obs_batch = {agent_id: {"obs": ob} for agent_id, ob in obs.items()}

    # Use the new compute_actions API
    actions = algo.compute_actions(obs_batch)


    obs, rewards, terminated, truncated, info = env.step(actions)
    done = all(terminated.values()) or all(truncated.values())
    # Accumulate rewards
    for agent_id, reward in rewards.items():
        total_reward[agent_id] += reward
    
    done = all(terminated.values()) or all(truncated.values())

# --- Print final rewards ---
print("Total episode rewards:")
for agent_id, reward in total_reward.items():
    print(f"  {agent_id}: {reward:.2f}")