
from ray.rllib.algorithms.algorithm import Algorithm
from ray.tune.registry import register_env
from predpreygrass_ import PredPreyGrass  # Import your custom environment

def env_creator(config):
    return PredPreyGrass(config)

# Register environment before loading checkpoint
register_env("PredPreyGrass", lambda config: PredPreyGrass(config))

def policy_mapping_fn(agent_id, *args, **kwargs):
    if "predator" in agent_id:
        return "predator_policy"
    elif "prey" in agent_id:
        return "prey_policy"
    return None

# Path to the latest checkpoint (update path accordingly)
checkpoint_path = "/home/doesburg/ray_results/PPO_2025-02-06_09-17-05/PPO_PredPreyGrass_c05c2_00000_0_2025-02-06_09-17-05/checkpoint_000003"  # Example: "~/ray_results/PPO/your_env/checkpoint_000010"

# Load the trained policy
trained_algo = Algorithm.from_checkpoint(
    checkpoint_path,
 )


print("Checkpoint loaded successfully!")

env = PredPreyGrass()  # Initialize environment
obs, _ = env.reset()


done = False
total_reward = 0

while not done:
    action_dict = {}

    for agent_id in env.agents:
        policy_id = policy_mapping_fn(agent_id)  # Safe way to access the function
        print(f"Agent: {agent_id}, Policy: {policy_id}")
        policy = trained_algo.get_policy("predator_policy")  # Or "prey_policy" based on agent_id
        action = policy.compute_single_action(obs[agent_id])

        action_dict[agent_id] = trained_algo.compute_actions(
            obs[agent_id], policy_id=policy_id
        )

    # Step in environment
    obs, rewards, terminations, truncations, _ = env.step(action_dict)
    total_reward += sum(rewards.values())

    # Check if episode is done
    done = terminations["__all__"] or truncations["__all__"]

print(f"Simulation complete! Total Reward: {total_reward}")
