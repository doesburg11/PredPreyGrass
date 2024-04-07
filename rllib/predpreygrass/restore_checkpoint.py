import numpy as np

from ray.rllib.policy.policy import Policy

# Use the `from_checkpoint` utility of the Policy class:
dir="/home/doesburg/ray_results/PPO_2024-03-29_18-08-44/PPO_pred_prey_grass_ffc89_00000_0_2024-03-29_18-08-44/checkpoint_000000"
my_restored_policy = Policy.from_checkpoint(dir)

print("my_restored_policy:", my_restored_policy)

# Use the restored policy for serving actions.
obs = np.array([0.0, 0.1, 0.2, 0.3])  # individual CartPole observation
action = my_restored_policy.compute_single_action(obs)

print(f"Computed action {action} from given CartPole observation.")
