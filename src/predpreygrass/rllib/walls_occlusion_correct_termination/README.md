# Walls & Occlusion Termination in PredPreyGrass

## Problem:
The original environment does not handle termination correctly: this results that the final reward of a prey which gets killed ("penalty_prey_caught") is not counted because the prey under consideration is already removed from the reward dict which is returned by the step function.

# Fix:
The reward, termination, runcation, infos dicts have to return the same agents as defined in the action_dict. Terminated agents have to return "True". Observations dict has to return on the other hand only the agents which are still alive (or maybe zeros/black death?)