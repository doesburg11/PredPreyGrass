import numpy as np

# Define movement directions (North, South, East, West)
DELTA = np.array(   [
                        [ 0, 0],
                        [ 0, 1], 
                        [ 0,-1], 
                        [ 1, 0], 
                        [-1, 0]
                    ]
                )  # North  # South  # East  # West

# Example: current positions of agents (N agents)
positions = np.array(
                [
                    [2, 3], 
                    [5, 5], 
                    [1, 7], 
                    [6, 2], 
                    [7, 6]
                ]
            )  # Shape: (N, 2)

# Generate all possible moves
# We want to add each delta to each position. We can use broadcasting to achieve this.

# Expand positions and DELTA for broadcasting
# positions[:, np.newaxis, :] will have shape (N, 5, 2)
# DELTA[np.newaxis, :, :] will have shape (1, 5, 2)
print("all_new_positions:")
all_new_positions = positions[:, np.newaxis, :] + DELTA[np.newaxis, :, :]

# The resulting shape of all_new_positions will be (N, 4, 2)
# where N is the number of agents, and each agent has 4 potential new positions.
print(all_new_positions)

# Assume actions is an array of shape (N,) where each value is 0, 1, 2, or 3
# representing the action chosen for each agent.
actions = np.array([0, 2, 1, 3, 3])  # Example actions for each agent

# Get the new positions based on the selected actions
new_positions = all_new_positions[np.arange(len(positions)), actions]
print("New positions:")
print(new_positions)  # Shape: (N, 2)
