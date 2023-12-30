"""
matrix=dict()

matrix={(1,1): "prey_0", (1,0): "prey_2"}

#print(matrix[1,0])  ###works

grid_location_to_prey_dict = {}
# initialization
for i in range(16):
    for j in range(16):
        grid_location_to_prey_dict[i,j] = []

print(grid_location_to_prey_dict)
"""
import numpy as np
observation =np.ones((15,15), dtype=int)
print(observation)
observation_range_agent = 3
max = 15
#mask is number of 'outer squares' of an observation surface set to zero
mask = int((max - observation_range_agent)/2)
if mask > 0: # observation_range agent is smaller than default max_observation_range
    for j in range(mask):
            observation[j,0:max] = 0
            observation[max-1-j,0:max] = 0
            observation[0:max,j] = 0
            observation[0:max,max-1-j] = 0

print(observation)