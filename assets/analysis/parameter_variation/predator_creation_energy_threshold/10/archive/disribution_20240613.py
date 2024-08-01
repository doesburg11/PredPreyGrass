import numpy as np
import matplotlib.pyplot as plt

# Define the size of the grid
grid_size = (16, 16)

# Define the parameters of the Gaussian distribution
mean = (3, 3)  # The center of the distribution
std_dev = (2, 2)  # The standard deviation of the distribution

# Generate a grid of coordinates
x = np.linspace(0, grid_size[0] - 1, grid_size[0])
y = np.linspace(0, grid_size[1] - 1, grid_size[1])
x, y = np.meshgrid(x, y)

# Generate the 2D Gaussian distribution
gaussian_distribution = np.exp(-((x - mean[0])**2 / (2 * std_dev[0]**2) + (y - mean[1])**2 / (2 * std_dev[1]**2)))

# Normalize the distribution so that it represents probabilities
gaussian_distribution /= np.sum(gaussian_distribution)
# set the probablity of mean the center cell to 1
factor = 1 / gaussian_distribution[mean[0], mean[1]]

# Plot the distribution
plt.imshow(gaussian_distribution, cmap='hot', interpolation='nearest')
plt.show()


"""
# Add annotations for each cell
for i in range(grid_size[0]):
    for j in range(grid_size[1]):
        print(f'{factor*gaussian_distribution[i, j]:.1f}', end="  ")
    print()
"""


