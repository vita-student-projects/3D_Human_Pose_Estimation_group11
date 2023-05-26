import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal

# Coordinates of the point
x = 5
y = 10

# Create a 2D grid
grid_size = 64
grid = np.zeros((grid_size, grid_size))


mean = [x, y]
covariance = [[10, 0], [0, 10]]
gaussian = multivariate_normal(mean=mean, cov=covariance)

for i in range(grid_size):
    for j in range(grid_size):
        grid[i, j] = gaussian.pdf([i, j])

grid = grid/np.max(grid)
# Create the heatmap
plt.imshow(grid, cmap='hot', interpolation='nearest')
plt.colorbar()

# Set the labels and title
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Gaussian Heatmap')

# Show the plot
plt.show()
