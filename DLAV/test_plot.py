import matplotlib.pyplot as plt
import numpy as np

# Given data
data = np.array([[ 0.0000e+00,  0.0000e+00,  0.0000e+00],
                [-8.9843e+01, -2.6863e+01,  1.0745e+02],
                [-3.9482e+01 , 4.5109e+02 , 1.8343e+02],
                [ 5.3856e+01,  8.8675e+02,  3.0373e+02],
                [ 8.9842e+01,  2.6863e+01, -1.0745e+02],
                [ 8.7195e+01,  5.1236e+02, -7.5314e+01],
                [ 1.2003e+02,  9.4992e+02,  6.7638e+01],
                [ 7.0670e-01, -2.6220e+02, -2.7285e+00],
                [-3.8920e+01, -4.9876e+02, -1.0309e+02],
                [-1.1029e+02, -5.2902e+02, -1.9390e+02],
                [-9.7597e+01, -6.4279e+02, -1.8298e+02],
                [ 7.8294e+01 ,-4.2872e+02 ,-1.6366e+02],
                [ 2.6530e+02 ,-1.9285e+02 ,-1.6447e+02],
                [ 3.5755e+02 , 4.3534e+01 ,-2.1065e+02],
                [-1.4279e+02 ,-4.6683e+02 ,-6.0156e-01],
                [-4.0125e+02 ,-3.1555e+02 , 2.9685e+01],
                [-4.3098e+02, -2.4058e+02, -2.1529e+02]])
data = 

# Extracting x, y, and z coordinates
x = data[:, 2]
y = -data[:, 0]
z = data[:, 1]

# Creating the 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot of the data points
ax.scatter(x, y, z, c='b', marker='o')

# Set labels for the axes
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

# Show the plot
plt.show()
