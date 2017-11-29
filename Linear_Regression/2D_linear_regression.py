import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# load in the data from 'data_2d.csv'
X = []
Y = []

for line in open('data_2d.csv'):
    x1, x2, y = line.split(',')
    X.append([float(x1), float(x2), 1])     # This is our bias w_0, x_0 which equals 1 term.
    Y.append(float(y))
    # We add a bis term to "move / adjust" our line of best fit.
# convert X and Y into Numpy arrays:
X = np.array(X)
Y = np.array(Y)

# Plot to see what our data looks like:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y)
plt.show()

# For multi-dimensional regression we have D equations and D unknowns. We want to isolate w to find the weights.
# Our model still takes the form (Y_hat = wT * Xi) and our error function does not change.
# Take derivative of error function and solving for w:

# w(XT X) = XT Y (this takes the form Ax = b, solving for w:
# w = ((XT X)^-1) XT Y convert into Numpy methods:

# calculate the weights for our data:
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Y_hat = np.dot(X, w)

# compute R2 to determine how good our model is:
d1 = Y - Y_hat
d2 = Y - Y.mean()
R2 = 1 - d1.dot(d1) / d2.dot(d2)
print(R2)
