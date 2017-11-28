import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


# load in the data
X = []
Y = []

for line in open('data_2d.csv'):
    x1, x2, y = line.split(',')
    X.append([float(x1), float(x2), 1])     # This is our bias w_0, x_0 which equals 1 term.
    Y.append(float(y))

# convert X and Y into numpy arrays:
X = np.array(X)
Y = np.array(Y)

# plot just to see what it looks like:
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], Y)
plt.show()

# calculate the weights for our data:
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Y_hat = np.dot(X, w)

# compute R^2 to see how good out model is:
d1 = Y - Y_hat
d2 = Y - Y.mean()
R2 = 1 - d1.dot(d1) / d2.dot(d2)
print(R2)
