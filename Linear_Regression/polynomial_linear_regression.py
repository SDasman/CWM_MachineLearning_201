import numpy as np
import matplotlib.pyplot as plt


# load in the data from 'data_poly.csv'
X = []
Y = []

for line in open('data_poly.csv'):
    x, y = line.split(',')
    x = float(x)
    X.append([1, x, x*x])
    Y.append(float(y))

# convert to Numpy arrays
X = np.array(X)
Y = np.array(Y)

# For multi-dimensional regression we have D equations and D unknowns. We want to isolate w to find the weights.
# Our model still takes the form (Y_hat = wT * Xi) and our error function does not change.
# Take derivative of error function and solving for w:

# w(XT X) = XT Y (this takes the form Ax = b, solving for w:
# w = ((XT X)^-1) XT Y convert into Numpy methods:

# calculate the weights for our data:
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Y_hat = np.dot(X, w)

# Plot to see the data and line of best fit.
plt.scatter(X[:, 1], Y)
plt.plot(sorted(X[:, 1]), sorted(Y_hat), color='black')
plt.show()

# Calculate R2 to determine how good our model is:
d1 = Y - Y_hat
d2 = Y - Y.mean()
R2 = 1 - d1.dot(d1) / d2.dot(d2)
print(R2)
