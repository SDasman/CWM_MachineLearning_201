import numpy as np
import matplotlib.pyplot as plt


# load the data
X = []
Y = []

for line in open('data_1d.csv'):
    x, y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

# turn into numpy arrays
X = np.array(X)
Y = np.array(Y)

# Just plot the data to see what it looks like:
# plt.scatter(X, Y)
# plt.axis('equal')
# plt.show()

# Apply the equations we learned to find a and b
# Denominator was the same for both a and b, store it as a variable:
denom = X.dot(X) - X.mean() * X.sum()

# now find a and b:
a = (X.dot(Y) - Y.mean() * X.sum()) / denom
b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denom

Y_hat = a * X + b

# Now plot everything:
plt.scatter(X, Y)
plt.plot(X, Y_hat)
# plt.axis('equal')
plt.show()
