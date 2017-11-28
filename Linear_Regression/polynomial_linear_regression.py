import numpy as np
import matplotlib.pyplot as plt


# load in the data
X = []
Y = []

for line in open('data_poly.csv'):
    x, y = line.split(',')
    x = float(x)
    X.append([1, x, x*x])
    Y.append(float(y))

# convert to numpy arrays
X = np.array(X)
Y = np.array(Y)

# calculate weights:
w = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Y_hat = np.dot(X, w)

plt.scatter(X[:, 1], Y)
plt.plot(sorted(X[:, 1]), sorted(Y_hat))
plt.show()

# calculate R^2
d1 = Y - Y_hat
d2 = Y - Y.mean()
R2 = 1 - d1.dot(d1) / d2.dot(d2)
print(R2)
