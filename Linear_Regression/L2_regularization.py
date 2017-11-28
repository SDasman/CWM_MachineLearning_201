import numpy as np
import matplotlib.pyplot as plt


# Generate our own data
N = 50
X = np.linspace(0, 10, N)
Y = 0.5*X + np.random.randn(N)
# the +np.random.randn(N) is generating some random noise.

# manually set the outliers
Y[-1] += 30
Y[-2] += 30

plt.scatter(X, Y)
plt.show()

# add the bias term:
X = np.vstack([np.ones(N), X]).T

# solve for the weights. Here w_ml is the maximum likelihood.
w_ml = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Y_hat_ml = X.dot(w_ml)

# the original data plotted without L2 regularization.
plt.scatter(X[:, 1], Y)
# the maximum likelihood line
plt.plot(X[:, 1], Y_hat_ml)
plt.show()

# Now we will do L2 regularization solution
L2 = 1000.0
w_map = np.linalg.solve(L2 * np.eye(2) + X.T.dot(X), X.T.dot(Y))
Y_hat_map = X.dot(w_map)

# now plot the L2 calculations:
plt.scatter(X[:, 1], Y)
plt.plot(X[:, 1], Y_hat_ml, label='maximum likelihood')
plt.plot(X[:, 1], Y_hat_map, label='MAP')
plt.legend()
plt.show()
