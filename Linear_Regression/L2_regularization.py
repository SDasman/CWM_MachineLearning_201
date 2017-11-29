import numpy as np
import matplotlib.pyplot as plt

# We will now look at L2 Regularization. One factor that arises out of linear regression is
# when we minimize the Squared Error, the log likelihood increases

# Our weights can be affected by outliers (over fitting) (Dynamic weights?).


# Generate our own random data using Numpy:
N = 50
X = np.linspace(0, 10, N)
Y = 0.5*X + np.random.randn(N)
# the +np.random.randn(N) is generating some random noise.

# We will manually set the outliers to see how our line of best fit changes.
Y[-1] += 30
Y[-2] += 30

plt.scatter(X, Y)
plt.show()

# We add our bias terms of 1s to our data:
X = np.vstack([np.ones(N), X]).T

# solve for the weights. Here w_ml is the maximum log likelihood.
# notice how this w_ml is the same as 1/2/poly - regression.
w_ml = np.linalg.solve(np.dot(X.T, X), np.dot(X.T, Y))
Y_hat_ml = X.dot(w_ml)

# the original data plotted WITHOUT L2 regularization.
plt.scatter(X[:, 1], Y)
# the maximum likelihood line
plt.plot(X[:, 1], Y_hat_ml, color='black')
plt.show()


# Now we will do L2 regularization solution.
# Change the error / cost function by adding a Squared magnitude to it.
# L2 Regularization (Ridge Regression) is used to help with over fitting
# We penalize overly large weights.
# Take derivative of the cost function and set = 0 and solve for our new w, which takes the form:

# w = (Lambda I + XT X)^-1 * XT Y

L2 = 1000.0
w_map = np.linalg.solve(L2 * np.eye(2) + X.T.dot(X), X.T.dot(Y))
# We can see the new L2 term
# Notice that our model stayed the same, it's still linear regression.
# Our error function, and our method to find w changed, which gives us new results.
Y_hat_map = X.dot(w_map)

# now plot the L2 calculations:
plt.scatter(X[:, 1], Y)
plt.plot(X[:, 1], Y_hat_ml, label='maximum likelihood', color='black')
plt.plot(X[:, 1], Y_hat_map, label='MAP', color='red')
plt.legend()
plt.show()
