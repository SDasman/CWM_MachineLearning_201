import numpy as np
import matplotlib.pyplot as plt


# load the data from 'data_1d.csv'
X = []
Y = []

for line in open('data_1d.csv'):
    x, y = line.split(',')
    X.append(float(x))
    Y.append(float(y))

# turn X and Y into Numpy arrays
X = np.array(X)
Y = np.array(Y)

# Just plot the data to see what it looks like:
plt.scatter(X, Y)
plt.show()

# 2 equations with 2 unknowns, this is from taking the partial derivative for the Squared Error function:
# these two equations take the form of:
# aC + bD = E
# aD + bN = F

# where C, D, E, F take the substitution form of sums of Xi^2, Xi, XiYi, and Yi

# Apply the above functions to find a and b, which takes the form Y = aX + b (Linear Regression)
# Denominator was the same for both a and b, store it as a variable:
denom = X.dot(X) - X.mean() * X.sum()

# # Solving for a and b with algebra we get:
a = (X.dot(Y) - Y.mean() * X.sum()) / denom
b = (Y.mean() * X.dot(X) - X.mean() * X.dot(Y)) / denom

Y_hat = a * X + b

# Now plot everything:
plt.scatter(X, Y)
plt.plot(X, Y_hat, color='black')
plt.show()

# We chose the line of best fit for our given data.
# We can see how good our model is (R Squared)

# Calculation of R^2:
d1 = Y - Y_hat
d2 = Y - Y.mean()

# R2 = 1 - (sum of squared residual terms / sum of squared total)
R2 = 1 - (d1.dot(d1) / d2.dot(d2))
print(R2)
