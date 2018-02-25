import numpy as np
from keras import Sequential, optimizers
from keras.layers import Dense

# Set the random seed to something so we get the same numbers each time.
# seed is the same random numbers every time.
np.random.seed(113017)

# Features are interesting data points we want to feed into the Neural Network as inputs.
num_features = 2

# This is the number of samples, with each sample containing a value per features, in our 
# data set.
num_samples = 10000
# 10 samples each with 2 features

# For demonstration purposes we will create some random data as input. The input will be
# an array of arrays. Each sample will be an array containing num_features worth of random
# data.
data_input = np.random.rand(num_samples, num_features)

# Our output will sum the three numbers together and divide the result in half
output = np.sum(data_input, axis=1) / 2

# Other Samples to try:
#       Div each feature by 3: output = np.divide(input, 5)
#       Average the features together:  output = np.mean(input, axis=1)

# We build a sequential NN with one layer
model = Sequential()
# adding the neural network. Dense means all the nodes from the previous layers are connected
# to all the nodes of the current layer
# 4 is the number of nodes. input_shape is actually a tuple, we need the comma.
# the number of samples is not explicitly coded in ever.
# it expects the dimension of the data (width of 2 for us)
model.add(Dense(4, input_shape=(num_features,)))
# this second dense is the same shape of the output
# in a dense node every node returns 1 output
# Dense(2) would create 2 results, with more results the weights change
# Keras automatically assigns different weights randomly
model.add(Dense(1))

# This will show a summary of our model, if we want to see it
# model.summary()

# Stochastic gradient descent optimizer with some sensible defaults.
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error',
              optimizer=sgd)

# Train the model.
model.fit(data_input, output, epochs=1)

# Create a random test vector of features to see how well we predict, multiply it by 10
# so it's out of the range of the input data. This ensures it's never been seen
# during training.

model.save("My_Model.h5")

print("-" * 30)
test_data = np.random.rand(1, num_features) * 10
print("    TEST: ", test_data[0])
# model.predict() run this new data on the trained neural network
result = model.predict(np.asarray(test_data))
print("  RESULT: ", result[0])
print("EXPECTED: ", np.sum(test_data, axis=1)/2)
