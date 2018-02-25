import numpy as np
from keras import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from keras.callbacks import LambdaCallback

# Features are interesting data points we want to feed into the Neural Network as inputs.
num_features = 2

# This is the number of samples, with each sample containing a value per features, in our 
# data set.
num_samples = 10

# For demonstration purposes we will create some random data as input. The input will be
# an array of arrays. Each sample will be an array containing num_features worth of random
# data.
input = np.random.rand(num_samples, num_features)

# Our output will sum the three numbers together and divide the result in half
output = np.sum(input, axis=1) / 2

# Other Samples to try:
#       Div each feature by 3: output = np.divide(input, 5)
#       Average the features together:  output = np.mean(input, axis=1)

# We build a sequential NN with one layer
model = Sequential()
model.add(Dense(4, input_shape=(num_features,)))
model.add(Dense(1))

# This will show a summary of our model, if we want to see it
# model.summary()

# Stochastic gradient descent optimizer with some sensible defaults.
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mean_squared_error',
              optimizer=sgd)

# Collect the starting weights and define a function to collect them whenever
# it's called.
ws = []
ws.append(model.layers[0].get_weights())
ws.append(model.layers[1].get_weights())


def collect_weights(batch, logs):
    weights = model.layers[0].get_weights()
    ws.append(weights)
    weights = model.layers[1].get_weights()
    ws.append(weights)


# Define a callback to pass to the trainer to run at the end of every epoch
print_weights = LambdaCallback(on_epoch_end=collect_weights)

# Train the model.
model.fit(input, output, epochs=3, callbacks=[print_weights])

# Create a random test vector of features to see how well we predict, multiply it by 10
# so it's out of the range of the input data. This ensures it's never been seen
# during training.

print("-"*30)
test_data = np.random.rand(1, num_features) * 10
print("    TEST: ", test_data[0])
result = model.predict(np.asarray(test_data))
print("  RESULT: ", result[0])
print("EXPECTED: ", np.sum(test_data, axis=1)/2)


# Print out all the weights we collected
epoch_name = "initial"
for i in range(0, len(ws), 2):
    print("Weights for Layer 0", epoch_name)
    print(ws[i][0])
    print("Weights for Layer 1", epoch_name)
    print(ws[i+1][0])
    epoch_name = "After epoch " + str(int(i/2 + 1))
