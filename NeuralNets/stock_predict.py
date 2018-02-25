import pandas as pd
import numpy as np
from datetime import datetime
from keras import Sequential
from keras.layers import Dense
from keras import optimizers


# Parse the time from a string so we can merge on dates.
def parse(x):
    return datetime.strptime(x, '%m/%d/%Y')


# Set the seed so that we have a reproducible output.
np.random.seed(113017)

# Read in the CSV file for the S&P 500 (data source: yahoo)
spxData = pd.read_csv("SPX.csv", parse_dates=["Date"], date_parser=parse)

# We want to predict if tomorrow's close will be higher than today's close. To
# do that we'll shift all of the data back a day and compare the shifted against
# the current. Effectively this is showing us:
#     Tomorrow's Close > Today's Close
spxData["nextDayGreen"] = (spxData["Adj Close"].shift(-1) > spxData["Adj Close"]) * 1

# To do this we'll compute some indicators to use as input data that we think will
# help our chances of predicting the market.

# Compute the 200 day moving average
spxData["200sma"] = spxData["Adj Close"].rolling(window=200).mean()
# Compute the 50 day moving average
spxData["50sma"] = spxData["Adj Close"].rolling(window=50).mean()
# Compute the 50 day moving volume average
spxData["50smavol"] = spxData["Volume"].rolling(window=50).mean()

# Remove columns with na values as they break training.
spxData.dropna(inplace=True)

# These are the columns we are interested in for training
input_cols = ["Adj Close", "Volume", "200sma", "50sma", "50smavol"]

# Now build the Training / Test data sets by splitting off this month.
# The last 20 rows of the Data Frame represent Nov, 2017. We'll use that to test with and the
# rest will be used for training
spxTestData = spxData[-20:]
spxData = spxData[0:-20]

# Convert them into NumPy arrays for processing
train_input_data_ = np.asarray(spxData[input_cols])
train_output_data = np.asarray(spxData["nextDayGreen"])

test_input_data = np.asarray(spxTestData[input_cols])
test_output_data = np.asarray(spxTestData["nextDayGreen"])

# We build a sequential NN 
model = Sequential()
model.add(Dense(units=20, input_shape=(len(input_cols),), kernel_initializer="uniform", activation="tanh"))
# If we over fit, can regularize by dropping some samples
# model.add(Dropout(0.1))
model.add(Dense(units=300, kernel_initializer="uniform", activation="tanh"))
model.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid"))

# Stochastic gradient descent optimizer with some sensible defaults.
sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

# Uncomment this to view the model summary
# model.summary()

# Train the model.
model.fit(train_input_data_, train_output_data, epochs=5)

results = model.evaluate(test_input_data, test_output_data)
print("Model evaluation results (loss, acc): " + str(results))
