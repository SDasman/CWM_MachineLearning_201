import matplotlib.pyplot as plt
import numpy as np
from keras import Sequential
from keras.layers import Dense
from keras import optimizers

# Make things reproducible but random
seed = 1130
np.random.seed(seed)

num_features = 1
max_samples = 1000

num_samples = [20, 200, max_samples]
num_epoch = [1, 5, 10]
num_units = [num_features, num_features * 10, num_features * 100]
num_layers = [1,3,5]
lr_varies = [1, 0.01, 0.001]

def createInput(num_samples, num_features):
    x = np.random.rand(num_samples, num_features) - 0.5
    #y = x*x*x
    y = x*x*x
    return x, y

def createModel(num_features, x, y, num_epochs=10, num_layers=3, num_units=100, learn_rate=0.01):
    # train
    model = Sequential()
    model.add(Dense(num_units, input_shape=(num_features,),activation="relu"))
    if (num_layers == 3):
        model.add(Dense(num_units,activation="relu"))
        model.add(Dense(num_units,activation="relu"))
    elif (num_layers == 4):
        model.add(Dense(num_units,activation="relu"))
        model.add(Dense(num_units,activation="relu"))
        model.add(Dense(num_units,activation="relu"))
        model.add(Dense(num_units,activation="relu"))
        
    model.add(Dense(1))
    
    rmsprop = optimizers.RMSprop(lr=learn_rate)
    model.compile(loss='mean_squared_error',
                  optimizer=rmsprop)

    hist = model.fit(x, y, epochs=num_epochs)
    return model, hist

def predict(model, num_samples, num_features):
    test_x, answ = createInput(num_samples, num_features)
    test_y = model.predict(test_x)
    return test_x, test_y

histories = []

plt.figure(1)
fig_num = 1

for i in num_epoch:    
    x,y = createInput(max_samples, num_features)
    model,h1 = createModel(num_features, x, y, i)
    histories.append(h1)
    test_x, test_y = predict(model, max_samples, num_features)
    plt.subplot(5,3,fig_num)
    plt.plot(x,y,'yo')
    plt.plot(test_x,test_y,'gx')
    plt.grid(True)
    plt.title("Epochs: "+str(i))    
    fig_num += 1

for i in num_samples:    
    x,y = createInput(i, num_features)
    model,h1 = createModel(num_features, x, y, 10)
    histories.append(h1)
    test_x, test_y = predict(model, i, num_features)
    plt.subplot(5,3,fig_num)
    plt.plot(x,y,'yo')
    plt.plot(test_x,test_y,'gx')
    plt.grid(True)
    plt.title("Samples: "+str(i))    
    fig_num += 1

for i in num_units:    
    x,y = createInput(max_samples, num_features)
    model,h1 = createModel(num_features, x, y, 10, 3, i)
    histories.append(h1)
    test_x, test_y = predict(model, max_samples, num_features)
    plt.subplot(5,3,fig_num)
    plt.plot(x,y,'yo')
    plt.plot(test_x,test_y,'gx')
    plt.grid(True)
    plt.title("Units: "+str(i))    
    fig_num += 1

for i in num_layers:    
    x,y = createInput(max_samples, num_features)
    model, h1 = createModel(num_features, x, y, 10, i)
    histories.append(h1)
    test_x, test_y = predict(model, max_samples, num_features)
    plt.subplot(5,3,fig_num)
    plt.plot(x,y,'yo')
    plt.plot(test_x,test_y,'gx')
    plt.grid(True)
    plt.title("Hidden Layers: "+str(i))    
    fig_num += 1


for i in lr_varies:    
    x,y = createInput(max_samples, num_features)
    model, h1 = createModel(num_features, x, y, 10, 3, 100, i)
    histories.append(h1)
    test_x, test_y = predict(model, max_samples, num_features)
    # plot(x-series, y-series, decorator)
    plt.subplot(5,3,fig_num)
    plt.plot(x,y,'yo')
    plt.plot(test_x,test_y,'gx')
    plt.grid(True)
    plt.title("LR: "+str(i))    
    fig_num += 1

plt.subplots_adjust(top=0.92, bottom=0.08, left=0.10, right=0.95, hspace=0.25,
                    wspace=0.25)

plt.figure(2)
fig_num = 1
for j in histories:
    plt.subplot(5 ,3, fig_num)
    plt.plot(j.history["loss"])
    fig_num+=1

plt.show()