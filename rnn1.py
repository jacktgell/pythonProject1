import pandas as pd
import tensorflow as tf
from scipy.signal import resample
from sklearn.model_selection import train_test_split
from Feature_testing import shape_data
import json
from Binance_read_modify_bitcoin_data import *
import random
from keras.layers import Dropout
from keras.regularizers import l1

def get_dataset_balance(arr):
    unique, counts = np.unique(arr, axis=0, return_counts=True)
    print("Unique rows: ", unique)
    print("Frequencies: ", counts)

def resample(x, y, percentage=0.5):
    _, counts = np.unique(y, axis=0, return_counts=True)
    Max, Min = np.amax(counts), np.amin(counts)
    diff = Max - Min
    Height = Min + diff*percentage

    def upsample_subarrays(x, y, target_val, Height):
        featidx = np.where((y == target_val).all(axis=1))
        sub_x = [np.array(x[idx]) for idx in featidx][0]
        sub_y = [np.array(y[idx]) for idx in featidx][0]

        while(sub_x.shape[0]<Height):
            # Select a random index from the original array
            random_index = np.random.randint(0, sub_x.shape[0])

            # Select the same row from both arrays using the random index
            random_row1 = sub_x[random_index, :]
            random_row2 = sub_y[random_index, :]

            # Append the random rows to both arrays
            sub_x = np.append(sub_x, [random_row1], axis=0)
            sub_y = np.append(sub_y, [random_row2], axis=0)
        return sub_x, sub_y

    def downsample_subarrays(subx, suby, Height):
        if subx.shape[0]>Height:
            while(subx.shape[0]>Height):
                # Select a random index from the original array
                random_index = np.random.randint(0, subx.shape[0])

                # Delete the same row from both arrays using the random index
                subx = np.delete(subx, random_index, axis=0)
                suby = np.delete(suby, random_index, axis=0)
        return subx, suby

    subarraysy, subarraysx = [], []
    for target in [[0, 0, 1], [0, 1, 0], [1, 0, 0]]:
        sub_x, sub_y = upsample_subarrays(x, y, target, Height)
        sub_x, sub_y = downsample_subarrays(sub_x, sub_y, Height)
        subarraysx.append(sub_x)
        subarraysy.append(sub_y)

    # Concatenate the arrays along the first axis (rows)
    x = np.concatenate([subarraysx[idx] for idx in range(len(subarraysx))], axis=0)
    y = np.concatenate([subarraysy[idx] for idx in range(len(subarraysy))], axis=0)

    return x, y

def test_model_on_target(x, y, model, target_val):
    featidx = np.where((y == target_val).all(axis=1))
    sub_x = [np.array(x[idx]) for idx in featidx][0]
    sub_y = [np.array(y[idx]) for idx in featidx][0]
    return model.evaluate(sub_x, sub_y, return_dict=True)
def test_model_on_target_lin(x, y, model):
    predictions = model.predict(x)
    accuracy = {'buy':[], 'sell':[], 'hold':[]}

    try:
        for result in zip(predictions, y):
            if result[1] > 1.001:
                if result[0] > 1.001:
                    accuracy['buy'].append(1)
                else:
                    accuracy['buy'].append(0)
            elif result[1] < 0.999:
                if result[0] < 0.999:
                    accuracy['sell'].append(1)
                else:
                    accuracy['sell'].append(0)
            else:
                if result[0] > 0.999 and result[1] < 1.001:
                    accuracy['hold'].append(1)
                else:
                    accuracy['hold'].append(0)
    except:
        a = 1

    accuracy['buy'] = sum(accuracy['buy'])/len(accuracy['buy']) if len(accuracy['buy']) else 'NA'
    accuracy['sell'] = sum(accuracy['sell'])/len(accuracy['sell']) if len(accuracy['sell']) else 'NA'
    accuracy['hold'] = sum(accuracy['hold'])/len(accuracy['hold']) if len(accuracy['hold']) else 'NA'
    return accuracy


def random_int(a, b):
    return random.randint(a, b)

def random_float(a, b):
    return random.uniform(a, b)

def rand_activation():
    activations = ['sigmoid',
                   'relu',
                   'tanh',
                   'lstm']
    return activations[random_int(0, len(activations)-1)]

def rand_optimizer(selection = 'Rand'):
    opt = [tf.keras.optimizers.Adam(),
            tf.keras.optimizers.SGD(),
            tf.keras.optimizers.RMSprop(),
            tf.keras.optimizers.Adadelta()]
    string_opt = ['Adam', 'SGD', 'RMSprop', 'Adadelta']

    if selection == 'Rand':
        pointer = random_int(0, len(opt)-1)
        object_opt = opt[pointer]
        str_opt = string_opt[pointer]
    else:
        pointer = string_opt.index(selection)
        str_opt = selection
        object_opt = opt[pointer]
    return object_opt, str_opt


def recommend(metric, epsilon_parameters):

    # Find the highest accuracy score
    max_accuracy = max(epsilon_parameters['accuracy'])
    # Find the index of the trial with the highest accuracy score
    max_accuracy_index = epsilon_parameters['accuracy'].index(max_accuracy)
    return epsilon_parameters[metric][max_accuracy_index]

epochs = 50

outputs = ['linear', 'softmax']
output = outputs[random_int(0,1)]
data, target, kline_interval = generate_features(output)
features = data.columns
best_sofar = 0
trial = 0
h_layer_range = 20

epsilon_parameters = {'feature': [],
                      'period': [],
                      'percentage': [],
                      'neurons': [],
                      'n_layers': [],
                      'output': [],
                      'accuracy': [0],
                      'learning_rate': [],
                      'batch_size': [],
                      'kline_interval': [],
                      'optimizer': []}
for i in range(h_layer_range+1):
    epsilon_parameters[f'activation_{i}'] = []
    epsilon_parameters[f'neurons_{i}'] = []
    epsilon_parameters[f'l1_{i}'] = []
    epsilon_parameters[f'dropout_{i}'] = []


while 1:
    learning_rate = random_float(0.01, 0.001) if random_float(0,1)>0.1 else recommend('learning_rate', epsilon_parameters) if trial else 0.01
    data, target, kline_interval = generate_features(output)
    epsilon_parameters['kline_interval'].append(kline_interval)
    feature = features[random_int(0, len(features)-1)] if random_int(0, len(features) - 1) == 0 else 'None'
    epsilon_parameters['feature'].append(feature)
    output = outputs[random_int(0, 1)] if random_float(0,1)>0.5 else recommend('output', epsilon_parameters) if trial else 'linear'
    epsilon_parameters['output'].append(output)

    if feature != 'None':
        data.drop(labels=feature, inplace=True,  axis=1)

    period = random_int(20, 300) if random_float(0, 1)>0.2 else recommend('period', epsilon_parameters) if trial else 100
    epsilon_parameters['period'].append(period)

    data, target = shape_data(data.values, target.values, output, rows_per_period=period)

    percentage = random_float(0.1, 0.9) if random_float(0, 1)>0.2 else recommend('percentage', epsilon_parameters) if trial else 0.5
    epsilon_parameters['percentage'].append(percentage)
    if output == 'softmax':
        data, target = resample(data, target, percentage=percentage)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

    model = tf.keras.Sequential()
    l1f = random_float(0, 0.2) if random_float(0, 1)>0.2 else recommend(f'l1_{0}', epsilon_parameters) if trial else 0.2
    neurons = random_int(30,1000) if random_float(0, 1)>0.2 else recommend(f'neurons_{0}', epsilon_parameters) if trial else 100

    epsilon_parameters[f'neurons_{0}'].append(neurons)
    epsilon_parameters[f'l1_{0}'].append(l1f)
    model.add(tf.keras.layers.LSTM(neurons, kernel_regularizer=l1(l1f), input_shape=(X_test.shape[1], X_test.shape[2]), return_sequences=True))

    n_layers = random_int(1,h_layer_range) if random_float(0, 1)>0.2 else recommend('n_layers', epsilon_parameters) if trial else 10
    epsilon_parameters['n_layers'].append(n_layers)

    for i in range(n_layers):
        activation = rand_activation()
        epsilon_parameters[f'activation_{i+1}'].append(activation)
        neurons = (random_int(30,2000) if random_float(0, 1)>0.2 else recommend(f'neurons_{i+1}', epsilon_parameters)) if trial else 100
        epsilon_parameters[f'neurons_{i+1}'].append(neurons)
        l1f = random_float(0, 0.2) if random_float(0, 1)>0.2 else recommend(f'l1_{i+1}', epsilon_parameters) if trial else 0
        epsilon_parameters[f'l1_{i+1}'].append(l1f)
        dropout = random_float(0,0.4) if random_float(0, 1)>0.2 else recommend(f'dropout_{i+1}', epsilon_parameters) if trial else 0
        epsilon_parameters[f'dropout_{i+1}'].append(dropout)

        if 'lstm' == activation:
            model.add(tf.keras.layers.LSTM(neurons, kernel_regularizer=l1(l1f), return_sequences=True))
        else:
            model.add(tf.keras.layers.Dense(neurons, kernel_regularizer=l1(l1f), activation=activation)) # returns a sequence of vectors of dimension 32
        model.add(Dropout(dropout))

    model.add(tf.keras.layers.Flatten())
    if output == 'softmax':
        model.add(tf.keras.layers.Dense(3, activation='softmax'))
        loss = 'categorical_crossentropy'
        metric = ['accuracy']
    else:
        model.add(tf.keras.layers.Dense(1, activation='linear'))
        loss='mean_squared_error'
        metric = ['val_loss']

    optimizer, str_opt = rand_optimizer(selection=('Rand'if random_float(0, 1)>0.7 else recommend('optimizer', epsilon_parameters)) if trial else 'Rand')

    model.compile(loss=loss, optimizer=optimizer)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor=metric[0], patience=5)

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: learning_rate - (learning_rate/100)*epoch)
    batch_size = (random_int(8, 128) if random_float(0, 1)>0.5 else recommend('batch_size', epsilon_parameters)) if trial else 32
    model.fit(X_train, y_train, verbose=1, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[early_stop, lr_schedule])

    if output == 'softmax':
        buy_dict = test_model_on_target(X_test, y_test, model, [1, 0, 0])
        sell_dict = test_model_on_target(X_test, y_test, model, [0, 1, 0])
        hold_dict = test_model_on_target(X_test, y_test, model, [0, 0, 1])
        loss_acc_dict = model.evaluate(X_test, y_test, return_dict=True)
        acc_dict = loss_acc_dict["accuracy"]
    else:
        accuracy = test_model_on_target_lin(X_test, y_test, model)
        buy_dict = accuracy['buy']
        sell_dict = accuracy['sell']
        hold_dict = accuracy['hold']
        acc_dict = (accuracy['buy']+accuracy['sell']+accuracy['hold'])/3

    if trial > 0:
        epsilon_parameters['accuracy'].append(acc_dict)
    else:
        epsilon_parameters['accuracy'][0] = acc_dict

    trial += 1
    epsilon_parameters['config'].append(model.get_config())
    best_sofar = max(epsilon_parameters['accuracy'], best_sofar)

    print(f'trial :{trial}')
    print(f'accuracies buy: {buy_dict} sell: {sell_dict} hold: {hold_dict} all: {acc_dict}')
    print(f'best so far is: {best_sofar}')
    print('x'*100)

    # Open a file to write the JSON data to
    with open('feature_tester.json', 'w') as outfile:
      # Use the json.dump() method to write the dictionary to the file
      json.dump(epsilon_parameters, outfile)