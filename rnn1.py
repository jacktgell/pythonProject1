import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from Feature_testing import shape_data, random_int, random_float, recommend, rand_activation, rand_optimizer, resample
import json
from assemble_data import *
import random
from keras.layers import Dropout
from keras.regularizers import l1

from evaluate_model import test_model_on_target, test_model_on_target_lin

epochs = 5000

outputs = ['linear', 'softmax']
output = outputs[random_int(0,1)]
data, target, kline_interval, normaliser = generate_features(output)
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
                      'optimizer': [],
                      'normaliser': [],
                      'config': []}

for i in range(h_layer_range+1):
    epsilon_parameters[f'activation_{i}'] = []
    epsilon_parameters[f'neurons_{i}'] = []
    epsilon_parameters[f'l1_{i}'] = []
    epsilon_parameters[f'dropout_{i}'] = []


while 1:
    learning_rate = random_float(0.01, 0.001) if random_float(0,1)>0.1 else recommend('learning_rate', epsilon_parameters) if trial else 0.01
    data, target, kline_interval, normaliser = generate_features(output)
    epsilon_parameters['normaliser'].append(normaliser)
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
        data, target = resample(data, target, output, percentage=percentage)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

    model = tf.keras.Sequential()
    l1f = random_float(0, 0.2) if random_float(0, 1)>0.2 else recommend(f'l1_{0}', epsilon_parameters) if trial else 0.2
    neurons = random_int(30,1000) if random_float(0, 1)>0.2 else recommend(f'neurons_{0}', epsilon_parameters) if trial else 100

    epsilon_parameters[f'neurons_{0}'].append(neurons)
    epsilon_parameters[f'l1_{0}'].append(l1f)
    model.add(tf.keras.layers.LSTM(neurons, kernel_regularizer=l1(l1f), input_shape=(X_test.shape[1], X_test.shape[2]), return_sequences=True))

    n_layers = random_int(2,h_layer_range) if random_float(0, 1)>0.2 else recommend('n_layers', epsilon_parameters) if trial else 2
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
        metric = ['val_loss']
        print(output)
    else:
        model.add(tf.keras.layers.Dense(1, activation='linear'))
        loss='mean_squared_error'
        metric = ['val_loss']

    optimizer, str_opt = rand_optimizer(selection=('Rand'if random_float(0, 1)>0.7 else recommend('optimizer', epsilon_parameters)) if trial else 'Rand')

    model.compile(loss=loss, optimizer=optimizer)
    early_stop = tf.keras.callbacks.EarlyStopping(monitor=metric[0], patience=5)

    lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: learning_rate / (1 + epoch * 0.02))
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
    with open('generated_data/feature_tester.json', 'w') as outfile:
      # Use the json.dump() method to write the dictionary to the file
      json.dump(epsilon_parameters, outfile)