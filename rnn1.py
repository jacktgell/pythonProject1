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

def random_int(a, b):
    return random.randint(a, b)

def random_float(a, b):
    return random.uniform(a, b)

def rand_activation():
    activations = ['sigmoid',
                   'relu',
                   'tanh',
                   'lstm',
                   'GRU']
    return activations[random_int(0, len(activations)-1)]

def rand_optimizer():
    opt = [tf.keras.optimizers.Adam(),
            tf.keras.optimizers.SGD(),
            tf.keras.optimizers.RMSprop(),
            tf.keras.optimizers.Adadelta()]
    return opt[random_int(0, len(opt)-1)]

epochs = 50

data, target, kline_interval = generate_features()
features = data.columns
best_sofar = 0

es = 0
feature_tester = {}

while 1:
    learning_rate = random_float(0.01, 0.001)
    losses = []
    data, target, kline_interval = generate_features()
    feature = features[random_int(0, len(features)-1)] if random_int(0, len(features) - 1) == 0 else None
    if feature:
        data.drop(labels=feature, inplace=True,  axis=1)
    period = random_int(20, 300)
    data, target = shape_data(data.values, target.values, rows_per_period=period)

    percentage = random_float(0.1, 0.9)
    data, target = resample(data, target, percentage=percentage)

    X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2)

    model = tf.keras.Sequential()
    activation = rand_activation()
    l1f = random_float(0, 0.3)
    if random_float(0, 0.5)<0.5:
        model.add(tf.keras.layers.GRU(random_int(30,300), kernel_regularizer=l1(l1f), input_shape=(X_test.shape[1], X_test.shape[2]), return_sequences=True, return_state=True))
    else:
        model.add(tf.keras.layers.LSTM(random_int(100,300), kernel_regularizer=l1(l1f), input_shape=(X_test.shape[1], X_test.shape[2]), return_sequences=True, return_state=True))

    for i in range(random_int(3,18)):

        if 'GRU' == activation:
            model.add(tf.keras.layers.GRU(random_int(30,300), kernel_regularizer=l1(l1f), return_sequences=True, return_state=True))
            model.add(Dropout(random_float(0,0.4)))
        if 'lstm' == activation:
            model.add(tf.keras.layers.LSTM(random_int(30,300), kernel_regularizer=l1(l1f), return_sequences=True, return_state=True))
            model.add(Dropout(random_float(0,0.4)))
        else:
            model.add(tf.keras.layers.Dense(random_int(100,1000), kernel_regularizer=l1(l1f), activation=activation)) # returns a sequence of vectors of dimension 32
            model.add(Dropout(random_float(0,0.4)))

    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(3, activation='softmax'))


    optimizer = rand_optimizer()

    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    early_stop = tf.keras.callbacks.EarlyStopping(monitor=f'accuracy', patience=5)

    #lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-2 - 1e-4*epoch*2)
    batch_size = random_int(8, 128)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test), callbacks=[early_stop], verbose=1)

    buy_dict = test_model_on_target(X_test, y_test, model, [1, 0, 0])
    sell_dict = test_model_on_target(X_test, y_test, model, [0, 1, 0])
    hold_dict = test_model_on_target(X_test, y_test, model, [0, 0, 1])
    loss_acc_dict = model.evaluate(X_test, y_test, return_dict=True)

    print(f'accuracies buy: {buy_dict["accuracy"]} sell: {sell_dict["accuracy"]} hold: {hold_dict["accuracy"]} all: {loss_acc_dict["accuracy"]}')

    es += 1
    feature_tester[f'Trail {es}'] = {'all data': loss_acc_dict,
                               'model': model.get_config(),
                               'buy': buy_dict,
                               'sell': sell_dict,
                               'hold': hold_dict,
                               'up sample percentage': percentage,
                               'learning_rate': learning_rate,
                                'batch_size': batch_size,
                                'feature': feature,
                                'periods': period,
                                'kline_interval': kline_interval}

    best_sofar = max(loss_acc_dict["accuracy"], best_sofar)
    print(f'best so far is: {best_sofar}')

    print('x'*100)

    # Open a file to write the JSON data to
    with open('feature_tester.json', 'w') as outfile:
      # Use the json.dump() method to write the dictionary to the file
      json.dump(feature_tester, outfile)