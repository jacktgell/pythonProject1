import random
import tensorflow as tf
from binance_get_klines import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer

def shape_data(data, Y, output, rows_per_period = 20):
    # Assume data is a 2D array of shape (timesteps, features)
    #data, Y = Normalise_data(feature)

    # Number of periods to break the data into
    periods = data.shape[0]-rows_per_period

    # Initialize a list to store the periods
    periods_list = []
    Y_periods_list = []
    # Iterate through the periods
    for i in range(periods):
        # Extract the data for this period
        start_index = i
        end_index = start_index + rows_per_period
        period_data = data[start_index:end_index, :]

        # Append the data for this period to the list
        Y_period = Y[i]

        # Append the data and Y element for this period to their respective lists
        periods_list.append(period_data)
        Y_periods_list.append(Y_period)

    # Convert the lists of periods and Y elements to numpy arrays
    periods_array = np.array(periods_list)
    Y_periods_array = np.array(Y_periods_list)

    Y_converted = np.zeros((Y_periods_array.shape[0], 3))

    # Convert the Y values
    if output == 'softmax':
        for i in range(Y_periods_array.shape[0]):
            if all(Y_periods_array[i] == [1, 0, 0]):
                Y_converted[i, :] = [1, 0, 0]
            elif all(Y_periods_array[i] == [0, 1, 0]):
                Y_converted[i, :] = [0, 1, 0]
            else:
                Y_converted[i, :] = [0, 0, 1]
    else:
        Y_converted = Y_periods_array

    # Replace Y_periods_array with the converted values
    Y_periods_array = Y_converted

    return periods_array, Y_periods_array

def resample(x, y, output, percentage=0.5):
    _, counts = np.unique(y, axis=0, return_counts=True)
    Max, Min = np.amax(counts), np.amin(counts)
    diff = Max - Min
    Height = Min + diff*percentage

    def upsample_subarrays(x, y, target_val, Height):
        if x.shape[0] == Height:
            return x, y

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
    def upsample_subarrays_lin(x, y, target_val, Height):
        if target_val > 1.001:
            featidx = np.where((y > 1.001).all(axis=1))
        elif target_val < 0.999:
            featidx = np.where((y < 0.999).all(axis=1))
        else:
            featidx = np.where((y >= 0.999 and y <= 1.001).all(axis=1))
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
    targeta = [[0, 0, 1], [0, 1, 0], [1, 0, 0]] if output == 'softmax' else [0, 1, 2]
    for target in targeta:
        if output == 'softmax':
            sub_x, sub_y = upsample_subarrays(x, y, target, Height)
        else:
            sub_x, sub_y = upsample_subarrays_lin(x, y, target, Height)
        sub_x, sub_y = downsample_subarrays(sub_x, sub_y, Height)
        subarraysx.append(sub_x)
        subarraysy.append(sub_y)

    # Concatenate the arrays along the first axis (rows)
    x = np.concatenate([subarraysx[idx] for idx in range(len(subarraysx))], axis=0)
    y = np.concatenate([subarraysy[idx] for idx in range(len(subarraysy))], axis=0)

    return x, y


def set_label(close_value):
    if close_value > 1.01:
        return [1, 0, 0]    #buy
    elif close_value < 0.99:
        return [0, 1, 0]    #sell
    else:
        return [0, 0, 1]    #hold

def calculate_rsi(prices, n=14):
    deltas = prices.diff()
    seed = deltas[:n+1]
    up = seed[seed>=0].sum()/n
    down = -seed[seed<0].sum()/n
    rs = up/down
    rsi = np.zeros_like(prices)
    rsi[:n] = 100. - 100./(1.+rs)

    for i in range(n, len(prices)):
        delta = deltas[i-1] # cause the diff is 1 shorter

        if delta>0:
            upval = delta
            downval = 0.
        else:
            upval = 0.
            downval = -delta

        up = (up*(n-1) + upval)/n
        down = (down*(n-1) + downval)/n

        rs = up/down
        rsi[i] = 100. - 100./(1.+rs)

    return rsi


# Min-Max Scaling
def min_max_scaler(data):
    scaler = MinMaxScaler()
    return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Z-Score Normalization
def z_score_normalizer(data):
    scaler = StandardScaler()
    return pd.DataFrame(scaler.fit_transform(data), columns=data.columns)

# Log Transformation
def log_transformer(data):
    transformer = PowerTransformer(method='yeo-johnson')
    return pd.DataFrame(transformer.fit_transform(data), columns=data.columns)



def scale_data(df, x=0):

    # Select the columns you want to scale
    df = df.rename(columns={'close': 'label'})
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    target = df['label']
    df_scaled = df.drop(columns=['label'])

    if x == 0:
        scaled_df = min_max_scaler(df_scaled)
        normaliser = 'min_max_scaler'
    elif x == 1:
        scaled_df = z_score_normalizer(df_scaled)
        normaliser = 'z_score_normalizer'
    elif x == 2:
        scaled_df = log_transformer(df_scaled)
        normaliser = 'log_transformer'

    # Replace the original data with the scaled data
    return scaled_df, target, normaliser


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