import math
from random import random

from normalise_date import *
import numpy as np

def shape_data(data, Y, rows_per_period = 20):
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
    for i in range(Y_periods_array.shape[0]):
        if all(Y_periods_array[i] == [1, 0, 0]):
            Y_converted[i, :] = [1, 0, 0]
        elif all(Y_periods_array[i] == [0, 1, 0]):
            Y_converted[i, :] = [0, 1, 0]
        else:
            Y_converted[i, :] = [0, 0, 1]

    # Replace Y_periods_array with the converted values
    Y_periods_array = Y_converted

    return periods_array, Y_periods_array

def rand_el(my_list, lower=0):
    return my_list[random.randint(lower, len(my_list)-1)]

def rand_int(upper, lower, interval=1):
    selection = random.randint(lower, upper)
    selection = math.ceil(selection/interval)*interval
    return selection


