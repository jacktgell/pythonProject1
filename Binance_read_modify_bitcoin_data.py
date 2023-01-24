import random

from binance_get_klines import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler, PowerTransformer

'''
['timestamp', 'open', 'high', 'low', 'close', 'volume', 'close_time',
       'quote_asset_volume', 'num_trades', 'taker_buy_base_asset_volume',
       'taker_buy_quote_asset_volume', 'ignore', '200ma', '50ma', 'label',
       'rsi']
'''


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
    target = df['label']
    df_scaled = df.drop(columns=['label'])

    if x == 0:
        scaled_df = min_max_scaler(df_scaled)
    elif x == 1:
        scaled_df = z_score_normalizer(df_scaled)
    elif x == 2:
        scaled_df = log_transformer(df_scaled)

    # Replace the original data with the scaled data
    return scaled_df, target

def generate_features():
    kline_interval = get_bitcoin_data()
    df = pd.read_csv('bitcoin_data.csv')

    df.drop(columns=['timestamp', 'ignore', 'close_time'], inplace=True)


    # Create a new column with the moving average
    num_samples = 200
    df["200ma"] = df['close'].rolling(num_samples).mean()
    num_samples = 50
    df["50ma"] = df['close'].rolling(num_samples).mean()
    # Shift the values in each column down by one row

    # Create a new dataframe with NaN values
    nan_df = pd.DataFrame([[pd.np.nan] * len(df.columns)], columns=df.columns)
    # Concatenate the NaN dataframe with the original dataframe
    df_shifted = pd.concat([nan_df, df]).reset_index(drop=True)
    # Divide the shifted DataFrame by the original DataFrame
    df = df_shifted / df

    label_df = df['close'].apply(set_label)

    df['rsi'] = calculate_rsi(df['close'])
    df.drop(columns=['close'], inplace=True)
    df = pd.concat([df, label_df], axis=1)
    df.dropna(inplace=True)
    df, target = scale_data(df, x=random.randint(0, 2))

    return df, target, kline_interval









