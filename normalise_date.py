import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

def Normalise_data(feature):
    df = pd.read_csv('bitcoin_data.csv')

    df = df.sort_values("Date")
    df["Target"] = df["High"].shift(1)
    df["Target"] = df["Target"].div(df["Close"])
    df["Target"] = (df["Target"] - 1) * 100

    df.drop(columns=['Date'], inplace=True)
    df = df.dropna()
    Y = df["Target"]
    df.drop(columns=['Target', 'Unnamed: 0'], inplace=True)
    if feature != '':
        df.drop(columns=[feature], inplace=True)

    scaler = MinMaxScaler()
    scaler.fit(df)
    df_scaled = scaler.transform(df)
    df_scaled = np.array(df_scaled)
    Y = Y.values

    return df_scaled, Y