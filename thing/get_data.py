import datetime
import pandas_datareader as pdr
import pandas as pd
import datetime
import requests

pd.set_option('display.max_columns', 20)
pd.options.display.max_columns = 20

today = datetime.date.today()
# Get the current date and time

df = pdr.get_data_yahoo("BTC-USD", start="2015-01-01", end=today)
df.drop('Adj Close', axis=1, inplace=True)
df = df.reset_index()
df = df.rename(columns={'index': 'date'})

current_date = datetime.datetime.now()
start_date = datetime.datetime(2018, 2, 1)
difference = current_date - start_date
response = requests.get(f"https://api.alternative.me/fng/?limit={difference.days-1}")
fear_greed_df = {}
for idx in range(len(dict(response.json())['data'])):
    fear_greed_df[f'{idx}'] = dict(response.json())['data'][idx]
fear_greed_df = pd.DataFrame(fear_greed_df)
fear_greed_df = fear_greed_df.transpose()
fear_greed_df['timestamp'] = pd.to_numeric(fear_greed_df['timestamp'], errors='coerce')
fear_greed_df['timestamp'] = pd.to_datetime(fear_greed_df['timestamp'], unit='s', origin='1970-01-01').dt.date


fear_greed_df.rename(columns={'timestamp': 'Date'}, inplace=True)
fear_greed_df['Date'] = fear_greed_df['Date'].astype(df['Date'].dtype)


df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)
df.reset_index(inplace=True)
df['200-week-ma'] = df['Close'].rolling(200*7).mean()
df['50-week-ma'] = df['Close'].rolling(50*7).mean()


# Calculate the change in price for each period
df["Change"] = df["Close"].diff()
# Calculate the gain and loss for each period
df["Gain"] = df["Change"].mask(df["Change"] < 0, 0)
df["Loss"] = df["Change"].mask(df["Change"] > 0, 0)
# Calculate the average gain and loss over the specified number of periods
periods = 14
df["Avg Gain"] = df["Gain"].ewm(span=periods, min_periods=periods).mean()
df["Avg Loss"] = df["Loss"].ewm(span=periods, min_periods=periods).mean().abs()
# Calculate the relative strength
df["Relative Strength"] = df["Avg Gain"] / df["Avg Loss"]
# Calculate the RSI
df["RSI"] = 100 - (100 / (1 + df["Relative Strength"]))

df = pd.merge(df, fear_greed_df, on='Date')
del fear_greed_df
df.rename(columns={'value': 'FG'}, inplace=True)
df.drop(columns=['value_classification', 'time_until_update'], inplace=True)


df = df.drop(columns=["Gain", "Loss"])

df.to_csv('bitcoin_data.csv')

del idx
del periods
del response
del today
del current_date
del difference
del start_date

print(df.tail())
print(df.head())

