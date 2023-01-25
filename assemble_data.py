import ccxt
import pandas as pd
import talib

# Initialize the ccxt exchange object
exchange = ccxt.binance()

# Download historical data for Bitcoin
ohlcv = exchange.fetch_ohlcv('BTC/USDT', '1d')

# Convert the data to a DataFrame
df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

# Apply technical indicators to the data
df['sma'] = talib.SMA(df['close'])
df['rsi'] = talib.RSI(df['close'])
df['macd'], df['macdsignal'], df['macdhist'] = talib.MACD(df['close'])

#print the dataframe
print(df)
