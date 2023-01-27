from binance.client import Client
import csv
import random

def get_bitcoin_data():
    # Create a client instance
    try:
        client = Client("api_key", "api_secret")

        intervals = [Client.KLINE_INTERVAL_4HOUR,
                     Client.KLINE_INTERVAL_6HOUR,
                     Client.KLINE_INTERVAL_8HOUR,
                     Client.KLINE_INTERVAL_12HOUR,
                     Client.KLINE_INTERVAL_1DAY]
        interval = intervals[random.randint(0, len(intervals)-1)]
        print(f'intervals {interval}')
        # Retrieve the data
        klines = client.get_historical_klines("BTCUSDT", interval, "1 Jan, 2017")

        # Open a CSV file for writing
        with open("generated_data/bitcoin_data.csv", "w", newline="") as file:
            # Create a CSV writer
            writer = csv.writer(file)

            # Write the header row
            writer.writerow(["timestamp", "open", "high", "low", "close", "volume", "close_time", "quote_asset_volume", "num_trades", "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"])

            # Loop through the candle sticks
            for kline in klines:
                # Write the data to the CSV file
                writer.writerow(kline)
        return interval
    except:
        return 0

