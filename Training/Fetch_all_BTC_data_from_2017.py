from binance.client import Client
import pandas as pd
import os
import time

# You can use empty API keys just to fetch historical klines
client = Client(api_key='', api_secret='')

# Set parameters
symbol = 'BTCUSDT'
interval = Client.KLINE_INTERVAL_1MINUTE
start_str = '1 Jan, 2017'

# Output CSV path
output_file = 'btc_1min_data_from_2017.csv'

# Fetch the klines (1 minute candles)
print("Fetching BTC/USDT data from Binance...")
klines = client.get_historical_klines(symbol, interval, start_str)

# Convert to dataframe
df = pd.DataFrame(klines, columns=[
    'timestamp', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_asset_volume', 'number_of_trades',
    'taker_buy_base_volume', 'taker_buy_quote_volume', 'ignore'
])

# Convert timestamp
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)

# Keep only useful columns
df = df[['open', 'high', 'low', 'close', 'volume']]
df = df.astype(float)

# Save to CSV
df.to_csv(output_file)
print(f"Data saved to {output_file}")
