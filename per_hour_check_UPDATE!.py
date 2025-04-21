import yfinance as yf
import pandas as pd
import ta
import matplotlib.pyplot as plt

# Download BTC-USD hourly data
print("Downloading data...")
btc = yf.download('BTC-USD', start='2020-01-01', interval='1h')

# Drop missing data
btc.dropna(inplace=True)

# Add indicators
btc['rsi'] = ta.momentum.RSIIndicator(btc['Close']).rsi()
btc['macd'] = ta.trend.MACD(btc['Close']).macd()
btc['ema_20'] = ta.trend.EMAIndicator(btc['Close'], window=20).ema_indicator()

# Show sample
print(btc.tail())

# Plot Close + EMA
btc[['Close', 'ema_20']].plot(figsize=(12, 6), title='BTC Price + EMA 20')
plt.show()
