import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib
from ScalpingStrategy1 import detect_fractal
import os

# --- Strategy Class for backtesting.py ---
class ScalpingStrategy1BT(Strategy):
    n_fractal = 2

    def init(self):
        # Compute indicators using talib
        close = self.data.Close
        high = self.data.High
        low = self.data.Low

        self.sma20 = self.I(talib.SMA, close, 20)
        self.sma50 = self.I(talib.SMA, close, 50)
        self.sma100 = self.I(talib.SMA, close, 100)

        # Fractals
        self.green_fractal = self.I(detect_fractal, low, self.n_fractal, 'green')
        self.red_fractal = self.I(detect_fractal, high, self.n_fractal, 'red')

    def next(self):
        # Get current values
        i = len(self.data.Close) - 1
        close = self.data.Close[-1]
        sma20 = self.sma20[-1]
        sma50 = self.sma50[-1]
        sma100 = self.sma100[-1]
        green_fractal = self.green_fractal[-1]
        red_fractal = self.red_fractal[-1]

        # --- Long Entry ---
        long_ma = sma20 > sma50 and sma50 > sma100
        long_pullback = close < sma20 or close < sma50
        long_fractal = green_fractal and close > sma100
        if long_ma and long_pullback and long_fractal:
            if not self.position.is_long:
                self.position.close()
                self.buy(size=1)  # Trade 1 unit of the scaled asset (e.g., 1 cBTC)

        # --- Short Entry ---
        short_ma = sma100 > sma50 and sma50 > sma20
        short_pullback = close > sma20
        short_fractal = red_fractal
        if short_ma and short_pullback and short_fractal:
            if not self.position.is_short:
                self.position.close()
                self.sell(size=1)  # Trade 1 unit of the scaled asset (e.g., 1 cBTC)

        # --- Exit on opposite signal ---
        # backtesting.py closes on opposite signal by default if you call buy()/sell()

# --- Main Execution ---
if __name__ == "__main__":
    # Ensure the Backtests folder exists for plot output
    os.makedirs("Backtests", exist_ok=True)
    
    # Load data
    CSV_FILE = 'btc_minute_data.csv'
    try:
        df = pd.read_csv(CSV_FILE, parse_dates=True)
    except FileNotFoundError:
        print(f"Error: {CSV_FILE} not found. Please place your historical data CSV in the root directory.")
        exit(1)

    # Rename columns to match backtesting.py requirements
    rename_map = {
        'timestamp': 'Date',
        'Timestamp': 'Date',
        'Date': 'Date',
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume',
        'Open': 'Open',
        'High': 'High',
        'Low': 'Low',
        'Close': 'Close',
        'Volume': 'Volume'
    }
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Ensure required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"CSV is missing required columns: {missing_cols}. Available columns: {df.columns.tolist()}")

    # Ensure Date column is datetime and set as index
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    else:
        raise ValueError("CSV must have a 'Date' or 'timestamp' column.")

    # Drop rows with NaN in required columns
    df = df.dropna(subset=required_cols)

    # Scale prices to trade a fraction of BTC (e.g., 0.01 BTC as 1 unit)
    # This makes 1 unit of the asset in backtesting correspond to 0.01 BTC.
    price_scalar = 0.01 
    df['Open'] = df['Open'] * price_scalar
    df['High'] = df['High'] * price_scalar
    df['Low'] = df['Low'] * price_scalar
    df['Close'] = df['Close'] * price_scalar

    # Run backtest (removed unsupported 'size' parameter)
    bt = Backtest(
        df,
        ScalpingStrategy1BT,
        cash=10_000,
        commission=0.0005,
        exclusive_orders=True,
        trade_on_close=True
    )

    stats = bt.run()
    print(stats)

    # Plot results (will open a window in VSCode or save to HTML if not interactive)
    try:
        bt.plot(filename='Backtests/backtest_plot.html', open_browser=False)
        print("Plot saved to Backtests/backtest_plot.html")
    except Exception as e:
        print("Plotting failed:", e)
        print("Trying to plot interactively...")
        bt.plot()

    print("\n--- Backtest Complete ---")
    print("Backtest process is fully set up and ready for use.")
    print("To test a new strategy, define a new class inheriting from backtesting.Strategy (e.g., in ScalpingStrategy1.py), then import and use it in backtest.py.")
    print("Run this script to backtest and view results in Backtests/backtest_plot.html.")
    print("No further setup is needed. Happy coding and algo trading!")
