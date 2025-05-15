import pandas as pd
import numpy as np
import talib

def detect_fractal(series, n=2, fractal_type='green'):
    """
    Detects Williams Fractals.
    n: number of bars on each side to check for high/low.
    fractal_type: 'green' (bullish) or 'red' (bearish).
    Returns a boolean Series or array.
    Compatible with both pandas Series and numpy arrays (for backtesting.py).
    """
    # Convert to pandas Series if input is a numpy array
    if not isinstance(series, pd.Series):
        series = pd.Series(series)
    fractals = pd.Series(False, index=series.index)
    for i in range(n, len(series) - n):
        window = series.iloc[i-n : i+n+1]
        target_val = series.iloc[i]

        if fractal_type == 'green':  # Bullish fractal (low point)
            if target_val == window.min() and np.count_nonzero(window == target_val) == 1:
                fractals.iloc[i] = True
        elif fractal_type == 'red':  # Bearish fractal (high point)
            if target_val == window.max() and np.count_nonzero(window == target_val) == 1:
                fractals.iloc[i] = True
    # Shift result by n bars to ensure fractal is confirmed only after n future bars
    result = fractals.shift(n)
    # For backtesting.py, return as numpy array
    return result.values if hasattr(series, "values") else result


def compute_indicators(df, n_fractal=2):
    """Computes technical indicators needed for the strategy."""
    if 'close' not in df.columns:
        raise ValueError("DataFrame must contain 'close' column.")
    if len(df) < 100: # Ensure enough data for longest MA
        return None

    df = df.copy() # Avoid SettingWithCopyWarning
    df['SMA_20'] = talib.SMA(df['close'], timeperiod=20)
    df['SMA_50'] = talib.SMA(df['close'], timeperiod=50)
    df['SMA_100'] = talib.SMA(df['close'], timeperiod=100)

    # Use 'high' for red fractals and 'low' for green fractals
    if 'high' in df.columns and 'low' in df.columns:
        df['Green_Fractal'] = detect_fractal(df['low'], n=n_fractal, fractal_type='green')
        df['Red_Fractal'] = detect_fractal(df['high'], n=n_fractal, fractal_type='red')
    else:
        # Fallback if high/low are not available (less accurate)
        df['Green_Fractal'] = detect_fractal(df['close'], n=n_fractal, fractal_type='green')
        df['Red_Fractal'] = detect_fractal(df['close'], n=n_fractal, fractal_type='red')

    # Fill initial NaNs created by indicators/fractals
    df.fillna(method='bfill', inplace=True) # Back-fill to avoid lookahead bias in signals
    df.fillna(False, inplace=True) # Fill remaining fractal NaNs with False

    return df

def generate_signals(df):
    """Generates BUY, SELL, or HOLD signals based on strategy rules."""
    signals = pd.Series("HOLD", index=df.index)

    # Ensure indicators are present
    required_cols = ['SMA_20', 'SMA_50', 'SMA_100', 'Green_Fractal', 'Red_Fractal', 'close']
    if not all(col in df.columns for col in required_cols):
        print("Warning: Missing required columns for signal generation.")
        return signals # Return default HOLD signals

    # Vectorized conditions for speed
    long_condition_ma = (df['SMA_20'] > df['SMA_50']) & (df['SMA_50'] > df['SMA_100'])
    long_condition_pullback = (df['close'] < df['SMA_20']) | (df['close'] < df['SMA_50'])
    long_condition_fractal = df['Green_Fractal'] & (df['close'] > df['SMA_100'])
    buy_signals = long_condition_ma & long_condition_pullback & long_condition_fractal

    short_condition_ma = (df['SMA_100'] > df['SMA_50']) & (df['SMA_50'] > df['SMA_20'])
    short_condition_pullback = df['close'] > df['SMA_20']
    short_condition_fractal = df['Red_Fractal']
    sell_signals = short_condition_ma & short_condition_pullback & short_condition_fractal

    signals[buy_signals] = "BUY"
    signals[sell_signals] = "SELL"

    return signals

# --- Removed the old run_backtest function ---
# Example usage (can be run independently for testing indicators/signals)
# if __name__ == "__main__":
#     # Create dummy data or load from CSV
#     try:
#         df_hist = pd.read_csv('historical_data.csv', index_col='timestamp', parse_dates=True)
#         # Ensure columns are named correctly (e.g., 'close', 'high', 'low')
#         if 'Close' in df_hist.columns and 'close' not in df_hist.columns:
#              df_hist.rename(columns={'Close': 'close', 'High': 'high', 'Low': 'low', 'Open': 'open'}, inplace=True)

#         df_with_indicators = compute_indicators(df_hist)
#         if df_with_indicators is not None:
#             df_with_indicators['signal'] = generate_signals(df_with_indicators)
#             print(df_with_indicators[['close', 'SMA_20', 'SMA_50', 'SMA_100', 'Green_Fractal', 'Red_Fractal', 'signal']].tail(20))
#             print("\nSignal Counts:")
#             print(df_with_indicators['signal'].value_counts())
#         else:
#             print("Not enough data to compute indicators.")
#     except FileNotFoundError:
#         print("Error: historical_data.csv not found. Please provide a data file.")
#     except Exception as e:
#         print(f"An error occurred: {e}")
