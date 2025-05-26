import pandas as pd
import numpy as np
import talib
import logging # Added for logging

# --- Logger Setup ---
# Note: The logger is configured in the main backtest.py script.
# This allows the DEBUG_LOGGING toggle in backtest.py to control logging for this module too.
logger = logging.getLogger(__name__)

# --- Strategy Configuration ---
DEFAULT_TRADE_SIZE_PERCENTAGE = 0.3 # Default trade size as a percentage of capital (e.g., 0.02 = 2%)

def detect_fractal(series, n=2, fractal_type='green'):
    """
    Detects Williams Fractals using a more vectoriczed approach.
    n: number of bars on each side to check for high/low.
    fractal_type: 'green' (bullish) or 'red' (bearish).
    Returns a boolean numpy array.
    """
    if not isinstance(series, pd.Series):
        series = pd.Series(series) # Ensure input is a pandas Series

    # Using rolling windows for min/max is more efficient than iterating
    # Pad with NaN to handle edges correctly with rolling window, then drop NaNs
    # The window size is 2*n + 1
    window_size = 2 * n + 1

    if fractal_type == 'green': # Bullish fractal (low point)
        # Find points that are the minimum in their rolling window
        is_min = series == series.rolling(window=window_size, center=True, min_periods=window_size).min()
        # Check uniqueness: Ensure the target_val is unique minimum in the window
        # This is harder to vectorize directly with rolling count of min values.
        # We can approximate or stick to a loop for this specific uniqueness check if critical,
        # or accept that multiple identical mins in a window might be flagged.
        # For now, let's keep the original loop for uniqueness, but optimize other parts.
    elif fractal_type == 'red': # Bearish fractal (high point)
        # Find points that are the maximum in their rolling window
        is_max = series == series.rolling(window=window_size, center=True, min_periods=window_size).max()
    else:
        raise ValueError("fractal_type must be 'green' or 'red'")

    # The original loop for uniqueness check is retained for accuracy, as pure vectorization of this specific
    # np.count_nonzero(window == target_val) == 1 part is complex with rolling operations.
    # However, the primary detection (min/max in window) can be faster.
    # Let's refine the loop for clarity and ensure it works with the pre-calculated is_min/is_max.

    fractals = pd.Series(False, index=series.index)
    # Iterate only over points that could potentially be fractals (already identified as min/max in window)
    # This reduces iterations compared to checking every point.

    # We still need the loop for the uniqueness condition (np.count_nonzero == 1)
    # and to ensure the fractal is strictly the min/max.
    for i in range(n, len(series) - n):
        target_val = series.iloc[i]
        window = series.iloc[i-n : i+n+1]

        if fractal_type == 'green':
            if target_val == window.min() and np.count_nonzero(window.values == target_val) == 1:
                fractals.iloc[i] = True
        elif fractal_type == 'red':
            if target_val == window.max() and np.count_nonzero(window.values == target_val) == 1:
                fractals.iloc[i] = True

    # Shift result by n bars to ensure fractal is confirmed only after n future bars
    # Original boolean series `fractals` is converted to float, shifted (introducing NaNs),
    # NaNs are filled with 0.0 (representing False), and then converted back to boolean.
    # This avoids FutureWarning with fillna on object arrays.
    result = fractals.astype(float).shift(n).fillna(0.0).astype(bool)

    logger.debug(f"Fractal detection complete for type '{fractal_type}'. Found {result.sum()} fractals.")
    return result.values # Return as numpy array for backtesting.py


def compute_indicators(df, n_fractal=2):
    """Computes technical indicators needed for the strategy."""
    logger.debug(f"Computing indicators for DataFrame with shape {df.shape}. n_fractal={n_fractal}")
    if 'close' not in df.columns:
        logger.error("DataFrame must contain 'close' column for compute_indicators.")
        raise ValueError("DataFrame must contain 'close' column.")
    if len(df) < 100: # Ensure enough data for longest MA
        logger.warning("Not enough data to compute all indicators (less than 100 rows).")
        return None # Or handle differently, e.g., compute what's possible

    df_copy = df.copy() # Avoid SettingWithCopyWarning
    df_copy['SMA_20'] = talib.SMA(df_copy['close'], timeperiod=20)
    df_copy['SMA_50'] = talib.SMA(df_copy['close'], timeperiod=50)
    df_copy['SMA_100'] = talib.SMA(df_copy['close'], timeperiod=100)

    if 'high' in df_copy.columns and 'low' in df_copy.columns:
        df_copy['Green_Fractal'] = detect_fractal(df_copy['low'], n=n_fractal, fractal_type='green')
        df_copy['Red_Fractal'] = detect_fractal(df_copy['high'], n=n_fractal, fractal_type='red')
    else:
        logger.warning("'high' or 'low' columns not found. Using 'close' for fractal detection (less accurate).")
        df_copy['Green_Fractal'] = detect_fractal(df_copy['close'], n=n_fractal, fractal_type='green')
        df_copy['Red_Fractal'] = detect_fractal(df_copy['close'], n=n_fractal, fractal_type='red')

    # Fill initial NaNs. Back-fill first to avoid lookahead bias where appropriate for signals.
    # For MAs, NaNs are at the beginning. For fractals (due to shift), NaNs are also at the beginning.
    # Consider the implications of bfill carefully. For indicators like MAs, it's usually fine.
    # For fractals, the shift already handles the "future data" aspect for confirmation.
    df_copy.fillna(method='bfill', inplace=True) # Back-fill NaNs from indicators
    # Ensure boolean columns (fractals) that might still be NaN (if all leading values were NaN)
    # are set to False.
    for col in ['Green_Fractal', 'Red_Fractal']:
        if col in df_copy.columns:
            if df_copy[col].isnull().any():
                 df_copy[col] = df_copy[col].fillna(False)
            df_copy[col] = df_copy[col].astype(bool) # Ensure boolean type

    logger.debug("Indicator computation complete.")
    return df_copy

def generate_signals(df):
    """
    Generates BUY, SELL, or HOLD signals and the corresponding trade size percentage.
    Returns two pandas Series: signal_actions, signal_sizes.
    """
    signal_actions = pd.Series("HOLD", index=df.index)
    signal_sizes = pd.Series(0.0, index=df.index) # Default size is 0.0

    # Ensure indicators are present
    required_cols = ['SMA_20', 'SMA_50', 'SMA_100', 'Green_Fractal', 'Red_Fractal', 'close']
    if not all(col in df.columns for col in required_cols):
        logger.warning("Missing required columns for signal generation. Returning HOLD signals with size 0.")
        return signal_actions, signal_sizes

    # Vectorized conditions for speed
    long_condition_ma = (df['SMA_20'] > df['SMA_50']) & (df['SMA_50'] > df['SMA_100'])
    long_condition_pullback = (df['close'] < df['SMA_20']) | (df['close'] < df['SMA_50'])
    long_condition_fractal = df['Green_Fractal'] & (df['close'] > df['SMA_100'])
    buy_signals_triggered = long_condition_ma & long_condition_pullback & long_condition_fractal

    short_condition_ma = (df['SMA_100'] > df['SMA_50']) & (df['SMA_50'] > df['SMA_20'])
    short_condition_pullback = df['close'] > df['SMA_20']
    short_condition_fractal = df['Red_Fractal'] & (df['close'] < df['SMA_100']) # Added check against SMA_100
    sell_signals_triggered = short_condition_ma & short_condition_pullback & short_condition_fractal

    signal_actions[buy_signals_triggered] = "BUY"
    signal_sizes[buy_signals_triggered] = DEFAULT_TRADE_SIZE_PERCENTAGE

    signal_actions[sell_signals_triggered] = "SELL"
    signal_sizes[sell_signals_triggered] = DEFAULT_TRADE_SIZE_PERCENTAGE

    logger.debug(f"Signal generation complete. Buy signals: {buy_signals_triggered.sum()}, Sell signals: {sell_signals_triggered.sum()}")
    return signal_actions, signal_sizes

# --- Removed the old run_backtest function ---
# Example usage (can be run independently for testing indicators/signals)
# if __name__ == "__main__":
#     # Create dummy data or load from CSV
#     try:
#         # Configure a basic logger for standalone testing
#         logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#         logger.info("Starting standalone test for ScalpingStrategy1.")

#         df_hist = pd.read_csv('historical_data.csv', index_col='timestamp', parse_dates=True)
#         # Ensure columns are named correctly (e.g., 'close', 'high', 'low')
#         if 'Close' in df_hist.columns and 'close' not in df_hist.columns:
#              df_hist.rename(columns={'Close': 'close', 'High': 'high', 'Low': 'low', 'Open': 'open'}, inplace=True)

#         df_with_indicators = compute_indicators(df_hist)
#         if df_with_indicators is not None:
#             signal_actions, signal_sizes = generate_signals(df_with_indicators)
#             df_with_indicators['signal_action'] = signal_actions
#             df_with_indicators['signal_size'] = signal_sizes
#             print(df_with_indicators[['close', 'SMA_20', 'SMA_50', 'SMA_100', 'Green_Fractal', 'Red_Fractal', 'signal_action', 'signal_size']].tail(20))
#             print("\nSignal Action Counts:")
#             print(df_with_indicators['signal_action'].value_counts())
#             print("\nSignal Size Distribution (for non-HOLD signals):")
#             print(df_with_indicators[df_with_indicators['signal_action'] != 'HOLD']['signal_size'].value_counts())
#         else:
#             print("Not enough data to compute indicators.")
#     except FileNotFoundError:
#         print("Error: historical_data.csv not found. Please provide a data file.")
#     except Exception as e:
#         print(f"An error occurred: {e}")
#         logger.error(f"An error occurred during standalone test: {e}", exc_info=True)
