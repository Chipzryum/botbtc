import pandas as pd
import numpy as np
import talib
import logging
from .base_strategy import BaseStrategy # MODIFIED: Changed to relative import

# --- Logger Setup ---
logger = logging.getLogger(__name__)

class ScalpingStrategy1(BaseStrategy):
    """
    Scalping strategy that uses Williams fractals and moving averages for signal generation.
    """
    
    def __init__(self, params=None):
        """
        Initialize the strategy with optional parameters.
        
        Args:
            params (dict, optional): Strategy-specific parameters.
                - n_fractal (int): Period for fractal detection. Default is 2.
                - trade_size (float): Default trade size as percentage. Default is 0.3.
        """
        super().__init__(params)
        
        # Set default parameters if not provided
        if not self.params:
            self.params = {}
        
        # Set strategy parameters with defaults
        self.n_fractal = self.params.get('n_fractal', 2)
        self.default_trade_size = self.params.get('trade_size', 0.3)
        
        logger.info(f"ScalpingStrategy1 initialized with n_fractal={self.n_fractal}, "
                   f"trade_size={self.default_trade_size}")
    
    def detect_fractal(self, series, n=2, fractal_type='green'):
        """
        Detects Williams Fractals using a more vectorized approach.
        
        Args:
            series (pd.Series): Price series to analyze
            n (int): Number of bars on each side to check for high/low
            fractal_type (str): 'green' (bullish) or 'red' (bearish)
            
        Returns:
            numpy.ndarray: Boolean array indicating fractal positions
        """
        if not isinstance(series, pd.Series):
            series = pd.Series(series)  # Ensure input is a pandas Series

        # Window size is 2*n + 1
        window_size = 2 * n + 1

        # Create empty result series
        fractals = pd.Series(False, index=series.index)
        
        # Process based on fractal type
        for i in range(n, len(series) - n):
            target_val = series.iloc[i]
            window = series.iloc[i-n : i+n+1]

            if fractal_type == 'green':  # Bullish fractal (low point)
                if target_val == window.min() and np.count_nonzero(window.values == target_val) == 1:
                    fractals.iloc[i] = True
            elif fractal_type == 'red':  # Bearish fractal (high point)
                if target_val == window.max() and np.count_nonzero(window.values == target_val) == 1:
                    fractals.iloc[i] = True
            else:
                raise ValueError("fractal_type must be 'green' or 'red'")

        # Shift result by n bars to ensure fractal is confirmed only after n future bars
        result = fractals.astype(float).shift(n).fillna(0.0).astype(bool)

        logger.debug(f"Fractal detection complete for type '{fractal_type}'. Found {result.sum()} fractals.")
        return result.values  # Return as numpy array
    
    def compute_indicators(self, df):
        """
        Computes technical indicators needed for the strategy.
        
        Args:
            df (pd.DataFrame): DataFrame with price data (must contain OHLCV columns)
            
        Returns:
            pd.DataFrame: DataFrame with added indicator columns
        """
        logger.debug(f"Computing indicators for DataFrame with shape {df.shape}. n_fractal={self.n_fractal}")
        
        if 'close' not in df.columns:
            logger.error("DataFrame must contain 'close' column for compute_indicators.")
            raise ValueError("DataFrame must contain 'close' column.")
            
        if len(df) < 100:  # Ensure enough data for longest MA
            logger.warning("Not enough data to compute all indicators (less than 100 rows).")
            return None  # Or handle differently, e.g., compute what's possible

        df_copy = df.copy()  # Avoid SettingWithCopyWarning
        
        # Compute moving averages
        df_copy['SMA_20'] = talib.SMA(df_copy['close'], timeperiod=20)
        df_copy['SMA_50'] = talib.SMA(df_copy['close'], timeperiod=50)
        df_copy['SMA_100'] = talib.SMA(df_copy['close'], timeperiod=100)

        # Compute fractals
        if 'high' in df_copy.columns and 'low' in df_copy.columns:
            df_copy['Green_Fractal'] = self.detect_fractal(df_copy['low'], n=self.n_fractal, fractal_type='green')
            df_copy['Red_Fractal'] = self.detect_fractal(df_copy['high'], n=self.n_fractal, fractal_type='red')
        else:
            logger.warning("'high' or 'low' columns not found. Using 'close' for fractal detection (less accurate).")
            df_copy['Green_Fractal'] = self.detect_fractal(df_copy['close'], n=self.n_fractal, fractal_type='green')
            df_copy['Red_Fractal'] = self.detect_fractal(df_copy['close'], n=self.n_fractal, fractal_type='red')

        # Fill initial NaNs with bfill and handle boolean columns
        df_copy.bfill(inplace=True)
        
        for col in ['Green_Fractal', 'Red_Fractal']:
            if col in df_copy.columns:
                if df_copy[col].isnull().any():
                    df_copy[col] = df_copy[col].fillna(False)
                df_copy[col] = df_copy[col].astype(bool)  # Ensure boolean type

        logger.debug("Indicator computation complete.")
        return df_copy
    
    def generate_signals(self, df):
        """
        Generates BUY, SELL, or HOLD signals and the corresponding trade size percentage.
        
        Args:
            df (pd.DataFrame): DataFrame with price data and indicators
            
        Returns:
            tuple: (signal_actions, signal_sizes) - Series of actions and sizes
        """
        signal_actions = pd.Series("HOLD", index=df.index)
        signal_sizes = pd.Series(0.0, index=df.index)  # Default size is 0.0

        # Ensure indicators are present
        required_cols = ['SMA_20', 'SMA_50', 'SMA_100', 'Green_Fractal', 'Red_Fractal', 'close']
        if not all(col in df.columns for col in required_cols):
            logger.warning("Missing required columns for signal generation. Returning HOLD signals with size 0.")
            return signal_actions, signal_sizes

        # Vectorized conditions for speed
        # Long conditions
        long_condition_ma = (df['SMA_20'] > df['SMA_50']) & (df['SMA_50'] > df['SMA_100'])
        long_condition_pullback = (df['close'] < df['SMA_20']) | (df['close'] < df['SMA_50'])
        long_condition_fractal = df['Green_Fractal'] & (df['close'] > df['SMA_100'])
        buy_signals_triggered = long_condition_ma & long_condition_pullback & long_condition_fractal

        # Short conditions
        short_condition_ma = (df['SMA_100'] > df['SMA_50']) & (df['SMA_50'] > df['SMA_20'])
        short_condition_pullback = df['close'] > df['SMA_20']
        short_condition_fractal = df['Red_Fractal'] & (df['close'] < df['SMA_100'])
        sell_signals_triggered = short_condition_ma & short_condition_pullback & short_condition_fractal

        # Set signals and sizes
        signal_actions[buy_signals_triggered] = "BUY"
        signal_sizes[buy_signals_triggered] = self.default_trade_size

        signal_actions[sell_signals_triggered] = "SELL"
        signal_sizes[sell_signals_triggered] = self.default_trade_size

        logger.debug(f"Signal generation complete. Buy signals: {buy_signals_triggered.sum()}, "
                    f"Sell signals: {sell_signals_triggered.sum()}")
        return signal_actions, signal_sizes


# Example usage for standalone testing
if __name__ == "__main__":
    # Configure a basic logger for standalone testing
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger.info("Starting standalone test for ScalpingStrategy1.")
    
    try:
        # Load test data
        df_hist = pd.read_csv('historical_data.csv', index_col='timestamp', parse_dates=True)
        
        # Ensure columns are named correctly
        if 'Close' in df_hist.columns and 'close' not in df_hist.columns:
            df_hist.rename(columns={
                'Close': 'close', 
                'High': 'high', 
                'Low': 'low', 
                'Open': 'open',
                'Volume': 'volume'
            }, inplace=True)
            
        # Initialize strategy with custom parameters
        strategy = ScalpingStrategy1(params={'n_fractal': 3, 'trade_size': 0.2})
        
        # Process data with the strategy
        df_with_indicators, signal_actions, signal_sizes = strategy.prepare_for_backtest(df_hist)
        
        if df_with_indicators is not None:
            # Add signals to DataFrame for visualization
            df_with_indicators['signal_action'] = signal_actions
            df_with_indicators['signal_size'] = signal_sizes
            
            # Print results
            print(df_with_indicators[['close', 'SMA_20', 'SMA_50', 'SMA_100', 
                                     'Green_Fractal', 'Red_Fractal', 
                                     'signal_action', 'signal_size']].tail(20))
            print("\nSignal Action Counts:")
            print(signal_actions.value_counts())
            print("\nSignal Size Distribution (for non-HOLD signals):")
            print(signal_sizes[signal_actions != 'HOLD'].value_counts())
        else:
            print("Not enough data to compute indicators.")
            
    except FileNotFoundError:
        print("Error: historical_data.csv not found. Please provide a data file.")
    except Exception as e:
        print(f"An error occurred: {e}")
        logger.error(f"An error occurred during standalone test: {e}", exc_info=True)
