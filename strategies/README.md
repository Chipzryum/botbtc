# Trading Strategy Development Guidelines

This document outlines the rules and best practices for creating new trading strategies within this framework. Adhering to these guidelines will ensure consistency, maintainability, and compatibility with the backtesting and live trading systems.

## Core Principles

1.  **Inheritance from `BaseStrategy`**:
    *   All new strategy classes **MUST** inherit from the `BaseStrategy` class (defined in `base_strategy.py`).
    *   This ensures that your strategy implements the necessary methods and properties expected by the system.

2.  **Mandatory Method Implementations**:
    *   Your strategy class **MUST** implement the following methods:
        *   `__init__(self, params=None)`:
            *   The constructor should call `super().__init__(params)`.
            *   It should define and initialize any strategy-specific parameters. It's good practice to allow parameters to be passed in via the `params` dictionary and to set sensible defaults if they are not provided.
            *   Example: `self.short_ma_period = params.get('short_ma', 20)`
        *   `compute_indicators(self, df)`:
            *   **Input**: A pandas DataFrame `df` containing at least 'open', 'high', 'low', 'close', and 'volume' columns (standard OHLCV). The column names should be lowercase (e.g., 'close', not 'Close').
            *   **Action**: This method should calculate all technical indicators required by your strategy and add them as new columns to the input DataFrame.
            *   **Output**: It **MUST** return the modified pandas DataFrame with the added indicator columns. If for some reason indicators cannot be computed (e.g., not enough data), it can return `None`, which the calling system should handle.
            *   **Important**: Aim for vectorized operations (using pandas/numpy capabilities) for efficiency instead of iterating row by row.
        *   `generate_signals(self, df)`:
            *   **Input**: A pandas DataFrame `df` that *already includes the indicator columns* computed by `compute_indicators`.
            *   **Action**: This method implements the core logic of your strategy to generate trading signals based on the price data and indicators.
            *   **Output**: It **MUST** return a tuple of two pandas Series, both indexed identically to the input `df`:
                1.  `signal_actions`: A Series of strings indicating the action for each bar. Expected values are "BUY", "SELL", or "HOLD".
                2.  `signal_sizes`: A Series of floats (between 0.0 and 1.0) indicating the percentage of capital/position to use for the trade. For "HOLD" actions, the size should typically be 0.0. For "BUY" or "SELL" signals that close a position, the size might also be 1.0 (to close 100% of the position) or another relevant fraction.
    *   These methods form the core interface used by the backtesting and trading engines.

3.  **Parameterization**:
    *   Strategies should be parameterizable through the `params` dictionary passed to the `__init__` method.
    *   Avoid hardcoding magic numbers directly in your strategy logic. Instead, define them as parameters. This makes optimization and testing different variations of your strategy much easier.

4.  **Data Handling**:
    *   Assume input DataFrames (`df`) to `compute_indicators` and `generate_signals` will have a `DatetimeIndex`.
    *   Ensure your indicator calculations and signal generation logic can handle `NaN` values gracefully (e.g., by `bfill`, `ffill`, `dropna`, or by starting calculations after enough data is available for indicators to be valid).
    *   The `compute_indicators` method should ideally make a copy of the input DataFrame (e.g., `df_copy = df.copy()`) before adding columns to avoid `SettingWithCopyWarning` and unintended side effects.

5.  **Logging**:
    *   Use the `logging` module for any informational messages, warnings, or errors within your strategy.
    *   Instantiate a logger at the beginning of your strategy file: `logger = logging.getLogger(__name__)`.
    *   This helps in debugging and understanding the strategy's behavior during backtesting or live execution.

6.  **Clarity and Comments**:
    *   Write clear, understandable code.
    *   Add comments to explain complex logic, assumptions, or the purpose of specific calculations.

7.  **Efficiency**:
    *   Prioritize vectorized operations with pandas and numpy over loops where possible, especially in `compute_indicators` and `generate_signals`, as these methods will be called frequently.
    *   Be mindful of the computational cost of your indicators and logic, especially if the strategy is intended for high-frequency data.

## Example Structure (Simplified)

```python
# In your_strategy_name.py
import pandas as pd
import talib # or other indicator libraries
import logging
from .base_strategy import BaseStrategy

logger = logging.getLogger(__name__)

class YourStrategyName(BaseStrategy):
    def __init__(self, params=None):
        super().__init__(params)
        # Define default parameters and override with provided params
        self.ma_period = self.params.get('ma_period', 50)
        self.trade_size_percent = self.params.get('trade_size', 0.5) # Example: trade 50% of available capital
        logger.info(f"{self.__class__.__name__} initialized with MA period: {self.ma_period}")

    def compute_indicators(self, df):
        df_copy = df.copy()
        if len(df_copy) < self.ma_period:
            logger.warning("Not enough data to compute MA.")
            return None # Or return df_copy without the indicator
        
        df_copy[f'SMA_{self.ma_period}'] = talib.SMA(df_copy['close'], timeperiod=self.ma_period)
        # Add other indicators here
        
        # Handle NaNs, e.g., by back-filling, but be cautious about lookahead bias
        # df_copy.bfill(inplace=True) 
        return df_copy

    def generate_signals(self, df):
        signal_actions = pd.Series("HOLD", index=df.index)
        signal_sizes = pd.Series(0.0, index=df.index)

        # Example: Buy if close is above SMA, Sell if below
        # Ensure the indicator column exists
        sma_col = f'SMA_{self.ma_period}'
        if sma_col not in df.columns:
            logger.warning(f"'{sma_col}' not found in DataFrame. Cannot generate signals.")
            return signal_actions, signal_sizes

        # Vectorized signal generation
        buy_condition = (df['close'] > df[sma_col]) & (df['close'].shift(1) <= df[sma_col].shift(1)) # Crossover
        sell_condition = (df['close'] < df[sma_col]) & (df['close'].shift(1) >= df[sma_col].shift(1)) # Crossover

        signal_actions[buy_condition] = "BUY"
        signal_sizes[buy_condition] = self.trade_size_percent

        signal_actions[sell_condition] = "SELL"
        signal_sizes[sell_condition] = self.trade_size_percent # Assuming same size for sell, adjust as needed

        return signal_actions, signal_sizes

```

## Testing
*   It's highly recommended to include an `if __name__ == "__main__":` block in your strategy file for standalone testing. This allows you to load sample data and test your `compute_indicators` and `generate_signals` methods independently.

By following these guidelines, we can build a robust and flexible library of trading strategies.

