"""
Configuration file for backtest settings.
This file contains default configuration parameters for the backtesting system.
"""

# Default configuration dictionary
DEFAULT_CONFIG = {
    # Basic setup
    'debug_logging': False,  # Toggle for debug logging
    'backtests_dir': "Backtests",  # Directory to save backtest results

    # Data settings
    'csv_file': 'btc_minute_data.csv',  # Historical data CSV file
    'price_scalar': 0.05,  # Scalar for price data (e.g., to trade fractions of an asset)

    # Backtest parameters
    'initial_cash': 10_000,  # Initial cash for backtesting
    'commission_fee': 0.0005,  # Commission fee per trade (0.05%)

    # Output settings
    'plot_results': True,  # Whether to plot results using backtesting.py
    'save_results': True,  # Whether to save results to JSON
    'custom_plot': False,  # Whether to create custom plots with matplotlib
}

# Strategy-specific default parameters
STRATEGY_PARAMS = {
    'ScalpingStrategy1': {
        'n_fractal': 2,  # Period for fractal detection
        'trade_size': 0.3  # Default trade size as percentage
    },

    # Add parameters for other strategies here
}

# Paths for data and results
PATHS = {
    'data_dir': 'data',
    'results_dir': 'Backtests',
    'logs_dir': 'logs'
}
