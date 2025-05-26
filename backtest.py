import pandas as pd
from backtesting import Backtest, Strategy
from ScalpingStrategy1 import compute_indicators, generate_signals
import os
import datetime
import logging
import json

# --- Configuration Variables ---
DEBUG_LOGGING = True  # Toggle for debug logging
BACKTESTS_DIR = "Backtests"  # Directory to save backtest results
CSV_FILE = 'btc_minute_data.csv'  # Historical data CSV file
INITIAL_CASH = 10_000  # Initial cash for backtesting
COMMISSION_FEE = 0.0005  # Commission fee per trade
PRICE_SCALAR = 0.05  # Scalar for price data (e.g., to trade fractions of an asset)
N_FRACTAL_PERIOD = 2  # Period for fractal detection

# --- Logger Setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if DEBUG_LOGGING else logging.INFO)
# Clear existing handlers to avoid duplicate messages if script is re-run in some environments
if logger.hasHandlers():
    logger.handlers.clear()
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# --- Strategy Class ---
class ScalpingStrategy1BT(Strategy):
    # n_fractal is defined globally as N_FRACTAL_PERIOD
    progress = None  # Required by the backtesting library

    def init(self):
        logger.info("Initializing ScalpingStrategy1BT: Preparing indicators and signals...")

        current_data_df = pd.DataFrame({
            'open': self.data.Open,
            'high': self.data.High,
            'low': self.data.Low,
            'close': self.data.Close,
        }, index=self.data.index)

        df_with_indicators = compute_indicators(current_data_df, n_fractal=N_FRACTAL_PERIOD)

        if df_with_indicators is None:
            logger.error("Indicator computation returned None. Strategy will not generate trade signals.")
            self.signal_actions = pd.Series("HOLD", index=current_data_df.index)
            self.signal_sizes = pd.Series(0.0, index=current_data_df.index)
        else:
            self.signal_actions, self.signal_sizes = generate_signals(df_with_indicators)

            if not self.signal_actions.index.equals(current_data_df.index):
                logger.warning("Signal actions index does not match data index. Reindexing to align.")
                self.signal_actions = self.signal_actions.reindex(current_data_df.index, fill_value="HOLD")
            if not self.signal_sizes.index.equals(current_data_df.index):
                logger.warning("Signal sizes index does not match data index. Reindexing to align.")
                self.signal_sizes = self.signal_sizes.reindex(current_data_df.index, fill_value=0.0)

        logger.info(
            f"ScalpingStrategy1BT initialized. Total data points: {len(current_data_df)}. Buy signals found: {(self.signal_actions == 'BUY').sum()}, Sell signals found: {(self.signal_actions == 'SELL').sum()}")

    def next(self):
        current_bar_iloc = len(self.data.Close) - 1

        if current_bar_iloc < 0 or current_bar_iloc >= len(self.signal_actions):
            logger.warning(
                f"current_bar_iloc {current_bar_iloc} is out of bounds for signal series (len {len(self.signal_actions)}). Skipping trade logic for this bar.")
            return

        current_action = self.signal_actions.iloc[current_bar_iloc]
        current_size_percentage = self.signal_sizes.iloc[current_bar_iloc]
        current_price = self.data.Close[-1]

        if DEBUG_LOGGING and current_bar_iloc > 0 and current_bar_iloc % 500 == 0:
            position_status = 'Flat'
            if self.position.is_long:
                position_status = 'Long'
            elif self.position.is_short:
                position_status = 'Short'
            logger.debug(
                f"Step: {current_bar_iloc}, Price: {current_price:.2f}, "
                f"Action: {current_action}, Size%: {current_size_percentage:.4f}, "
                f"Position: {position_status}, Equity: {self.equity:.2f}"
            )

        if current_action == "BUY" and current_size_percentage > 0:
            if not self.position.is_long:
                self.buy(size=current_size_percentage)
                logger.debug(
                    f"Executed BUY: Price={current_price:.2f}, Size%={current_size_percentage * 100:.2f}, Equity={self.equity:.2f}")

        elif current_action == "SELL" and current_size_percentage > 0:
            if not self.position.is_short:
                self.sell(size=current_size_percentage)
                logger.debug(
                    f"Executed SELL: Price={current_price:.2f}, Size%={current_size_percentage * 100:.2f}, Equity={self.equity:.2f}")


# --- Main Execution ---
if __name__ == "__main__":
    os.makedirs(BACKTESTS_DIR, exist_ok=True)
    logger.info(f"Backtests directory '{BACKTESTS_DIR}' ensured.")

    try:
        logger.info(f"Loading data from {CSV_FILE}...")
        df = pd.read_csv(CSV_FILE, parse_dates=['Date'])
        logger.info("Data loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Error: {CSV_FILE} not found. Please ensure it's in the root directory or update CSV_FILE path.")
        exit(1)
    except Exception as e:
        logger.error(f"Error loading or parsing {CSV_FILE}: {e}")
        exit(1)

    if not isinstance(df.index, pd.DatetimeIndex) and 'Date' not in df.columns:
        logger.error("CSV must have a 'Date' column (or 'timestamp' parsable as 'Date').")
        raise ValueError("CSV must have a 'Date' column (or 'timestamp' parsable as 'Date').")

    if 'Date' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['Date']):
            df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
        logger.debug("Set 'Date' column as index.")
    elif not isinstance(df.index, pd.DatetimeIndex):
        logger.error("Date column not found or not properly set as DatetimeIndex.")
        raise ValueError("Date column not found or not properly set as DatetimeIndex.")

    required_input_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing_cols = [col for col in required_input_cols if col not in df.columns]
    if missing_cols:
        logger.error(f"CSV is missing required columns: {missing_cols}. Available columns: {df.columns.tolist()}")
        raise ValueError(f"CSV is missing required columns: {missing_cols}.")

    df = df.dropna(subset=required_input_cols)
    logger.debug(f"DataFrame shape after dropping NaNs in required columns: {df.shape}")

    df['Open'] = df['Open'] * PRICE_SCALAR
    df['High'] = df['High'] * PRICE_SCALAR
    df['Low'] = df['Low'] * PRICE_SCALAR
    df['Close'] = df['Close'] * PRICE_SCALAR
    logger.debug(f"Prices scaled by factor {PRICE_SCALAR}.")

    start_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Starting backtest at: {start_time_str}")

    bt = Backtest(
        df,
        ScalpingStrategy1BT,
        cash=INITIAL_CASH,
        commission=COMMISSION_FEE,
        exclusive_orders=True,
        trade_on_close=True
    )

    stats = bt.run(progress=True)
    logger.info("Backtest run completed.")
    if DEBUG_LOGGING:
        logger.debug(f"--- Raw Stats ---\n{stats}\n-----------------")
    else:
        logger.info(
            f"Final Equity: {stats.get('Equity Final [$]', 'N/A')}, Return [%]: {stats.get('Return [%]', 'N/A')}, # Trades: {stats.get('# Trades', 'N/A')}")

    # Prepare data for JSON serialization
    results_data = {}
    for key, value in stats.items():
        if isinstance(value, pd.Timestamp):
            results_data[key] = value.isoformat()
        elif isinstance(value, pd.Timedelta):
            results_data[key] = str(value)  # Or value.total_seconds() for numerical representation
        elif isinstance(value, (pd.Series, pd.DataFrame)):
            if key not in ['_trades', '_equity_curve', '_strategy']:  # Exclude these for now, handle explicitly
                try:
                    results_data[key] = value.to_dict()
                except:  # Fallback for complex structures within series if any
                    results_data[key] = str(value)
            elif key == '_strategy':  # Avoid serializing the full strategy object
                results_data[key] = str(value)

        else:
            results_data[key] = value

    # Handle _trades DataFrame
    if '_trades' in stats and isinstance(stats['_trades'], pd.DataFrame):
        trades_df = stats['_trades'].copy()
        # Convert Timedelta and Timestamp columns to string for JSON compatibility
        for col_name, col_type in trades_df.dtypes.items():
            if pd.api.types.is_timedelta64_ns_dtype(col_type):
                trades_df[col_name] = trades_df[col_name].astype(str)
            elif pd.api.types.is_datetime64_any_dtype(col_type):
                trades_df[col_name] = trades_df[col_name].dt.strftime('%Y-%m-%d %H:%M:%S')
        results_data['_trades'] = trades_df.to_dict(orient='records')

    # Handle _equity_curve DataFrame
    if '_equity_curve' in stats and isinstance(stats['_equity_curve'], pd.DataFrame):
        equity_df = stats['_equity_curve'].copy()
        equity_df.index = equity_df.index.strftime('%Y-%m-%d %H:%M:%S')  # Convert DatetimeIndex to string
        results_data['_equity_curve'] = equity_df.reset_index().to_dict(orient='records')

    # Add configuration and run time to the results file
    results_data['run_config'] = {
        'backtest_start_time': start_time_str,
        'csv_file': CSV_FILE,
        'initial_cash': INITIAL_CASH,
        'commission_fee': COMMISSION_FEE,
        'price_scalar': PRICE_SCALAR,
        'n_fractal_period': N_FRACTAL_PERIOD
    }

    # Save backtest stats to JSON
    stats_json_filepath = os.path.join(BACKTESTS_DIR, 'backtest_stats_and_data.json')
    try:
        with open(stats_json_filepath, 'w') as f:
            json.dump(results_data, f, indent=4, default=str)  # default=str for any missed complex types
        logger.info(f"Backtest stats and data saved to {stats_json_filepath}")
    except Exception as e:
        logger.error(f"Error saving stats to JSON: {e}")

    logger.info("--- Backtest Core Processing Complete ---")
    logger.info(f"Results (JSON data) saved in '{BACKTESTS_DIR}' directory.")
    logger.info("To generate custom reports, use the reporting.py script that reads this JSON file.")