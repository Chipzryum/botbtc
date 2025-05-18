import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from backtesting.lib import crossover
import talib
from ScalpingStrategy1 import detect_fractal
import os
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import warnings
import datetime
# import sys # No longer needed

# Suppress specific warnings if necessary (e.g., from matplotlib or pandas)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
# You can be more specific with warnings if you know which ones are bothering you

# --- Strategy Class for backtesting.py ---
class ScalpingStrategy1BT(Strategy):
    n_fractal = 2
    progress = None # May not be strictly needed with progress=True in bt.run(), but harmless.

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

# --- Helper functions for HTML metrics (no longer for progress) ---

def get_gradient_rgb(value, good_threshold, bad_threshold, lower_is_better=False):
    """Calculates an RGB color on a red-yellow-green gradient."""
    if pd.isna(value) or good_threshold is None or bad_threshold is None:
        return "rgb(128, 128, 128)"  # Grey for N/A or unconfigurable

    try:
        value = float(value)
    except (ValueError, TypeError):
        return "rgb(128, 128, 128)"  # Grey for non-numeric

    if good_threshold == bad_threshold: # Avoid division by zero and define behavior
        return "rgb(255, 193, 7)" # Neutral yellow if thresholds are the same

    if lower_is_better:
        norm_val = (bad_threshold - value) / (bad_threshold - good_threshold)
    else:
        norm_val = (value - bad_threshold) / (good_threshold - bad_threshold)

    norm_val = max(0, min(1, norm_val))  # Clamp between 0 and 1

    # Colors: Bad (Red), Mid (Yellow), Good (Green)
    r_bad, g_bad, b_bad = 220, 53, 69
    r_mid, g_mid, b_mid = 255, 193, 7
    r_good, g_good, b_good = 40, 167, 69

    if norm_val < 0.5:
        # Interpolate Bad to Mid
        t = norm_val * 2
        r = int(r_bad * (1 - t) + r_mid * t)
        g = int(g_bad * (1 - t) + g_mid * t)
        b = int(b_bad * (1 - t) + b_mid * t)
    else:
        # Interpolate Mid to Good
        t = (norm_val - 0.5) * 2
        r = int(r_mid * (1 - t) + r_good * t)
        g = int(g_mid * (1 - t) + g_good * t)
        b = int(b_mid * (1 - t) + b_good * t)

    return f"rgb({r},{g},{b})"

def format_metric_display_value(value):
    if pd.isna(value):
        return "N/A"
    if isinstance(value, pd.Timedelta):
        # Format timedelta to a more readable string, e.g., "X days HH:MM:SS" or total hours/minutes
        total_seconds = value.total_seconds()
        days, remainder = divmod(total_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        if days > 0:
            return f"{int(days)}d {int(hours)}h {int(minutes)}m"
        elif hours > 0:
            return f"{int(hours)}h {int(minutes)}m {int(seconds)}s"
        elif minutes > 0:
            return f"{int(minutes)}m {int(seconds)}s"
        else:
            return f"{total_seconds:.2f}s"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)

# --- Plotting and Reporting Functions ---

def _generate_plots_and_metrics_html(stats, metrics_to_display_config):
    """Generates Matplotlib plots, base64 encodes them, and prepares HTML for metrics."""
    # Custom visualizations using Matplotlib (Equity, Drawdown, Trade Hists)
    plt.figure(figsize=(12, 10))

    # Equity Curve
    if '_equity_curve' in stats and 'Equity' in stats['_equity_curve']:
        plt.subplot(2, 2, 1)
        plt.plot(stats['_equity_curve']['Equity'], label='Equity')
        plt.title('Equity Curve')
        plt.xlabel('Time')
        plt.ylabel('Equity')
        plt.legend()

    # Drawdown
    if '_equity_curve' in stats and 'Drawdown' in stats['_equity_curve']:
        plt.subplot(2, 2, 2)
        plt.plot(stats['_equity_curve']['Drawdown'], label='Drawdown', color='r')
        plt.title('Drawdown')
        plt.xlabel('Time')
        plt.ylabel('Drawdown')
        plt.legend()

    # Additional Metrics (Trade Returns and Duration Histograms)
    if '_trades' in stats:
        trades = stats['_trades'].copy()
        plt.subplot(2, 2, 3)
        plt.hist(trades['ReturnPct'], bins=50, alpha=0.75, label='Trade Returns')
        plt.title('Trade Returns Distribution')
        plt.xlabel('Return %')
        plt.ylabel('Frequency')
        plt.legend()

        if 'Duration' in trades.columns:
            if not pd.api.types.is_timedelta64_dtype(trades['Duration']):
                trades['Duration'] = pd.to_timedelta(trades['Duration'])
            trades['Duration_seconds'] = trades['Duration'].dt.total_seconds()
            plt.subplot(2, 2, 4)
            plt.hist(trades['Duration_seconds'], bins=50, alpha=0.75, label='Trade Duration (seconds)')
            plt.title('Trade Duration Distribution')
            plt.xlabel('Duration (seconds)')
            plt.ylabel('Frequency')
            plt.legend()
        else:
            print("Warning: 'Duration' column not found in trades data. Skipping duration histogram.")

    plt.tight_layout()
    buf = BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close()

    # Generate HTML for Performance Metrics
    metrics_html_content = ""
    for m_info in metrics_to_display_config:
        value = stats.get(m_info['key'])
        display_value = format_metric_display_value(value)
        color_style = get_gradient_rgb(value, m_info['good'], m_info['bad'], m_info.get('lower_is_better', False))
        metrics_html_content += f'''
        <div class="metric-card">
            <span class="metric-label">{m_info['label']}</span>
            <span class="metric-value" style="color: {color_style};">{display_value}</span>
        </div>
        '''
    return plot_data_base64, metrics_html_content

def _write_combined_html_report(backtests_dir, plot_data_base64, metrics_html_content, color_key_html):
    """Writes the combined_results.html file."""
    html_file_path = os.path.join(backtests_dir, 'combined_results.html')
    with open(html_file_path, 'w') as f:
        f.write(f"""
        <html>
            <head>
                <title>Backtest Results</title>
                <style>
                  body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f4f4f4; color: #333; }}
                  .container {{ max-width: 1200px; margin: 20px auto; padding: 20px; background-color: #fff; box-shadow: 0 0 10px rgba(0,0,0,0.1); }}
                  h1, h2 {{ color: #333; text-align: center; }}
                  iframe {{ border: 1px solid #ddd; margin-bottom: 20px; }}
                  .metrics-header {{ text-align: left; margin-top: 30px; margin-bottom:10px; padding-left: 20px;}}
                  .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                    gap: 15px;
                    padding: 0 20px 20px 20px;
                  }}
                  .metric-card {{
                    border: 1px solid #ddd;
                    border-radius: 8px;
                    padding: 20px;
                    text-align: center;
                    background-color: #fff;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                    transition: transform 0.2s ease-in-out;
                  }}
                  .metric-card:hover {{ transform: translateY(-5px); }}
                  .metric-card .metric-label {{
                    display: block;
                    font-size: 0.95em;
                    color: #555;
                    margin-bottom: 10px;
                  }}
                  .metric-card .metric-value {{
                    display: block;
                    font-size: 1.6em;
                    font-weight: bold;
                  }}
                  .color-key {{
                    padding: 15px 20px;
                    text-align: center;
                    border-top: 1px solid #eee;
                    margin-top: 20px;
                    font-size: 0.9em;
                  }}
                  .gradient-bar {{
                    display: inline-block;
                    width: 120px;
                    height: 18px;
                    background: linear-gradient(to right, rgb(220, 53, 69), rgb(255, 193, 7), rgb(40, 167, 69));
                    vertical-align: middle;
                    margin: 0 8px;
                    border-radius: 4px;
                    border: 1px solid #ccc;
                  }}
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Backtest Results</h1>
                    <iframe src="backtest_plot.html" width="100%" height="600px" style="border:none;"></iframe>
                    
                    <h2>Performance Plots</h2>
                    <img src="data:image/png;base64,{plot_data_base64}" width="100%">
                    
                    <h2 class="metrics-header">Key Performance Indicators</h2>
                    {color_key_html}
                    <div class="metrics-grid">
                        {metrics_html_content}
                    </div>
                </div>
            </body>
        </html>
        """)
    print(f"Combined results saved to {html_file_path}")

def _write_summary_text_file(stats, backtests_dir):
    """Writes the summary.txt file."""
    summary_file_path = os.path.join(backtests_dir, 'summary.txt')
    with open(summary_file_path, 'w') as f:
        f.write("Backtest Summary\n")
        f.write("================\n\n")

        f.write("Overall Performance:\n")
        f.write(f"  Start: {stats.get('Start', 'N/A')}\n")
        f.write(f"  End: {stats.get('End', 'N/A')}\n")
        f.write(f"  Duration: {stats.get('Duration', 'N/A')}\n")
        f.write(f"  Exposure Time [%]: {stats.get('Exposure Time [%]', 'N/A')}\n\n")

        f.write("Equity & Returns:\n")
        f.write(f"  Equity Final [$]: {stats.get('Equity Final [$]', 'N/A')}\n")
        f.write(f"  Equity Peak [$]: {stats.get('Equity Peak [$]', 'N/A')}\n")
        f.write(f"  Return [%]: {stats.get('Return [%]', 'N/A')}\n")
        f.write(f"  Return (Ann.) [%]: {stats.get('Return (Ann.) [%]', 'N/A')}\n")
        f.write(f"  Buy & Hold Return [%]: {stats.get('Buy & Hold Return [%]', 'N/A')}\n")
        f.write(f"  Volatility (Ann.) [%]: {stats.get('Volatility (Ann.) [%]', 'N/A')}\n\n")
        
        f.write("Risk & Ratios:\n")
        f.write(f"  Sharpe Ratio: {stats.get('Sharpe Ratio', 'N/A')}\n")
        f.write(f"  Sortino Ratio: {stats.get('Sortino Ratio', 'N/A')}\n")
        f.write(f"  Calmar Ratio: {stats.get('Calmar Ratio', 'N/A')}\n")
        f.write(f"  SQN: {stats.get('SQN', 'N/A')}\n\n")

        f.write("Drawdown:\n")
        f.write(f"  Max Drawdown [%]: {stats.get('Max. Drawdown [%]', 'N/A')}\n")
        f.write(f"  Avg Drawdown [%]: {stats.get('Avg. Drawdown [%]', stats.get('Avg Drawdown [%]', 'N/A'))}\n")
        f.write(f"  Max Drawdown Duration: {stats.get('Max. Drawdown Duration', stats.get('Max Drawdown Duration', 'N/A'))}\n")
        f.write(f"  Avg Drawdown Duration: {stats.get('Avg. Drawdown Duration', stats.get('Avg Drawdown Duration', 'N/A'))}\n\n")

        f.write("Trades:\n")
        f.write(f"  # Trades: {stats.get('# Trades', 'N/A')}\n")
        f.write(f"  Win Rate [%]: {stats.get('Win Rate [%]', 'N/A')}\n")
        f.write(f"  Profit Factor: {stats.get('Profit Factor', 'N/A')}\n")
        f.write(f"  Expectancy [%]: {stats.get('Expectancy [%]', 'N/A')}\n")
        f.write(f"  Avg Trade [%]: {stats.get('Avg. Trade [%]', stats.get('Avg Trade [%]', 'N/A'))}\n")
        f.write(f"  Best Trade [%]: {stats.get('Best Trade [%]', 'N/A')}\n")
        f.write(f"  Worst Trade [%]: {stats.get('Worst Trade [%]', 'N/A')}\n")
        f.write(f"  Avg. Trade Duration: {stats.get('Avg. Trade Duration', 'N/A')}\n")
        f.write(f"  Max. Trade Duration: {stats.get('Max. Trade Duration', 'N/A')}\n")
    print(f"Summary saved to {summary_file_path}")

# --- Main Execution ---
if __name__ == "__main__":
    BACKTESTS_DIR = "Backtests"
    CSV_FILE = 'btc_minute_data.csv' # Define CSV_FILE here
    
    os.makedirs(BACKTESTS_DIR, exist_ok=True)

    # Load data
    try:
        df = pd.read_csv(CSV_FILE, parse_dates=True)
    except FileNotFoundError:
        print(f"Error: {CSV_FILE} not found. Please place your historical data CSV in the root directory.")
        exit(1)
    # ... (rest of data loading and preprocessing remains the same) ...
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
    price_scalar = 0.01
    df['Open'] = df['Open'] * price_scalar
    df['High'] = df['High'] * price_scalar
    df['Low'] = df['Low'] * price_scalar
    df['Close'] = df['Close'] * price_scalar

    # Run backtest
    start_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Starting backtest at: {start_time}")
    
    bt = Backtest(
        df,
        ScalpingStrategy1BT,
        cash=10_000,
        commission=0.0005,
        exclusive_orders=True,
        trade_on_close=True
    )

    stats = bt.run(progress=True) # Use built-in tqdm progress
    print(stats)

    # Generate the standard backtesting.py plot
    bt.plot(filename=os.path.join(BACKTESTS_DIR, 'backtest_plot.html'), open_browser=False)

    # Define metrics configuration
    metrics_config = [
        {'label': 'Sharpe Ratio', 'key': 'Sharpe Ratio', 'good': 1, 'bad': 0, 'lower_is_better': False},
        {'label': 'Sortino Ratio', 'key': 'Sortino Ratio', 'good': 1.5, 'bad': 0, 'lower_is_better': False},
        {'label': 'Max Drawdown [%]', 'key': 'Max. Drawdown [%]', 'good': -10, 'bad': -25, 'lower_is_better': True},
        {'label': 'Win Rate [%]', 'key': 'Win Rate [%]', 'good': 55, 'bad': 40, 'lower_is_better': False},
        {'label': 'Profit Factor', 'key': 'Profit Factor', 'good': 1.5, 'bad': 1, 'lower_is_better': False},
        {'label': 'Expectancy [%]', 'key': 'Expectancy [%]', 'good': 0.1, 'bad': 0, 'lower_is_better': False},
        {'label': 'Avg. Trade Duration', 'key': 'Avg. Trade Duration', 'good': None, 'bad': None},
        {'label': 'Return [%]', 'key': 'Return [%]', 'good': 10, 'bad': 0, 'lower_is_better': False},
        {'label': 'Buy & Hold Return [%]', 'key': 'Buy & Hold Return [%]', 'good': None, 'bad': None},
    ]
    
    color_key_html_content = """
    <div class="color-key">
        <strong style="margin-right: 5px;">Color Key:</strong>
        <span style="color: rgb(220, 53, 69);">Bad</span>
        <span class="gradient-bar"></span>
        <span style="color: rgb(40, 167, 69); margin-left: 5px;">Good</span>
    </div>
    """

    # Generate plots and HTML content
    plot_data_base64, metrics_html = _generate_plots_and_metrics_html(stats, metrics_config)
    
    # Write reports
    _write_combined_html_report(BACKTESTS_DIR, plot_data_base64, metrics_html, color_key_html_content)
    _write_summary_text_file(stats, BACKTESTS_DIR)

    print("\n--- Backtest Complete ---")
    print("Backtest process is fully set up and ready for use.")
    print("To test a new strategy, define a new class inheriting from backtesting.Strategy (e.g., in ScalpingStrategy1.py), then import and use it in backtest.py.")
    print("Run this script to backtest and view results in Backtests/combined_results.html.")
    print("No further setup is needed. Happy coding and algo trading!")
