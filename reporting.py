import pandas as pd
import numpy as np
from backtesting import Backtest, Strategy
from ScalpingStrategy1 import compute_indicators, generate_signals
import os
import datetime
import logging
import json
import base64
from io import BytesIO
import matplotlib.pyplot as plt
import warnings
import sys  # Used by reporting part if it were standalone, not strictly needed now

# --- Configuration Variables ---
DEBUG_LOGGING = True
BACKTESTS_DIR = "Backtests"  # Unified directory for all outputs
CSV_FILE = 'btc_minute_data.csv'
INITIAL_CASH = 10_000
COMMISSION_FEE = 0.0005
PRICE_SCALAR = 0.05
N_FRACTAL_PERIOD = 2

# --- Logger Setup ---
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG if DEBUG_LOGGING else logging.INFO)
if logger.hasHandlers():
    logger.handlers.clear()
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

# Suppress specific warnings (moved here for global effect)
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="matplotlib")


# --- Strategy Class (from original backtesting script) ---
class ScalpingStrategy1BT(Strategy):
    progress = None

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
            # This warning can be very verbose if signals don't cover the whole range.
            # logger.warning(f"current_bar_iloc {current_bar_iloc} is out of bounds for signal series (len {len(self.signal_actions)}).")
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
                # logger.debug(f"Executed BUY: Price={current_price:.2f}, Size%={current_size_percentage * 100:.2f}, Equity={self.equity:.2f}")
        elif current_action == "SELL" and current_size_percentage > 0:
            if not self.position.is_short:
                self.sell(size=current_size_percentage)
                # logger.debug(f"Executed SELL: Price={current_price:.2f}, Size%={current_size_percentage * 100:.2f}, Equity={self.equity:.2f}")


# --- Reporting Script Functions (Integrated) ---

# Configuration for metrics display in HTML report (specific to reporting)
METRICS_CONFIG_REPORTING = [
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

COLOR_KEY_HTML_CONTENT_REPORTING = """
<div class="color-key">
    <strong style="margin-right: 5px;">Color Key:</strong>
    <span style="color: rgb(220, 53, 69);">Bad</span>
    <span class="gradient-bar"></span>
    <span style="color: rgb(40, 167, 69); margin-left: 5px;">Good</span>
</div>
"""


def get_gradient_rgb(value, good_threshold, bad_threshold, lower_is_better=False):
    if pd.isna(value) or good_threshold is None or bad_threshold is None: return "rgb(128, 128, 128)"
    try:
        value = float(value)
    except (ValueError, TypeError):
        return "rgb(128, 128, 128)"
    if good_threshold == bad_threshold: return "rgb(255, 193, 7)"
    norm_val = ((bad_threshold - value) / (bad_threshold - good_threshold)) if lower_is_better else (
                (value - bad_threshold) / (good_threshold - bad_threshold))
    norm_val = max(0, min(1, norm_val))
    r_b, g_b, b_b, r_m, g_m, b_m, r_g, g_g, b_g = 220, 53, 69, 255, 193, 7, 40, 167, 69
    if norm_val < 0.5:
        t = norm_val * 2; r = int(r_b * (1 - t) + r_m * t); g = int(g_b * (1 - t) + g_m * t); b = int(
            b_b * (1 - t) + b_m * t)
    else:
        t = (norm_val - 0.5) * 2; r = int(r_m * (1 - t) + r_g * t); g = int(g_m * (1 - t) + g_g * t); b = int(
            b_m * (1 - t) + b_g * t)
    return f"rgb({r},{g},{b})"


def format_metric_display_value(value):
    if pd.isna(value): return "N/A"
    if isinstance(value, pd.Timedelta):
        s = value.total_seconds()
        if pd.isna(s): return "N/A"
        d, rem = divmod(s, 86400);
        h, rem = divmod(rem, 3600);
        m, sec = divmod(rem, 60)
        if d > 0: return f"{int(d)}d {int(h)}h {int(m)}m"
        if h > 0: return f"{int(h)}h {int(m)}m {int(sec)}s"
        if m > 0: return f"{int(m)}m {int(sec)}s"
        return f"{s:.2f}s"
    if isinstance(value, pd.Timestamp): return value.strftime('%Y-%m-%d %H:%M:%S')
    if isinstance(value, float): return f"{value:.2f}"
    return str(value)


def _generate_plots_and_metrics_html_reporting(stats_data, metrics_config):
    logger.debug("Reporting: Generating plots and metrics HTML.")
    plot_data_base64 = ""
    fig = None
    try:
        fig = plt.figure(figsize=(12, 10))
        plot_count = 0
        # Equity Curve
        if '_equity_curve' in stats_data and isinstance(stats_data['_equity_curve'], dict) and 'Equity' in stats_data[
            '_equity_curve']:
            equity_data = stats_data['_equity_curve']['Equity']
            if equity_data is not None and (
                    isinstance(equity_data, list) and len(equity_data) > 0 or isinstance(equity_data,
                                                                                         pd.Series) and not equity_data.empty):
                plot_count += 1;
                plt.subplot(2, 2, plot_count)
                x_label = 'Data Points'
                if isinstance(equity_data, pd.Series) and isinstance(equity_data.index, pd.DatetimeIndex):
                    plt.plot(equity_data.index, equity_data, label='Equity');
                    x_label = 'Time'
                else:
                    plt.plot(equity_data, label='Equity')
                plt.title('Equity Curve');
                plt.xlabel(x_label);
                plt.ylabel('Equity');
                plt.legend();
                plt.xticks(rotation=15)
        # Drawdown
        if '_equity_curve' in stats_data and isinstance(stats_data['_equity_curve'], dict) and 'Drawdown' in stats_data[
            '_equity_curve']:
            drawdown_data = stats_data['_equity_curve']['Drawdown']
            if drawdown_data is not None and (
                    isinstance(drawdown_data, list) and len(drawdown_data) > 0 or isinstance(drawdown_data,
                                                                                             pd.Series) and not drawdown_data.empty):
                plot_count += 1;
                plt.subplot(2, 2, plot_count)
                x_label = 'Data Points'
                if isinstance(drawdown_data, pd.Series) and isinstance(drawdown_data.index, pd.DatetimeIndex):
                    plt.plot(drawdown_data.index, drawdown_data, label='Drawdown', color='r');
                    x_label = 'Time'
                else:
                    plt.plot(drawdown_data, label='Drawdown', color='r')
                plt.title('Drawdown');
                plt.xlabel(x_label);
                plt.ylabel('Drawdown Pct');
                plt.legend();
                plt.xticks(rotation=15)

        if '_trades' in stats_data and isinstance(stats_data['_trades'], pd.DataFrame) and not stats_data[
            '_trades'].empty:
            trades_df = stats_data['_trades']
            if 'ReturnPct' in trades_df.columns and not trades_df['ReturnPct'].dropna().empty:
                plot_count += 1;
                plt.subplot(2, 2, plot_count)
                plt.hist(trades_df['ReturnPct'].dropna() * 100, bins=50, alpha=0.75, label='Trade Returns (%)')
                plt.title('Trade Returns Distribution');
                plt.xlabel('Return %');
                plt.ylabel('Frequency');
                plt.legend()
            if 'Duration' in trades_df.columns and pd.api.types.is_timedelta64_dtype(trades_df['Duration']):
                duration_s = trades_df['Duration'].dt.total_seconds().dropna()
                if not duration_s.empty:
                    plot_count += 1;
                    plt.subplot(2, 2, plot_count)
                    plt.hist(duration_s, bins=50, alpha=0.75, label='Trade Duration (seconds)')
                    plt.title('Trade Duration Distribution');
                    plt.xlabel('Duration (s)');
                    plt.ylabel('Frequency');
                    plt.legend()
        if plot_count > 0:
            plt.tight_layout();
            buf = BytesIO();
            plt.savefig(buf, format='png');
            buf.seek(0)
            plot_data_base64 = base64.b64encode(buf.read()).decode('utf-8')
        else:
            logger.warning(
                "Reporting: No custom plots generated due to missing/empty data in _equity_curve or _trades.")
    except Exception as e:
        logger.error(f"Reporting: Error during plot generation: {e}", exc_info=True)
    finally:
        if fig: plt.close(fig)
    metrics_html = ""
    for m_info in metrics_config:
        val = stats_data.get(m_info['key']);
        c_val = val.total_seconds() if isinstance(val, pd.Timedelta) and not pd.isna(val) else val
        disp_val = format_metric_display_value(val);
        c_style = get_gradient_rgb(c_val, m_info['good'], m_info['bad'], m_info.get('lower_is_better', False))
        metrics_html += f'<div class="metric-card"><span class="metric-label">{m_info["label"]}</span><span class="metric-value" style="color: {c_style};">{disp_val}</span></div>'
    return plot_data_base64, metrics_html


def _write_combined_html_report_reporting(output_dir, plot_b64, metrics_html, color_key_html, source_json_name=""):
    path = os.path.join(output_dir, 'combined_results.html')
    title = f"Backtest Analysis (Source: {os.path.basename(source_json_name)})" if source_json_name else "Backtest Analysis"
    with open(path, 'w', encoding='utf-8') as f:
        f.write(f"""<!DOCTYPE html><html lang="en"><head><meta charset="UTF-8"><title>{title}</title><style>
        body{{font-family:Arial,sans-serif;margin:0;padding:0;background-color:#f4f4f4;color:#333}}
        .container{{max-width:1200px;margin:20px auto;padding:20px;background-color:#fff;box-shadow:0 0 10px rgba(0,0,0,0.1);border-radius:8px}}
        h1,h2{{color:#333;text-align:center;margin-top:25px;margin-bottom:15px}} iframe{{border:1px solid #ddd;margin-bottom:20px;border-radius:4px}}
        .metrics-header{{text-align:left;margin-top:30px;margin-bottom:10px;padding-left:20px}}
        .metrics-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:15px;padding:0 20px 20px}}
        .metric-card{{border:1px solid #ddd;border-radius:8px;padding:20px;text-align:center;background-color:#fff;box-shadow:0 2px 4px rgba(0,0,0,0.05);transition:transform .2s ease-in-out}}.metric-card:hover{{transform:translateY(-5px)}}
        .metric-label{{display:block;font-size:.95em;color:#555;margin-bottom:10px}}
        .metric-value{{display:block;font-size:1.6em;font-weight:700}}
        .plot-section img{{max-width:100%;height:auto;display:block;margin:20px auto;border:1px solid #ddd;border-radius:4px;box-shadow:0 0 8px rgba(0,0,0,0.1)}}
        .plot-section p{{text-align:center;color:#777;font-style:italic;padding:10px}}
        .color-key{{padding:15px 20px;text-align:center;border-top:1px solid #eee;margin-top:20px;font-size:.9em}}
        .gradient-bar{{display:inline-block;width:120px;height:18px;background:linear-gradient(to right,rgb(220,53,69),rgb(255,193,7),rgb(40,167,69));vertical-align:middle;margin:0 8px;border-radius:4px;border:1px solid #ccc}}
        </style></head><body><div class="container"><h1>Backtest Analysis</h1>
        <h2>Main Backtest Plot (Interactive)</h2><iframe src="backtest_plot.html" width="100%" height="600px"></iframe>
        {"<h2>Performance Plots (Static)</h2><div class='plot-section'><img src='data:image/png;base64," + plot_b64 + "' alt='Static Performance Plots'></div>" if plot_b64 else "<h2>Performance Plots (Static)</h2><div class='plot-section'><p>Static plots could not be generated (check logs for details regarding _equity_curve or _trades data).</p></div>"}
        <h2 class="metrics-header">Key Performance Indicators</h2>{color_key_html}
        <div class="metrics-grid">{metrics_html}</div></div></body></html>""")
    logger.info(f"Reporting: Combined HTML report saved to {path}")


def _write_summary_text_file_reporting(stats_data, output_dir, gen_time_str, source_json_name=""):
    path = os.path.join(output_dir, 'summary.txt')
    run_conf = stats_data.get('run_config', {})
    with open(path, 'w', encoding='utf-8') as f:
        f.write(
            f"Backtest Summary (Source: {os.path.basename(source_json_name) if source_json_name else 'N/A'})\n===================================\n\n")
        f.write(f"Backtest Configuration (from JSON):\n")
        f.write(f"  Original Data File: {run_conf.get('csv_file', 'N/A')}\n")
        f.write(f"  Backtest Execution Start Time: {run_conf.get('backtest_start_time', 'N/A')}\n")
        f.write(f"  Initial Capital: ${run_conf.get('initial_cash', 0):,.2f}\n")
        f.write(f"  Commission Fee: {run_conf.get('commission_fee', 0) * 100:.4f}%\n")
        f.write(f"  Price Scalar: {run_conf.get('price_scalar', 'N/A')}\n")
        f.write(f"  Fractal Period (n): {run_conf.get('n_fractal_period', 'N/A')}\n\n")
        f.write(
            f"Report Generation Time: {gen_time_str}\nStrategy Identifier (JSON): {stats_data.get('_strategy', 'N/A')}\n\nOverall Performance:\n")
        for k in ['Start', 'End', 'Duration', 'Exposure Time [%]']: f.write(
            f"  {k}: {format_metric_display_value(stats_data.get(k, 'N/A'))}\n")
        f.write("\nEquity & Returns:\n")
        for k in ['Equity Final [$]', 'Equity Peak [$]', 'Return [%]', 'Return (Ann.) [%]', 'Buy & Hold Return [%]',
                  'Volatility (Ann.) [%]', 'CAGR [%]']: f.write(
            f"  {k}: {format_metric_display_value(stats_data.get(k, 'N/A'))}\n")
        f.write("\nRisk & Ratios:\n")
        for k in ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'SQN', 'Kelly Criterion', 'Alpha [%]',
                  'Beta']: f.write(f"  {k}: {format_metric_display_value(stats_data.get(k, 'N/A'))}\n")
        f.write("\nDrawdown:\n")
        for k in ['Max. Drawdown [%]', 'Avg. Drawdown [%]', 'Max. Drawdown Duration',
                  'Avg. Drawdown Duration']: f.write(
            f"  {k}: {format_metric_display_value(stats_data.get(k, stats_data.get(k.replace('. ', ''), 'N/A')))}\n")  # Handle key variations
        f.write("\nTrades:\n")
        for k in ['# Trades', 'Win Rate [%]', 'Profit Factor', 'Expectancy [%]', 'Avg. Trade [%]', 'Best Trade [%]',
                  'Worst Trade [%]', 'Avg. Trade Duration', 'Max. Trade Duration', 'Commissions [$]']: f.write(
            f"  {k}: {format_metric_display_value(stats_data.get(k, stats_data.get(k.replace('. ', ''), 'N/A')))}\n")
    logger.info(f"Reporting: Summary text file saved to {path}")


def preprocess_stats_data_reporting(stats_json_data):
    logger.debug("Reporting: Preprocessing stats data from JSON.")
    s = stats_json_data  # Use a shorter alias
    for k in ['Start', 'End']:
        if k in s and isinstance(s[k], str): s[k] = pd.to_datetime(s[k], errors='coerce')
    for k in ['Duration', 'Max. Drawdown Duration', 'Avg. Drawdown Duration', 'Avg. Trade Duration',
              'Max. Trade Duration']:
        if k in s and isinstance(s[k], str): s[k] = pd.to_timedelta(s[k], errors='coerce')

    if '_trades' in s and isinstance(s['_trades'], list):
        if s['_trades']:
            tdf = pd.DataFrame(s['_trades'])
            for col in ['EntryTime', 'ExitTime']:
                if col in tdf: tdf[col] = pd.to_datetime(tdf[col], errors='coerce')
            if 'Duration' in tdf: tdf['Duration'] = pd.to_timedelta(tdf['Duration'], errors='coerce')
            s['_trades'] = tdf
        else:
            s['_trades'] = pd.DataFrame()  # Empty list means empty DataFrame
    elif '_trades' not in s:
        s['_trades'] = pd.DataFrame()

    if '_equity_curve' in s and isinstance(s['_equity_curve'], dict):
        eq_json = s['_equity_curve']
        if all(key in eq_json for key in ['Index', 'Equity', 'Drawdown']):
            try:
                dt_idx = pd.to_datetime(eq_json['Index'])
                s['_equity_curve'] = {
                    'Equity': pd.Series(eq_json['Equity'], index=dt_idx, name='Equity'),
                    'Drawdown': pd.Series(eq_json['Drawdown'], index=dt_idx, name='Drawdown')
                }
                logger.info("Reporting: Reconstructed _equity_curve with DatetimeIndex.")
            except Exception as e:
                logger.warning(f"Reporting: Could not convert _equity_curve Index: {e}. Plotting vs sequence.",
                               exc_info=DEBUG_LOGGING)
                s['_equity_curve'] = {'Equity': eq_json.get('Equity', []),
                                      'Drawdown': eq_json.get('Drawdown', [])}  # Fallback to lists
        elif 'Equity' in eq_json and 'Drawdown' in eq_json:  # If Index is missing but data is there
            s['_equity_curve'] = {'Equity': eq_json['Equity'], 'Drawdown': eq_json['Drawdown']}
            logger.warning(
                "Reporting: _equity_curve from JSON is missing 'Index' list. Plots will use sequence numbers.")
        else:
            s['_equity_curve'] = {}  # Mark as unusable
    elif '_equity_curve' not in s:
        s['_equity_curve'] = {}

    return s


def generate_custom_reports_from_json(json_filepath, output_dir):
    logger.info(f"Reporting: Generating custom reports from {json_filepath}")
    try:
        with open(json_filepath, 'r', encoding='utf-8') as f:
            stats_raw = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Reporting: Error decoding JSON {json_filepath}: {e}"); return
    except Exception as e:
        logger.error(f"Reporting: Error loading JSON {json_filepath}: {e}", exc_info=True); return

    stats_processed = preprocess_stats_data_reporting(stats_raw)
    gen_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    plot_b64, metrics_html = _generate_plots_and_metrics_html_reporting(stats_processed, METRICS_CONFIG_REPORTING)
    _write_combined_html_report_reporting(output_dir, plot_b64, metrics_html, COLOR_KEY_HTML_CONTENT_REPORTING,
                                          source_json_name=json_filepath)
    _write_summary_text_file_reporting(stats_processed, output_dir, gen_time, source_json_name=json_filepath)
    logger.info(f"Reporting: Custom reports generated in '{output_dir}'.")


# --- Main Execution (Backtesting + Reporting) ---
if __name__ == "__main__":
    os.makedirs(BACKTESTS_DIR, exist_ok=True)
    logger.info(f"Backtests directory '{BACKTESTS_DIR}' ensured.")

    try:
        logger.info(f"Loading data from {CSV_FILE}...")
        df = pd.read_csv(CSV_FILE, parse_dates=['Date'])
        logger.info("Data loaded successfully.")
    except FileNotFoundError:
        logger.error(f"Error: {CSV_FILE} not found.");
        exit(1)
    except Exception as e:
        logger.error(f"Error loading or parsing {CSV_FILE}: {e}");
        exit(1)

    if 'Date' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['Date']): df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)
    elif not isinstance(df.index, pd.DatetimeIndex):
        logger.error("Date column not found or not properly set as DatetimeIndex.");
        exit(1)

    req_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    missing = [col for col in req_cols if col not in df.columns]
    if missing: logger.error(f"CSV missing required columns: {missing}."); exit(1)

    df.dropna(subset=req_cols, inplace=True)
    for col in ['Open', 'High', 'Low', 'Close']: df[col] *= PRICE_SCALAR
    logger.debug(f"Prices scaled by {PRICE_SCALAR}. Data shape: {df.shape}")

    bt_start_time_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"Starting backtest at: {bt_start_time_str}")

    bt = Backtest(df, ScalpingStrategy1BT, cash=INITIAL_CASH, commission=COMMISSION_FEE, exclusive_orders=True,
                  trade_on_close=True)
    stats = bt.run(progress=True)  # stats object from backtesting.py
    logger.info("Backtest run completed.")

    # 1. Save the standard backtesting.py plot
    plot_filepath = os.path.join(BACKTESTS_DIR, 'backtest_plot.html')
    bt.plot(filename=plot_filepath, open_browser=False)
    logger.info(f"Standard backtest plot saved to {plot_filepath}")

    # 2. Prepare data and save detailed stats to JSON
    json_output_data = {}
    for key, value in stats.items():  # Iterate over the original stats object
        if isinstance(value, pd.Timestamp):
            json_output_data[key] = value.isoformat()
        elif isinstance(value, pd.Timedelta):
            json_output_data[key] = str(value)
        elif key == '_strategy':
            json_output_data[key] = str(value)  # Avoid serializing full object
        elif key not in ['_trades', '_equity_curve']:  # Handle _trades and _equity_curve separately
            try:
                json_output_data[key] = value.to_dict() if isinstance(value, (pd.Series, pd.DataFrame)) else value
            except:
                json_output_data[key] = str(value)  # Fallback

    if '_trades' in stats and isinstance(stats['_trades'], pd.DataFrame):
        trades_df_json = stats['_trades'].copy()
        for col, dtype in trades_df_json.dtypes.items():
            if pd.api.types.is_timedelta64_ns_dtype(dtype):
                trades_df_json[col] = trades_df_json[col].astype(str)
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                trades_df_json[col] = trades_df_json[col].dt.strftime('%Y-%m-%d %H:%M:%S')
        json_output_data['_trades'] = trades_df_json.to_dict(orient='records')

    if '_equity_curve' in stats and isinstance(stats['_equity_curve'], pd.DataFrame):
        equity_df_raw = stats['_equity_curve']  # This has 'Equity', 'DrawdownPct', 'DrawdownDuration'
        json_output_data['_equity_curve'] = {
            'Equity': equity_df_raw['Equity'].tolist(),
            'Drawdown': equity_df_raw['DrawdownPct'].tolist(),  # Map DrawdownPct to 'Drawdown' for reporting
            'Index': equity_df_raw.index.strftime('%Y-%m-%d %H:%M:%S').tolist()
        }

    json_output_data['run_config'] = {
        'backtest_start_time': bt_start_time_str, 'csv_file': CSV_FILE, 'initial_cash': INITIAL_CASH,
        'commission_fee': COMMISSION_FEE, 'price_scalar': PRICE_SCALAR, 'n_fractal_period': N_FRACTAL_PERIOD
    }

    stats_json_filepath = os.path.join(BACKTESTS_DIR, 'backtest_stats_and_data.json')
    try:
        with open(stats_json_filepath, 'w', encoding='utf-8') as f:
            json.dump(json_output_data, f, indent=4, default=str)  # default=str for any missed complex types
        logger.info(f"Detailed backtest stats and data saved to {stats_json_filepath}")
    except Exception as e:
        logger.error(f"Error saving stats to JSON: {e}", exc_info=True)

    # 3. Generate custom reports using the saved JSON
    if os.path.exists(stats_json_filepath):
        logger.info("Proceeding to generate custom HTML and TXT reports...")
        generate_custom_reports_from_json(stats_json_filepath, BACKTESTS_DIR)
    else:
        logger.error(f"Could not generate custom reports because JSON file was not created: {stats_json_filepath}")

    logger.info(f"--- All Processing Complete ---")
    logger.info(f"All outputs (interactive plot, JSON data, custom HTML report, summary TXT) are in '{BACKTESTS_DIR}'.")