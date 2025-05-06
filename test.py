import time
import datetime
import requests
import pytz
import numpy as np
import pandas as pd
import joblib
import talib
from datetime import datetime as dt, timedelta
from tensorflow.keras.models import load_model
from flask import Flask, render_template_string, jsonify # Import Flask components
import traceback # Import traceback for detailed error info
import os # For path joining

# ── CONFIG ─────────────────────────────────────────────────────────
SYMBOL    = "BTC"
SEQ_LEN   = 60
THRESH    = 0.001  # 0.1% threshold for buy/sell signal
# Define the path to your historical data CSV
# Use os.path.join for better cross-platform compatibility
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # Get the directory of the current script
HISTORICAL_DATA_PATH = os.path.join(BASE_DIR, "backtest", "data", "BTC-1m-rand.csv")
MODEL_PATH = os.path.join(BASE_DIR, "backtest", "Trained", "btc_lstm_fixed_version.h5")
FEAT_SCALER_PATH = os.path.join(BASE_DIR, "bot_data", "feature_scaler.pkl")
TGT_SCALER_PATH = os.path.join(BASE_DIR, "bot_data", "target_scaler.pkl")


# Define the exact order of all 37 features as expected by the scaler
# This list includes the original OHLCV (excluding close) and the 33 indicators.
ALL_FEATURES = [
    "open", "high", "low", "volume",
    # Hilbert Transform (6)
    "HT_DCPERIOD","HT_PHASOR","HT_QUADRATURE",
    "HT_SINE","HT_SINE_LEAD","HT_TRENDMODE",
    # Candlestick Patterns (3)
    "CDL_HAMMER","CDL_ENGULFING","CDL_DOJI",
    # Price Transforms (6)
    "LOG","LINEARREG","WCLPRICE","TYPPRICE","MEDPRICE","AVGPRICE",
    # Volume (2)
    "AD","OBV",
    # Momentum (8)
    "MACD","MACD_signal","MACD_hist",
    "WILLR","STOCH_K","STOCH_D","RSI","STDDEV",
    # Volatility (6)
    "TRANGE","ATR","NATR","BB_upper","BB_middle","BB_lower",
    # Moving Averages (2)
    "EMA","SMA"
]

# ── INITIALIZE ─────────────────────────────────────────────────────
print("Initializing model and scalers...")
# Load the trained model and scalers
try:
    model      = load_model(MODEL_PATH, compile=False)
    feat_scaler= joblib.load(FEAT_SCALER_PATH)
    tgt_scaler = joblib.load(TGT_SCALER_PATH)
    print("✅ Model and scalers loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading model or scalers: {e}. Check paths:\nModel: {MODEL_PATH}\nFeat Scaler: {FEAT_SCALER_PATH}\nTgt Scaler: {TGT_SCALER_PATH}")
    exit() # Exit if essential files are missing
except Exception as e:
    print(f"An unexpected error occurred during initialization: {e}")
    traceback.print_exc() # Print detailed traceback
    exit()

# ── INDICATOR COMPUTATION ──────────────────────────────────────────
def compute_indicators(df):
    """
    Compute the 33 technical indicators and combine with original OHLCV data
    to match the 37 features expected by the scaler.

    Args:
        df (pd.DataFrame): DataFrame with OHLCV data (and timestamp index).

    Returns:
        pd.DataFrame: DataFrame with 37 features (OHLCV + Indicators)
                      for the last SEQ_LEN periods, in the correct order.
                      Returns None if not enough data after dropping NaNs.
    """
    # Debug: Check input DataFrame size
    # print(f"compute_indicators: Input DataFrame shape: {df.shape}")

    # Ensure df has the necessary columns before proceeding
    required_cols = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: Input DataFrame to compute_indicators is missing required columns: {required_cols}")
        return None

    # Check if DataFrame has enough data for sequence length after potential NaNs from TA-Lib
    # A buffer is needed for TA-Lib lookback, handled by fetching more data in the main loop.
    # We check if we have at least SEQ_LEN rows *after* computing indicators and dropping NaNs.

    o,h,l,c,v = df.open.values, df.high.values, df.low.values, df.close.values, df.volume.values
    A = {} # Dictionary to hold calculated indicators

    # --- Compute all 33 indicators using TA-Lib ---
    # (Indicator computation code remains the same as before)
    # ── Hilbert Transform (6) ─────────────────────
    A["HT_DCPERIOD"]   = talib.HT_DCPERIOD(c)
    inph, quad         = talib.HT_PHASOR(c)
    A["HT_PHASOR"] = inph
    A["HT_QUADRATURE"] = quad
    sine, lead         = talib.HT_SINE(c)
    A["HT_SINE"] = sine
    A["HT_SINE_LEAD"]   = lead
    A["HT_TRENDMODE"] = talib.HT_TRENDMODE(c)
    # ── Candlestick Patterns (3) ──────────────────
    A["CDL_HAMMER"]    = talib.CDLHAMMER(o,h,l,c)
    A["CDL_ENGULFING"] = talib.CDLENGULFING(o,h,l,c)
    A["CDL_DOJI"]      = talib.CDLDOJI(o,h,l,c)
    # ── Price Transforms (6) ──────────────────────
    A["LOG"]       = np.log(c, where=c>0) # Handle potential log(0)
    A["LINEARREG"] = talib.LINEARREG(c, timeperiod=14)
    A["WCLPRICE"]  = (h + l + 2*c) / 4
    A["TYPPRICE"]  = (h + l + c) / 3
    A["MEDPRICE"]  = (h + l) / 2
    A["AVGPRICE"]  = (o + h + l + c) / 4
    # ── Volume‑based (2) ─────────────────────────
    A["AD"]  = talib.AD(h, l, c, v)
    A["OBV"] = talib.OBV(c, v)
    # ── Momentum (8) ─────────────────────────────
    macd,sig,hist = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
    A["MACD"] = macd
    A["MACD_signal"] = sig
    A["MACD_hist"] = hist
    A["WILLR"]   = talib.WILLR(h,l,c,timeperiod=14)
    st_k,st_d     = talib.STOCH(h,l,c,fastk_period=14, slowk_period=3, slowd_period=3)
    A["STOCH_K"] = st_k
    A["STOCH_D"] = st_d
    A["RSI"]     = talib.RSI(c,timeperiod=14)
    A["STDDEV"]  = talib.STDDEV(c,timeperiod=14, nbdev=1)
    # ── Volatility (6) ────────────────────────────
    A["TRANGE"] = talib.TRANGE(h,l,c)
    A["ATR"]    = talib.ATR(h,l,c,timeperiod=14)
    A["NATR"]   = talib.NATR(h,l,c,timeperiod=14)
    up,mid,dn     = talib.BBANDS(c,timeperiod=14, nbdevup=2, nbdevdn=2)
    A["BB_upper"] = up
    A["BB_middle"] = mid
    A["BB_lower"] = dn
    # ── Moving Averages (2) ───────────────────────
    A["EMA"] = talib.EMA(c,timeperiod=14)
    A["SMA"] = talib.SMA(c,timeperiod=14)
    # --- End of indicator computation ---


    # Create a DataFrame for the calculated indicators
    df_ind = pd.DataFrame(A, index=df.index)

    # Select the relevant OHLCV columns from the original DataFrame
    df_ohlcv = df[['open', 'high', 'low', 'volume']]

    # Concatenate OHLCV and indicators DataFrames
    df_combined = pd.concat([df_ohlcv, df_ind], axis=1)

    # Drop any rows that contain NaN values after indicator computation
    initial_rows = len(df_combined)
    df_combined.dropna(inplace=True)

    # Ensure the combined DataFrame has enough rows after dropping NaNs
    if len(df_combined) < SEQ_LEN:
        # print(f"Warning: Not enough data ({len(df_combined)} rows) after dropping NaNs for sequence length {SEQ_LEN}. Required: {SEQ_LEN}. Skipping prediction for this cycle.")
        return None # Return None if not enough data after dropping NaNs

    # Slice to get the last SEQ_LEN periods
    df_sliced = df_combined.iloc[-SEQ_LEN:]

    # Reorder columns to match the scaler's expectation (defined in ALL_FEATURES)
    try:
        df_ordered = df_sliced[ALL_FEATURES].copy()
    except KeyError as e:
         print(f"Error: Column mismatch during reordering in compute_indicators: {e}")
         print(f"Columns in df_sliced: {df_sliced.columns.tolist()}")
         print(f"Expected columns (ALL_FEATURES): {ALL_FEATURES}")
         return None

    return df_ordered

# --- METRICS CALCULATION ---
def calculate_metrics(trades_df, initial_capital, equity_curve):
    """Calculates performance metrics from a list of trades and equity curve."""
    if trades_df.empty:
        return {
            "Total Trades": 0, "Net Profit": 0, "Profit Factor": 0,
            "Winning Trades": 0, "Losing Trades": 0, "Win Rate (%)": 0,
            "Average Trade PnL": 0, "Max Drawdown (%)": 0,
            "Final Capital": initial_capital, "Cumulative Return (%)": 0,
        }

    # PnL is already calculated during backtest simulation
    total_trades = len(trades_df)
    net_profit = trades_df['PnL'].sum()
    gross_profit = trades_df[trades_df['PnL'] > 0]['PnL'].sum()
    gross_loss = abs(trades_df[trades_df['PnL'] < 0]['PnL'].sum())

    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    winning_trades = len(trades_df[trades_df['PnL'] > 0])
    losing_trades = len(trades_df[trades_df['PnL'] < 0])
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    avg_trade_pnl = net_profit / total_trades if total_trades > 0 else 0

    # Max Drawdown calculation from equity curve
    peak = equity_curve.cummax()
    drawdown = (equity_curve - peak) / peak
    max_drawdown = abs(drawdown.min()) * 100 if not drawdown.empty and not drawdown.isnull().all() else 0

    final_capital = initial_capital + net_profit
    cumulative_return = (net_profit / initial_capital) * 100 if initial_capital > 0 else 0

    metrics = {
        "Initial Capital": initial_capital,
        "Final Capital": round(final_capital, 2),
        "Net Profit": round(net_profit, 2),
        "Cumulative Return (%)": round(cumulative_return, 2),
        "Total Trades": total_trades,
        "Winning Trades": winning_trades,
        "Losing Trades": losing_trades,
        "Win Rate (%)": round(win_rate, 2),
        "Average Trade PnL": round(avg_trade_pnl, 2),
        "Profit Factor": round(profit_factor, 2) if profit_factor != np.inf else "inf",
        "Max Drawdown (%)": round(max_drawdown, 2),
    }
    return metrics

# ── BACKTESTING LOGIC ────────────────────────────────────────────
def run_backtest(df_historical, initial_capital=10000):
    """
    Runs the backtest over historical data, simulates trades, and collects results.

    Args:
        df_historical (pd.DataFrame): DataFrame with historical OHLCV data.
        initial_capital (float): Starting capital for the backtest.

    Returns:
        tuple: (trades_df, equity_curve, raw_results_df)
               - trades_df: DataFrame of executed trades.
               - equity_curve: Series representing portfolio value over time.
               - raw_results_df: DataFrame with price, prediction, signal per timestep.
    """
    raw_results = [] # Store raw price, prediction, signal per step
    trades = []      # Store executed trades
    position = None  # 'Long', 'Short', or None
    entry_price = 0
    entry_time = None
    equity = initial_capital
    equity_curve_list = [] # List to build equity curve data

    lookback_buffer = 100 # Buffer for TA-Lib indicator calculation
    start_index = SEQ_LEN + lookback_buffer
    if start_index >= len(df_historical):
        print(f"Error: Not enough historical data ({len(df_historical)} rows) for backtesting.")
        return pd.DataFrame(), pd.Series(dtype=float), pd.DataFrame()

    print(f"Starting backtest from index {start_index} (Timestamp: {df_historical.index[start_index]})...")

    for i in range(start_index, len(df_historical)):
        curr_timestamp = df_historical.index[i]
        curr_price = df_historical['close'].iloc[i]

        # Record equity at the start of the timestep
        equity_curve_list.append({'timestamp': curr_timestamp, 'equity': equity})

        # Get data window for prediction
        window_start_index = max(0, i - (SEQ_LEN + lookback_buffer) + 1)
        df_window = df_historical.iloc[window_start_index:i+1] # Include current row for prediction base

        features_df = compute_indicators(df_window)

        signal = "HOLD" # Default signal
        y_pred = np.nan # Default prediction

        if features_df is not None:
            try:
                scaled_features = feat_scaler.transform(features_df)
                X = scaled_features.reshape(1, SEQ_LEN, scaled_features.shape[1])
                y_scaled = model.predict(X, verbose=0)[0]
                y_pred = tgt_scaler.inverse_transform([y_scaled])[0][0]

                # Determine trading signal based on prediction vs *previous* close
                # (Predicting the *next* close based on data up to *current* close)
                if y_pred > curr_price * (1 + THRESH): signal = "BUY"
                elif y_pred < curr_price * (1 - THRESH): signal = "SELL"
                else: signal = "HOLD"

            except Exception as e:
                print(f"Error during prediction/scaling at index {i}: {e}")
                # Keep signal as HOLD if prediction fails

        # Store raw results for this timestamp (for potential downsampling later)
        raw_results.append({
            "timestamp": curr_timestamp, # Keep as datetime object for now
            "price": curr_price,
            "predicted_price": y_pred,
            "signal": signal
        })

        # --- Trade Execution Logic (Simplified: exit on opposite signal) ---
        if position == 'Long':
            if signal == 'SELL':
                pnl = curr_price - entry_price # Assume 1 unit trade size
                equity += pnl
                trades.append({
                    "Entry Time": entry_time, "Exit Time": curr_timestamp,
                    "Direction": position, "Entry Price": entry_price,
                    "Exit Price": curr_price, "PnL": pnl
                })
                position = None
        elif position == 'Short':
            if signal == 'BUY':
                pnl = entry_price - curr_price # Assume 1 unit trade size
                equity += pnl
                trades.append({
                    "Entry Time": entry_time, "Exit Time": curr_timestamp,
                    "Direction": position, "Entry Price": entry_price,
                    "Exit Price": curr_price, "PnL": pnl
                })
                position = None

        # --- Entry Logic ---
        if position is None: # Only enter if flat
            if signal == 'BUY':
                position = 'Long'
                entry_price = curr_price
                entry_time = curr_timestamp
            elif signal == 'SELL':
                position = 'Short'
                entry_price = curr_price
                entry_time = curr_timestamp

        # Optional: Print progress
        if (i - start_index + 1) % 500 == 0: # Print every 500 steps
            print(f"Processed {i - start_index + 1}/{len(df_historical) - start_index} timestamps...")


    print("Backtest finished.")

    # Convert results to DataFrames/Series
    trades_df = pd.DataFrame(trades)
    raw_results_df = pd.DataFrame(raw_results).set_index('timestamp')

    # Create equity curve Series
    equity_curve_df = pd.DataFrame(equity_curve_list).set_index('timestamp')
    # Ensure equity curve covers the full range and fill gaps
    equity_curve = equity_curve_df['equity'].reindex(df_historical.index[start_index:]).ffill()
    equity_curve.fillna(initial_capital, inplace=True) # Fill initial NaNs if any

    return trades_df, equity_curve, raw_results_df

# ── FLASK WEB SERVER ───────────────────────────────────────────────
app = Flask(__name__)

# Store processed backtest results globally
PROCESSED_BACKTEST_DATA = {}

@app.route('/')
def index():
    """Serves the main HTML page with metrics and chart areas."""
    # Updated HTML with Metrics Section
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Trading Bot Backtest Visualization</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
        <!-- Note: Annotation plugin might need updates or different approach -->
        <!-- <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@1.0.2"></script> -->
        <style>
            body { font-family: sans-serif; background-color: #f0f0f0; margin: 0; padding: 20px; display: flex; flex-direction: column; align-items: center; }
            .container { background-color: #fff; padding: 30px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1); width: 95%; max-width: 1400px; margin-bottom: 20px; }
            h1, h2 { text-align: center; color: #333; margin-bottom: 20px; }
            .chart-container { position: relative; width: 100%; height: 450px; margin-bottom: 30px; }
            canvas { display: block; width: 100% !important; height: 100% !important; }
            .metrics-container { background-color: #f9f9f9; padding: 20px; border-radius: 8px; margin-bottom: 30px; border: 1px solid #eee; }
            .metrics-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; }
            .metric-item { background-color: #fff; padding: 15px; border-radius: 5px; box-shadow: 0 2px 5px rgba(0,0,0,0.05); text-align: center; }
            .metric-label { font-size: 0.9em; color: #666; margin-bottom: 5px; display: block; }
            .metric-value { font-size: 1.2em; font-weight: bold; color: #333; }
            #loadingMessage, #errorMessage { text-align: center; padding: 20px; font-size: 1.1em; color: #555; }
            #errorMessage { color: red; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Trading Bot Backtest Results</h1>

            <!-- Metrics Section -->
            <div class="metrics-container">
                <h2>Performance Metrics</h2>
                <div id="metricsGrid" class="metrics-grid">
                    <div id="loadingMetrics">Loading metrics...</div>
                </div>
            </div>

            <!-- Price Chart Section -->
            <h2>Price Chart (Downsampled)</h2>
            <div class="chart-container">
                <canvas id="backtestChart"></canvas>
            </div>

             <!-- Equity Curve Section -->
            <h2>Equity Curve</h2>
            <div class="chart-container">
                <canvas id="equityChart"></canvas>
            </div>

            <!-- Loading/Error Messages -->
            <div id="loadingMessage" style="display: none;">Loading chart data...</div>
            <div id="errorMessage" style="display: none;"></div>

        </div> <!-- End of container -->
        <script src="/static/script.js"></script>
    </body>
    </html>
    """
    return render_template_string(html_content)

@app.route('/static/script.js')
def serve_script():
    """Serves the updated JavaScript file."""
    # Updated JS to handle metrics, downsampled chart, and equity curve
    js_content = """
document.addEventListener('DOMContentLoaded', function() {
    const loadingMessage = document.getElementById('loadingMessage');
    const errorMessage = document.getElementById('errorMessage');
    const metricsGrid = document.getElementById('metricsGrid');
    const loadingMetrics = document.getElementById('loadingMetrics');
    const priceChartCanvas = document.getElementById('backtestChart');
    const equityChartCanvas = document.getElementById('equityChart');

    loadingMessage.style.display = 'block';
    errorMessage.style.display = 'none';
    if(loadingMetrics) loadingMetrics.style.display = 'block';

    fetch('/backtest_data')
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            loadingMessage.style.display = 'none';
            if(loadingMetrics) loadingMetrics.style.display = 'none';

            // --- 1. Display Metrics ---
            displayMetrics(data.metrics);

            // --- 2. Render Price Chart (Downsampled) ---
            if (data.chart_data && priceChartCanvas) {
                renderPriceChart(data.chart_data);
            } else {
                 if(priceChartCanvas) priceChartCanvas.parentElement.innerHTML = '<p style="text-align:center;">Price chart data not available.</p>';
            }

            // --- 3. Render Equity Curve Chart ---
            if (data.equity_curve && equityChartCanvas) {
                renderEquityCurveChart(data.equity_curve);
            } else {
                 if(equityChartCanvas) equityChartCanvas.parentElement.innerHTML = '<p style="text-align:center;">Equity curve data not available.</p>';
            }

        })
        .catch(error => {
            console.error('Error fetching backtest data:', error);
            loadingMessage.style.display = 'none';
            if(loadingMetrics) loadingMetrics.style.display = 'none';
            errorMessage.textContent = `Error loading backtest data: ${error.message}. Check Flask server logs.`;
            errorMessage.style.display = 'block';
            if(metricsGrid) metricsGrid.innerHTML = '<p style="color: red;">Failed to load metrics.</p>';
            if(priceChartCanvas) priceChartCanvas.parentElement.innerHTML = '<p style="color: red;">Failed to load price chart.</p>';
            if(equityChartCanvas) equityChartCanvas.parentElement.innerHTML = '<p style="color: red;">Failed to load equity chart.</p>';
        });

    function displayMetrics(metrics) {
        if (!metricsGrid) return;
        metricsGrid.innerHTML = ''; // Clear loading/previous metrics
        if (!metrics || Object.keys(metrics).length === 0) {
            metricsGrid.innerHTML = '<p>No metrics available.</p>';
            return;
        }
        for (const [key, value] of Object.entries(metrics)) {
            const metricItem = document.createElement('div');
            metricItem.classList.add('metric-item');
            const label = document.createElement('span');
            label.classList.add('metric-label');
            label.textContent = key;
            const val = document.createElement('span');
            val.classList.add('metric-value');
             // Basic formatting
            if (key.toLowerCase().includes('capital') || key.toLowerCase().includes('profit') || key.toLowerCase().includes('pnl')) {
                 val.textContent = typeof value === 'number' ? '$' + value.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2}) : value;
            } else if (key.toLowerCase().includes('%')) {
                 val.textContent = typeof value === 'number' ? `${value.toFixed(2)}%` : value;
            } else {
                val.textContent = value;
            }
            metricItem.appendChild(label);
            metricItem.appendChild(val);
            metricsGrid.appendChild(metricItem);
        }
    }

    function renderPriceChart(chartData) {
        const ctx = priceChartCanvas.getContext('2d');
        const timestamps = chartData.map(item => new Date(item.timestamp));
        const prices = chartData.map(item => item.price);
        const predictedPrices = chartData.map(item => item.predicted_price);

        new Chart(ctx, {
            type: 'line',
            data: {
                labels: timestamps,
                datasets: [
                    { label: 'Actual Price', data: prices, borderColor: 'blue', borderWidth: 1.5, pointRadius: 0, fill: false, tension: 0.1 },
                    { label: 'Predicted Price', data: predictedPrices, borderColor: 'rgba(255, 99, 132, 0.7)', borderDash: [5, 5], borderWidth: 1, pointRadius: 0, fill: false, tension: 0.1 }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: {
                    x: { type: 'time', time: { unit: 'hour', tooltipFormat: 'MMM d, HH:mm' }, title: { display: true, text: 'Timestamp' } },
                    y: { title: { display: true, text: 'Price (USD)' }, ticks: { callback: value => '$' + value.toLocaleString() } }
                },
                plugins: { tooltip: { mode: 'index', intersect: false } }
            }
        });
    }

     function renderEquityCurveChart(equityCurveData) {
        const ctx = equityChartCanvas.getContext('2d');
        const timestamps = equityCurveData.map(item => new Date(item.timestamp));
        const equityValues = equityCurveData.map(item => item.equity);

        new Chart(ctx, {
            type: 'line',
            data: {
                labels: timestamps,
                datasets: [{
                    label: 'Equity Curve', data: equityValues, borderColor: 'rgba(40, 167, 69, 0.8)', backgroundColor: 'rgba(40, 167, 69, 0.1)',
                    borderWidth: 2, pointRadius: 0, fill: true, tension: 0.1
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                scales: {
                    x: { type: 'time', time: { unit: 'day', tooltipFormat: 'MMM d, yyyy' }, title: { display: true, text: 'Timestamp' } },
                    y: { title: { display: true, text: 'Portfolio Equity (USD)' }, ticks: { callback: value => '$' + value.toLocaleString() } }
                },
                plugins: { tooltip: { mode: 'index', intersect: false }, legend: { display: false } }
            }
        });
    }
});
    """
    from flask import Response
    return Response(js_content, mimetype='application/javascript')


@app.route('/backtest_data')
def backtest_data():
    """Returns the processed backtest results (metrics, downsampled chart data, equity curve) as JSON."""
    # Return the globally stored *processed* backtest data
    if not PROCESSED_BACKTEST_DATA:
         return jsonify({"error": "Backtest data not processed yet or failed."}), 500
    return jsonify(PROCESSED_BACKTEST_DATA)

# ── MAIN EXECUTION ───────────────────────────────────────────────
if __name__ == "__main__":
    INITIAL_CAPITAL = 10000
    DOWNSAMPLE_RULE = 'H' # Downsample to Hourly ('D' for Daily, '15T' for 15-min, etc.)

    print(f"Loading historical data from {HISTORICAL_DATA_PATH}...")
    try:
        # Load historical data
        df_historical = pd.read_csv(
            HISTORICAL_DATA_PATH,
            header=None, # No header row
            names=['timestamp', 'open', 'high', 'low', 'close', 'volume'], # Provide column names
            parse_dates=['timestamp'], # Parse the 'timestamp' column as dates
            index_col='timestamp' # Set timestamp as index directly
        )
        df_historical.sort_index(inplace=True) # Ensure data is sorted by timestamp
        print(f"✅ Historical data loaded. Shape: {df_historical.shape}")
        print(f"   Time range: {df_historical.index.min()} to {df_historical.index.max()}")

    except FileNotFoundError:
        print(f"Error: Historical data file not found at {HISTORICAL_DATA_PATH}.")
        exit()
    except Exception as e:
        print(f"An error occurred while loading historical data: {e}")
        traceback.print_exc()
        exit()

    # --- Run Backtest and Process Results ---
    print("\nRunning backtest...")
    trades_df, equity_curve, raw_results_df = run_backtest(df_historical, INITIAL_CAPITAL)

    if not raw_results_df.empty:
        print("Calculating metrics...")
        metrics = calculate_metrics(trades_df, INITIAL_CAPITAL, equity_curve)

        print("Downsampling chart data...")
        # Resample raw results for the chart (e.g., hourly)
        chart_df_downsampled = raw_results_df[['price', 'predicted_price']].resample(DOWNSAMPLE_RULE).agg({
            'price': 'last',             # Take the last price in the period
            'predicted_price': 'mean'    # Average prediction in the period (or 'last')
        }).dropna() # Drop periods with no data

        # Resample equity curve for consistency (optional, but good practice)
        equity_curve_downsampled = equity_curve.resample(DOWNSAMPLE_RULE).last().dropna()


        print("Preparing data for web server...")
        # Prepare the final data structure for the web server
        PROCESSED_BACKTEST_DATA = {
            "metrics": metrics,
            "trades": trades_df.to_dict(orient='records'), # Keep trades if needed, convert times
             # Convert downsampled chart data to list of dicts
            "chart_data": [
                {"timestamp": idx.isoformat(), "price": row['price'], "predicted_price": row['predicted_price']}
                for idx, row in chart_df_downsampled.iterrows()
            ],
            # Convert downsampled equity curve data to list of dicts
            "equity_curve": [
                {"timestamp": idx.isoformat(), "equity": val}
                for idx, val in equity_curve_downsampled.items()
            ]
        }
        print(f"✅ Backtest processed. Metrics calculated. Chart data downsampled to '{DOWNSAMPLE_RULE}'.")
        print(f"   Downsampled chart points: {len(PROCESSED_BACKTEST_DATA['chart_data'])}")

        # --- Optional: Save processed data to JSON ---
        # try:
        #     processed_json_path = os.path.join(BASE_DIR, "processed_backtest_results.json")
        #     with open(processed_json_path, 'w') as f:
        #         # Need custom handler for Timestamps if not using isoformat above
        #         json.dump(PROCESSED_BACKTEST_DATA, f, indent=4, default=str)
        #     print(f"   Processed results saved to {processed_json_path}")
        # except Exception as e:
        #     print(f"   Warning: Could not save processed results to JSON: {e}")

    else:
        print("Backtest did not produce results. Cannot start web server.")
        exit()


    # Start the Flask web server
    print("\nStarting Flask web server...")
    print("Open your browser and go to http://127.0.0.1:5000/ to see the backtest results.")
    try:
        # Use host='0.0.0.0' to make it accessible on your network if needed
        app.run(host='127.0.0.1', port=5000, debug=False) # Turn debug off for stability if backtest is long
    except Exception as e:
        print(f"Error starting Flask server: {e}")
        traceback.print_exc()
