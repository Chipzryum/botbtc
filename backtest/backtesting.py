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

# ── CONFIG ─────────────────────────────────────────────────────────
SYMBOL    = "BTC"
SEQ_LEN   = 60
THRESH    = 0.001  # 0.1% threshold for buy/sell signal
# Define the path to your historical data CSV
# FIXED: Use a raw string (r"...") for Windows paths to avoid SyntaxError
HISTORICAL_DATA_PATH = r"C:\Users\Chipz\Documents\GitHub\botbtc\backtest\data\BTC-1m-rand.csv"


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
    model      = load_model("backtest/Trained/btc_lstm_fixed_version.h5", compile=False)
    feat_scaler= joblib.load("bot_data/feature_scaler.pkl")
    tgt_scaler = joblib.load("bot_data/target_scaler.pkl")
    print("✅ Model and scalers loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading model or scalers: {e}. Make sure 'btc_lstm_fixed_version.h5', 'feature_scaler.pkl', and 'target_scaler.pkl' are in the same directory.")
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
    # Handle potential log(0) by replacing with NaN and dropping later
    A["LOG"]       = np.log(c, where=c>0)
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

    # Create a DataFrame for the calculated indicators
    df_ind = pd.DataFrame(A, index=df.index)
    # Debug: Check indicator DataFrame shape
    # print(f"compute_indicators: Indicator DataFrame shape: {df_ind.shape}")


    # Select the relevant OHLCV columns from the original DataFrame
    df_ohlcv = df[['open', 'high', 'low', 'volume']]
    # Debug: Check OHLCV DataFrame shape
    # print(f"compute_indicators: OHLCV DataFrame shape: {df_ohlcv.shape}")


    # Concatenate OHLCV and indicators DataFrames
    # Debug: Check columns before concatenation
    # print(f"compute_indicators: OHLCV columns: {df_ohlcv.columns.tolist()}")
    # print(f"compute_indicators: Indicator columns: {df_ind.columns.tolist()}")
    df_combined = pd.concat([df_ohlcv, df_ind], axis=1)
    # Debug: Check combined DataFrame shape
    # print(f"compute_indicators: Combined DataFrame shape before dropna: {df_combined.shape}")


    # Drop any rows that contain NaN values after indicator computation
    initial_rows = len(df_combined)
    df_combined.dropna(inplace=True)
    # Debug: Check combined DataFrame shape after dropna
    # print(f"compute_indicators: Combined DataFrame shape after dropna: {df_combined.shape} (Dropped {initial_rows - len(df_combined)} rows)")


    # Ensure the combined DataFrame has enough rows after dropping NaNs
    if len(df_combined) < SEQ_LEN:
        print(f"Warning: Not enough data ({len(df_combined)} rows) after dropping NaNs for sequence length {SEQ_LEN}. Required: {SEQ_LEN}. Skipping prediction for this cycle.")
        return None # Return None if not enough data after dropping NaNs

    # Slice to get the last SEQ_LEN periods
    df_sliced = df_combined.iloc[-SEQ_LEN:]
    # Debug: Check sliced DataFrame shape
    # print(f"compute_indicators: Sliced DataFrame shape: {df_sliced.shape}")


    # Reorder columns to match the scaler's expectation (defined in ALL_FEATURES)
    # Use .copy() to avoid SettingWithCopyWarning
    # Debug: Check columns before reordering
    # print(f"compute_indicators: Sliced DataFrame columns before reordering: {df_sliced.columns.tolist()}")
    # print(f"compute_indicators: Expected ALL_FEATURES order: {ALL_FEATURES}")
    try:
        df_ordered = df_sliced[ALL_FEATURES].copy()
        # Debug: Check ordered DataFrame shape and columns
        # print(f"compute_indicators: Ordered DataFrame shape: {df_ordered.shape}")
        # print(f"compute_indicators: Ordered DataFrame columns: {df_ordered.columns.tolist()}")
    except KeyError as e:
         print(f"Error: Column mismatch during reordering in compute_indicators: {e}")
         print(f"Columns in df_sliced: {df_sliced.columns.tolist()}")
         print(f"Expected columns (ALL_FEATURES): {ALL_FEATURES}")
         return None


    return df_ordered

# ── BACKTESTING LOGIC ────────────────────────────────────────────
def run_backtest(df_historical):
    """
    Runs the backtest over historical data and collects results.

    Args:
        df_historical (pd.DataFrame): DataFrame with historical OHLCV data.

    Returns:
        list: A list of dictionaries, each containing results for a timestamp.
    """
    backtest_results = []
    # Need enough data for the initial SEQ_LEN + lookback for TA-Lib
    # Using a fixed buffer (e.g., 100 minutes) as talib.get_compatibility_flags() is not available.
    lookback_buffer = 100 # A reasonable buffer for most TA-Lib indicators
    # Start index should be after the initial lookback for indicators + SEQ_LEN for the first sequence
    start_index = SEQ_LEN + lookback_buffer
    # Ensure start_index does not exceed the total number of rows
    if start_index >= len(df_historical):
        print(f"Error: Not enough historical data ({len(df_historical)} rows) for backtesting with SEQ_LEN={SEQ_LEN} and lookback_buffer={lookback_buffer}.")
        print(f"Need at least {start_index + 1} rows.")
        return []

    print(f"Starting backtest from index {start_index} (Timestamp: {df_historical.index[start_index]})...")

    # Iterate through the historical data, starting from where enough history is available
    for i in range(start_index, len(df_historical)):
        # Get the data window needed for indicators and the sequence
        # We need data from i - (SEQ_LEN + lookback_buffer) + 1 to i (inclusive)
        window_start_index = max(0, i - (SEQ_LEN + lookback_buffer) + 1)
        df_window = df_historical.iloc[window_start_index:i+1]

        # Debug: Print window information
        # print(f"Processing index {i}. Window from {df_historical.index[window_start_index]} to {df_historical.index[i]}. Window size: {len(df_window)}")


        # Compute features (OHLCV + Indicators) for the window
        features_df = compute_indicators(df_window)

        if features_df is not None:
            # Debug: Check features_df shape before scaling
            # print(f"Backtest: features_df shape before scaling: {features_df.shape}")
            # print(f"Backtest: features_df columns before scaling: {features_df.columns.tolist()}")

            try:
                # Scale the features using the loaded scaler
                # Pass the DataFrame directly to the scaler to retain feature names and avoid the warning
                scaled_features = feat_scaler.transform(features_df)
                # Debug: Check scaled_features shape
                # print(f"Backtest: scaled_features shape after scaling: {scaled_features.shape}")

                # Reshape the scaled features for the LSTM model: (batch_size, SEQ_LEN, num_features)
                # Batch size is 1 for single sequence prediction
                X = scaled_features.reshape(1, SEQ_LEN, scaled_features.shape[1])
                # Debug: Check X shape
                # print(f"Backtest: X shape before prediction: {X.shape}")


                # Predict the next close price using the model
                y_scaled = model.predict(X, verbose=0)[0]
                # Debug: Check y_scaled shape
                # print(f"Backtest: y_scaled shape after prediction: {y_scaled.shape}")


                # Inverse transform the prediction to get the actual price
                y_pred   = tgt_scaler.inverse_transform([y_scaled])[0][0]
                # Debug: Print prediction
                # print(f"Backtest: Predicted price: {y_pred}")


                # Get the current close price and timestamp
                curr_price = df_historical['close'].iloc[i]
                curr_timestamp = df_historical.index[i]

                # Determine trading signal
                if   y_pred > curr_price*(1+THRESH): signal = "BUY"
                elif y_pred < curr_price*(1-THRESH): signal = "SELL"
                else:                             signal = "HOLD"

                # Store the results for this timestamp
                backtest_results.append({
                    "timestamp": curr_timestamp.isoformat(), # Use ISO format for easy parsing in JS
                    "price": curr_price,
                    "predicted_price": y_pred,
                    "signal": signal
                })

                # Optional: Print progress
                if (i - start_index + 1) % 100 == 0: # Print every 100 processed steps
                    print(f"Processed {i - start_index + 1}/{len(df_historical) - start_index} timestamps in backtest.")

            except Exception as e:
                print(f"Error during backtest processing at index {i}: {e}")
                traceback.print_exc() # Print detailed traceback
                # Continue to the next iteration even if one step fails


        else:
            # If compute_indicators returned None, it means there wasn't enough valid data in the window
            print(f"Warning: Skipping backtest step at index {i} due to insufficient valid data in window.")


    print("Backtest finished.")
    return backtest_results

# ── FLASK WEB SERVER ───────────────────────────────────────────────
app = Flask(__name__)

# Store backtest results globally (or load from a file if backtest is long)
# For simplicity, we'll run the backtest once when the script starts
BACKTEST_DATA = []

@app.route('/')
def index():
    """Serves the main HTML page."""
    # We'll embed the HTML directly for simplicity in this example
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Trading Bot Backtest Visualization</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns/dist/chartjs-adapter-date-fns.bundle.min.js"></script>
        <script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@1.0.2"></script>
        <style>
            body {
                font-family: sans-serif;
                display: flex;
                justify-content: center;
                align-items: center;
                min-height: 100vh;
                background-color: #f0f0f0;
                margin: 0;
            }
            .container {
                background-color: #fff;
                padding: 20px;
                border-radius: 8px;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
                width: 95%;
                max-width: 1200px;
                margin: 20px auto; /* Add margin for better spacing */
            }
            canvas {
                max-height: 600px;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Trading Bot Backtest Results</h1>
            <canvas id="backtestChart"></canvas>
        </div>
        <script src="/static/script.js"></script>
    </body>
    </html>
    """
    return render_template_string(html_content)

@app.route('/static/script.js')
def serve_script():
    """Serves the JavaScript file."""
    # Use the content from the 'backtest_js' immersive
    js_content = """
document.addEventListener('DOMContentLoaded', function() {
    fetch('/backtest_data')
        .then(response => response.json())
        .then(data => {
            // Map data for Chart.js
            const timestamps = data.map(item => new Date(item.timestamp));
            const prices = data.map(item => item.price);
            const predictedPrices = data.map(item => item.predicted_price);
            const signals = data.map(item => item.signal);

            const ctx = document.getElementById('backtestChart').getContext('2d');

            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: timestamps, // Use timestamps for the x-axis labels
                    datasets: [
                        {
                            label: 'Actual Price',
                            data: prices,
                            borderColor: 'blue',
                            backgroundColor: 'rgba(0, 0, 255, 0.1)',
                            borderWidth: 1,
                            pointRadius: 0, // Hide points for a cleaner line
                            fill: false
                        },
                        {
                            label: 'Predicted Price',
                            data: predictedPrices,
                            borderColor: 'red',
                            backgroundColor: 'rgba(255, 0, 0, 0.1)',
                            borderWidth: 1,
                            pointRadius: 0, // Hide points
                            fill: false,
                            borderDash: [5, 5] // Dashed line for prediction
                        }
                    ]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false, // Allow chart to fill container width
                    scales: {
                        x: {
                            type: 'time', // Use time scale for timestamps
                            time: {
                                unit: 'minute', // Display unit as minutes
                                tooltipFormat: 'MMM d, HH:mm' // Custom tooltip format
                            },
                            title: {
                                display: true,
                                text: 'Timestamp'
                            }
                        },
                        y: {
                            title: {
                                display: true,
                                text: 'Price (USD)'
                            }
                        }
                    },
                    plugins: {
                        tooltip: {
                            mode: 'index',
                            intersect: false,
                            callbacks: {
                                label: function(context) {
                                    let label = context.dataset.label || '';
                                    if (label) {
                                        label += ': ';
                                    }
                                    if (context.parsed.y !== null) {
                                        // Format price as currency
                                        label += new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD' }).format(context.parsed.y);
                                    }
                                    // Add signal to tooltip for the actual price dataset
                                    if (context.dataset.label === 'Actual Price') {
                                        const index = context.dataIndex;
                                        label += ' | Signal: ' + signals[index];
                                    }
                                    return label;
                                }
                            }
                            /*
                            // Annotation plugin for BUY/SELL signals - currently not working correctly with time scale
                            // Keeping commented out for now, might require more complex logic or a different approach
                            annotation: {
                                annotations: signals.map((signal, index) => {
                                    if (signal === 'BUY') {
                                        return {
                                            type: 'point',
                                            xValue: timestamps[index], // Use timestamp for xValue
                                            yValue: prices[index],   // Use price for yValue
                                            backgroundColor: 'green',
                                            radius: 5,
                                            tooltip: { // Add tooltip for annotation
                                                enabled: true,
                                                content: 'BUY Signal',
                                                position: 'top'
                                            }
                                        };
                                    } else if (signal === 'SELL') {
                                         return {
                                            type: 'point',
                                            xValue: timestamps[index], // Use timestamp for xValue
                                            yValue: prices[index],   // Use price for yValue
                                            backgroundColor: 'red',
                                            radius: 5,
                                            tooltip: { // Add tooltip for annotation
                                                enabled: true,
                                                content: 'SELL Signal',
                                                position: 'top'
                                            }
                                        };
                                    }
                                    return null; // No annotation for HOLD
                                }).filter(annotation => annotation !== null) // Filter out null annotations
                            }
                            */
                        }
                    }
                }
            });
        })
        .catch(error => console.error('Error fetching backtest data:', error));
});
    """
    from flask import Response
    return Response(js_content, mimetype='application/javascript')


@app.route('/backtest_data')
def backtest_data():
    """Returns the backtest results as JSON."""
    # Return the globally stored backtest data
    return jsonify(BACKTEST_DATA)

# ── MAIN EXECUTION ───────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Loading historical data from {HISTORICAL_DATA_PATH}...")
    try:
        # Load historical data
        # FIXED: Specify no header and provide column names
        df_historical = pd.read_csv(
            HISTORICAL_DATA_PATH,
            header=None, # No header row
            names=['timestamp', 'open', 'high', 'low', 'close', 'volume'], # Provide column names
            parse_dates=['timestamp'] # Parse the 'timestamp' column as dates
        )
        # Debug: Print columns after initial load
        # print(f"Initial columns after loading: {df_historical.columns.tolist()}")
        # Ensure expected columns are present before setting index - This check is now less critical
        # since we are providing names, but keeping it doesn't hurt.
        required_initial_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
        if not all(col in df_historical.columns for col in required_initial_cols):
            print(f"Error: Historical data file is missing expected columns after renaming: {required_initial_cols}")
            exit()

        df_historical = df_historical.set_index("timestamp")[['open','high','low','close','volume']]
        df_historical.sort_index(inplace=True) # Ensure data is sorted by timestamp
        # Debug: Print DataFrame shape and index info after processing
        # print(f"DataFrame shape after setting index and sorting: {df_historical.shape}")
        # print(f"DataFrame index type: {df_historical.index.dtype}")

        print(f"✅ Historical data loaded. Shape: {df_historical.shape}")
    except FileNotFoundError:
        print(f"Error: Historical data file not found at {HISTORICAL_DATA_PATH}.")
        print("Please make sure the path is correct and the file exists.")
        exit()
    except Exception as e:
        print(f"An error occurred while loading historical data: {e}")
        traceback.print_exc() # Print detailed traceback
        exit()

    # Run the backtest and store the results
    BACKTEST_DATA = run_backtest(df_historical)

    # Start the Flask web server
    print("\nStarting Flask web server...")
    print("Open your browser and go to http://127.0.0.1:5000/ to see the backtest results.")
    try:
        app.run(debug=True) # debug=True allows for auto-reloading during development
    except Exception as e:
        print(f"Error starting Flask server: {e}")
        traceback.print_exc() # Print detailed traceback

