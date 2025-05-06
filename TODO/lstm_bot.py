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

# ── CONFIG ─────────────────────────────────────────────────────────
SYMBOL    = "BTC"
SEQ_LEN   = 60
THRESH    = 0.001  # 0.1% threshold for buy/sell signal

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
# Load the trained model and scalers
model      = load_model("backtest/Trained/btc_lstm_fixed_version.h5", compile=False)
feat_scaler= joblib.load("bot_data/feature_scaler.pkl")
tgt_scaler = joblib.load("bot_data/target_scaler.pkl")

# ── COINBASE FETCH FUNCTIONS ────────────────────────────────────────
def fetch_historical_data(symbol, hours=3):
    """Fetch and print the past `hours` of 1m OHLCV from Coinbase in Sydney time."""
    utc_now = datetime.datetime.utcnow()
    start   = utc_now - timedelta(hours=hours)
    # Corrected URL, removed Markdown formatting
    endpoint = f"https://api.exchange.coinbase.com/products/{symbol}-USD/candles"
    params   = {'start': start.isoformat(), 'end': utc_now.isoformat(), 'granularity': 60}
    print("=== Backfilling past 3h from Coinbase ===")
    try:
        resp = requests.get(endpoint, params=params, timeout=10)
        resp.raise_for_status()
        candles = resp.json()[::-1] # Reverse to get oldest first
        sydney_tz = pytz.timezone('Australia/Sydney')
        for ts, low, high, o, c, v in candles:
            utc_dt   = datetime.datetime.utcfromtimestamp(ts).replace(tzinfo=pytz.utc)
            syd_str  = utc_dt.astimezone(sydney_tz).strftime('%Y-%m-%d %H:%M:%S')
            print(f"{syd_str} O:{o:.2f}, H:{high:.2f}, L:{low:.2f}, C:{c:.2f}, V:{v:.2f}")
    except requests.exceptions.RequestException as e:
        print(f"Error fetching historical data: {e}")


def fetch_completed_minute_candle(symbol, retries=5, delay=2):
    """Return the last completed 1m candle [ts, low, high, o, c, v] from Coinbase."""
    for _ in range(retries):
        now      = datetime.datetime.utcnow()
        # Get the start time for the last completed minute candle
        end_time = now.replace(second=0, microsecond=0)
        start    = end_time - timedelta(minutes=1)
        # Corrected URL, removed Markdown formatting
        endpoint = f"https://api.exchange.coinbase.com/products/{symbol}-USD/candles"
        params   = {'start': start.isoformat(), 'end': end_time.isoformat(), 'granularity': 60}
        try:
            resp = requests.get(endpoint, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data:
                # Coinbase returns most recent first, so data[0] is the latest completed candle
                return data[0]
        except requests.exceptions.RequestException as e:
            print(f"Error fetching completed candle: {e}")
        time.sleep(delay) # Wait before retrying
    return None # Return None if fetching fails after retries


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

    # Select the relevant OHLCV columns from the original DataFrame
    df_ohlcv = df[['open', 'high', 'low', 'volume']]

    # Concatenate OHLCV and indicators DataFrames
    df_combined = pd.concat([df_ohlcv, df_ind], axis=1)

    # Drop any rows that contain NaN values after indicator computation
    df_combined.dropna(inplace=True)

    # Ensure the combined DataFrame has enough rows after dropping NaNs
    if len(df_combined) < SEQ_LEN:
        print(f"Warning: Not enough data ({len(df_combined)} rows) after dropping NaNs for sequence length {SEQ_LEN}. Skipping prediction for this cycle.")
        return None # Return None if not enough data after dropping NaNs

    # Slice to get the last SEQ_LEN periods
    df_sliced = df_combined.iloc[-SEQ_LEN:]

    # Reorder columns to match the scaler's expectation (defined in ALL_FEATURES)
    # Use .copy() to avoid SettingWithCopyWarning
    df_ordered = df_sliced[ALL_FEATURES].copy()

    return df_ordered

# ── MAIN LOOP ──────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) Backfill and print initial data
    fetch_historical_data(SYMBOL)
    print("\n=== Entering LSTM trading loop ===\n")

    last_ts = None # To track the last processed candle timestamp

    while True:
        try:
            # a) Fetch and print Coinbase live candle (optional, for monitoring)
            # This uses the corrected fetch_completed_minute_candle function
            cb = fetch_completed_minute_candle(SYMBOL)
            if cb and cb[0] != last_ts:
                ts, low, high, o, c, v = cb
                utc_dt   = datetime.datetime.utcfromtimestamp(ts).replace(tzinfo=pytz.utc)
                syd_tz = pytz.timezone('Australia/Sydney')
                syd_str  = utc_dt.astimezone(syd_tz).strftime('%Y-%m-%d %H:%M:%S')
                print(f"{syd_str} O:{o:.2f}, H:{high:.2f}, L:{low:.2f}, C:{c:.2f}, V:{v:.2f}")
                last_ts = ts # Update last processed timestamp

            # b) Fetch enough historical data to compute indicators and build sequence
            # Need at least SEQ_LEN + a buffer for TA-Lib lookback periods.
            # Using a fixed buffer (e.g., 100 minutes) as talib.get_compatibility_flags() is not available.
            lookback_buffer = 100 # A reasonable buffer for most TA-Lib indicators
            required_minutes = SEQ_LEN + lookback_buffer
            end_time = datetime.datetime.utcnow().replace(second=0, microsecond=0)
            start = end_time - timedelta(minutes=required_minutes)

            # Corrected URL for fetching historical data in the main loop
            endpoint = f"https://api.exchange.coinbase.com/products/{SYMBOL}-USD/candles"
            params = {'start': start.isoformat(), 'end': end_time.isoformat(), 'granularity': 60}

            resp = requests.get(endpoint, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()[::-1] # Reverse to get oldest first

            # Create DataFrame from fetched data
            df = pd.DataFrame(data, columns=["timestamp","low","high","open","close","volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')
            df = df.set_index("timestamp")[['open','high','low','close','volume']]

            # Compute features (OHLCV + Indicators), this now returns 37 features
            features_df = compute_indicators(df)

            # Check if compute_indicators returned data (it returns None if not enough data)
            if features_df is not None:
                # Scale the features using the loaded scaler
                # Pass the DataFrame directly to the scaler to retain feature names and avoid the warning
                scaled_features = feat_scaler.transform(features_df)

                # Reshape the scaled features for the LSTM model: (batch_size, SEQ_LEN, num_features)
                # Batch size is 1 for single sequence prediction
                X = scaled_features.reshape(1, SEQ_LEN, scaled_features.shape[1])

                # Predict the next close price using the model
                y_scaled = model.predict(X, verbose=0)[0]

                # Inverse transform the prediction to get the actual price
                y_pred   = tgt_scaler.inverse_transform([y_scaled])[0][0]

                # Get the current close price from the latest fetched data
                curr     = df['close'].iloc[-1]

                # Determine trading signal
                now_str = dt.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                if   y_pred > curr*(1+THRESH): signal = "BUY"
                elif y_pred < curr*(1-THRESH): signal = "SELL"
                else:                             signal = "HOLD"

                # Print prediction and signal
                print(f"[{now_str}] Price: ${curr:.2f} | Pred: ${y_pred:.2f} | {signal}")
            else:
                 # If compute_indicators returned None, print a message and wait
                 print(f"[{dt.utcnow().isoformat()}] Skipping prediction due to insufficient data.")


            # Wait for the next minute
            time.sleep(60)

        except KeyboardInterrupt:
            print("\nStopping by user.")
            break
        except requests.exceptions.RequestException as e:
            print(f"[{dt.utcnow().isoformat()}] Network Error: {e}")
            time.sleep(30) # Wait longer on network errors
        except Exception as e:
            print(f"[{dt.utcnow().isoformat()}] An unexpected Error occurred: {e}")
            time.sleep(30) # Wait longer on other errors

