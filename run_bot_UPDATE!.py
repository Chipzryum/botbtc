import requests
import os
import time
import numpy as np
import pandas as pd
import joblib
import talib
from datetime import datetime, timedelta
from tensorflow.keras.models import load_model
import hmac
import hashlib
import json
import base64

# ── CONFIG ─────────────────────────────────────────────────────────
API_KEY = "176a0278-99da-47e6-be9b-e2d5a43b40db/apiKeys/2a44f44c-d365-4e2e-9d0e-d284207a8661"  # Replace with your Coinbase API key
API_SECRET = "-----BEGIN EC PRIVATE KEY-----\nMHcCAQEEIEx2iLfinx2yYb7weFBbsTdm+eqFQwLINpPjvZPk6Pi/oAoGCCqGSM49\nAwEHoUQDQgAE5IXT7fdJQvjAEx6OM4Fy/P0bKdsSN1KSadf078u7ZZXXyzNwviDD\nNJL/cgHN7h29cvnfrgpJdzGDaLUPiqNBHQ==\n-----END EC PRIVATE KEY-----\n"  # Replace with your Coinbase API secret
BASE_URL = "https://api.exchange.coinbase.com"
SYMBOL = "BTC-USD"  # Coinbase uses this format instead of BTCUSDT
SEQ_LEN = 60

# ── INITIALIZE ─────────────────────────────────────────────────────
model = load_model("btc_lstm_full_model.h5")
feat_scaler = joblib.load("feature_scaler.pkl")
tgt_scaler = joblib.load("target_scaler.pkl")

def get_coinbase_signature(request_path, method, body='', timestamp=None):
    """
    Create signature for Coinbase API authentication
    """
    if timestamp is None:
        timestamp = str(int(time.time()))
    
    message = timestamp + method + request_path
    if body:
        message += body
    
    signature = hmac.new(
        base64.b64decode(API_SECRET),
        message.encode('ascii'),
        hashlib.sha256
    )
    signature_b64 = base64.b64encode(signature.digest()).decode('utf-8')
    
    return timestamp, signature_b64

def coinbase_request(method, endpoint, params=None, data=None):
    """
    Make a request to Coinbase API with proper authentication
    """
    url = BASE_URL + endpoint
    
    body = ''
    if data:
        body = json.dumps(data)
    
    timestamp, signature = get_coinbase_signature(endpoint, method, body)
    
    headers = {
        'CB-ACCESS-KEY': API_KEY,
        'CB-ACCESS-SIGN': signature,
        'CB-ACCESS-TIMESTAMP': timestamp,
        'CB-ACCESS-PASSPHRASE': 'your_passphrase',  # You need to add your Coinbase API passphrase here
        'Content-Type': 'application/json'
    }
    
    response = requests.request(method, url, headers=headers, params=params, data=body)
    if response.status_code != 200:
        print(f"Error: {response.status_code}, {response.text}")
        return None
    
    return response.json()

def fetch_ohlcv(symbol, limit):
    """
    Fetch the last `limit` 1-minute candles from Coinbase and
    return a DataFrame with columns open, high, low, close, volume.
    """
    # Calculate start time - need to request enough data to account for any missing minutes
    # Adding a buffer to ensure we get at least 'limit' bars
    end_time = datetime.utcnow()
    start_time = end_time - timedelta(minutes=limit*2)  # Request more to handle any gaps
    
    start_iso = start_time.isoformat()
    end_iso = end_time.isoformat()
    
    endpoint = f"/products/{symbol}/candles"
    params = {
        'start': start_iso,
        'end': end_iso,
        'granularity': 60  # 60 seconds = 1 minute candles
    }
    
    candles = coinbase_request('GET', endpoint, params=params)
    
    if not candles:
        print("Failed to retrieve candles from Coinbase")
        return None
    
    # Coinbase returns data in format [timestamp, low, high, open, close, volume]
    df = pd.DataFrame(candles, columns=['timestamp', 'low', 'high', 'open', 'close', 'volume'])
    
    # Convert timestamp from Unix time to datetime and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df = df.set_index('timestamp')
    
    # Sort by timestamp in ascending order and get the most recent 'limit' entries
    df = df.sort_index().tail(limit)
    
    # Reorder columns to match the original code
    return df[['open', 'high', 'low', 'close', 'volume']]

def compute_indicators(df):
    """Add exactly the 33+6 HT indicators we trained on, using TA‑Lib."""
    o = df["open"].values; h = df["high"].values
    l = df["low"].values;  c = df["close"].values
    v = df["volume"].values
    A = {}  # container for indicator arrays

    # 6 Hilbert Transform
    A["HT_DCPERIOD"]   = talib.HT_DCPERIOD(c)
    inph, quad         = talib.HT_PHASOR(c)
    A["HT_PHASOR"]     = inph
    A["HT_QUADRATURE"] = quad
    sine, lead         = talib.HT_SINE(c)
    A["HT_SINE"]       = sine
    A["HT_SINE_LEAD"]  = lead
    A["HT_TRENDMODE"]  = talib.HT_TRENDMODE(c)

    # 3 Candles
    A["CDL_HAMMER"]    = talib.CDLHAMMER(o,h,l,c)
    A["CDL_ENGULFING"] = talib.CDLENGULFING(o,h,l,c)
    A["CDL_DOJI"]      = talib.CDLDOJI(o,h,l,c)

    # 6 Price
    A["LOG"]           = np.log(c, where=c>0)
    A["LINEARREG"]     = talib.LINEARREG(c, timeperiod=14)
    A["WCLPRICE"]      = (h + l + 2*c) / 4
    A["TYPPRICE"]      = (h + l + c) / 3
    A["MEDPRICE"]      = (h + l) / 2
    A["AVGPRICE"]      = (o + h + l + c) / 4

    # 2 Volume
    A["AD"]            = talib.AD(h, l, c, v)
    A["OBV"]           = talib.OBV(c, v)

    # 8 Momentum
    macd, sig, hist    = talib.MACD(c, fastperiod=12, slowperiod=26, signalperiod=9)
    A["MACD"], A["MACD_signal"], A["MACD_hist"] = macd, sig, hist
    A["WILLR"]         = talib.WILLR(h, l, c, timeperiod=14)
    slowk, slowd       = talib.STOCH(h, l, c,
                                     fastk_period=14,
                                     slowk_period=3, slowd_period=3)
    A["STOCH_K"], A["STOCH_D"] = slowk, slowd
    A["RSI"]           = talib.RSI(c, timeperiod=14)
    A["STDDEV"]        = talib.STDDEV(c, timeperiod=14, nbdev=1)

    # 6 Volatility
    A["TRANGE"]        = talib.TRANGE(h, l, c)
    A["ATR"]           = talib.ATR(h, l, c, timeperiod=14)
    A["NATR"]          = talib.NATR(h, l, c, timeperiod=14)
    up, mid, dn        = talib.BBANDS(c, timeperiod=14, nbdevup=2, nbdevdn=2)
    A["BB_upper"], A["BB_middle"], A["BB_lower"] = up, mid, dn

    # 2 Moving Averages
    A["EMA"]           = talib.EMA(c, timeperiod=14)
    A["SMA"]           = talib.SMA(c, timeperiod=14)

    return pd.DataFrame(A, index=df.index)

if __name__ == "__main__":
    while True:
        try:
            # 1) fetch last 61 minutes to build a 60‑min window
            raw = fetch_ohlcv(SYMBOL, SEQ_LEN+1)
            
            if raw is None or len(raw) < SEQ_LEN:
                print(f"Not enough data points, got {len(raw) if raw is not None else 0}, need {SEQ_LEN}")
                time.sleep(60)
                continue

            # 2) compute indicators, then discard the oldest row so we have exactly 60
            ind = compute_indicators(raw).iloc[-SEQ_LEN:]
            
            # Check for NaN values that might cause problems
            if ind.isna().any().any():
                print("Warning: NaN values detected in indicators, filling with method 'ffill'")
                ind = ind.fillna(method='ffill').fillna(method='bfill')

            # 3) scale & reshape into (1,60,37)
            X = feat_scaler.transform(ind.values)
            X = X.reshape(1, SEQ_LEN, X.shape[1])

            # 4) predict & inverse‑scale
            y_scaled = model.predict(X)[0]
            y_pred = tgt_scaler.inverse_transform([y_scaled])[0][0]
            curr_price = raw["close"].iloc[-1]

            # 5) simple signal logic
            if y_pred > curr_price * 1.001: signal = "BUY"
            elif y_pred < curr_price * 0.999: signal = "SELL"
            else: signal = "HOLD"

            # 6) output
            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now}] Curr: ${curr_price:.2f} → Pred: ${y_pred:.2f} → {signal}")

            # 7) here you can call your news/sentiment modules and combine signals…
            #    e.g. final_decision = decision_engine([signal, news_signal, onchain_signal])

        except Exception as e:
            print(f"Error in main loop: {e}")
        
        time.sleep(60)  # repeat once per minute