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

# ── INITIALIZE ─────────────────────────────────────────────────────
model      = load_model("btc_lstm_fixed_version.h5", compile=False)
feat_scaler= joblib.load("feature_scaler.pkl")
tgt_scaler = joblib.load("target_scaler.pkl")

FEATURES = [
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

# ── COINBASE FETCH FUNCTIONS ────────────────────────────────────────
def fetch_historical_data(symbol, hours=3):
    """Fetch and print the past `hours` of 1m OHLCV from Coinbase in Sydney time."""
    utc_now = datetime.datetime.utcnow()
    start   = utc_now - timedelta(hours=hours)
    endpoint = f"https://api.exchange.coinbase.com/products/{symbol}-USD/candles"
    params   = {'start': start.isoformat(), 'end': utc_now.isoformat(), 'granularity': 60}
    print("=== Backfilling past 3h from Coinbase ===")
    resp = requests.get(endpoint, params=params, timeout=10)
    resp.raise_for_status()
    candles = resp.json()[::-1]
    sydney_tz = pytz.timezone('Australia/Sydney')
    for ts, low, high, o, c, v in candles:
        utc_dt   = datetime.datetime.utcfromtimestamp(ts).replace(tzinfo=pytz.utc)
        syd_str  = utc_dt.astimezone(sydney_tz).strftime('%Y-%m-%d %H:%M:%S')
        print(f"{syd_str} O:{o:.2f}, H:{high:.2f}, L:{low:.2f}, C:{c:.2f}, V:{v:.2f}")


def fetch_completed_minute_candle(symbol, retries=5, delay=2):
    """Return the last completed 1m candle [ts, low, high, o, c, v] from Coinbase."""
    for _ in range(retries):
        now      = datetime.datetime.utcnow()
        end_time = now.replace(second=0, microsecond=0)
        start    = end_time - timedelta(minutes=1)
        endpoint = f"https://api.exchange.coinbase.com/products/{symbol}-USD/candles"
        params   = {'start': start.isoformat(), 'end': end_time.isoformat(), 'granularity': 60}
        resp = requests.get(endpoint, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        if data:
            return data[0]
        time.sleep(delay)
    return None


# ── INDICATOR COMPUTATION ──────────────────────────────────────────
def compute_indicators(df):
    """Compute the exact 37 features matching the training set."""
    o,h,l,c,v = df.open.values, df.high.values, df.low.values, df.close.values, df.volume.values
    A = {}
    # Hilbert Transform
    A["HT_DCPERIOD"]   = talib.HT_DCPERIOD(c)
    inph, quad         = talib.HT_PHASOR(c)
    A["HT_PHASOR"], A["HT_QUADRATURE"] = inph, quad
    sine, lead         = talib.HT_SINE(c)
    A["HT_SINE"], A["HT_SINE_LEAD"]   = sine, lead
    A["HT_TRENDMODE"] = talib.HT_TRENDMODE(c)
    # Candles
    A["CDL_HAMMER"]    = talib.CDLHAMMER(o,h,l,c)
    A["CDL_ENGULFING"] = talib.CDLENGULFING(o,h,l,c)
    A["CDL_DOJI"]      = talib.CDLDOJI(o,h,l,c)
    # Price
    A["LOG"]       = np.log(c, where=c>0)
    A["LINEARREG"] = talib.LINEARREG(c, timeperiod=14)
    A["WCLPRICE"]  = (h + l + 2*c) / 4
    A["TYPPRICE"]  = (h + l + c) / 3
    A["MEDPRICE"]  = (h + l) / 2
    A["AVGPRICE"]  = (o + h + l + c) / 4
    # Volume
    A["AD"]  = talib.AD(h, l, c, v)
    A["OBV"] = talib.OBV(c, v)
    # Momentum
    macd,sig,hist = talib.MACD(c, 12,26,9)
    A["MACD"],A["MACD_signal"],A["MACD_hist"] = macd,sig,hist
    A["WILLR"]   = talib.WILLR(h,l,c,14)
    st_k,st_d     = talib.STOCH(h,l,c,14,3,3)
    A["STOCH_K"],A["STOCH_D"] = st_k,st_d
    A["RSI"]     = talib.RSI(c,14)
    A["STDDEV"]  = talib.STDDEV(c,14,1)
    # Volatility
    A["TRANGE"] = talib.TRANGE(h,l,c)
    A["ATR"]    = talib.ATR(h,l,c,14)
    A["NATR"]   = talib.NATR(h,l,c,14)
    up,mid,dn     = talib.BBANDS(c,14,2,2)
    A["BB_upper"],A["BB_middle"],A["BB_lower"] = up,mid,dn
    # Moving Averages
    A["EMA"] = talib.EMA(c,14)
    A["SMA"] = talib.SMA(c,14)
    df_ind = pd.DataFrame(A, index=df.index)
    return df_ind.iloc[-SEQ_LEN:]

# ── MAIN LOOP ──────────────────────────────────────────────────────
if __name__ == "__main__":
    # 1) Backfill and print
    fetch_historical_data(SYMBOL)
    print("=== Entering LSTM trading loop ===\n")

    last_ts = None
    while True:
        try:
            # a) Print Coinbase live candle
            cb = fetch_completed_minute_candle(SYMBOL)
            if cb and cb[0] != last_ts:
                ts, low, high, o, c, v = cb
                utc_dt   = datetime.datetime.utcfromtimestamp(ts).replace(tzinfo=pytz.utc)
                syd_str  = utc_dt.astimezone(pytz.timezone('Australia/Sydney')).strftime('%Y-%m-%d %H:%M:%S')
                print(f"{syd_str} O:{o:.2f}, H:{high:.2f}, L:{low:.2f}, C:{c:.2f}, V:{v:.2f}")
                last_ts = ts

            # b) Build dataframe of last SEQ_LEN+1 candles via Coinbase
            # Reuse fetch_historical_data with limit=SEQ_LEN+1 (internal)
            # Instead, fetch raw list then DataFrame:
            endpoint = f"https://api.exchange.coinbase.com/products/{SYMBOL}-USD/candles"
            end_time = datetime.datetime.utcnow().replace(second=0, microsecond=0)
            start = end_time - timedelta(minutes=SEQ_LEN)
            params = {'start': start.isoformat(), 'end': end_time.isoformat(), 'granularity': 60}
            resp = requests.get(endpoint, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()[::-1]
            df = pd.DataFrame(data, columns=["timestamp","low","high","open","close","volume"])
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit='s')
            df = df.set_index("timestamp")[['open','high','low','close','volume']]

            # Compute features, scale, predict
            features = compute_indicators(df)
            X = feat_scaler.transform(features[FEATURES].values).reshape(1, SEQ_LEN, len(FEATURES))
            y_scaled = model.predict(X, verbose=0)[0]
            y_pred   = tgt_scaler.inverse_transform([y_scaled])[0][0]
            curr     = df['close'].iloc[-1]

            now_str = dt.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            if   y_pred > curr*(1+THRESH): signal = "BUY"
            elif y_pred < curr*(1-THRESH): signal = "SELL"
            else:                             signal = "HOLD"
            print(f"[{now_str}] Price: ${curr:.2f} | Pred: ${y_pred:.2f} | {signal}")

            time.sleep(60)
        except KeyboardInterrupt:
            print("Stopping by user.")
            break
        except Exception as e:
            print(f"[{dt.utcnow().isoformat()}] Error:", e)
            time.sleep(30)
