# lstm_bot.py

import time
import numpy as np
import pandas as pd
import joblib
import talib
from datetime import datetime
from hyperliquid import HyperliquidSync
from tensorflow.keras.models import load_model

# ── CONFIG ─────────────────────────────────────────────────────────
SYMBOL   = "BTC"        # Hyperliquid perpetual symbol :cite[8]
# SYMBOL = "BTC-USDC"   # Alternative for spot trading
SEQ_LEN  = 60           
THRESH   = 0.001      # 0.1% threshold for buy/sell signal

# ── INITIALIZE ─────────────────────────────────────────────────────
exchange     = HyperliquidSync({})               # public data only
model        = load_model("btc_lstm_fixed_version.h5", compile=False)
feat_scaler  = joblib.load("feature_scaler.pkl")
tgt_scaler   = joblib.load("target_scaler.pkl")

# List of the 37 features in order
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

def fetch_ohlcv(symbol, limit):
    """Fetch the last `limit` 1 m bars from Hyperliquid as a DataFrame."""
    bars = exchange.fetch_ohlcv(symbol, timeframe="1m", limit=limit)
    df = pd.DataFrame(bars, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    return df.set_index("timestamp")

def compute_indicators(df):
    """Compute the exact 37 features matching the training set."""
    o, h, l, c, v = df.open.values, df.high.values, df.low.values, df.close.values, df.volume.values
    A = {}
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
    st_k, st_d         = talib.STOCH(h, l, c,
                                     fastk_period=14, slowk_period=3, slowd_period=3)
    A["STOCH_K"], A["STOCH_D"] = st_k, st_d
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

    # Build DataFrame, keep only the last SEQ_LEN rows
    df_ind = pd.DataFrame(A, index=df.index)
    return df_ind.iloc[-SEQ_LEN:]

if __name__ == "__main__":
    while True:
        try:
            # 1) Fetch raw candles + compute indicators
            raw = fetch_ohlcv(SYMBOL, SEQ_LEN + 1)
            features = compute_indicators(raw)

            # 2) Scale & reshape into (1,60,37)
            X = feat_scaler.transform(features[FEATURES].values)
            X = X.reshape(1, SEQ_LEN, len(FEATURES))

            # 3) Predict & inverse‐scale
            y_scaled  = model.predict(X, verbose=0)[0]
            y_pred    = tgt_scaler.inverse_transform([y_scaled])[0][0]
            curr_price= raw["close"].iloc[-1]

            # 4) Decision logic
            if   y_pred > curr_price * (1 + THRESH): signal = "BUY"
            elif y_pred < curr_price * (1 - THRESH): signal = "SELL"
            else:                                     signal = "HOLD"

            # 5) Log
            now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            print(f"[{now}] Price: ${curr_price:.2f} | Pred: ${y_pred:.2f} | {signal}")

            time.sleep(60)

        except Exception as e:
            print(f"[{datetime.utcnow().isoformat()}] Error:", e)
            time.sleep(30)
