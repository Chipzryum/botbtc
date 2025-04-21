# add_ta_indicators.py
import pandas as pd
import numpy as np
import talib

def add_indicators(df):
    # extract arrays
    open_   = df['open'].values
    high    = df['high'].values
    low     = df['low'].values
    close   = df['close'].values
    volume  = df['volume'].values

    # ── Hilbert Transform (6) ─────────────────────
    df['HT_DCPERIOD']      = talib.HT_DCPERIOD(close)
    inphase, quadrature    = talib.HT_PHASOR(close)
    df['HT_PHASOR']        = inphase
    df['HT_QUADRATURE']    = quadrature
    sine, lead             = talib.HT_SINE(close)
    df['HT_SINE']          = sine
    df['HT_SINE_LEAD']     = lead
    df['HT_TRENDMODE']     = talib.HT_TRENDMODE(close)

    # ── Candlestick Patterns (3) ──────────────────
    df['CDL_HAMMER']       = talib.CDLHAMMER(open_, high, low, close)
    df['CDL_ENGULFING']    = talib.CDLENGULFING(open_, high, low, close)
    df['CDL_DOJI']         = talib.CDLDOJI(open_, high, low, close)

    # ── Price Transforms (6) ──────────────────────
    df['LOG']              = np.log(close, where=close>0)
    df['LINEARREG']        = talib.LINEARREG(close, timeperiod=14)
    df['WCLPRICE']         = (high + low + 2*close) / 4
    df['TYPPRICE']         = (high + low + close) / 3
    df['MEDPRICE']         = (high + low) / 2
    df['AVGPRICE']         = (open_ + high + low + close) / 4

    # ── Volume‑based (2) ─────────────────────────
    df['AD']               = talib.AD(high, low, close, volume)
    df['OBV']              = talib.OBV(close, volume)

    # ── Momentum (8) ─────────────────────────────
    macd, macdsig, macdhist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['MACD']             = macd
    df['MACD_signal']      = macdsig
    df['MACD_hist']        = macdhist
    df['WILLR']            = talib.WILLR(high, low, close, timeperiod=14)
    slowk, slowd           = talib.STOCH(high, low, close,
                                         fastk_period=14,
                                         slowk_period=3, slowk_matype=0,
                                         slowd_period=3, slowd_matype=0)
    df['STOCH_K']          = slowk
    df['STOCH_D']          = slowd
    df['RSI']              = talib.RSI(close, timeperiod=14)
    df['STDDEV']           = talib.STDDEV(close, timeperiod=14, nbdev=1)

    # ── Volatility (6) ────────────────────────────
    df['TRANGE']           = talib.TRANGE(high, low, close)
    df['ATR']              = talib.ATR(high, low, close, timeperiod=14)
    df['NATR']             = talib.NATR(high, low, close, timeperiod=14)
    upper, middle, lower   = talib.BBANDS(close,
                                         timeperiod=14,
                                         nbdevup=2, nbdevdn=2, matype=0)
    df['BB_upper']         = upper
    df['BB_middle']        = middle
    df['BB_lower']         = lower

    # ── Moving Averages (2) ───────────────────────
    df['EMA']              = talib.EMA(close, timeperiod=14)
    df['SMA']              = talib.SMA(close, timeperiod=14)

    return df

if __name__ == "__main__":
    # 1) load your 6‑column CSV
    df = pd.read_csv("btc_minute_data.csv", parse_dates=['timestamp'])
    # 2) compute indicators
    df = add_indicators(df)
    # 3) save enriched CSV
    df.to_csv("btc_minute_data_with_indicators.csv", index=False)
    print("✅ Done! 39 columns in btc_minute_data_with_indicators.csv")
