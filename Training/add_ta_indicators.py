# add_ta_indicators.py
import pandas as pd
import numpy as np
import talib

def add_indicators(df):
    # Extract arrays
    open_   = df['open'].values
    high    = df['high'].values
    low     = df['low'].values
    close   = df['close'].values
    volume  = df['volume'].values

    # ── Core Price/Volume (Always Keep) ────────────
    # Already included in input data

    # ── Candlestick Patterns (1) ──────────────────
    df['CDL_ENGULFING'] = talib.CDLENGULFING(open_, high, low, close)

    # ── Price Transforms (3) ──────────────────────
    df['WCLPRICE'] = (high + low + 2*close) / 4  # Weighted Close Price
    df['TYPPRICE'] = (high + low + close) / 3     # Typical Price
    df['LINEARREG'] = talib.LINEARREG(close, timeperiod=14)  # Trend Slope

    # ── Volume (1) ────────────────────────────────
    df['OBV'] = talib.OBV(close, volume)  # On-Balance Volume

    # ── Momentum (5) ──────────────────────────────
    # MACD Histogram captures momentum shifts
    _, _, df['MACD_hist'] = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    df['RSI'] = talib.RSI(close, timeperiod=14)  # Overbought/Oversold
    # Stochastic K/D for reversal signals
    df['STOCH_K'], df['STOCH_D'] = talib.STOCH(high, low, close, 
                                               fastk_period=14, slowk_period=3, slowd_period=3)

    # ── Volatility (4) ────────────────────────────
    df['ATR'] = talib.ATR(high, low, close, timeperiod=14)  # Average True Range
    # Bollinger Bands (drop middle band)
    df['BB_upper'], df['BB_middle'], df['BB_lower'] = talib.BBANDS(close, timeperiod=14, nbdevup=2, nbdevdn=2)


    # ── Moving Averages (1) ───────────────────────
    df['EMA'] = talib.EMA(close, timeperiod=14)  # Faster than SMA

    # ── Fibonacci Confluence (4) ──────────────────
    # 1. Calculate recent swing highs/lows (14-period)
    # ── Fibonacci Confluence (4) ──────────────────
    # Use pandas rolling instead of numpy array
    df['swing_high'] = df['high'].rolling(14).max()
    df['swing_low'] = df['low'].rolling(14).min()
        
    # 2. Compute key Fibonacci retracement levels
    df['fib_0.382'] = df['swing_high'] - 0.382*(df['swing_high'] - df['swing_low'])
    df['fib_0.618'] = df['swing_high'] - 0.618*(df['swing_high'] - df['swing_low'])
    
    # 3. Check if price is near Fibonacci level (±1%)
    df['near_fib_0.382'] = ((close >= 0.99*df['fib_0.382']) & (close <= 1.01*df['fib_0.382'])).astype(int)
    df['near_fib_0.618'] = ((close >= 0.99*df['fib_0.618']) & (close <= 1.01*df['fib_0.618'])).astype(int)

    # Cleanup intermediate columns
    df.drop(['swing_high', 'swing_low'], axis=1, inplace=True)

    return df

if __name__ == "__main__":
    # Load data
    df = pd.read_csv("Training/btc_minute_data.csv", parse_dates=['timestamp'])
    # Compute indicators
    df = add_indicators(df)
    # Save
    df.to_csv("Training/btc_minute_data_with_indicators2.csv", index=False)
    print(f"✅ Done! {len(df.columns)} columns in output CSV")