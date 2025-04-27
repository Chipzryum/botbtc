# preprocess_lstm.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib

# 1) Load & clean
df = pd.read_csv("Training/btc_minute_data_with_indicators2.csv",
                 parse_dates=['timestamp'], index_col='timestamp')

# Keep only rows with enough historical data (skip indicator warmup period)
min_non_nan = 14  # Matches your EMA/RSI periods
df = df.iloc[min_non_nan:].dropna(how='any', axis=0)
print("Final dataset shape:", df.shape)

# 2) Split features/target with time-series order
features = [c for c in df.columns if c != 'close']
target = ['close']

# 3) Scale features differently for better LSTM convergence
cont_features = ['open', 'high', 'low', 'WCLPRICE', 'TYPPRICE', 
                'LINEARREG', 'OBV', 'MACD_hist', 'RSI', 'STOCH_K',
                'STOCH_D', 'ATR', 'BB_upper', 'BB_middle', 'BB_lower', 'EMA',
                'fib_0.382', 'fib_0.618']

binary_features = ['CDL_ENGULFING', 'near_fib_0.382', 'near_fib_0.618']

# Scale continuous features (0-1), keep binaries as 0/1
cont_scaler = MinMaxScaler()
tgt_scaler = MinMaxScaler()

# Split before scaling to prevent lookahead bias
train_size = int(len(df) * 0.8)
train_df = df.iloc[:train_size]
test_df = df.iloc[train_size:]

# Fit scalers on training data only
train_cont = cont_scaler.fit_transform(train_df[cont_features])
train_tgt = tgt_scaler.fit_transform(train_df[target])

# Transform all data using training scalers
scaled_cont = cont_scaler.transform(df[cont_features])
scaled_tgt = tgt_scaler.transform(df[target])

# Combine scaled continuous + binary features
scaled_X = np.hstack([scaled_cont, df[binary_features].values]).astype(np.float32)
scaled_y = scaled_tgt.astype(np.float32)

# 4) Create LSTM sequences with lookback window
def create_sequences(X, y, look_back=60):
    X_seq, y_seq = [], []
    for i in range(len(X) - look_back):
        X_seq.append(X[i:i+look_back])
        y_seq.append(y[i+look_back])
    return np.array(X_seq), np.array(y_seq)

look_back = 60  # 1-hour window for scalping
X_seq, y_seq = create_sequences(scaled_X, scaled_y, look_back)

# 5) Final train/test split with sequences
X_train, X_test, y_train, y_test = train_test_split(
    X_seq, y_seq, test_size=0.2, shuffle=False
)

# 6) Save datasets and scalers
np.save("bot_data/X_train.npy", X_train)
np.save("bot_data/X_test.npy", X_test)
np.save("bot_data/y_train.npy", y_train)
np.save("bot_data/y_test.npy", y_test)
joblib.dump(cont_scaler, "bot_data/cont_scaler.pkl")
joblib.dump(tgt_scaler, "bot_data/tgt_scaler.pkl")

print(f"✅ Training shape: {X_train.shape}, Test shape: {X_test.shape}")