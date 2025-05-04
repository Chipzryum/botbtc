# preprocess_lstm_final.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import joblib
import gc

# Configuration ────────────────────────────────────────────────────────────────
LOOK_BACK = 30                   # Historical time steps (30-60 for scalping)
TEST_SIZE = 0.2                  # 80/20 train/test split
RANDOM_STATE = 42                # For reproducibility
CONT_FEATURES = [
    'open', 'high', 'low', 'WCLPRICE', 'TYPPRICE', 'LINEARREG',
    'OBV', 'MACD_hist', 'RSI', 'STOCH_K', 'STOCH_D', 'ATR',
    'BB_upper', 'BB_middle', 'BB_lower', 'EMA', 'fib_0.382', 'fib_0.618'
]
BINARY_FEATURES = [
    'CDL_ENGULFING', 'near_fib_0.382', 'near_fib_0.618'
]
TARGET = 'close'

def create_sequences(data, targets, look_back):
    """Memory-efficient sequence creation using pre-allocated arrays"""
    n_samples = len(data) - look_back
    X = np.empty((n_samples, look_back, data.shape[1]), dtype=np.float32)
    y = np.empty((n_samples, targets.shape[1]), dtype=np.float32)
    
    for i in range(n_samples):
        X[i] = data[i:i+look_back]
        y[i] = targets[i+look_back]
    return X, y

def main():
    # 1. Load Data ─────────────────────────────────────────────────────────────
    print("Loading data...")
    df = pd.read_csv("Training/btc_minute_data_with_indicators2.csv", 
                    parse_dates=['timestamp'],
                    index_col='timestamp')
    
    # Validate columns
    missing = [c for c in CONT_FEATURES + BINARY_FEATURES + [TARGET] 
              if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns: {missing}")
    print(f"Loaded data shape: {df.shape}")

    # 2. Train/Test Split (Time-Based) ─────────────────────────────────────────
    print("Splitting data...")
    train_size = int(len(df) * (1 - TEST_SIZE))
    train_df = df.iloc[:train_size]
    test_df = df.iloc[train_size:]
    
    # Early garbage collection
    del df
    gc.collect()

    # 3. Feature Scaling ───────────────────────────────────────────────────────
    print("Scaling features...")
    cont_scaler = MinMaxScaler()
    tgt_scaler = MinMaxScaler()

    # Fit only on training data
    train_cont = cont_scaler.fit_transform(train_df[CONT_FEATURES])
    train_tgt = tgt_scaler.fit_transform(train_df[[TARGET]])

    # Transform test data
    test_cont = cont_scaler.transform(test_df[CONT_FEATURES])
    test_tgt = tgt_scaler.transform(test_df[[TARGET]])

    # Combine features
    X_train = np.hstack([
        train_cont.astype(np.float32),
        train_df[BINARY_FEATURES].values.astype(np.float32)
    ])
    X_test = np.hstack([
        test_cont.astype(np.float32),
        test_df[BINARY_FEATURES].values.astype(np.float32)
    ])
    y_train = train_tgt.astype(np.float32)
    y_test = test_tgt.astype(np.float32)

    # 4. Create Sequences ──────────────────────────────────────────────────────
    print("Creating training sequences...")
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, LOOK_BACK)
    
    print("Creating test sequences...")
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, LOOK_BACK)

    # 5. Save Everything ───────────────────────────────────────────────────────
    print("Saving files...")
    np.save("bot_data/X_train.npy", X_train_seq)
    np.save("bot_data/X_test.npy", X_test_seq)
    np.save("bot_data/y_train.npy", y_train_seq)
    np.save("bot_data/y_test.npy", y_test_seq)
    
    joblib.dump(cont_scaler, "bot_data/cont_scaler.pkl")
    joblib.dump(tgt_scaler, "bot_data/tgt_scaler.pkl")

    print(f"""
    ✅ Preprocessing Complete
    Final Shapes:
    - X_train: {X_train_seq.shape} (~{(X_train_seq.nbytes / 1024**3):.2f}GB)
    - X_test:  {X_test_seq.shape} (~{(X_test_seq.nbytes / 1024**3):.2f}GB)
    Training time estimate: 30-90 mins on i7-4790
    """)

if __name__ == "__main__":
    main()