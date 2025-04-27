import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense

# ---- Parameters ----
SEQ_LEN    = 60
BATCH_SIZE = 64
EPOCHS     = 2

# ---- Data Generator ----
class SequenceGenerator(tf.keras.utils.Sequence):
    def __init__(self, features_path, target_path, seq_len, batch_size):
        # memory-map to avoid loading full 3D into RAM
        self.X = np.load(features_path, mmap_mode='r')
        self.y = np.load(target_path,   mmap_mode='r')
        self.seq_len    = seq_len
        self.batch_size = batch_size
        self.indices    = np.arange(seq_len, len(self.X))

    def __len__(self):
        return int(np.ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, idx):
        batch_idx = self.indices[idx * self.batch_size:(idx+1) * self.batch_size]
        # build X_batch by slicing 60 rows per index
        X_batch = np.stack([self.X[i-self.seq_len:i] for i in batch_idx])
        y_batch = self.y[batch_idx]
        return X_batch, y_batch

# ---- Create Generator ----
gen = SequenceGenerator("bot_data/scaled_features.npy",
                        "bot_data/scaled_target.npy",
                        SEQ_LEN, BATCH_SIZE)

# ---- Build Model ----
model = Sequential([
    LSTM(128, return_sequences=True,
         input_shape=(SEQ_LEN, gen.X.shape[1])),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse')

# ---- Train ----
history = model.fit(gen, epochs=EPOCHS)

# ---- Save ----
model.save("btc_lstm_full_model.h5")
print("✅ Model trained and saved.")
