# continue_train.py

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# ── CONFIG ─────────────────────────────────────────────────────────
SEQ_LEN     = 60
FEATURES_N  = 37    # number of features per timestep
BATCH_SIZE  = 512
EPOCHS      = 1     # how many extra epochs to train
MODEL_IN    = "backtest/Trained/btc_lstm_fixed_version.h5"
MODEL_OUT   = "btc_lstm_continued.h5"

# ── LOAD DATA ───────────────────────────────────────────────────────
print("Loading preprocessed data...")
X = np.load("bot_data/scaled_features.npy")  # shape (n_samples, FEATURES_N)
y = np.load("bot_data/scaled_target.npy")    # shape (n_samples, 1)

# reshape X to (samples, SEQ_LEN, FEATURES_N)
n_samples = X.shape[0] - (SEQ_LEN - 1)
# we need to build sliding windows
print("Building sliding windows...")
X_windows = np.stack([X[i : i + SEQ_LEN] 
                      for i in range(n_samples)], axis=0)
y_windows = y[(SEQ_LEN - 1) : (SEQ_LEN - 1) + n_samples]

print("X_windows shape:", X_windows.shape)
print("y_windows shape:", y_windows.shape)

# ── LOAD MODEL ─────────────────────────────────────────────────────
print(f"Loading model from {MODEL_IN} (compile=False)...")
model = load_model(MODEL_IN, compile=False)

model.compile(optimizer="adam", loss="mse")
model.summary()

# ── CALLBACKS ─────────────────────────────────────────────────────
# Save best weights by val_loss
checkpoint = ModelCheckpoint(
    filepath="best_continued.h5",
    save_best_only=True,
    monitor="val_loss",
    mode="min",
)
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    mode="min",
    restore_best_weights=True,
)

# ── TRAIN ──────────────────────────────────────────────────────────
history = model.fit(
    X_windows, y_windows,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
    callbacks=[checkpoint, early_stop],
    shuffle=True,
)

# ── SAVE FINAL MODEL ───────────────────────────────────────────────
print(f"Saving continued model to {MODEL_OUT}...")
model.save(MODEL_OUT, include_optimizer=True)
print("✅ Done!")
