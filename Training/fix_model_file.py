# fix_model_file.py - Run this script ONCE to create a compatible .h5 file

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dropout, Dense, Input # Import Input

# ---- Parameters (match your training script) ----
SEQ_LEN = 60
# BATCH_SIZE and EPOCHS are not needed for just loading/saving

# ---- Define Model (exactly as in your training script) ----
# Use Input layer for explicit shape, recommended for robustness
model = Sequential([
    # Assuming gen.X.shape[1] was 37 based on your lstm_bot.py FEATURES list
    Input(shape=(SEQ_LEN, 37)),
    LSTM(128, return_sequences=True), # NO time_major=False here
    Dropout(0.2),
    LSTM(64), # NO time_major=False here
    Dropout(0.2),
    Dense(1)
])

# Compile the model structure - needed before loading weights in some TF versions
model.compile(optimizer='adam', loss='mse') # Use the optimizer/loss you actually used

print("Model architecture defined.")
model.summary()

# ---- Load Weights from the OLD model file ----
old_model_path = "backtest/Trained/btc_lstm_full_model.h5"
print(f"Attempting to load weights from: {old_model_path}")
try:
    # This loads ONLY the weights, ignoring the old configuration format issues
    model.load_weights(old_model_path)
    print("✅ Successfully loaded weights into the new model structure.")
except Exception as e:
    print(f"❌ Error loading weights: {e}")
    print("Please ensure the model architecture defined in this script EXACTLY matches")
    print(f"the architecture that was saved in {old_model_path}.")
    print("If you changed layer types, units, etc., weights cannot be loaded.")
    exit() # Stop if weights couldn't be loaded

# ---- Save the model using your CURRENT Keras version's format ----
new_model_path = "backtest/Trained/btc_lstm_fixed_version.h5"
print(f"Saving model with current Keras format to: {new_model_path}")
# Saving the full model captures the architecture and weights in a new, compatible format
model.save(new_model_path, include_optimizer=True) # include_optimizer=True is often helpful

print(f"✅ Model successfully re-saved to {new_model_path}.")
print(f"You can now use '{new_model_path}' in your lstm_bot.py script.")