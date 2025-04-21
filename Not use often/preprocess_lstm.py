# preprocess_lstm.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib

# 1) Load & clean
df = pd.read_csv("btc_minute_data_with_indicators.csv",
                 parse_dates=['timestamp'], index_col='timestamp')
df.dropna(inplace=True)
print("After dropna:", df.shape)

# 2) Split features / target
features = [c for c in df.columns if c != 'close']
target   = ['close']

# 3) Scale
feat_scaler = MinMaxScaler()
tgt_scaler  = MinMaxScaler()

scaled_X = feat_scaler.fit_transform(df[features]).astype(np.float32)
scaled_y = tgt_scaler .fit_transform(df[target]  ).astype(np.float32)

# 4) Save
np.save("scaled_features.npy", scaled_X)
np.save("scaled_target.npy",   scaled_y)
joblib.dump(feat_scaler, "feature_scaler.pkl")
joblib.dump(tgt_scaler,  "target_scaler.pkl")

print("✅ Saved scaled_features.npy (shape", scaled_X.shape,
      ") and scaled_target.npy (shape", scaled_y.shape, ")")
