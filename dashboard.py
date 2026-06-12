# ==========================================
# TRAIN RANDOM FOREST WITH SMALLER FILE SIZE
# ==========================================

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import joblib
import os
import json
from google.colab import files

print("="*60)
print("🌲 TRAINING RANDOM FOREST (OPTIMIZED SIZE)")
print("="*60)

# Upload dataset
print("\n📁 Upload energy_data_full.csv")
uploaded = files.upload()
filename = list(uploaded.keys())[0]
df = pd.read_csv(filename)

print(f"✅ Loaded: {df.shape}")

# Features
features = [
    'temperature', 'humidity', 'hour', 'dayofweek', 'month',
    'floor_area', 'occupants', 'retrofit',
    'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'is_weekend', 'temp_humidity', 'occ_per_area'
]

X = df[features]
y = df['energy_consumption']

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# TRAIN WITH SMALLER FILE SIZE
# Reduce n_estimators from 200 to 100
# Reduce max_depth from 15 to 12
# ==========================================
print("\n🌲 Training Random Forest (optimized size)...")

rf_model = RandomForestRegressor(
    n_estimators=100,      # ← turunkan dari 200 ke 100 (saiz file jadi separuh)
    max_depth=12,          # ← turunkan dari 15 ke 12
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

# Evaluate
y_pred = rf_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print(f"\n📈 Performance: R²={r2:.4f}, MAE={mae:.4f}")

# Save model
os.makedirs('models', exist_ok=True)

# ==========================================
# COMPRESS MODEL (smaller file)
# Use compress=3 (0-9, higher = smaller but slower)
# ==========================================
joblib.dump(rf_model, 'models/thesis_mv_random_forest.pkl', compress=3)
print("✅ Model saved with compression!")

# Save features
with open('models/thesis_mv_features.txt', 'w') as f:
    for feat in features:
        f.write(f"{feat}\n")

# Save metrics
with open('models/model_metrics.json', 'w') as f:
    json.dump({'r2_score': float(r2), 'mae': float(mae)}, f)

# Check file size
import os
size_mb = os.path.getsize('models/thesis_mv_random_forest.pkl') / (1024 * 1024)
print(f"\n📦 Model file size: {size_mb:.2f} MB")

# Compress whole folder
!zip -r models.zip models/
files.download('models.zip')

print(f"\n✅ models.zip downloaded! File size should be under 25MB")
