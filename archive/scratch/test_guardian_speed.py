import time
import joblib
import numpy as np
import os
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"

lgbm_model = joblib.load(MODEL_DIR / "guardian_best.pkl")
scaler = joblib.load(MODEL_DIR / "guardian_scaler.pkl")

# Generate dummy features
x = np.random.randn(1, 111)

# Method 1: Original
t0 = time.time()
for _ in range(2000):
    xs = scaler.transform(x)
    p = lgbm_model.predict_proba(xs)[0]
t1 = time.time()
print(f"Original: {t1 - t0:.4f} seconds")

# Method 2: Optimized scaling + predict_proba
mean = scaler.mean_
scale = scaler.scale_
t0 = time.time()
for _ in range(2000):
    xs = (x - mean) / scale
    p = lgbm_model.predict_proba(xs)[0]
t1 = time.time()
print(f"Optimized scaling: {t1 - t0:.4f} seconds")

# Method 3: Optimized scaling + direct booster predict
mean = scaler.mean_
scale = scaler.scale_
booster = lgbm_model._Booster
t0 = time.time()
for _ in range(2000):
    xs = (x - mean) / scale
    p = booster.predict(xs)[0]
t1 = time.time()
print(f"Optimized scaling + booster predict: {t1 - t0:.4f} seconds")
