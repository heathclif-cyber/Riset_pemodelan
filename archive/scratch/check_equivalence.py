import joblib
import numpy as np
from pathlib import Path

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"

lgbm_model = joblib.load(MODEL_DIR / "guardian_best.pkl")
scaler = joblib.load(MODEL_DIR / "guardian_scaler.pkl")

# Generate dummy features
x = np.random.randn(5, 111)

xs = scaler.transform(x)
p1 = lgbm_model.predict_proba(xs)

xs_fast = (x - scaler.mean_) / scaler.scale_
p2 = lgbm_model._Booster.predict(xs_fast)

print("P1 shape:", p1.shape)
print("P2 shape:", p2.shape)
print("Are close:", np.allclose(p1, p2, atol=1e-7))
print("P1 first row:", p1[0])
print("P2 first row:", p2[0])
