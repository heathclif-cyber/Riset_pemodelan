"""
tools/inspect_importance.py — Menampilkan Feature Importance dari model LGBM yang telah ditraining
"""

import json
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).parent.parent
MODEL_DIR = ROOT / "models"

def main():
    model_path = MODEL_DIR / "lgbm_baseline.pkl"
    feat_path = MODEL_DIR / "feature_cols_v2.json"

    if not model_path.exists():
        print(f"Model tidak ditemukan di: {model_path}")
        print("Harap pastikan Anda telah menyelesaikan training LGBM (04_train_lgbm.py).")
        return

    if not feat_path.exists():
        print(f"File feature columns tidak ditemukan di: {feat_path}")
        return

    # Load model & feature list
    print("Loading model and features...")
    model = joblib.load(model_path)
    with open(feat_path, "r") as f:
        features = json.load(f)

    # Dapatkan feature importance (split & gain)
    importance_split = model.feature_importances_
    # LGBM booster memungkinkan kita mengambil importance berbasis gain
    booster = model.booster_
    importance_gain = booster.feature_importance(importance_type="gain")

    # Buat DataFrame
    df_importance = pd.DataFrame({
        "Feature": features,
        "Split_Importance": importance_split,
        "Gain_Importance": importance_gain
    })

    # Hitung persentase kontribusi gain
    total_gain = df_importance["Gain_Importance"].sum()
    df_importance["Gain_Pct"] = (df_importance["Gain_Importance"] / total_gain * 100) if total_gain > 0 else 0.0

    # Sort berdasarkan Gain (yang biasanya lebih representatif untuk kontribusi keputusan model)
    df_importance = df_importance.sort_values(by="Gain_Importance", ascending=False).reset_index(drop=True)

    print("\n" + "=" * 80)
    print(f" FEATURE IMPORTANCE REPORT - {model_path.name}")
    print("=" * 80)
    print(f"{'Rank':<5} | {'Feature Name':<35} | {'Split Count':<12} | {'Gain':<15} | {'Gain %':<8}")
    print("-" * 80)
    for idx, row in df_importance.head(30).iterrows():
        print(f"{idx+1:<5} | {row['Feature']:<35} | {int(row['Split_Importance']):<12d} | {row['Gain_Importance']:<15.2f} | {row['Gain_Pct']:<8.2f}%")
    print("-" * 80)
    print(f"Menampilkan 30 dari {len(df_importance)} total fitur.")
    print("=" * 80)

if __name__ == "__main__":
    main()
