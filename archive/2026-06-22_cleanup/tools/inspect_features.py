"""
tools/inspect_features.py — Audit data quality & fitur gejolak market
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path("data/training/processed")
FEATURES_DIR  = Path("data/training/labeled")
SYMBOL = "BTCUSDT"

print("=" * 70)
print("STEP 1: Cek kolom di BTCUSDT_clean.parquet")
print("=" * 70)
clean = pd.read_parquet(PROCESSED_DIR / f"{SYMBOL}_clean.parquet")
print(f"Shape: {clean.shape}")
print("Kolom:")
for c in sorted(clean.columns):
    nn   = int(clean[c].notna().sum())
    uniq = int(clean[c].nunique())
    last = clean[c].dropna().iloc[-1] if nn > 0 else "ALL_NULL"
    tag  = " <- SEMUA NULL" if nn == 0 else (" <- MOSTLY ZERO" if uniq <= 2 and nn > 0 and clean[c].abs().max() < 1e-9 else "")
    print(f"  {c:50s} notna={nn:6d}  unique={uniq:5d}  last={str(last)[:20]}{tag}")

print()
print("=" * 70)
print("STEP 2: Cek kolom di BTCUSDT_features_v3.parquet")
print("=" * 70)
feat_files = list(FEATURES_DIR.glob(f"{SYMBOL}_features_v3.parquet"))
if feat_files:
    feat = pd.read_parquet(feat_files[0])
    print(f"Shape: {feat.shape}")

    # Fitur volatility/gejolak yang kita cari
    targets = [
        "funding_rate", "funding_price_div", "btc_dominance",
        "vol_regime", "ultra_high_vol", "wyckoff_phase", "spring_upthrust",
        "dist_liq_50x_long", "dist_liq_20x_long",
        "dist_liq_50x_short", "dist_liq_20x_short",
        "whale_retail_divergence",
        "open_interest", "long_short_ratio", "fear_greed",
        "hmm_regime_enc",
        "atr_percent_h4", "atr_14_h4",
    ]
    print("\nFitur gejolak/volatility yang dicari:")
    for t in targets:
        if t in feat.columns:
            s    = feat[t]
            nn   = int(s.notna().sum())
            uniq = int(s.nunique())
            zeros = int((s == 0).sum())
            mn   = float(s.min()) if nn > 0 else float("nan")
            mx   = float(s.max()) if nn > 0 else float("nan")
            pct_zero = zeros / len(s) * 100
            flag = ""
            if nn == 0:
                flag = " [!] ALL NULL"
            elif pct_zero > 90:
                flag = f" [!] {pct_zero:.0f}% ZEROS"
            elif uniq <= 2:
                flag = f" [!] BINARY (unique={uniq})"
            print(f"  [OK] {t:40s} notna={nn:6d}  0%={pct_zero:5.1f}%  min={mn:.4f}  max={mx:.4f}{flag}")
        else:
            print(f"  [--] {t:40s} NOT FOUND IN FEATURES")
else:
    print(f"File tidak ditemukan: {FEATURES_DIR}/{SYMBOL}_features_v3.parquet")

print()
print("=" * 70)
print("STEP 3: Spot-check nilai funding_rate dari raw data")
print("=" * 70)
raw_files = list(Path("data/raw").glob(f"{SYMBOL}_*.parquet"))
print(f"Raw files for {SYMBOL}: {len(raw_files)}")
for f in sorted(raw_files):
    try:
        df = pd.read_parquet(f)
        fr_cols = [c for c in df.columns if "funding" in c.lower()]
        if fr_cols:
            print(f"  {f.name}: funding cols = {fr_cols}")
            for fc in fr_cols:
                nn    = int(df[fc].notna().sum())
                zeros = int((df[fc] == 0).sum())
                mx    = float(df[fc].abs().max()) if nn > 0 else 0.0
                print(f"    {fc}: notna={nn}, zeros={zeros}, max_abs={mx:.6f}")
        else:
            print(f"  {f.name}: NO funding cols")
    except Exception as e:
        print(f"  {f.name}: ERROR — {e}")

print()
print("=" * 70)
print("STEP 4: Cek apakah dist_liq & whale_retail sudah di-engineer")
print("=" * 70)
all_feat = list(FEATURES_DIR.glob("*_features_v3.parquet"))
if all_feat:
    sample = pd.read_parquet(all_feat[0])
    all_cols = set(sample.columns)
    missing_volatility = [
        "dist_liq_50x_long", "dist_liq_20x_long",
        "dist_liq_50x_short", "dist_liq_20x_short",
        "whale_retail_divergence",
        "atr_zscore_20d", "vol_spike_3sigma",
    ]
    print(f"Sample file: {all_feat[0].name} ({len(all_cols)} cols)")
    for m in missing_volatility:
        status = "[OK] ADA" if m in all_cols else "[--] BELUM DIIMPLEMENTASI"
        print(f"  {status}: {m}")
else:
    print("Tidak ada file features_v3 ditemukan!")

print()
print("SELESAI.")
