import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LABEL_DIR, TRAIN_CUTOFF_DATE

feats = [
    "atr_zscore_20d", "atr_percent_h4", "atr_percentile_h1", "funding_rate",
    "ofi_h4_delta", "ofi_raw", "cvd_slope_h4", "cvd_momentum_adv",
    "rsi_6", "rsi_h4", "log_ret_1", "log_ret_5",
    "swing_momentum", "trend_strength", "ema_50_slope_h4",
]

syms = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT"]
cols_df = {}
for sym in syms:
    p = LABEL_DIR / f"{sym}_features_v3.parquet"
    if not p.exists():
        continue
    df = pd.read_parquet(p)
    cutoff = pd.Timestamp(TRAIN_CUTOFF_DATE)
    if cutoff.tzinfo is None:
        cutoff = cutoff.tz_localize("UTC")
    df = df[df.index < cutoff]
    cols_df[sym] = df

print(f"\n{'Feature':<22} " + "  ".join(f"{s[:6]:>8}" for s in syms) + "  AVG")
print("-" * 75)
for feat in feats:
    row = []
    for sym in syms:
        df = cols_df.get(sym)
        if df is None or feat not in df.columns:
            row.append("MISS")
        else:
            pct = df[feat].notna().mean() * 100
            row.append(f"{pct:.0f}%")
    print(f"{feat:<22} " + "  ".join(f"{r:>8}" for r in row))

# Also check date breakdown for OFI/CVD in BTC
print("\n--- BTC: OFI/CVD non-NaN by year ---")
df_btc = cols_df.get("BTCUSDT")
if df_btc is not None:
    for feat in ["ofi_h4_delta", "ofi_raw", "cvd_slope_h4", "cvd_momentum_adv", "funding_rate"]:
        if feat not in df_btc.columns:
            print(f"{feat}: MISSING")
            continue
        by_year = df_btc.groupby(df_btc.index.year)[feat].apply(lambda x: x.notna().mean() * 100)
        print(f"\n{feat}:")
        print(by_year.to_string())
