"""Diagnose TRXUSDT dist_liq parity warning."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT / "scratch"
SWINT = Path(r"D:\Apps-Dev\swint_tradev2")

with open(SCRATCH / "parity_latest.json", encoding="utf-8") as f:
    parity = json.load(f)

trx = next(c for c in parity["coins"] if c["symbol"] == "TRXUSDT")
print("=== TRX parity status:", trx["status"])
for c in trx["checks"]:
    if c["status"] in ("warning", "error"):
        print(
            f"  {c['feature']}: live={c['live']:.4g} mean={c['train_mean']:.4g} "
            f"z={c['z_score']} p95={c['train_p95']:.4g} -> {c['message']}"
        )

df = pd.read_parquet(SCRATCH / "TRXUSDT_live.parquet")
last = df.iloc[-1]
print("\n=== Last bar:", df.index[-1])
cols = [
    "close", "atr_14_h1", "h4_swing_low", "h4_swing_high",
    "dist_liq_50x_long", "dist_liq_20x_long",
    "dist_liq_50x_short", "dist_liq_20x_short",
    "long_short_ratio", "hmm_regime_enc",
]
for col in cols:
    if col not in df.columns:
        continue
    val = last[col]
    if isinstance(val, (float, np.floating)):
        print(f"  {col}: {val:.6g}")
    else:
        print(f"  {col}: {val}")

c = float(last["close"])
atr = float(last.get("atr_14_h1", last.get("atr_h1", np.nan)))
sl = float(last.get("h4_swing_low", np.nan))
sh = float(last.get("h4_swing_high", np.nan))
if np.isfinite(sl) and np.isfinite(atr) and atr > 0:
    liq50 = sl * 0.98
    liq20 = sl * 0.95
    print("\n=== Manual recompute")
    print(f"  close={c:.6g} swing_low={sl:.6g} swing_high={sh:.6g} atr={atr:.6g}")
    print(f"  liq_50x_long={liq50:.6g} liq_20x_long={liq20:.6g}")
    print(f"  dist_50x_long={(c - liq50) / atr:.4g} dist_20x_long={(c - liq20) / atr:.4g}")
    print(f"  pct above swing_low: {(c / sl - 1) * 100:.2f}%")

if "h4_swing_low" in df.columns:
    tail = df[["dist_liq_50x_long", "dist_liq_20x_long", "close", "h4_swing_low", "atr_14_h1"]].tail(48)
else:
    tail = df[["dist_liq_50x_long", "dist_liq_20x_long"]].tail(48)
print("\n=== Last 48h dist_liq stats")
print(tail.describe().round(3).to_string())

# When did swing_low last change?
if "h4_swing_low" in df.columns:
    sl_series = df["h4_swing_low"].dropna()
    changes = sl_series[sl_series != sl_series.shift(1)]
    print("\n=== Recent swing_low changes (last 10)")
    for ts, val in changes.tail(10).items():
        print(f"  {ts}: {val:.6g}")

train_path = SWINT / "data" / "training" / "labeled" / "TRXUSDT_features_v3.parquet"
if train_path.exists():
    tdf = pd.read_parquet(train_path)
    for feat in ["dist_liq_50x_long", "dist_liq_20x_long"]:
        if feat not in tdf.columns:
            continue
        s = tdf[feat].dropna()
        live = float(last[feat])
        print(f"\n=== TRX training {feat}: mean={s.mean():.4g} std={s.std():.4g} "
              f"p95={s.quantile(0.95):.4g} max={s.max():.4g}")
        print(f"    live={live:.4g} z={abs(live - s.mean()) / s.std():.2f}")

    hold = tdf[tdf.index >= "2026-04-01"]
    if len(hold):
        for feat in ["dist_liq_50x_long", "dist_liq_20x_long"]:
            s = hold[feat].dropna()
            print(f"  holdout {feat}: mean={s.mean():.4g} p95={s.quantile(0.95):.4g} max={s.max():.4g}")