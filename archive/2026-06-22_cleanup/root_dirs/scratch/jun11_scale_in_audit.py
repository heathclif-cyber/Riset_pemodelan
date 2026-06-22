# -*- coding: utf-8 -*-
"""Audit scale_in holdout trades on 2026-06-11."""
import datetime
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from importlib import import_module
from core.models import load_lstm
from config import ALL_COINS, MODEL_DIR
from pipeline.ic32_fusion_shared import load_b_dir_hmm_cfg

h07 = import_module("pipeline.07h_holdout_ic32_scale_in_diag")

live_cfg = h07._apply_live_config()
hmm_cfg = load_b_dir_hmm_cfg()
gdn = h07._load_guardian_cont()
with open(MODEL_DIR / "feature_cols_ic32_regime.json", encoding="utf-8") as f:
    feat_cols = json.load(f)
with open(MODEL_DIR / "feature_cols_lstm_temporal.json", encoding="utf-8") as f:
    lstm_feats = json.load(f)[:11]
lgbm = joblib.load(MODEL_DIR / "runs/ic32_regime_v1/lgbm.pkl")
lstm = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
variant = {"label": "pyr2_scale_in", "enabled": True, "max_per_coin": 2, "exit_mode": "scale_in"}

trades = []
for sym in ALL_COINS:
    trades.extend(
        h07._run_holdout(sym, hmm_cfg, live_cfg, gdn, feat_cols, lstm_feats,
                         lgbm, lstm, lstm_scaler, variant)
    )

target = datetime.date(2026, 6, 11)
rows = []
for t in trades:
    ts = pd.Timestamp(t["ts_in"])
    if ts.date() == target:
        rows.append({
            "symbol": t["symbol"],
            "direction": t["direction"],
            "ts_in_utc": str(ts),
            "ts_in_wita": str(ts.tz_convert("Asia/Makassar")),
            "ts_out_utc": str(pd.Timestamp(t["ts_out"])),
            "net_pnl": round(t["net_pnl"], 4),
            "n_legs": t.get("n_legs", 1),
            "modal_used": t.get("modal_used", 10),
            "outcome": t.get("outcome"),
        })

df = pd.DataFrame(rows).sort_values("ts_in_utc")
print("=== SCALE_IN trades entry date 2026-06-11 (UTC calendar) ===")
print(f"count={len(df)} long={(df.direction == 'LONG').sum()} short={(df.direction == 'SHORT').sum()}")
print(f"total_pnl={df.net_pnl.sum():.2f}")
print(df.to_string(index=False))
print()

# SHORT on adjacent UTC days
for d in [datetime.date(2026, 6, 10), datetime.date(2026, 6, 12)]:
    sub = [t for t in trades if pd.Timestamp(t["ts_in"]).date() == d]
    shorts = [t for t in sub if t["direction"] == "SHORT"]
    print(f"--- {d} UTC: {len(sub)} trades, {len(shorts)} SHORT ---")
    for t in sorted(shorts, key=lambda x: x["ts_in"]):
        ts = pd.Timestamp(t["ts_in"])
        print(f"  {t['symbol']:14} SHORT in={ts} WITA={ts.tz_convert('Asia/Makassar')} pnl={t['net_pnl']:.2f}")

# Per-coin window around Jun 11
syms = ["SUIUSDT", "ARBUSDT", "1000PEPEUSDT", "BNBUSDT", "HBARUSDT", "DOTUSDT"]
print("\n=== Per-coin scale_in Jun 10-12 WITA ===")
for sym in syms:
    coin_trades = [t for t in trades if t["symbol"] == sym]
    sub = [
        t for t in coin_trades
        if datetime.date(2026, 6, 10) <= pd.Timestamp(t["ts_in"]).date() <= datetime.date(2026, 6, 12)
    ]
    print(f"--- {sym} ---")
    for t in sorted(sub, key=lambda x: x["ts_in"]):
        ts_in = pd.Timestamp(t["ts_in"]).tz_convert("Asia/Makassar")
        ts_out = pd.Timestamp(t["ts_out"]).tz_convert("Asia/Makassar")
        print(
            f"  {t['direction']} in={ts_in} out={ts_out} "
            f"pnl={t['net_pnl']:.2f} legs={t.get('n_legs', 1)}"
        )