# -*- coding: utf-8 -*-
"""Daily holdout breakdown for scale_in variant."""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pipeline.backtest_utils as btu
from pipeline import ic32_fusion_shared as ifs
from pipeline.ic32_fusion_shared import build_per_bar_thresholds, load_b_dir_hmm_cfg
from pipeline.backtest_utils import hierarchical_predict, compute_guardian_static_array
from core.evaluator import full_trading_report
from core.models import load_lstm
from core.utils import ensure_utc_index
from config import (
    ALL_COINS, HOLDOUT_DIR, MODEL_DIR, LABEL_MAP, OOS_START,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, GUARDIAN_DYNAMIC_FEATURES, GUARDIAN_MIN_HOLD_BARS,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    TRAILING_STOP_ENABLED, TRAILING_STOP_ATR, TRAILING_STOP_MIN_BARS,
)
import joblib

# Reuse helpers from 07h
from importlib import import_module
h07 = import_module("pipeline.07h_holdout_ic32_scale_in_diag")
_apply_live_config = h07._apply_live_config
_add_momentum_feats = h07._add_momentum_feats
_load_guardian_cont = h07._load_guardian_cont
_attach_ts = h07._attach_ts
_run_holdout = h07._run_holdout

VARIANT = {"label": "pyr2_scale_in", "enabled": True, "max_per_coin": 2, "exit_mode": "scale_in"}
OUT_CSV = ROOT / "models/runs/ic32_regime_v1/holdout_scale_in_daily_apr_jun26.csv"
OUT_JSON = ROOT / "models/runs/ic32_regime_v1/holdout_scale_in_daily_summary_apr_jun26.json"


def main():
    live_cfg = _apply_live_config()
    hmm_cfg = load_b_dir_hmm_cfg()
    gdn = _load_guardian_cont()
    with open(MODEL_DIR / "feature_cols_ic32_regime.json", encoding="utf-8") as f:
        feat_cols = json.load(f)
    with open(MODEL_DIR / "feature_cols_lstm_temporal.json", encoding="utf-8") as f:
        lstm_feats = json.load(f)[:11]
    lgbm = joblib.load(MODEL_DIR / "runs/ic32_regime_v1/lgbm.pkl")
    lstm = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
    lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")

    trades = []
    for sym in ALL_COINS:
        trades.extend(
            _run_holdout(sym, hmm_cfg, live_cfg, gdn, feat_cols, lstm_feats,
                         lgbm, lstm, lstm_scaler, VARIANT)
        )

    df = pd.DataFrame(trades)
    df["ts_in"] = pd.to_datetime(df["ts_in"], utc=True)
    df["date"] = df["ts_in"].dt.date
    df["is_win"] = df["net_pnl"] > 0

    daily = df.groupby("date").agg(
        trades=("net_pnl", "count"),
        wins=("is_win", "sum"),
        losses=("net_pnl", lambda s: int((s <= 0).sum())),
        pnl=("net_pnl", "sum"),
        gross_win=("net_pnl", lambda s: float(s[s > 0].sum())),
        gross_loss=("net_pnl", lambda s: abs(float(s[s <= 0].sum()))),
    ).reset_index()
    daily["loss_pnl"] = daily.apply(
        lambda r: round(-r["gross_loss"], 2) if r["gross_loss"] > 0 else 0.0, axis=1
    )
    daily["win_pnl"] = daily["gross_win"].round(2)
    daily["pnl"] = daily["pnl"].round(2)
    daily["wr_pct"] = (daily["wins"] / daily["trades"] * 100).round(1)
    daily["day_type"] = daily["pnl"].apply(lambda x: "good" if x > 0 else ("flat" if x == 0 else "bad"))
    daily = daily.sort_values("date")
    daily["cum_pnl"] = daily["pnl"].cumsum().round(2)
    daily.to_csv(OUT_CSV, index=False)

    bad = daily[daily["pnl"] < 0].sort_values("pnl")
    good = daily[daily["pnl"] > 0].sort_values("pnl", ascending=False)
    flat = daily[daily["pnl"] == 0]

    summary = {
        "variant": VARIANT["label"],
        "period": {"start": str(daily["date"].min()), "end": str(daily["date"].max())},
        "total_trades": int(len(df)),
        "trading_days": int(len(daily)),
        "good_days": int(len(good)),
        "bad_days": int(len(bad)),
        "flat_days": int(len(flat)),
        "total_pnl": round(float(df["net_pnl"].sum()), 2),
        "avg_trades_per_day": round(float(daily["trades"].mean()), 2),
        "median_trades_per_day": round(float(daily["trades"].median()), 1),
        "max_trades_day": int(daily["trades"].max()),
        "avg_pnl_per_day": round(float(daily["pnl"].mean()), 2),
        "avg_loss_on_bad_days": round(float(bad["pnl"].mean()), 2) if len(bad) else 0,
        "total_loss_bad_days": round(float(bad["pnl"].sum()), 2) if len(bad) else 0,
        "avg_win_on_good_days": round(float(good["pnl"].mean()), 2) if len(good) else 0,
        "worst_5_days": bad.head(5)[["date", "trades", "wins", "losses", "pnl", "loss_pnl"]].astype(str).to_dict("records"),
        "best_5_days": good.head(5)[["date", "trades", "wins", "losses", "pnl"]].astype(str).to_dict("records"),
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    print(f"Saved {OUT_CSV} ({len(daily)} days)")
    print(f"Saved {OUT_JSON}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()