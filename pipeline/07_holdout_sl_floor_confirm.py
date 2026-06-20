"""
07_holdout_sl_floor_confirm.py — Konfirmasi holdout SL% floor (SEKALI SAJA)

Keputusan min_sl_pct=0.008 SUDAH di-freeze dari OOF (05t_sl_floor_sweep.json).
Ini KONFIRMASI, bukan tuning: bandingkan stack aktif (tb_genuine_v2_dynsize_lstm_cond)
dengan SL floor 0.0 (current) vs 0.008 (kandidat) pada holdout Apr-Jun 2026.

Aturan 1: tidak ada parameter yang dipilih dari hasil ini. Floor sudah dipilih di OOF.

Usage:
  python pipeline/07_holdout_sl_floor_confirm.py
"""

CONFIRMED = True  # holdout sudah dikonfirmasi 2026-06-17 — jangan re-run (Aturan 1)

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    ALL_COINS, OOS_START, OOS_END, TRAIN_CUTOFF_DATE,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP,
    SWING_LABEL_MAX_SL, TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    GUARDIAN_ACTIVATION_ATR, LSTM_SEQ_LEN, MODEL_DIR, HOLDOUT_DIR, REPORT_DIR,
)
from core.cascade_utils import apply_conditional_momentum_fusion_pre
from core.evaluator import simulate_trades_swing
from core.models import load_lstm
from core.utils import ensure_utc_index
from pipeline.lstm_fusion_shared import (
    DYNAMIC_FEATS, DYNSIZE_CFG, LONG, SHORT,
    apply_hmm_thr, build_y_pred, compute_dynamic_modal,
    load_guardian_params, load_hmm_cfg, summarize_trades,
)

LSTM_RUN = "tb_lstm_genuine_v2"
LSTM_DIR = MODEL_DIR / "runs" / LSTM_RUN
INFERENCE_CFG = MODEL_DIR / "inference_config.json"

SL_FLOORS = [0.000, 0.008]   # current vs kandidat (frozen dari OOF)
LOWVOL_ATR_PCT = 0.004


def load_promoted_models():
    lgbm = joblib.load(MODEL_DIR / "lgbm_baseline.pkl")
    with open(MODEL_DIR / "feature_cols_v2.json", encoding="utf-8") as f:
        lgbm_feats = json.load(f)
    guard = joblib.load(MODEL_DIR / "guardian_best.pkl")
    scaler = joblib.load(MODEL_DIR / "guardian_scaler.pkl")
    with open(MODEL_DIR / "guardian_feature_cols.json", encoding="utf-8") as f:
        guard_feats = json.load(f)
    g_static = [f for f in guard_feats if f not in DYNAMIC_FEATS]
    with open(LSTM_DIR / "lstm_v4_selected_features.json", encoding="utf-8") as f:
        lstm_feats = json.load(f)
    lstm_model = load_lstm(LSTM_DIR / "lstm_momentum.pt", device="cpu")
    lstm_scaler = joblib.load(LSTM_DIR / "lstm_momentum_scaler.pkl")

    with open(INFERENCE_CFG, encoding="utf-8") as f:
        inf = json.load(f)
    mom = inf.get("cascade", {}).get("lstm_momentum", {})
    fusion_cfg = {
        "fusion": "lstm", "mode": "conditional_momentum",
        "bull_thr": float(mom.get("bull_thr", 0.38)),
        "bear_thr": float(mom.get("bear_thr", 0.50)),
        "boost": float(mom.get("boost", 0.10)),
        "opposite_pen": float(mom.get("opposite_pen", 0.14)),
        "near_miss_gap": float(mom.get("near_miss_gap", 0.03)),
        "vol_thr": float(mom.get("vol_thr", 2.0)),
        "proportional": bool(mom.get("proportional", True)),
        "enable_boost": bool(mom.get("enable_boost", True)),
        "enable_penalty": bool(mom.get("enable_penalty", True)),
        "label": "ref_lstm_cond_from_inference_config",
    }
    return {
        "lgbm": lgbm, "lgbm_feats": lgbm_feats,
        "guard": guard, "guard_scaler": scaler, "g_static": g_static,
        "lstm_model": lstm_model, "lstm_scaler": lstm_scaler, "lstm_feats": lstm_feats,
        "fusion_cfg": fusion_cfg,
    }


def lstm_predict_proba(X_raw, lstm_model, lstm_scaler, seq_len):
    n, f = X_raw.shape
    probs = np.full((n, 3), 1.0 / 3.0, dtype=np.float32)
    if n < seq_len:
        return probs
    X_sc = lstm_scaler.transform(X_raw.reshape(-1, f)).reshape(n, f).astype(np.float32)
    seqs = np.stack([X_sc[i - seq_len + 1: i + 1] for i in range(seq_len - 1, n)])
    chunks = []
    with torch.no_grad():
        for b in range(0, len(seqs), 512):
            t = torch.from_numpy(seqs[b: b + 512])
            chunks.append(torch.softmax(lstm_model(t), dim=1).cpu().numpy())
    probs[seq_len - 1:] = np.concatenate(chunks, axis=0)
    return probs


def load_holdout_coin(sym, models):
    path = HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    df = ensure_utc_index(df).sort_index()
    df = df[(df.index >= OOS_START) & (df.index < OOS_END)]
    if len(df) < LSTM_SEQ_LEN + 10:
        return None
    n = len(df)
    lgbm_feats = models["lgbm_feats"]
    X_lgbm = np.zeros((n, len(lgbm_feats)), dtype=np.float64)
    for i, col in enumerate(lgbm_feats):
        if col in df.columns:
            X_lgbm[:, i] = df[col].ffill().fillna(0).values
    proba = models["lgbm"].predict_proba(X_lgbm)
    p0, p2 = proba[:, 0].astype(np.float32), proba[:, 2].astype(np.float32)

    lstm_feats = models["lstm_feats"]
    X_lstm = np.zeros((n, len(lstm_feats)), dtype=np.float64)
    for i, col in enumerate(lstm_feats):
        if col in df.columns:
            X_lstm[:, i] = df[col].ffill().fillna(0).values
    lstm_p = lstm_predict_proba(X_lstm, models["lstm_model"], models["lstm_scaler"], LSTM_SEQ_LEN)
    lstm_valid = np.isfinite(lstm_p).all(axis=1)
    vol_spike = (df["vol_spike_zscore"].fillna(-99).values.astype(np.float32)
                 if "vol_spike_zscore" in df.columns else np.full(n, -99.0, np.float32))

    g_static = models["g_static"]
    X_grd = np.zeros((n, len(g_static)), dtype=np.float64)
    for i, col in enumerate(g_static):
        if col in df.columns:
            X_grd[:, i] = df[col].ffill().fillna(0).values

    return {
        "sym": sym, "ts": df.index, "p0": p0, "p2": p2,
        "hmm": df["hmm_regime_enc"].fillna(-1).values.astype(np.int8)
        if "hmm_regime_enc" in df.columns else np.full(n, -1, np.int8),
        "lstm_p": lstm_p, "lstm_valid": lstm_valid, "vol_spike": vol_spike,
        "close": df["close"].values.astype(np.float64),
        "high": df["high"].values.astype(np.float64),
        "low": df["low"].values.astype(np.float64),
        "atr": df["atr_14_h1"].values.astype(np.float64),
        "h4_sh": df["h4_swing_high"].values.astype(np.float64)
        if "h4_swing_high" in df.columns else np.full(n, np.nan),
        "h4_sl": df["h4_swing_low"].values.astype(np.float64)
        if "h4_swing_low" in df.columns else np.full(n, np.nan),
        "X_grd": X_grd,
    }


def eval_stack(coins, cfg, hmm_cfg, g_model, g_scaler, g_params, min_sl_pct):
    common = dict(
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        max_hold=MAX_HOLDING_BARS,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        min_sl_pct=min_sl_pct,
    )
    all_trades, lowvol_trades = [], []
    for c in coins:
        y = build_y_pred(c, cfg, hmm_cfg)
        p0, p2 = c["p0"].copy(), c["p2"].copy()
        _, _, tl, ts = apply_hmm_thr(p0, p2, c["hmm"], hmm_cfg)
        p0, p2 = apply_conditional_momentum_fusion_pre(
            p0, p2, c["lstm_p"], tl, ts, c["vol_spike"],
            vol_thr=cfg["vol_thr"], bull_thr=cfg["bull_thr"], bear_thr=cfg["bear_thr"],
            near_miss_gap=cfg["near_miss_gap"], boost=cfg["boost"],
            opposite_pen=cfg["opposite_pen"], enable_boost=cfg["enable_boost"],
            enable_penalty=cfg["enable_penalty"], lstm_valid=c["lstm_valid"],
            proportional=cfg["proportional"],
        )
        _, conf, tl, ts = apply_hmm_thr(p0, p2, c["hmm"], hmm_cfg)
        modal_arr = compute_dynamic_modal(p0, p2, c["hmm"], y, MODAL_PER_TRADE, DYNSIZE_CFG, tl, ts)
        rep = simulate_trades_swing(
            y_pred=y, guardian_enabled=True,
            guardian_model=g_model, guardian_scaler=g_scaler, X_guardian=c["X_grd"],
            guardian_exit_threshold=g_params["exit_threshold"],
            guardian_min_hold_bars=g_params["min_hold_bars"],
            guardian_activation_atr=GUARDIAN_ACTIVATION_ATR,
            modal_arr=modal_arr,
            close=c["close"], high=c["high"], low=c["low"], atr=c["atr"],
            h4_swing_highs=c["h4_sh"], h4_swing_lows=c["h4_sl"],
            **common,
        )
        close, atr = c["close"], c["atr"]
        for t in rep.get("trades", []):
            bi = t.get("bar_in", 0)
            all_trades.append(t)
            atr_pct = (atr[bi] / close[bi]) if (bi < len(close) and close[bi] > 0) else np.nan
            if not np.isnan(atr_pct) and atr_pct < LOWVOL_ATR_PCT:
                lowvol_trades.append(t)
    return {"portfolio": summarize_trades(all_trades), "lowvol": summarize_trades(lowvol_trades)}


def main():
    if CONFIRMED:
        raise RuntimeError("Konfirmasi SL floor sudah dijalankan — jangan re-run (kontaminasi holdout).")
    sep = "=" * 80
    print(f"\n{sep}")
    print("  KONFIRMASI HOLDOUT - SL% Floor (stack aktif tb_genuine_v2_dynsize_lstm_cond)")
    print(f"  Holdout: {OOS_START.date()} -> {OOS_END.date()} | floors={SL_FLOORS} (frozen dari OOF)")
    print(sep)

    models = load_promoted_models()
    hmm_cfg = load_hmm_cfg()
    g_params = load_guardian_params()
    cfg = models["fusion_cfg"]

    coins = [c for c in (load_holdout_coin(s, models) for s in ALL_COINS) if c is not None]
    print(f"  Coins evaluated: {len(coins)}\n")
    if not coins:
        raise RuntimeError("No holdout coin data found.")

    rows = []
    base = None
    print(f"  {'floor':>7} | {'N':>5} {'WR%':>6} {'PF':>6} {'PnL':>9} {'pptN':>8} {'SL%':>6} "
          f"|| LOWVOL {'N':>4} {'WR%':>6} {'PF':>6} {'SL%':>6}")
    for floor in SL_FLOORS:
        r = eval_stack(coins, cfg, hmm_cfg, models["guard"], models["guard_scaler"], g_params, floor)
        r["min_sl_pct"] = floor
        if floor == 0.0:
            base = r
        p, lv = r["portfolio"], r["lowvol"]
        rows.append(r)
        print(f"  {floor:>7.3f} | {p['n']:>5,} {p['wr']:>6.1f} {p['pf']:>6.2f} {p['pnl']:>9.2f} "
              f"{p['ppt_norm']:>8.4f} {p['sl_pct']:>6.1f} || {'':6}{lv['n']:>4,} "
              f"{lv['wr']:>6.1f} {lv['pf']:>6.2f} {lv['sl_pct']:>6.1f}")

    cand = next(r for r in rows if r["min_sl_pct"] == 0.008)
    pb, pc = base["portfolio"], cand["portfolio"]
    lb, lc = base["lowvol"], cand["lowvol"]
    print(f"\n{sep}")
    print("  DELTA 0.008 vs current(0.000)")
    print(f"  Portfolio: WR {pc['wr']-pb['wr']:+.2f}pp  PF {pc['pf']-pb['pf']:+.3f}  "
          f"PnL {pc['pnl']-pb['pnl']:+.2f}  pptN {pc['ppt_norm']-pb['ppt_norm']:+.4f}  "
          f"SL% {pc['sl_pct']-pb['sl_pct']:+.2f}pp")
    print(f"  Low-vol  : WR {lc['wr']-lb['wr']:+.2f}pp  PF {lc['pf']-lb['pf']:+.3f}  "
          f"SL% {lc['sl_pct']-lb['sl_pct']:+.2f}pp  (N {lb['n']}->{lc['n']})")
    print(sep)

    today = datetime.now().strftime("%Y-%m-%d")
    rdir = REPORT_DIR / "experiments"
    rdir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created": datetime.now().isoformat(),
        "script": "07_holdout_sl_floor_confirm.py",
        "model_version": "tb_genuine_v2_dynsize_lstm_cond",
        "methodology": "confirmation_only_floor_frozen_from_oof",
        "train_cutoff": str(TRAIN_CUTOFF_DATE.date()),
        "holdout_start": str(OOS_START.date()),
        "holdout_end": str(OOS_END.date()),
        "n_coins": len(coins),
        "fusion_cfg": cfg,
        "lowvol_atr_pct_threshold": LOWVOL_ATR_PCT,
        "results": rows,
    }
    json_path = rdir / f"{today}_sl_floor_holdout_confirm.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\n  JSON: {json_path}")
    print("  INGAT: set CONFIRMED = True setelah review.\n")


if __name__ == "__main__":
    main()
