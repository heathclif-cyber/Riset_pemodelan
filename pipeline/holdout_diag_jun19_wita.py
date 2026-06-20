"""
pipeline/holdout_diag_jun19_wita.py
Diagnostic holdout backtest khusus 19 Juni 2026 WITA (UTC+8).

Entry window: 2026-06-18 16:00 UTC  --  2026-06-19 15:59 UTC
(= 00:00 - 23:59 WITA Jun 19)

Config aktif: ic32_b_dir_combined · hard_consensus · Guardian 32f
Model: LGBM=ic32_regime_v1/lgbm.pkl, LSTM=lstm_best.pt, Guardian=guardian_best.pkl

CATATAN: ini BUKAN evaluasi holdout resmi — ini diagnostik harian.
Tidak menyentuh flag .holdout_b_dir_combined_evaluated.
"""
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pipeline.backtest_utils as btu
from pipeline.backtest_utils import hierarchical_predict, compute_guardian_static_array
from core.evaluator import full_trading_report
from core.models import load_lstm
from core.utils import setup_logger, ensure_utc_index
from config import (
    ALL_COINS, HOLDOUT_DIR, MODEL_DIR, LABEL_MAP,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, GUARDIAN_DYNAMIC_FEATURES,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    TRAILING_STOP_ENABLED, TRAILING_STOP_ATR, TRAILING_STOP_MIN_BARS,
)

logger = setup_logger("holdout_diag_jun19")

# ─── Tanggal target ───────────────────────────────────────────────────────────
# 19 Juni 2026 WITA = 18 Juni 2026 16:00 UTC s/d 19 Juni 2026 15:59 UTC
JUN19_START_UTC = datetime(2026, 6, 18, 16, 0, tzinfo=timezone.utc)
JUN19_END_UTC   = datetime(2026, 6, 19, 16, 0, tzinfo=timezone.utc)

HOLDOUT_LABEL_DIR = HOLDOUT_DIR / "labeled"
RUN_DIR           = MODEL_DIR / "runs" / "ic32_regime_v1"
FROZEN_PATH       = RUN_DIR / "b_dir_combined_frozen.json"
INF_CFG_PATH      = MODEL_DIR / "inference_config.json"


def _load_frozen_cfg() -> dict:
    with open(FROZEN_PATH, encoding="utf-8") as f:
        data = json.load(f)
    raw = data["per_state_thresholds"]
    return {int(k): (float(v[0]), float(v[1])) for k, v in raw.items()}


def _apply_live_config():
    with open(INF_CFG_PATH, encoding="utf-8") as f:
        cfg = json.load(f)
    cascade  = cfg.get("cascade", {})
    guardian = cfg.get("guardian", {})
    ra       = cfg.get("regime_alignment", {})

    import config as project_config
    project_config.CONFIDENCE_THRESHOLD_ENTRY       = float(cascade.get("confidence_threshold_entry", 0.59))
    project_config.LSTM_ADJUST_AGREE_BOOST          = float(cascade.get("lstm_adjust_agree_boost", 0.05))
    project_config.LSTM_ADJUST_NEUTRAL_PEN          = float(cascade.get("lstm_adjust_neutral_pen", 0.0))
    project_config.LSTM_ADJUST_OPPOSITE_PEN         = float(cascade.get("lstm_adjust_opposite_pen", 0.65))
    project_config.LSTM_DIRECTIONAL_REVIEW_THRESHOLD = float(cascade.get("lstm_directional_review_threshold", 0.35))
    project_config.LSTM_FLAT_REVIEW_ENABLED         = bool(cascade.get("lstm_flat_review_enabled", True))
    project_config.LSTM_CONFIRMATION_ENABLED        = bool(cascade.get("lstm_confirmation_enabled", True))
    project_config.REGIME_AWARE_ALIGNMENT           = bool(ra.get("enabled", True))
    project_config.GUARDIAN_EXIT_THRESHOLD          = float(guardian.get("exit_threshold", 0.65))
    project_config.GUARDIAN_MIN_HOLD_BARS           = int(guardian.get("min_hold_bars", 4))

    btu.CONFIDENCE_THRESHOLD_ENTRY          = project_config.CONFIDENCE_THRESHOLD_ENTRY
    btu.LSTM_ADJUST_AGREE_BOOST             = project_config.LSTM_ADJUST_AGREE_BOOST
    btu.LSTM_ADJUST_NEUTRAL_PEN             = project_config.LSTM_ADJUST_NEUTRAL_PEN
    btu.LSTM_ADJUST_OPPOSITE_PEN            = project_config.LSTM_ADJUST_OPPOSITE_PEN
    btu.LSTM_DIRECTIONAL_REVIEW_THRESHOLD   = project_config.LSTM_DIRECTIONAL_REVIEW_THRESHOLD
    btu.LSTM_FLAT_REVIEW_ENABLED            = project_config.LSTM_FLAT_REVIEW_ENABLED
    btu.LSTM_CONFIRMATION_ENABLED           = project_config.LSTM_CONFIRMATION_ENABLED
    btu.REGIME_AWARE_ALIGNMENT              = project_config.REGIME_AWARE_ALIGNMENT
    btu.SMART_ENTRY_MODE                    = "disabled"
    btu.MOMENTUM_DYNAMIC_THRESHOLD_ENABLED  = False
    btu.TREND_DYNAMIC_THRESHOLD_ENABLED     = False
    btu.LSTM_STANDALONE_ENABLED             = False

    return {
        "conf_entry":   project_config.CONFIDENCE_THRESHOLD_ENTRY,
        "gdn_exit":     project_config.GUARDIAN_EXIT_THRESHOLD,
        "gdn_min_hold": project_config.GUARDIAN_MIN_HOLD_BARS,
        "flip":         project_config.REGIME_AWARE_ALIGNMENT,
    }


def _build_per_bar_thresholds(hmm_enc: np.ndarray, hmm_cfg: dict):
    n = len(hmm_enc)
    default_tl, default_ts = hmm_cfg[-1]
    tl_arr = np.full(n, default_tl, dtype=np.float64)
    ts_arr = np.full(n, default_ts, dtype=np.float64)
    for state, (tl, ts) in hmm_cfg.items():
        if state == -1:
            continue
        mask = hmm_enc == state
        tl_arr[mask] = tl
        ts_arr[mask] = ts
    return tl_arr, ts_arr


def backtest_coin(sym, hmm_cfg, live_cfg,
                  lgbm_model, lstm_model, lstm_scaler, lstm_feat_cols,
                  feat_cols, guardian_model, guardian_scaler, g_static):
    p = HOLDOUT_LABEL_DIR / f"{sym}_features_v3.parquet"
    if not p.exists():
        return None

    df = pd.read_parquet(p)
    df = ensure_utc_index(df).sort_index()

    rp = HOLDOUT_LABEL_DIR / f"{sym}_regime_h1.parquet"
    if rp.exists():
        try:
            reg = pd.read_parquet(rp)
            for col in ["hmm_regime_enc", "hmm_regime"]:
                if col in df.columns:
                    df = df.drop(columns=[col])
            cols = [c for c in ["hmm_regime_enc", "hmm_regime"] if c in reg.columns]
            df = df.join(reg[cols], how="left")
            df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
            if "hmm_regime" in df.columns:
                df["hmm_regime"] = df["hmm_regime"].fillna("RANGING_LOW_VOL")
        except Exception:
            df["hmm_regime_enc"] = 1
    else:
        df["hmm_regime_enc"] = 1
        df["hmm_regime"] = "RANGING_LOW_VOL"

    mask = df["label"].astype(str).isin(LABEL_MAP)
    df = df[mask].copy()
    if len(df) < 30:
        return None

    n = len(df)
    X = np.zeros((n, len(feat_cols)), dtype=np.float64)
    for idx, col in enumerate(feat_cols):
        if col in df.columns:
            X[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)

    hmm_enc = df["hmm_regime_enc"].values.astype(np.int32)
    thr_l, thr_s = _build_per_bar_thresholds(hmm_enc, hmm_cfg)

    y_pred, confidence = hierarchical_predict(
        None, lgbm_model, lstm_model, lstm_scaler,
        X, feat_cols, [], df,
        model_dir=RUN_DIR,
        lstm_feat_cols=lstm_feat_cols,
        per_bar_thr_long=thr_l,
        per_bar_thr_short=thr_s,
    )

    conf_thr = live_cfg["conf_entry"]
    below = (y_pred != 1) & (confidence < conf_thr)
    y_pred[below] = 1

    y   = df["label"].map(LABEL_MAP).values.astype(np.int64)
    atr = df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(n)
    h4t = df["h4_trend"].values if "h4_trend" in df.columns else None
    volr = df["vol_ratio_20"].values if "vol_ratio_20" in df.columns else None
    X_guardian = compute_guardian_static_array(df, g_static)

    report = full_trading_report(
        y_pred=y_pred, y_actual=y,
        atr=atr, close=df["close"].values, high=df["high"].values, low=df["low"].values,
        h4_swing_highs=df["h4_swing_high"].values if "h4_swing_high" in df.columns else None,
        h4_swing_lows=df["h4_swing_low"].values  if "h4_swing_low" in df.columns else None,
        index=df.index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP, max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        max_hold=MAX_HOLDING_BARS, symbol=sym, confidence=confidence,
        guardian_model=guardian_model, guardian_scaler=guardian_scaler,
        X_guardian=X_guardian,
        guardian_exit_threshold=live_cfg["gdn_exit"],
        guardian_min_hold_bars=live_cfg["gdn_min_hold"],
        trailing_stop_enabled=TRAILING_STOP_ENABLED,
        trailing_stop_atr=TRAILING_STOP_ATR,
        trailing_stop_min_bars=TRAILING_STOP_MIN_BARS,
        h4_trend=h4t, vol_ratio=volr,
    )
    return report


def main():
    hmm_cfg  = _load_frozen_cfg()
    live_cfg = _apply_live_config()

    with open(MODEL_DIR / "feature_cols_ic32_regime.json", encoding="utf-8") as f:
        feat_cols = json.load(f)
    with open(MODEL_DIR / "feature_cols_lstm_temporal.json", encoding="utf-8") as f:
        lstm_feat_cols = json.load(f)

    # Guardian 32f (live model)
    guardian_model  = joblib.load(MODEL_DIR / "guardian_best.pkl")
    guardian_scaler = joblib.load(MODEL_DIR / "guardian_scaler.pkl")
    with open(MODEL_DIR / "guardian_feature_cols.json", encoding="utf-8") as f:
        guardian_feat_cols = json.load(f)
    g_static = [c for c in guardian_feat_cols if c not in set(GUARDIAN_DYNAMIC_FEATURES)]

    lgbm_model  = joblib.load(RUN_DIR / "lgbm.pkl")
    lstm_model  = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
    lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")

    sep = "=" * 80
    print(f"\n{sep}")
    print("  DIAGNOSTIC: ic32 B-dir-combined | 19 Juni 2026 WITA")
    print(f"  Entry window UTC: {JUN19_START_UTC} -> {JUN19_END_UTC}")
    print(f"  HMM thr: {hmm_cfg}")
    print(f"  CONF={live_cfg['conf_entry']} GDN_EXIT={live_cfg['gdn_exit']} GDN_HOLD={live_cfg['gdn_min_hold']} FLIP={live_cfg['flip']}")
    print(f"  Guardian: guardian_best.pkl ({len(guardian_feat_cols)} feat | static={len(g_static)})")
    print(f"{sep}\n")

    all_trades_jun19 = []
    coin_rows = []

    for sym in ALL_COINS:
        try:
            r = backtest_coin(
                sym, hmm_cfg, live_cfg,
                lgbm_model, lstm_model, lstm_scaler, lstm_feat_cols,
                feat_cols, guardian_model, guardian_scaler, g_static,
            )
            if r is None:
                continue
            trades = r.get("trades", [])
            # Filter: hanya trade yang ENTRY pada 19 Juni WITA
            jun19_trades = [
                t for t in trades
                if "entry_time" in t and
                   JUN19_START_UTC <= pd.Timestamp(t["entry_time"]).tz_convert("UTC") < JUN19_END_UTC
            ]
            if not jun19_trades:
                continue
            all_trades_jun19.extend(jun19_trades)
            n_t   = len(jun19_trades)
            n_win = sum(1 for t in jun19_trades if t.get("net_pnl", 0) > 0)
            pnl_c = sum(t.get("net_pnl", 0) for t in jun19_trades)
            wr_c  = n_win / n_t * 100
            coin_rows.append((sym, n_t, wr_c, pnl_c, jun19_trades))
            logger.info(f"  [{sym}] {n_t} trades | WR={wr_c:.0f}% | PnL=${pnl_c:+.2f}")
        except Exception as exc:
            import traceback
            logger.error(f"  [{sym}] Error: {exc}")
            logger.error(traceback.format_exc())

    if not all_trades_jun19:
        print("\n  Tidak ada trade yang masuk pada 19 Juni 2026 WITA.")
        return

    n_total = len(all_trades_jun19)
    n_wins  = sum(1 for t in all_trades_jun19 if t.get("net_pnl", 0) > 0)
    total_pnl = sum(t.get("net_pnl", 0) for t in all_trades_jun19)
    gpnl = sum(t["net_pnl"] for t in all_trades_jun19 if t.get("net_pnl", 0) > 0)
    gloss = abs(sum(t["net_pnl"] for t in all_trades_jun19 if t.get("net_pnl", 0) <= 0))
    pf   = gpnl / gloss if gloss > 0 else float("inf")
    wr   = n_wins / n_total * 100
    ppt  = total_pnl / n_total

    long_t  = [t for t in all_trades_jun19 if t.get("direction") == "LONG"]
    short_t = [t for t in all_trades_jun19 if t.get("direction") == "SHORT"]

    outcome_counts = {}
    for t in all_trades_jun19:
        oc = t.get("outcome", "UNKNOWN")
        outcome_counts[oc] = outcome_counts.get(oc, 0) + 1
    gd_rate = sum(v for k, v in outcome_counts.items() if "GUARDIAN" in k) / n_total * 100

    print(f"\n{sep}")
    print("  SCORECARD — 19 Juni 2026 WITA (ic32_b_dir_combined)")
    print(f"{sep}")
    print(f"  Trades hari ini : {n_total}")
    print(f"  Win Rate        : {wr:.1f}%")
    print(f"  Net PnL         : ${total_pnl:+.2f}")
    print(f"  PnL/trade       : ${ppt:+.4f}")
    print(f"  Profit Factor   : {pf:.3f}")
    print(f"  LONG            : {len(long_t)} trades  ({len(long_t)/n_total*100:.0f}%)")
    print(f"  SHORT           : {len(short_t)} trades  ({len(short_t)/n_total*100:.0f}%)")
    print(f"  Guardian exits  : {gd_rate:.1f}%")
    print(f"\n  Outcome breakdown: {outcome_counts}")

    print(f"\n  Per-coin breakdown:")
    print(f"  {'Koin':<16} {'Trades':>6} {'WR':>6} {'PnL':>8}  Trades detail")
    for sym, n_t, wr_c, pnl_c, trades in sorted(coin_rows, key=lambda x: -abs(x[3])):
        trade_info = " | ".join(
            f"{t.get('direction','?')} {t.get('outcome','?')} ${t.get('net_pnl',0):+.2f}"
            for t in trades
        )
        print(f"  {sym:<16} {n_t:>6} {wr_c:>5.0f}% ${pnl_c:>7.2f}  {trade_info}")

    # Trade detail lengkap
    print(f"\n  Trade detail ({n_total} entries):")
    print(f"  {'Koin':<16} {'Dir':>5} {'Entry UTC':>20} {'Conf':>5} {'HMM':>4} {'Outcome':>12} {'PnL':>8}  Regime")
    for t in sorted(all_trades_jun19, key=lambda x: x.get("entry_time", "")):
        sym = t.get("symbol", "?")
        d   = t.get("direction", "?")
        et  = str(t.get("entry_time", "?"))[:16]
        cf  = t.get("confidence", 0)
        oc  = t.get("outcome", "?")
        pnl = t.get("net_pnl", 0)
        regime_label = "?"
        try:
            # Read regime from features file
            pass
        except Exception:
            pass
        print(f"  {sym:<16} {d:>5} {et:>20} {cf:>5.3f} {'':>4} {oc:>12} ${pnl:>7.2f}")

    print(f"\n{sep}\n")

    # Simpan hasil
    out = {
        "meta": {
            "run":           "holdout_diag_jun19_wita",
            "date_wita":     "2026-06-19",
            "entry_window":  f"{JUN19_START_UTC.isoformat()} / {JUN19_END_UTC.isoformat()}",
            "generated_at":  datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_config":  "ic32_b_dir_combined",
        },
        "aggregate": {
            "total_trades":    n_total,
            "win_rate":        round(wr, 2),
            "total_pnl":       round(total_pnl, 2),
            "pnl_per_trade":   round(ppt, 4),
            "profit_factor":   round(pf, 3),
            "long_trades":     len(long_t),
            "short_trades":    len(short_t),
            "guardian_exit_pct": round(gd_rate, 2),
            "outcome_counts":  outcome_counts,
        },
        "trades": [
            {
                "symbol":     t.get("symbol"),
                "direction":  t.get("direction"),
                "entry_time": str(t.get("entry_time")),
                "exit_time":  str(t.get("exit_time")),
                "confidence": round(float(t.get("confidence", 0)), 4),
                "outcome":    t.get("outcome"),
                "net_pnl":    round(float(t.get("net_pnl", 0)), 4),
                "bars_held":  t.get("bars_held"),
            }
            for t in sorted(all_trades_jun19, key=lambda x: x.get("entry_time", ""))
        ],
    }
    out_path = RUN_DIR / "holdout_diag_jun19_wita.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"  Hasil disimpan -> {out_path}")


if __name__ == "__main__":
    main()
