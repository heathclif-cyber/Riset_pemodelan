"""
pipeline/12_holdout_ic32_apr_jun26.py
ic32_regime_v1 holdout Apr-Jun 2026 — apples-to-apples vs TB widyawardhana

Config: V2.5 Hybrid (dari config.py aktif)
  LGBM   : ic32_regime_v1/lgbm.pkl (33 fitur)
  LSTM   : models/lstm_best.pt (temporal features)
  Guardian: models/guardian_clean_v2.pkl (ic32_guardian_clean_v2, 40 fitur)
  Thresholds: LONG=0.69, SHORT=0.59, CONF=0.59, GDN=0.65
  Exit    : Swing-based TP/SL (native ic32) + Guardian early exit

Outputs:
  - Console scorecard
  - models/runs/ic32_regime_v1/holdout_apr_jun26.json
"""
import json, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import pipeline.backtest_utils as btu
from pipeline.backtest_utils import hierarchical_predict, compute_guardian_static_array
from core.evaluator import full_trading_report
from core.models import load_lstm
from core.utils import setup_logger, ensure_utc_index
from config import (
    ALL_COINS, HOLDOUT_DIR, MODEL_DIR, LABEL_MAP,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, CONFIDENCE_THRESHOLD_ENTRY,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    GUARDIAN_ENABLED, GUARDIAN_EXIT_THRESHOLD, GUARDIAN_DYNAMIC_FEATURES,
    TRAILING_STOP_ENABLED, TRAILING_STOP_ATR, TRAILING_STOP_MIN_BARS,
)

logger = setup_logger("12_ic32_holdout")

HOLDOUT_LABEL_DIR = HOLDOUT_DIR / "labeled"
RUN_DIR = MODEL_DIR / "runs" / "ic32_regime_v1"

# ── Force V2.5 Hybrid / default cascade settings ─────────────────────────────
btu.SMART_ENTRY_MODE = "disabled"       # standard LGBM-gates-first cascade
btu.MOMENTUM_DYNAMIC_THRESHOLD_ENABLED = False
btu.TREND_DYNAMIC_THRESHOLD_ENABLED    = False
btu.LSTM_STANDALONE_ENABLED            = False

# ── Load models ───────────────────────────────────────────────────────────────
logger.info("Loading ic32_regime_v1 models...")

lgbm_model  = joblib.load(RUN_DIR / "lgbm.pkl")
lstm_model  = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")

with open(MODEL_DIR / "feature_cols_ic32_regime.json") as f:
    feat_cols = json.load(f)
with open(MODEL_DIR / "feature_cols_lstm_temporal.json") as f:
    lstm_feat_cols = json.load(f)

# ic32_guardian_clean_v2
guardian_model  = joblib.load(MODEL_DIR / "guardian_clean_v2.pkl")
guardian_scaler = joblib.load(MODEL_DIR / "guardian_clean_v2_scaler.pkl")
with open(MODEL_DIR / "guardian_clean_v2_feature_cols.json") as f:
    guardian_feat_cols = json.load(f)
_GDN_DYN   = set(GUARDIAN_DYNAMIC_FEATURES)
g_static   = [c for c in guardian_feat_cols if c not in _GDN_DYN]

logger.info(f"LGBM({len(lgbm_model.feature_name_)}f) LSTM({len(lstm_feat_cols)}f) "
            f"Guardian({len(guardian_feat_cols)}f={len(g_static)}static+{len(guardian_feat_cols)-len(g_static)}dyn)")
logger.info(f"Thresholds: LONG={btu.LGBM_THRESHOLD_LONG} SHORT={btu.LGBM_THRESHOLD_SHORT} "
            f"CONF={CONFIDENCE_THRESHOLD_ENTRY} GDN={GUARDIAN_EXIT_THRESHOLD}")


# ── Per-coin backtest ─────────────────────────────────────────────────────────
def backtest_coin(sym):
    p = HOLDOUT_LABEL_DIR / f"{sym}_features_v3.parquet"
    if not p.exists():
        return None

    df = pd.read_parquet(p)
    df = ensure_utc_index(df).sort_index()

    # Merge HMM regime
    rp = HOLDOUT_LABEL_DIR / f"{sym}_regime_h1.parquet"
    if rp.exists():
        try:
            reg = pd.read_parquet(rp)
            for col in ["hmm_regime_enc", "hmm_regime"]:
                if col in df.columns:
                    df = df.drop(columns=[col])
            df = df.join(reg[["hmm_regime_enc", "hmm_regime"]], how="left")
            df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
            df["hmm_regime"] = df["hmm_regime"].fillna("RANGING_LOW_VOL")
        except Exception:
            pass
    else:
        if "hmm_regime_enc" not in df.columns:
            df["hmm_regime_enc"] = 1
        if "hmm_regime" not in df.columns:
            df["hmm_regime"] = "RANGING_LOW_VOL"

    # Filter to valid labels only
    mask = df["label"].astype(str).isin(LABEL_MAP)
    df   = df[mask].copy()
    y    = df["label"].map(LABEL_MAP).values.astype(np.int64)
    n    = len(df)
    if n < 50:
        return None

    # Align LGBM features
    X = np.zeros((n, len(feat_cols)), dtype=np.float64)
    for idx, col in enumerate(feat_cols):
        if col in df.columns:
            X[:, idx] = df[col].ffill().fillna(0).values.astype(np.float64)

    # 2-model cascade: LGBM + LSTM
    y_pred, confidence = hierarchical_predict(
        None, lgbm_model, lstm_model, lstm_scaler,
        X, feat_cols, [], df,
        model_dir=RUN_DIR,
        lstm_feat_cols=lstm_feat_cols,
    )

    # Confidence filter
    below = (y_pred != 1) & (confidence < CONFIDENCE_THRESHOLD_ENTRY)
    y_pred[below] = 1
    n_filtered = int(below.sum())

    atr   = df["atr_14_h1"].values  if "atr_14_h1"    in df.columns else np.ones(n)
    close = df["close"].values       if "close"         in df.columns else np.ones(n)
    high  = df["high"].values        if "high"          in df.columns else close
    low   = df["low"].values         if "low"           in df.columns else close
    h4_sh = df["h4_swing_high"].values if "h4_swing_high" in df.columns else None
    h4_sl = df["h4_swing_low"].values  if "h4_swing_low"  in df.columns else None
    h4t   = df["h4_trend"].values    if "h4_trend"      in df.columns else None
    volr  = df["vol_ratio_20"].values if "vol_ratio_20" in df.columns else None

    # Guardian static feature matrix
    X_guardian = compute_guardian_static_array(df, g_static)

    report = full_trading_report(
        y_pred         = y_pred,
        y_actual       = y,
        atr            = atr,
        close          = close,
        high           = high,
        low            = low,
        h4_swing_highs = h4_sh,
        h4_swing_lows  = h4_sl,
        index          = df.index,
        modal          = MODAL_PER_TRADE,
        leverages      = LEVERAGE_SIM,
        fee_per_side   = FEE_PER_SIDE,
        slippage       = SLIPPAGE_PER_SIDE,
        min_rr         = SWING_LABEL_MIN_RR,
        min_tp_atr     = SWING_LABEL_MIN_TP,
        max_sl_atr     = SWING_LABEL_MAX_SL,
        tp_fallback_atr = TP_SL_FALLBACK_TP,
        sl_fallback_atr = TP_SL_FALLBACK_SL,
        max_hold        = MAX_HOLDING_BARS,
        symbol          = sym,
        confidence      = confidence,
        guardian_model  = guardian_model,
        guardian_scaler = guardian_scaler,
        X_guardian      = X_guardian,
        guardian_exit_threshold = GUARDIAN_EXIT_THRESHOLD,
        trailing_stop_enabled   = TRAILING_STOP_ENABLED,
        trailing_stop_atr       = TRAILING_STOP_ATR,
        trailing_stop_min_bars  = TRAILING_STOP_MIN_BARS,
        h4_trend        = h4t,
        vol_ratio       = volr,
    )
    report["n_filtered"] = n_filtered
    report["n_bars"]     = n
    return report


# ── Run all coins ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*80}")
    print(f"  ic32_regime_v1 Holdout — Apr-Jun 2026 | 21 koin | $10/trade 5x")
    print(f"  LGBM: 0.69/0.59 | CONF: {CONFIDENCE_THRESHOLD_ENTRY} | GDN: {GUARDIAN_EXIT_THRESHOLD}")
    print(f"{'='*80}\n")

    results   = {}
    all_trades = []
    success, failed = [], []

    for sym in ALL_COINS:
        try:
            r = backtest_coin(sym)
            if r is None:
                failed.append(sym)
                continue
            results[sym] = r
            success.append(sym)
            pnl  = sum(t.get("net_pnl", 0) for t in r.get("trades", []))
            n_tr = r.get("total_trades", 0)
            wr   = r.get("winrate", 0) * 100
            logger.info(f"  [{sym}] {n_tr} trades | WR={wr:.1f}% | PnL=${pnl:+.2f} | "
                        f"filtered={r.get('n_filtered', 0)}")
            all_trades.extend(r.get("trades", []))
        except Exception as e:
            import traceback
            logger.error(f"  [{sym}] Error: {e}")
            logger.error(traceback.format_exc())
            failed.append(sym)

    if not results:
        logger.error("Tidak ada hasil — semua koin gagal.")
        sys.exit(1)

    # ── Aggregate ─────────────────────────────────────────────────────────────
    n_total  = len(all_trades)
    n_wins   = sum(1 for t in all_trades if t.get("net_pnl", 0) > 0)
    total_pnl = sum(t.get("net_pnl", 0) for t in all_trades)
    long_trades  = [t for t in all_trades if t.get("direction") == "LONG"]
    short_trades = [t for t in all_trades if t.get("direction") == "SHORT"]
    n_long_win   = sum(1 for t in long_trades  if t.get("net_pnl", 0) > 0)
    n_short_win  = sum(1 for t in short_trades if t.get("net_pnl", 0) > 0)

    # Outcome breakdown
    outcome_counts = {}
    for t in all_trades:
        oc = t.get("outcome", "UNKNOWN")
        outcome_counts[oc] = outcome_counts.get(oc, 0) + 1

    wins_pnl   = [t["net_pnl"] for t in all_trades if t.get("net_pnl", 0) > 0]
    losses_pnl = [t["net_pnl"] for t in all_trades if t.get("net_pnl", 0) <= 0]
    gross_profit = sum(wins_pnl)
    gross_loss   = abs(sum(losses_pnl))
    pf           = gross_profit / gross_loss if gross_loss > 0 else float("inf")
    hold_bars    = [t.get("hold_bars", t.get("bar_out", 0) - t.get("bar_in", 0))
                    for t in all_trades if "bar_in" in t or "hold_bars" in t]

    # Approximate hold bars from all_trades if structured differently
    try:
        avg_hold = np.mean([t["bar_out"] - t["bar_in"] for t in all_trades
                            if "bar_out" in t and "bar_in" in t])
    except Exception:
        avg_hold = 0.0

    HOLDOUT_MONTHS = 2.5

    wr_pct    = n_wins / n_total * 100 if n_total else 0
    long_wr   = n_long_win / len(long_trades) * 100  if long_trades  else 0
    short_wr  = n_short_win / len(short_trades) * 100 if short_trades else 0
    long_pct  = len(long_trades) / n_total * 100 if n_total else 0
    tp_rate   = outcome_counts.get("WIN", 0) / n_total * 100 if n_total else 0
    sl_rate   = outcome_counts.get("LOSS", 0) / n_total * 100 if n_total else 0
    gd_rate   = sum(v for k, v in outcome_counts.items() if "GUARDIAN" in k) / n_total * 100 if n_total else 0
    to_rate   = outcome_counts.get("TIMEOUT", 0) / n_total * 100 if n_total else 0

    print(f"\n{'='*80}")
    print(f"  IC32_REGIME_V1 — Holdout Apr-Jun 2026 | {len(success)} koin | ${MODAL_PER_TRADE}/trade 5x")
    print(f"{'='*80}")
    print(f"  Total Trades   : {n_total:,}")
    print(f"  Trades/bulan   : {n_total / HOLDOUT_MONTHS:.0f}")
    print(f"  Win Rate       : {wr_pct:.1f}%")
    print(f"  LONG WR        : {long_wr:.1f}% ({len(long_trades)} trades, {long_pct:.1f}%)")
    print(f"  SHORT WR       : {short_wr:.1f}% ({len(short_trades)} trades)")
    print(f"  Net PnL        : ${total_pnl:+.2f}")
    print(f"  PnL/bulan      : ${total_pnl / HOLDOUT_MONTHS:+.2f}")
    print(f"  PnL/trade      : ${total_pnl / n_total:+.3f}" if n_total else "  PnL/trade : N/A")
    print(f"  Profit Factor  : {pf:.2f}")
    print(f"  Avg Hold Bars  : {avg_hold:.1f}")
    print(f"\n  Exit breakdown:")
    print(f"    TP (WIN)     : {tp_rate:.1f}%")
    print(f"    SL (LOSS)    : {sl_rate:.1f}%")
    print(f"    Guardian     : {gd_rate:.1f}%")
    print(f"    Time Exit    : {to_rate:.1f}%")
    for k, v in sorted(outcome_counts.items(), key=lambda x: -x[1]):
        pct = v / n_total * 100 if n_total else 0
        print(f"    {k:<20}: {v:>5} ({pct:.1f}%)")

    # ── Comparison table vs TB ────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"  PERBANDINGAN — Apr-Jun 2026 (periode sama) — $10/trade 5x")
    print(f"{'='*80}")
    print(f"  {'Metrik':<20} {'ic32_regime_v1':>18} {'TB widyawardhana':>18}")
    print(f"  {'-'*58}")
    tb_ref = {"trades": 680, "wr": 60.0, "long_pct": 45.0, "pnl": 209,
              "pnl_pm": 84, "pnl_pt": 0.307, "exit": "SL+time+Guardian"}
    ic32_ref = {"trades": n_total, "wr": wr_pct, "long_pct": long_pct,
                "pnl": total_pnl, "pnl_pm": total_pnl / HOLDOUT_MONTHS,
                "pnl_pt": total_pnl / n_total if n_total else 0,
                "exit": "SwingTP+SL+Guardian"}
    rows = [
        ("Total Trades",   f"{ic32_ref['trades']:,}",         f"{tb_ref['trades']:,}"),
        ("Trades/bulan",   f"{ic32_ref['trades']/HOLDOUT_MONTHS:.0f}", f"{tb_ref['pnl_pm']/tb_ref['pnl']*tb_ref['trades']:.0f}"),
        ("Win Rate",       f"{ic32_ref['wr']:.1f}%",          f"{tb_ref['wr']:.1f}%"),
        ("LONG %",         f"{ic32_ref['long_pct']:.1f}%",    f"{tb_ref['long_pct']:.1f}%"),
        ("Net PnL",        f"${ic32_ref['pnl']:+.0f}",        f"${tb_ref['pnl']:+.0f}"),
        ("PnL/bulan",      f"${ic32_ref['pnl_pm']:+.0f}",     f"${tb_ref['pnl_pm']:+.0f}"),
        ("PnL/trade",      f"${ic32_ref['pnl_pt']:+.3f}",     f"${tb_ref['pnl_pt']:+.3f}"),
        ("Exit mode",      "Swing TP/SL",                      "Fixed SL+time"),
    ]
    for label, ic32_val, tb_val in rows:
        print(f"  {label:<20} {ic32_val:>18} {tb_val:>18}")

    print(f"\n  NOTE: Exit mekanisme berbeda — ic32 pakai Swing-based TP/SL,")
    print(f"        TB pakai fixed SL=1.5xATR + max_hold=36. Perbandingan tidak 100% apple-to-apple.")

    # ── Save ──────────────────────────────────────────────────────────────────
    out = {
        "meta": {
            "run": "ic32_regime_v1",
            "holdout_period": "2026-04-01 to 2026-06-13",
            "n_coins": len(success),
            "config": "V2.5 Hybrid: LONG=0.69 SHORT=0.59 CONF=0.59 GDN=0.65",
            "exit_mode": "swing_based_tp_sl + guardian_clean_v2",
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        },
        "aggregate": {
            "total_trades": n_total,
            "trades_per_month": round(n_total / HOLDOUT_MONTHS, 1),
            "win_rate": round(wr_pct, 2),
            "long_wr": round(long_wr, 2),
            "short_wr": round(short_wr, 2),
            "long_pct": round(long_pct, 2),
            "total_pnl": round(total_pnl, 2),
            "pnl_per_month": round(total_pnl / HOLDOUT_MONTHS, 2),
            "pnl_per_trade": round(total_pnl / n_total, 4) if n_total else 0,
            "profit_factor": round(pf, 3),
            "avg_hold_bars": round(float(avg_hold), 1),
            "tp_rate_pct": round(tp_rate, 2),
            "sl_rate_pct": round(sl_rate, 2),
            "guardian_exit_pct": round(gd_rate, 2),
            "time_exit_pct": round(to_rate, 2),
            "outcome_counts": outcome_counts,
        },
        "per_coin": {
            sym: {
                "trades": r.get("total_trades", 0),
                "wr":     round(r.get("winrate", 0) * 100, 2),
                "pnl":    round(sum(t.get("net_pnl", 0) for t in r.get("trades", [])), 2),
            }
            for sym, r in results.items()
        },
        "failed": failed,
    }

    out_path = RUN_DIR / "holdout_apr_jun26.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"Saved -> {out_path}")
    print(f"\n  Saved -> {out_path}")
    print(f"{'='*80}\n")
