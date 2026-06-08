"""
test_lstm_oof_extended.py — GENUINE OOF Extended Backtest
================================================================
Purged CV (8-fold, retrain LGBM per fold) — model tidak pernah
lihat data uji. Bandingkan 3 skenario LSTM:

  A) LSTM ON  + Flat Review ON
  B) LSTM ON  + Flat Review OFF
  C) LSTM OFF (baseline LGBM only)

Data: Training 2020 -> TRAIN_CUTOFF_DATE (2025-11-01)
Metode: 8-fold purged CV per coin, LGBM retrain per fold
Model lain (Guardian) tetap fixed dari deployed models
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json, joblib, numpy as np, pandas as pd
from pathlib import Path
import lightgbm as lgb

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pipeline.backtest_utils as btu
from pipeline.backtest_utils import compute_guardian_static_array, hierarchical_predict
from core.evaluator import simulate_trades_swing
from core.models import load_lstm
from pipeline.shared import build_purged_folds
from config import (
    MODEL_DIR, LABEL_DIR, TRAINING_COINS, LABEL_MAP,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL, GUARDIAN_EXIT_THRESHOLD,
    GUARDIAN_DYNAMIC_FEATURES, TRAIN_CUTOFF_DATE,
    CONFIDENCE_THRESHOLD_ENTRY,
    N_FOLDS, PURGE_GAP_BARS,
    LSTM_OVERRIDE_THRESHOLD,
)

SEP = "=" * 72

def hdr(t):
    print(f"\n{SEP}\n  {t}\n{SEP}")

# ─────────────────────────────────────────────────────────────
# Load fixed deployed models (Guardian + LSTM — NOT retrained)
# ─────────────────────────────────────────────────────────────
hdr("Loading Fixed Models (Guardian + LSTM)")

guardian       = joblib.load(MODEL_DIR / "guardian_best.pkl")
g_scaler       = joblib.load(MODEL_DIR / "guardian_scaler.pkl")
g_feats        = json.load(open(MODEL_DIR / "guardian_feature_cols.json"))
g_static       = [c for c in g_feats if c not in set(GUARDIAN_DYNAMIC_FEATURES)]
lstm_feat_cols = json.load(open(MODEL_DIR / "feature_cols_lstm_temporal.json"))
lstm_scaler    = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
lstm_model     = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
lstm_model.eval()

lgbm_feat_list = json.load(open(MODEL_DIR / "feature_cols_v2.json"))

print(f"  Guardian: {len(g_feats)} feats ({len(g_static)} static + 7 dyn)")
print(f"  LSTM:     {len(lstm_feat_cols)} feats (FIXED, no fold retrain)")
print(f"  LGBM:     {len(lgbm_feat_list)} feats (RETRAIN per fold — genuine OOF)")

# ─────────────────────────────────────────────────────────────
# LGBM training params (consistent per fold)
# ─────────────────────────────────────────────────────────────
LGBM_PARAMS = {
    'objective': 'multiclass', 'num_class': 3, 'n_estimators': 500,
    'learning_rate': 0.05, 'max_depth': 6, 'num_leaves': 31,
    'min_child_samples': 50, 'subsample': 0.8, 'colsample_bytree': 0.8,
    'verbose': -1, 'n_jobs': -1, 'random_state': 42,
}

CONF_THRESHOLD = CONFIDENCE_THRESHOLD_ENTRY  # 0.59

# ─────────────────────────────────────────────────────────────
# Scenario definitions
# ─────────────────────────────────────────────────────────────
SCENARIOS = [
    {"label": "A: LSTM ON  + Flat Review ON  [LIVE]", "short": "A",
     "lstm_on": True,  "flat_review": True},
    {"label": "B: LSTM ON  + Flat Review OFF",         "short": "B",
     "lstm_on": True,  "flat_review": False},
    {"label": "C: LSTM OFF (baseline LGBM only)",      "short": "C",
     "lstm_on": False, "flat_review": False},
]


def run_fold_predictions(df_test, lgbm_fold, sc):
    """Run predictions for ONE test fold with given LSTM scenario."""
    btu.SMART_ENTRY_MODE          = "disabled"
    btu.LSTM_CONFIRMATION_ENABLED = sc["lstm_on"]
    btu.LSTM_FLAT_REVIEW_ENABLED  = sc["flat_review"]

    n_te = len(df_test)
    feat_cols = [c for c in lgbm_feat_list if c in df_test.columns]
    X_te = np.zeros((n_te, len(feat_cols)), dtype=np.float64)
    for i, col in enumerate(feat_cols):
        if col in df_test.columns:
            X_te[:, i] = df_test[col].ffill().fillna(0).values

    # LSTM input matrix
    X_lstm = np.zeros((n_te, len(lstm_feat_cols)), dtype=np.float64)
    for i, col in enumerate(lstm_feat_cols):
        if col in df_test.columns:
            X_lstm[:, i] = df_test[col].ffill().fillna(0).values

    _lstm_m = lstm_model if sc["lstm_on"] else None
    _lstm_s = lstm_scaler if sc["lstm_on"] else None

    yp, cf = hierarchical_predict(
        None, lgbm_fold, _lstm_m, _lstm_s,
        X_lstm, feat_cols, [], df_test,
        trend_alignment_enabled=False,
        regime_aware_alignment=True,
    )

    # Confidence gate
    below = (yp != 1) & (cf < CONF_THRESHOLD)
    yp[below] = 1

    # Guardian simulation
    Xg    = compute_guardian_static_array(df_test, g_static)
    atr   = df_test["atr_14_h1"].values if "atr_14_h1" in df_test.columns else np.ones(n_te)
    close = df_test["close"].values
    high  = df_test["high"].values  if "high"  in df_test.columns else close
    low   = df_test["low"].values   if "low"   in df_test.columns else close
    sh    = df_test["h4_swing_high"].values if "h4_swing_high" in df_test.columns else np.full(n_te, np.nan)
    sl    = df_test["h4_swing_low"].values  if "h4_swing_low"  in df_test.columns else np.full(n_te, np.nan)

    r = simulate_trades_swing(
        y_pred=yp, close=close, high=high, low=low, atr=atr,
        h4_swing_highs=sh, h4_swing_lows=sl,
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        max_hold=MAX_HOLDING_BARS, min_rr=SWING_LABEL_MIN_RR,
        min_tp_atr=SWING_LABEL_MIN_TP, max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        confidence=cf, guardian_enabled=True,
        guardian_model=guardian, guardian_scaler=g_scaler,
        X_guardian=Xg, guardian_exit_threshold=GUARDIAN_EXIT_THRESHOLD,
        guardian_min_hold_bars=2,
    )

    for t in r.get("trades", []):
        t["coin"]      = coin
        t["timestamp"] = df_test.index[t.get("bar_in", 0)]
    return r.get("trades", [])


def calc_stats(trades):
    if not trades:
        return {}
    n    = len(trades)
    wins = [t for t in trades if t.get("net_pnl", 0) > 0]
    lt   = [t for t in trades if t.get("direction") == "LONG"]
    st   = [t for t in trades if t.get("direction") == "SHORT"]
    gw   = sum(t["net_pnl"] for t in wins)
    gl   = abs(sum(t["net_pnl"] for t in trades if t.get("net_pnl", 0) <= 0))
    pnl  = sum(t.get("net_pnl", 0) for t in trades)

    months = {}
    for t in trades:
        m = pd.Timestamp(t["timestamp"]).strftime("%Y-%m")
        months.setdefault(m, {"pnl": 0, "trades": 0, "wins": 0})
        months[m]["pnl"]    += t.get("net_pnl", 0)
        months[m]["trades"] += 1
        if t.get("net_pnl", 0) > 0:
            months[m]["wins"] += 1

    yearly = {}
    for m, d in months.items():
        y = m[:4]
        yearly.setdefault(y, {"pnl": 0, "trades": 0})
        yearly[y]["pnl"]    += d["pnl"]
        yearly[y]["trades"] += d["trades"]

    mpnl  = [d["pnl"] for d in months.values()]
    neg_m = sum(1 for p in mpnl if p < 0)

    max_cl = cur = 0
    for t in sorted(trades, key=lambda x: str(x.get("timestamp", ""))):
        if t.get("net_pnl", 0) <= 0:
            cur += 1; max_cl = max(max_cl, cur)
        else:
            cur = 0

    holds = [t.get("bar_out", 0) - t.get("bar_in", 0)
             for t in trades if "bar_in" in t and "bar_out" in t]
    avg_hold = np.mean(holds) if holds else 0

    sl_hits = sum(1 for t in trades if str(t.get("outcome", "")).lower() == "loss")
    sl_pct  = sl_hits / n * 100 if n else 0

    return dict(
        trades=n, wr=len(wins)/n*100,
        lwr=len([t for t in lt if t.get("net_pnl",0)>0])/len(lt)*100 if lt else 0,
        swr=len([t for t in st if t.get("net_pnl",0)>0])/len(st)*100 if st else 0,
        long_n=len(lt), short_n=len(st),
        pnl=pnl, pf=gw/gl if gl>0 else float("inf"),
        avg=pnl/n, avg_hold=avg_hold,
        sl_hits=sl_hits, sl_pct=sl_pct,
        neg_months=neg_m, n_months=len(months),
        pnl_per_month=pnl/max(len(months),1),
        mean_mpnl=np.mean(mpnl), std_mpnl=np.std(mpnl),
        max_cl=max_cl, months=months, yearly=yearly,
    )


# ─────────────────────────────────────────────────────────────
# MAIN: Purged CV extended backtest
# ─────────────────────────────────────────────────────────────
hdr(f"GENUINE OOF EXTENDED BACKTEST — {N_FOLDS}-fold Purged CV, {len(TRAINING_COINS)} Coins")

all_trades = {sc["short"]: [] for sc in SCENARIOS}
total_folds = 0
skipped_folds = 0

for coin in TRAINING_COINS:
    feat_path = LABEL_DIR / f"{coin}_features_v3.parquet"
    reg_path  = LABEL_DIR / f"{coin}_regime_h1.parquet"
    if not feat_path.exists():
        print(f"  [!] SKIP {coin} — no features")
        continue

    df = pd.read_parquet(feat_path).sort_index()
    df = df[df.index < TRAIN_CUTOFF_DATE]

    if reg_path.exists():
        reg = pd.read_parquet(reg_path)
        if "hmm_regime_enc" in df.columns:
            df = df.drop(columns=["hmm_regime_enc"])
        df = df.join(reg[["hmm_regime_enc"]], how="left")
        df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")

    mask = df["label"].astype(str).isin(LABEL_MAP)
    df   = df[mask].copy()
    if len(df) < 500:
        print(f"  [!] SKIP {coin} — too few bars ({len(df)})")
        continue

    ts_index = pd.DatetimeIndex(df.index)
    folds    = build_purged_folds(ts_index, N_FOLDS, PURGE_GAP_BARS)

    n_mo = (df.index[-1].year - df.index[0].year) * 12 + df.index[-1].month - df.index[0].month
    print(f"\n  [{coin}] {len(df):,} bars, ~{n_mo}mo, {len(folds)} folds")

    for fi, (tr_idx, te_idx) in enumerate(folds):
        if len(te_idx) < 100:
            skipped_folds += 1
            continue

        df_tr = df.iloc[tr_idx]
        df_te = df.iloc[te_idx]

        # Features available in THIS coin
        feat_cols = [c for c in lgbm_feat_list if c in df_tr.columns]
        X_tr = df_tr[feat_cols].ffill().fillna(0)
        y_tr = df_tr["label"].map(LABEL_MAP).values.astype(np.int64)

        if len(np.unique(y_tr)) < 3:
            skipped_folds += 1
            continue

        # Retrain LGBM from scratch on THIS fold
        fold_model = lgb.LGBMClassifier(**LGBM_PARAMS)
        fold_model.fit(X_tr, y_tr)

        # Run all 3 scenarios on THE SAME test fold
        for sc in SCENARIOS:
            trades = run_fold_predictions(df_te, fold_model, sc)
            all_trades[sc["short"]].extend(trades)

        total_folds += 1
        if total_folds % 10 == 0:
            print(f"    ... {total_folds} folds done", end="", flush=True)

print(f"\n\n  Total folds processed: {total_folds}")
print(f"  Skipped folds:         {skipped_folds}")

# ─────────────────────────────────────────────────────────────
# SCORECARD
# ─────────────────────────────────────────────────────────────
all_results = []
for sc in SCENARIOS:
    stats = calc_stats(all_trades[sc["short"]])
    stats["label"] = sc["label"]
    stats["short"] = sc["short"]
    all_results.append(stats)

hdr(f"GENUINE OOF SCORECARD — {total_folds} folds, purged CV")

print()
FMT = "  {:<42} {:>7} {:>6} {:>6} {:>6} {:>9} {:>5} {:>7} {:>7} {:>6} {:>6}"
print(FMT.format("Scenario","Trades","WR%","L_WR%","S_WR%","PnL $","PF","$/Mo","NegMo","MaxCL","Avg/T"))
print("  " + "-" * 106)

base_r = all_results[-1]  # C is baseline
for r in all_results:
    delta = f"  ({r['pnl']-base_r['pnl']:+.0f})" if r["short"] != "C" else ""
    print(FMT.format(
        r["label"][:42], f"{r['trades']:,}",
        f"{r['wr']:.1f}%", f"{r['lwr']:.1f}%", f"{r['swr']:.1f}%",
        f"${r['pnl']:.0f}", f"{r['pf']:.2f}",
        f"${r['pnl_per_month']:.0f}",
        f"{r['neg_months']}/{r['n_months']}",
        r["max_cl"], f"{r['avg']:+.2f}",
    ) + delta)

# Yearly breakdown
print(f"\n  YEARLY PnL BREAKDOWN")
print("  " + "-" * 75)
all_years = sorted(set(y for r in all_results for y in r["yearly"].keys()))
labels = [r["short"] for r in all_results]
print("  " + f"{'Year':<8}" + "".join(f"  {l:>12}" for l in labels) + "  Delta A-C")
print("  " + "-" * 65)
tot_vals = [r["pnl"] for r in all_results]
for y in all_years:
    row = f"  {y:<8}"
    vals = [r["yearly"].get(y, {}).get("pnl", 0) for r in all_results]
    for v in vals:
        row += f"  {v:>+12.0f}"
    row += f"  {vals[0]-vals[-1]:>+10.0f}"
    print(row)
print("  " + "-" * 65)
tot_row = f"  {'TOTAL':<8}"
for v in tot_vals:
    tot_row += f"  {v:>+12.0f}"
tot_row += f"  {tot_vals[0]-tot_vals[-1]:>+10.0f}"
print(tot_row)

# Monthly breakdown
print(f"\n  MONTHLY PnL — all {all_results[0]['n_months']} months")
print("  " + "-" * 60)
all_months = sorted(set(m for r in all_results for m in r["months"].keys()))
print("  Month   " + "".join(f"  {l:>10}" for l in labels))
print("  " + "-" * 50)
for m in all_months:
    row = f"  {m}"
    for r in all_results:
        v = r["months"].get(m, {}).get("pnl", 0)
        row += f"  {v:>+10.0f}"
    print(row)

# Delta summary
hdr("DELTA vs Baseline C (LGBM Only)")
print()
print(f"  {'Scenario':<42} {'dTrades':>9} {'dWR%':>7} {'dPnL':>9} {'dPF':>7} {'d$/Mo':>8}")
print("  " + "-" * 82)
for r in all_results:
    if r["short"] == "C":
        continue
    dt   = r["trades"]        - base_r["trades"]
    dwr  = r["wr"]            - base_r["wr"]
    dp   = r["pnl"]           - base_r["pnl"]
    dpf  = r["pf"]            - base_r["pf"]
    dmo  = r["pnl_per_month"] - base_r["pnl_per_month"]
    print(f"  {r['label']:<42} {dt:>+9,} {dwr:>+6.1f}% {dp:>+9.0f} {dpf:>+7.2f} {dmo:>+8.0f}")

print(f"\n  FLAT REVIEW ISOLATION (A vs B)")
sc_a = all_results[0]; sc_b = all_results[1]
print(f"  dTrades: {sc_a['trades']-sc_b['trades']:+,}")
print(f"  dPnL:    ${sc_a['pnl']-sc_b['pnl']:+.0f}")

best  = max(all_results, key=lambda x: x["pnl"])
worst = min(all_results, key=lambda x: x["pnl"])
print(f"\n  BEST  : {best['label']} → Trades={best['trades']:,} WR={best['wr']:.1f}% PnL=${best['pnl']:.0f} PF={best['pf']:.2f}")
print(f"  WORST : {worst['label']} → Trades={worst['trades']:,} WR={worst['wr']:.1f}% PnL=${worst['pnl']:.0f}")
print(f"\n{SEP}\n")
