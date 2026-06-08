"""
test_lstm_flat_review.py
========================
Perbandingan scorecard:
  A) LSTM Confirmation ON  + Flat Review ON  (konfigurasi live saat ini)
  B) LSTM Confirmation ON  + Flat Review OFF (LSTM hanya soft-adjust, tidak override FLAT)
  C) LSTM Confirmation OFF                   (baseline murni LGBM)

Holdout OOS: Nov 2025 - Mar 2026, 10 koin pertama.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json, joblib, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pipeline.backtest_utils as btu
from pipeline.backtest_utils import hierarchical_predict, compute_guardian_static_array
from core.evaluator import simulate_trades_swing
from config import (
    MODEL_DIR, HOLDOUT_DIR, TRAINING_COINS, LABEL_MAP,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL, GUARDIAN_EXIT_THRESHOLD,
    GUARDIAN_DYNAMIC_FEATURES,
    LSTM_CONFIRMATION_ENABLED, LSTM_FLAT_REVIEW_ENABLED,
    LSTM_OVERRIDE_THRESHOLD, LSTM_DIRECTIONAL_REVIEW_THRESHOLD,
    LGBM_FLAT_REVIEW_THRESHOLD,
)

SEP = "=" * 68

def hdr(title):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)

# ─────────────────────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────────────────────
hdr("Loading Models")
from core.models import load_lstm

lgbm_model  = joblib.load(MODEL_DIR / "lgbm_baseline.pkl")
feat_cols   = json.load(open(MODEL_DIR / "feature_cols_v2.json"))
guardian    = joblib.load(MODEL_DIR / "guardian_best.pkl")
g_scaler    = joblib.load(MODEL_DIR / "guardian_scaler.pkl")
g_feats     = json.load(open(MODEL_DIR / "guardian_feature_cols.json"))
g_static    = [c for c in g_feats if c not in set(GUARDIAN_DYNAMIC_FEATURES)]

lstm_feat_cols = json.load(open(MODEL_DIR / "feature_cols_lstm_temporal.json"))
lstm_scaler    = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
lstm_model     = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")

print(f"  [OK] lgbm ({len(feat_cols)} feat) | lstm ({len(lstm_feat_cols)} feat) | guardian ({len(g_feats)} feat)")
print(f"  Config live: LSTM_CONFIRMATION={LSTM_CONFIRMATION_ENABLED} | FLAT_REVIEW={LSTM_FLAT_REVIEW_ENABLED}")
print(f"               LSTM_OVERRIDE_THR={LSTM_OVERRIDE_THRESHOLD} | DIR_REVIEW_THR={LSTM_DIRECTIONAL_REVIEW_THRESHOLD}")
print(f"               FLAT_REVIEW_THR={LGBM_FLAT_REVIEW_THRESHOLD}")

# ─────────────────────────────────────────────────────────────
# Load holdout data
# ─────────────────────────────────────────────────────────────
hdr("Loading Holdout Data (10 coins, OOS)")
all_data = {}
for coin in TRAINING_COINS[:10]:
    path = HOLDOUT_DIR / "labeled" / f"{coin}_features_v3.parquet"
    rp   = HOLDOUT_DIR / "labeled" / f"{coin}_regime_h1.parquet"
    if not path.exists():
        print(f"  [!] Skip {coin} — no parquet")
        continue
    df = pd.read_parquet(path).sort_index()
    if rp.exists():
        reg = pd.read_parquet(rp)
        if "hmm_regime_enc" in df.columns:
            df = df.drop(columns=["hmm_regime_enc"])
        df = df.join(reg[["hmm_regime_enc"]], how="left")
        df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
    mask = df["label"].astype(str).isin(LABEL_MAP)
    df = df[mask].copy()
    if len(df) >= 50:
        all_data[coin] = df
        print(f"  [-] {coin}: {len(df)} bars  [{df.index[0].date()} -> {df.index[-1].date()}]")

print(f"\n  Total: {len(all_data)} coins loaded")

# ─────────────────────────────────────────────────────────────
# Scenario definitions
# ─────────────────────────────────────────────────────────────
SCENARIOS = [
    {
        "label":              "A: LSTM ON  + Flat Review ON  [LIVE]",
        "lstm_confirmation":  True,
        "flat_review":        True,
        "short_label":        "A",
    },
    {
        "label":              "B: LSTM ON  + Flat Review OFF",
        "lstm_confirmation":  True,
        "flat_review":        False,
        "short_label":        "B",
    },
    {
        "label":              "C: LSTM OFF (baseline LGBM only)",
        "lstm_confirmation":  False,
        "flat_review":        False,
        "short_label":        "C",
    },
]

# ─────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────
def run_scenario(sc, all_data):
    """Run one scenario, return trade list."""
    btu.SMART_ENTRY_MODE          = "disabled"
    btu.LSTM_CONFIRMATION_ENABLED = sc["lstm_confirmation"]
    btu.LSTM_FLAT_REVIEW_ENABLED  = sc["flat_review"]

    all_trades = []
    for coin, df in all_data.items():
        n = len(df)

        # Build LGBM feature matrix
        gbm_feats = lgbm_model.feature_name_
        X_lgbm = np.zeros((n, len(gbm_feats)), dtype=np.float64)
        for idx, col in enumerate(gbm_feats):
            if col in df.columns:
                X_lgbm[:, idx] = df[col].ffill().fillna(0).values

        # Build LSTM feature matrix
        X_lstm = np.zeros((n, len(lstm_feat_cols)), dtype=np.float64)
        for idx, col in enumerate(lstm_feat_cols):
            if col in df.columns:
                X_lstm[:, idx] = df[col].ffill().fillna(0).values

        # Use lstm_model only if confirmation enabled
        _lstm = lstm_model if sc["lstm_confirmation"] else None
        _lstm_scaler = lstm_scaler if sc["lstm_confirmation"] else None

        yp, cf = hierarchical_predict(
            None, lgbm_model, _lstm, _lstm_scaler,
            X_lstm, feat_cols, [], df,
            trend_alignment_enabled=False,
            regime_aware_alignment=True,
        )

        Xg    = compute_guardian_static_array(df, g_static)
        atr   = df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(n)
        close = df["close"].values
        high  = df["high"].values  if "high"  in df.columns else close
        low   = df["low"].values   if "low"   in df.columns else close
        sh    = df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan)
        sl    = df["h4_swing_low"].values  if "h4_swing_low"  in df.columns else np.full(n, np.nan)

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
            t["coin"] = coin
            t["timestamp"] = df.index[t.get("bar_in", 0)]
        all_trades.extend(r.get("trades", []))

    return all_trades


def calc_stats(trades):
    n    = len(trades)
    if n == 0:
        return dict(trades=0, wr=0, lwr=0, swr=0, pnl=0, pf=0,
                    avg=0, long_n=0, short_n=0, sl_hits=0, sl_pct=0,
                    neg_months=0, pnl_per_month=0)
    wins = [t for t in trades if t.get("net_pnl", 0) > 0]
    lt   = [t for t in trades if t.get("direction") == "LONG"]
    st   = [t for t in trades if t.get("direction") == "SHORT"]
    sl_h = [t for t in trades if t.get("exit_reason", "") == "SL"]
    gw   = sum(t.get("net_pnl", 0) for t in wins)
    gl   = abs(sum(t.get("net_pnl", 0) for t in trades if t.get("net_pnl", 0) <= 0))
    pnl  = sum(t.get("net_pnl", 0) for t in trades)
    pf   = gw / gl if gl > 0 else float("inf")

    # Monthly breakdown
    monthly = {}
    for t in trades:
        m = pd.Timestamp(t.get("timestamp", "2000-01")).strftime("%Y-%m")
        monthly[m] = monthly.get(m, 0) + t.get("net_pnl", 0)
    neg_months = sum(1 for v in monthly.values() if v < 0)
    n_months   = max(len(monthly), 1)

    return dict(
        trades    = n,
        wr        = len(wins) / n * 100,
        lwr       = len([t for t in lt if t.get("net_pnl", 0) > 0]) / len(lt) * 100 if lt else 0,
        swr       = len([t for t in st if t.get("net_pnl", 0) > 0]) / len(st) * 100 if st else 0,
        pnl       = pnl,
        pf        = pf,
        avg       = pnl / n,
        long_n    = len(lt),
        short_n   = len(st),
        sl_hits   = len(sl_h),
        sl_pct    = len(sl_h) / n * 100,
        neg_months= neg_months,
        pnl_per_month = pnl / n_months,
        monthly   = monthly,
    )


# ─────────────────────────────────────────────────────────────
# Run all scenarios
# ─────────────────────────────────────────────────────────────
hdr(f"Running {len(SCENARIOS)} Scenarios on {len(all_data)} coins")
results = []
for sc in SCENARIOS:
    print(f"\n  -> {sc['label']} ...", end="", flush=True)
    trades = run_scenario(sc, all_data)
    stats  = calc_stats(trades)
    stats["label"]  = sc["label"]
    stats["short"]  = sc["short_label"]
    stats["trades_list"] = trades
    results.append(stats)
    print(f" done  ({stats['trades']} trades | WR {stats['wr']:.1f}% | PnL ${stats['pnl']:.1f})")

# ─────────────────────────────────────────────────────────────
# Scorecard Table
# ─────────────────────────────────────────────────────────────
hdr("SCORECARD COMPARISON — LSTM Flat Review ON vs OFF")
COLS = ["Scenario", "Trades", "LONG", "SHORT", "WR%", "L_WR%", "S_WR%",
        "PnL $", "PF", "Avg/T", "SL%", "Neg Mo", "$/Mo"]
W = [34, 7, 6, 6, 6, 6, 6, 8, 5, 7, 5, 7, 7]

def fmt_row(r, is_header=False):
    vals = [
        r[0], r[1], r[2], r[3],
        f"{r[4]:.1f}%", f"{r[5]:.1f}%", f"{r[6]:.1f}%",
        f"${r[7]:.1f}", f"{r[8]:.2f}", f"{r[9]:+.2f}",
        f"{r[10]:.1f}%", f"{r[11]}", f"${r[12]:.1f}"
    ] if not is_header else r
    return "  " + "  ".join(str(v).ljust(w) for v, w in zip(vals, W))

print()
print(fmt_row(COLS, is_header=True))
print("  " + "-" * (sum(W) + 2 * len(W)))

base_pnl = results[0]["pnl"]
for r in results:
    delta = f"  ({r['pnl'] - base_pnl:+.1f})" if r["label"] != results[0]["label"] else ""
    row_data = [
        r["label"], r["trades"], r["long_n"], r["short_n"],
        r["wr"], r["lwr"], r["swr"],
        r["pnl"], r["pf"], r["avg"],
        r["sl_pct"], r["neg_months"], r["pnl_per_month"],
    ]
    print(fmt_row(row_data) + delta)

# ─────────────────────────────────────────────────────────────
# Monthly PnL breakdown (side by side)
# ─────────────────────────────────────────────────────────────
print(f"\n\n  MONTHLY PnL BREAKDOWN")
print("  " + "-" * 60)

all_months = sorted(set(
    m for r in results for m in r["monthly"].keys()
))
labels = [r["short"] for r in results]
header = "  Month    " + "  ".join(f"{l:>10}" for l in labels)
print(header)
print("  " + "-" * (len(header) - 2))

for m in all_months:
    row = f"  {m}"
    for r in results:
        v = r["monthly"].get(m, 0)
        sign = "+" if v > 0 else ""
        row += f"  {sign}{v:>9.1f}"
    print(row)

# Totals
print("  " + "-" * (len(header) - 2))
tot_row = "  TOTAL   "
for r in results:
    sign = "+" if r["pnl"] > 0 else ""
    tot_row += f"  {sign}{r['pnl']:>9.1f}"
print(tot_row)

# ─────────────────────────────────────────────────────────────
# Delta Analysis
# ─────────────────────────────────────────────────────────────
hdr("DELTA ANALYSIS vs Baseline [C: LGBM Only]")
base = results[-1]  # C is baseline

print()
print(f"  {'Scenario':<35} {'dTrades':>8} {'dWR%':>7} {'dPnL $':>9} {'dPF':>7} {'dSL%':>7}")
print("  " + "-" * 75)
for r in results:
    if r["short"] == base["short"]:
        continue
    dt  = r["trades"]   - base["trades"]
    dwr = r["wr"]       - base["wr"]
    dp  = r["pnl"]      - base["pnl"]
    dpf = r["pf"]       - base["pf"]
    dsl = r["sl_pct"]   - base["sl_pct"]
    print(f"  {r['label']:<35} {dt:>+8} {dwr:>+6.1f}% {dp:>+9.1f} {dpf:>+7.2f} {dsl:>+6.1f}%")

# ─────────────────────────────────────────────────────────────
# Flat Review specific: trades yang HANYA muncul karena flat review
# (trades yang LGBM FLAT tapi LSTM override)
# ─────────────────────────────────────────────────────────────
hdr("FLAT REVIEW CONTRIBUTION")
sc_a = results[0]  # LSTM ON + Flat Review ON
sc_b = results[1]  # LSTM ON + Flat Review OFF
diff_trades = sc_a["trades"] - sc_b["trades"]
diff_pnl    = sc_a["pnl"]    - sc_b["pnl"]
if sc_b["trades"] > 0:
    print(f"\n  Trades ditambahkan Flat Review     : {diff_trades:+d}")
    print(f"  PnL ditambahkan Flat Review        : ${diff_pnl:+.1f}")
    if diff_trades != 0:
        pnl_per_override = diff_pnl / diff_trades if diff_trades != 0 else 0
        print(f"  Avg PnL per override trade         : ${pnl_per_override:+.2f}")
        print(f"  WR delta (A vs B)                  : {sc_a['wr'] - sc_b['wr']:+.1f}%")

# Best
print()
best = max(results, key=lambda x: x["pnl"])
print(f"  [BEST] {best['label']}")
print(f"         Trades={best['trades']} | WR={best['wr']:.1f}% | PnL=${best['pnl']:.1f} | PF={best['pf']:.2f}")

print(f"\n{SEP}\n")
