"""
test_lstm_flat_review_extended_fixed.py
========================================
Extended backtest (2020-2025, ~63 bulan) menggunakan model FIXED
yang sudah di-deploy — TANPA retrain per fold.

Model yang dipakai:
  - LGBM : lgbm_baseline.pkl  (ic32_regime_v1, 33 feat)
  - LSTM : lstm_best.pt       (ic32_lstm_momentum_v2, 11 feat)
  - Guard: guardian_best.pkl  (ic32_guardian_clean_v2, 40 feat)

Perbandingan:
  A) LSTM ON  + Flat Review ON  [konfigurasi live]
  B) LSTM ON  + Flat Review OFF
  C) LSTM OFF (baseline LGBM only)

Data: LABEL_DIR (training data 2020 -> TRAIN_CUTOFF_DATE), per coin.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json, joblib, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import pipeline.backtest_utils as btu
from pipeline.backtest_utils import compute_guardian_static_array, hierarchical_predict
from core.evaluator import simulate_trades_swing
from core.models import load_lstm
from config import (
    MODEL_DIR, LABEL_DIR, TRAINING_COINS, LABEL_MAP,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL, GUARDIAN_EXIT_THRESHOLD,
    GUARDIAN_DYNAMIC_FEATURES, TRAIN_CUTOFF_DATE,
    CONFIDENCE_THRESHOLD_ENTRY,
    LSTM_OVERRIDE_THRESHOLD, LSTM_DIRECTIONAL_REVIEW_THRESHOLD,
    LGBM_FLAT_REVIEW_THRESHOLD,
)

SEP = "=" * 72

def hdr(t): print(f"\n{SEP}\n  {t}\n{SEP}")

# ─────────────────────────────────────────────────────────────
# Load fixed deployed models
# ─────────────────────────────────────────────────────────────
hdr("Loading Deployed Models (FIXED)")

lgbm_model     = joblib.load(MODEL_DIR / "lgbm_baseline.pkl")
lgbm_feats     = json.load(open(MODEL_DIR / "feature_cols_v2.json"))
guardian       = joblib.load(MODEL_DIR / "guardian_best.pkl")
g_scaler       = joblib.load(MODEL_DIR / "guardian_scaler.pkl")
g_feats        = json.load(open(MODEL_DIR / "guardian_feature_cols.json"))
g_static       = [c for c in g_feats if c not in set(GUARDIAN_DYNAMIC_FEATURES)]
lstm_feat_cols = json.load(open(MODEL_DIR / "feature_cols_lstm_temporal.json"))
lstm_scaler    = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
lstm_model     = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
lstm_model.eval()

print(f"  [OK] lgbm_baseline.pkl      -> {len(lgbm_model.feature_name_)} feats")
print(f"  [OK] lstm_best.pt           -> {len(lstm_feat_cols)} feats (FIXED, no retrain)")
print(f"  [OK] guardian_best.pkl      -> {len(g_feats)} feats ({len(g_static)} static + 7 dyn)")
print(f"  Config: LSTM_OVERRIDE_THR={LSTM_OVERRIDE_THRESHOLD} | DIR_THR={LSTM_DIRECTIONAL_REVIEW_THRESHOLD}")
print(f"          LGBM_FLAT_THR={LGBM_FLAT_REVIEW_THRESHOLD} | CONF_ENTRY={CONFIDENCE_THRESHOLD_ENTRY}")

# ─────────────────────────────────────────────────────────────
# Load all training data (2020 → TRAIN_CUTOFF_DATE)
# ─────────────────────────────────────────────────────────────
hdr("Loading Training Data (2020 -> 2025-11)")

all_data = {}
for coin in TRAINING_COINS:
    fp = LABEL_DIR / f"{coin}_features_v3.parquet"
    rp = LABEL_DIR / f"{coin}_regime_h1.parquet"
    if not fp.exists():
        print(f"  [!] Skip {coin} — no parquet")
        continue
    df = pd.read_parquet(fp).sort_index()
    df = df[df.index < TRAIN_CUTOFF_DATE]
    if rp.exists():
        reg = pd.read_parquet(rp)
        if "hmm_regime_enc" in df.columns:
            df = df.drop(columns=["hmm_regime_enc"])
        df = df.join(reg[["hmm_regime_enc"]], how="left")
        df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
    mask = df["label"].astype(str).isin(LABEL_MAP)
    df   = df[mask].copy()
    if len(df) >= 500:
        all_data[coin] = df
        n_mo = (df.index[-1].year - df.index[0].year) * 12 + df.index[-1].month - df.index[0].month
        print(f"  [-] {coin:<15} {len(df):>6} bars  [{df.index[0].date()} -> {df.index[-1].date()}]  ~{n_mo}mo")

n_total_mo = sum(
    (df.index[-1].year - df.index[0].year)*12 + df.index[-1].month - df.index[0].month
    for df in all_data.values()
) // max(len(all_data), 1)
print(f"\n  Total: {len(all_data)} coins | ~{n_total_mo} months per coin")

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

CONF_THRESHOLD = CONFIDENCE_THRESHOLD_ENTRY  # 0.59

# ─────────────────────────────────────────────────────────────
# Runner: apply fixed model to all coins, all data
# ─────────────────────────────────────────────────────────────
def run_scenario(sc, all_data):
    btu.SMART_ENTRY_MODE          = "disabled"
    btu.LSTM_CONFIRMATION_ENABLED = sc["lstm_on"]
    btu.LSTM_FLAT_REVIEW_ENABLED  = sc["flat_review"]

    all_trades = []
    for coin, df in all_data.items():
        n = len(df)

        # LSTM input matrix
        X_lstm = np.zeros((n, len(lstm_feat_cols)), dtype=np.float64)
        for i, col in enumerate(lstm_feat_cols):
            if col in df.columns:
                X_lstm[:, i] = df[col].ffill().fillna(0).values

        _lstm   = lstm_model if sc["lstm_on"] else None
        _lstm_s = lstm_scaler if sc["lstm_on"] else None

        yp, cf = hierarchical_predict(
            None, lgbm_model, _lstm, _lstm_s,
            X_lstm, lgbm_feats, [], df,
            trend_alignment_enabled=False,
            regime_aware_alignment=True,
        )

        # Gate: below confidence threshold → FLAT
        below = (yp != 1) & (cf < CONF_THRESHOLD)
        yp[below] = 1

        # Guardian simulation
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
            t["coin"]      = coin
            t["timestamp"] = df.index[t.get("bar_in", 0)]
        all_trades.extend(r.get("trades", []))

    return all_trades


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

    return dict(
        trades=n, wr=len(wins)/n*100,
        lwr=len([t for t in lt if t.get("net_pnl",0)>0])/len(lt)*100 if lt else 0,
        swr=len([t for t in st if t.get("net_pnl",0)>0])/len(st)*100 if st else 0,
        long_n=len(lt), short_n=len(st),
        pnl=pnl, pf=gw/gl if gl>0 else float("inf"),
        avg=pnl/n,
        neg_months=neg_m, n_months=len(months),
        pnl_per_month=pnl/max(len(months),1),
        mean_mpnl=np.mean(mpnl), std_mpnl=np.std(mpnl),
        max_cl=max_cl, months=months, yearly=yearly,
    )


# ─────────────────────────────────────────────────────────────
# Run all scenarios
# ─────────────────────────────────────────────────────────────
all_results = []
for sc in SCENARIOS:
    print(f"\n  -> {sc['label']} ...", end="", flush=True)
    trades = run_scenario(sc, all_data)
    stats  = calc_stats(trades)
    stats["label"] = sc["label"]
    stats["short"] = sc["short"]
    all_results.append(stats)
    print(f" done  ({stats['trades']:,} trades | WR {stats['wr']:.1f}% | PnL ${stats['pnl']:.0f})")

# ─────────────────────────────────────────────────────────────
# MAIN SCORECARD
# ─────────────────────────────────────────────────────────────
hdr(f"EXTENDED BACKTEST SCORECARD — {len(all_data)} Coins, Fixed Deployed Model")

print()
FMT = "  {:<38} {:>7} {:>6} {:>6} {:>6} {:>9} {:>5} {:>7} {:>7} {:>6} {:>6}"
print(FMT.format("Scenario","Trades","WR%","L_WR%","S_WR%","PnL $","PF","$/Mo","NegMo","MaxCL","Avg/T"))
print("  " + "-" * 100)

base_r = all_results[-1]  # C is baseline
for r in all_results:
    delta = f"  ({r['pnl']-base_r['pnl']:+.0f})" if r["short"] != "C" else ""
    print(FMT.format(
        r["label"][:38], f"{r['trades']:,}",
        f"{r['wr']:.1f}%", f"{r['lwr']:.1f}%", f"{r['swr']:.1f}%",
        f"${r['pnl']:.0f}", f"{r['pf']:.2f}",
        f"${r['pnl_per_month']:.0f}",
        f"{r['neg_months']}/{r['n_months']}",
        r["max_cl"], f"{r['avg']:+.2f}",
    ) + delta)

# ─────────────────────────────────────────────────────────────
# YEARLY BREAKDOWN
# ─────────────────────────────────────────────────────────────
print(f"\n  YEARLY PnL BREAKDOWN (21 coins × full year)")
print("  " + "-" * 75)
all_years = sorted(set(y for r in all_results for y in r["yearly"].keys()))
labels = [r["short"] for r in all_results]
print("  " + f"{'Year':<8}" + "".join(f"  {l:>12}" for l in labels) + "  Delta A-C")
print("  " + "-" * 65)

for y in all_years:
    row = f"  {y:<8}"
    vals = [r["yearly"].get(y, {}).get("pnl", 0) for r in all_results]
    for v in vals:
        row += f"  {v:>+12.0f}"
    d_ac = vals[0] - vals[-1]
    row += f"  {d_ac:>+10.0f}"
    print(row)

print("  " + "-" * 65)
tot_row = f"  {'TOTAL':<8}"
tot_vals = [r["pnl"] for r in all_results]
for v in tot_vals:
    tot_row += f"  {v:>+12.0f}"
d_tot = tot_vals[0] - tot_vals[-1]
tot_row += f"  {d_tot:>+10.0f}"
print(tot_row)

# ─────────────────────────────────────────────────────────────
# MONTHLY TABLE (all months)
# ─────────────────────────────────────────────────────────────
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

# ─────────────────────────────────────────────────────────────
# KEY MARKET PERIODS
# ─────────────────────────────────────────────────────────────
print(f"\n  KEY MARKET PERIOD ANALYSIS")
print("  " + "-" * 90)
key_periods = [
    ("2021 Bull (Jan-May)",      "2021-01", "2021-05"),
    ("2021 ATH + Corr (Oct-Dec)","2021-10", "2021-12"),
    ("2022 Luna Bear (May-Jul)", "2022-05", "2022-07"),
    ("2022 FTX Crash (Nov-Dec)", "2022-11", "2022-12"),
    ("2023 Recovery (Jan-Apr)",  "2023-01", "2023-04"),
    ("2024 Bull Run (Oct-Dec)",  "2024-10", "2024-12"),
    ("2025 Ranging (Jan-Jun)",   "2025-01", "2025-06"),
]
print("  " + f"{'Period':<30}" + "".join(f"  {l:>10}" for l in labels) + "  Delta(A-C)  Note")
print("  " + "-" * 90)
for name, start, end in key_periods:
    row = f"  {name:<30}"
    vals = []
    for r in all_results:
        v = sum(d["pnl"] for m, d in r["months"].items() if start <= m <= end)
        vals.append(v)
        row += f"  {v:>+10.0f}"
    d_ac = vals[0] - vals[-1]
    tag  = "LSTM+" if d_ac > 50 else ("LSTM-" if d_ac < -50 else "~same")
    row += f"  {d_ac:>+10.0f}  [{tag}]"
    print(row)

# ─────────────────────────────────────────────────────────────
# DELTA SUMMARY
# ─────────────────────────────────────────────────────────────
hdr("DELTA vs Baseline C (LGBM Only)")
print()
print(f"  {'Scenario':<38} {'dTrades':>9} {'dWR%':>7} {'dPnL':>9} {'dPF':>7} {'d$/Mo':>8}")
print("  " + "-" * 82)
for r in all_results:
    if r["short"] == "C":
        continue
    dt   = r["trades"]        - base_r["trades"]
    dwr  = r["wr"]            - base_r["wr"]
    dp   = r["pnl"]           - base_r["pnl"]
    dpf  = r["pf"]            - base_r["pf"]
    dmo  = r["pnl_per_month"] - base_r["pnl_per_month"]
    print(f"  {r['label']:<38} {dt:>+9,} {dwr:>+6.1f}% {dp:>+9.0f} {dpf:>+7.2f} {dmo:>+8.0f}")

# A vs B (flat review isolation)
print(f"\n  FLAT REVIEW ISOLATION (A vs B)")
print("  " + "-" * 50)
sc_a = all_results[0]; sc_b = all_results[1]
d_t  = sc_a["trades"] - sc_b["trades"]
d_p  = sc_a["pnl"]    - sc_b["pnl"]
d_wr = sc_a["wr"]     - sc_b["wr"]
print(f"  Trades dari Flat Review override : {d_t:+,}")
print(f"  PnL dari Flat Review override    : ${d_p:+.0f}")
print(f"  WR delta A vs B                  : {d_wr:+.2f}%")
if d_t == 0:
    print(f"  => FLAT REVIEW tidak trigger sama sekali")
    print(f"     (LSTM confidence < LSTM_OVERRIDE_THR={LSTM_OVERRIDE_THRESHOLD} di semua bar)")
else:
    print(f"  => Avg PnL per override trade    : ${d_p/d_t:+.2f}")

# Best
hdr("KESIMPULAN")
best = max(all_results, key=lambda x: x["pnl"])
worst = min(all_results, key=lambda x: x["pnl"])
print(f"\n  BEST  : {best['label']}")
print(f"          Trades={best['trades']:,} | WR={best['wr']:.1f}% | PnL=${best['pnl']:.0f} | PF={best['pf']:.2f} | $/mo=${best['pnl_per_month']:.0f}")
print(f"\n  WORST : {worst['label']}")
print(f"          Trades={worst['trades']:,} | WR={worst['wr']:.1f}% | PnL=${worst['pnl']:.0f}")
print(f"\n  LSTM ON vs OFF delta   : {all_results[0]['pnl'] - all_results[-1]['pnl']:+.0f} PnL | {all_results[0]['trades'] - all_results[-1]['trades']:+,} trades")
print(f"  Flat Review ON vs OFF  : {all_results[0]['pnl'] - all_results[1]['pnl']:+.0f} PnL | {all_results[0]['trades'] - all_results[1]['trades']:+,} trades")
print(f"\n{SEP}\n")
