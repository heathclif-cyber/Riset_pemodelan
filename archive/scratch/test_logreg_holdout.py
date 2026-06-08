"""
Holdout test: LogReg Meta-Combiner as trade filter
Compares BASELINE vs LOGREG FILTER on holdout Apr-Jun 2026
"""
import sys, json, joblib, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))
import config
config.LSTM_CONFIRMATION_ENABLED = True
from config import *
from pipeline.backtest_utils import hierarchical_predict, compute_guardian_static_array
from core.evaluator import simulate_trades_swing
import pipeline.backtest_utils as btu

HOLDOUT_LABEL_DIR = HOLDOUT_DIR / "labeled"
COINANK_DIR = ROOT / "data" / "coinank"
MACRO_DIR = ROOT / "data" / "macro"

# Load LogReg model
logreg = joblib.load(MODEL_DIR / "runs" / "logreg_meta_v1" / "logreg_meta.pkl")
logreg_scaler = joblib.load(MODEL_DIR / "runs" / "logreg_meta_v1" / "logreg_scaler.pkl")
info = json.load(open(MODEL_DIR / "runs" / "logreg_meta_v1" / "logreg_meta_info.json"))

# Load other models
lgbm = joblib.load(MODEL_DIR / "lgbm_baseline.pkl")
lstm_f = json.load(open(MODEL_DIR / "feature_cols_lstm_temporal.json"))
guard = joblib.load(MODEL_DIR / "guardian_best.pkl")
gs = joblib.load(MODEL_DIR / "guardian_scaler.pkl")
gf = json.load(open(MODEL_DIR / "guardian_feature_cols.json"))
gst = [c for c in gf if c not in set(GUARDIAN_DYNAMIC_FEATURES)]
from core.models import load_lstm
lstm = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
lsc = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
btu.SMART_ENTRY_MODE = "disabled"

# Load OI for positioning context
def load_oi():
    pos = {}
    for coin in TRAINING_COINS:
        p = COINANK_DIR / f"{coin}_oi.parquet"
        if not p.exists(): continue
        oi = pd.read_parquet(p).sort_index()
        oi_c = [c for c in oi.columns if c.startswith("oi_") or "oi" in c.lower()]
        if not oi_c: continue
        ot = oi[oi_c[0]]
        om = ot.rolling(20).mean(); os = ot.rolling(20).std().clip(lower=1e-8)
        pos[coin] = pd.DataFrame({"oi_zscore": (ot - om) / os}, index=oi.index)
    return pos

pos_data = load_oi()
fg_data = None
fg_p = MACRO_DIR / "fear_greed.parquet"
if fg_p.exists(): fg_data = pd.read_parquet(fg_p)

def compute_logreg_score(row, coin, bar_ts):
    """Compute the 7 features for LogReg at a specific bar."""
    hmm = int(row.get("hmm_regime_enc", 1))

    # LGBM conf - will be filled from cascade
    lgbm_conf = row.get("_lgbm_conf", 0.5)

    # LSTM support - from cascade
    lstm_sup = row.get("_lstm_support", 0.35)

    # With trend
    h4_t = float(row.get("h4_trend", 0))
    lgbm_dir = int(row.get("_lgbm_dir", 1))
    is_with = 1.0 if (lgbm_dir == 2 and h4_t > 0) or (lgbm_dir == 0 and h4_t < 0) else 0.0

    # OI zscore
    oi_z = 0.0
    if coin in pos_data:
        daily = pos_data[coin]
        entry = pd.Timestamp(bar_ts.date(), tz="UTC")
        avail = daily[daily.index <= entry]
        if len(avail) > 0:
            oi_z = float(avail["oi_zscore"].iloc[-1]) if pd.notna(avail["oi_zscore"].iloc[-1]) else 0.0

    # Fear & Greed zscore
    fg_z = 0.0
    if fg_data is not None:
        entry = pd.Timestamp(bar_ts.date(), tz="UTC")
        avail = fg_data[fg_data.index <= entry]
        if len(avail) > 0:
            fg_z = (float(avail["fear_greed_value"].iloc[-1]) - 50.0) / 25.0 if pd.notna(avail["fear_greed_value"].iloc[-1]) else 0.0

    # Is trending
    is_trending = 1.0 if hmm in (0, 3) else 0.0

    feats = np.array([[hmm, lgbm_conf, lstm_sup, is_with, oi_z, fg_z, is_trending]])
    feats_scaled = logreg_scaler.transform(feats)
    proba = logreg.predict_proba(feats_scaled)[0, 1]
    return proba


def run_holdout(coins, use_logreg_filter):
    trades = []; logreg_scores = []; n_blocked = 0; n_passed = 0

    for ci, coin in enumerate(coins):
        fp = HOLDOUT_LABEL_DIR / f"{coin}_features_v3.parquet"
        rp = HOLDOUT_LABEL_DIR / f"{coin}_regime_h1.parquet"
        if not fp.exists(): continue

        df = pd.read_parquet(fp).sort_index()
        if rp.exists():
            reg = pd.read_parquet(rp)
            if "hmm_regime_enc" in df.columns: df = df.drop(columns=["hmm_regime_enc"])
            df = df.join(reg[["hmm_regime_enc"]], how="left")
            df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
        df = df[df["label"].astype(str).isin(LABEL_MAP)].copy()
        if len(df) < 200: continue

        n = len(df)
        X = np.zeros((n, len(lstm_f)))
        for i, col in enumerate(lstm_f):
            if col in df.columns: X[:, i] = df[col].ffill().fillna(0).values

        yp, cf = hierarchical_predict(None, lgbm, lstm, lsc, X, lstm_f, [], df,
                                       trend_alignment_enabled=False, regime_aware_alignment=True)
        below = (yp != 1) & (cf < CONFIDENCE_THRESHOLD_ENTRY)

        # Compute LogReg scores for each bar
        for i in range(n):
            if yp[i] == 1: continue  # FLAT
            score = compute_logreg_score(df.iloc[i], coin, df.index[i])
            logreg_scores.append(score)

            if use_logreg_filter and score < 0.45:
                yp[i] = 1  # block → FLAT
                cf[i] = 0.0
                n_blocked += 1
            else:
                n_passed += 1

        below = (yp != 1) & (cf < CONFIDENCE_THRESHOLD_ENTRY); yp[below] = 1

        Xg = compute_guardian_static_array(df, gst)
        atr = df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(n)
        c = df["close"].values; h = df["high"].values if "high" in df.columns else c
        l = df["low"].values if "low" in df.columns else c
        sh = df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan)
        sl = df["h4_swing_low"].values if "h4_swing_low" in df.columns else np.full(n, np.nan)

        r = simulate_trades_swing(
            y_pred=yp, close=c, high=h, low=l, atr=atr, h4_swing_highs=sh, h4_swing_lows=sl,
            modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            max_hold=MAX_HOLDING_BARS, min_rr=SWING_LABEL_MIN_RR,
            min_tp_atr=SWING_LABEL_MIN_TP, max_sl_atr=SWING_LABEL_MAX_SL,
            tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
            confidence=cf, guardian_enabled=True,
            guardian_model=guard, guardian_scaler=gs,
            X_guardian=Xg, guardian_exit_threshold=GUARDIAN_EXIT_THRESHOLD,
            guardian_min_hold_bars=2)
        for t in r.get("trades", []):
            t["coin"] = coin; t["timestamp"] = df.index[t.get("bar_in", 0)]
        trades.extend(r.get("trades", []))

    n = len(trades); wins = [t for t in trades if t.get("net_pnl",0)>0]
    pnl = sum(t.get("net_pnl",0) for t in trades)
    gw = sum(t["net_pnl"] for t in wins)
    gl = abs(sum(t["net_pnl"] for t in trades if t.get("net_pnl",0)<=0))
    return dict(trades=n, wr=len(wins)/n*100 if n else 0, pnl=pnl,
                pf=gw/gl if gl else 0, blocked=n_blocked, passed=n_passed,
                logreg_avg=np.mean(logreg_scores) if logreg_scores else 0)

coins = TRAINING_COINS[:5]
print("BASELINE..."); b = run_holdout(coins, False)
print("LOGREG FILTER..."); f = run_holdout(coins, True)

print(f"""
{'='*60}
  HOLDOUT TEST: LogReg Meta-Combiner (Apr-Jun 2026)
  LogReg AUC={info['cv_auc_mean']:.4f} | Features: {len(info['features'])}
  {'Metric':<20} {'BASELINE':>12} {'LOGREG':>12} {'Delta':>10}
  {'-'*55}""")
for k, label in [("trades","Trades"),("wr","WR %"),("pnl","PnL $"),("pf","PF"),
                  ("blocked","LogReg Blocked"),("passed","LogReg Passed")]:
    bv = b[k]; fv = f[k]
    if isinstance(bv, float):
        print(f"  {label:<20} {bv:>12.1f} {fv:>12.1f} {fv-bv:>+10.1f}")
    else:
        print(f"  {label:<20} {bv:>12,} {fv:>12,} {fv-bv:>+10,}")
print(f"  {'='*55}")
print(f"  LogReg avg score: {b['logreg_avg']:.4f} (baseline) vs {f['logreg_avg']:.4f} (filtered)")
print(f"  Conclusion: LogReg filter {'HELPS' if f['pnl']>b['pnl'] else 'DOES NOT HELP'} PnL")
print(f"{'='*60}")
