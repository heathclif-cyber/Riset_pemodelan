"""
Extended Scorecard: HMM Gate LSTM + Positioning Size Multiplier
21 coins, full training period (2020 -> TRAIN_CUTOFF_DATE)
Compares: BASELINE vs HMM_GATE vs POS+HMM
"""
import sys, json, joblib, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))
import config
config.LSTM_CONFIRMATION_ENABLED = True
config.LSTM_FLAT_REVIEW_ENABLED = True
from config import *
from pipeline.backtest_utils import hierarchical_predict, compute_guardian_static_array, _last_size_mult
from core.evaluator import simulate_trades_swing
import pipeline.backtest_utils as btu

C = ROOT / "data" / "coinank"

def add_pos(df, coin):
    oi_p = C / f"{coin}_oi.parquet"; lsp_p = C / f"{coin}_ls_position.parquet"
    if not oi_p.exists(): return df
    oi = pd.read_parquet(oi_p).sort_index()
    lsp = pd.read_parquet(lsp_p).sort_index() if lsp_p.exists() else None
    daily = pd.DataFrame(index=oi.index)
    if lsp is not None and "top_trader_position_ls" in lsp.columns:
        ls = lsp["top_trader_position_ls"]
        lm = ls.rolling(20).mean(); ls_s = ls.rolling(20).std().clip(lower=1e-8)
        daily["ls_extreme"] = ((ls - lm) / ls_s).abs().gt(POSITIONING_LS_EXTREME_THR).astype(float)
    df["_d"] = pd.to_datetime(df.index.date, utc=True)
    daily["_d"] = pd.to_datetime(daily.index.date, utc=True)
    daily = daily.dropna(subset=["_d"]).set_index("_d")
    daily = daily[~daily.index.duplicated(keep="last")]
    df["pos_extreme"] = daily["ls_extreme"].reindex(df["_d"]).ffill().fillna(0).values if "ls_extreme" in daily.columns else 0.0
    return df.drop(columns=["_d"])

def run_backtest(coins, pos_enabled, hmm_gate_enabled, label):
    config.POSITIONING_ENGINE_ENABLED = pos_enabled
    config.HMM_GATE_LSTM_ENABLED = hmm_gate_enabled
    btu.SMART_ENTRY_MODE = "disabled"
    lgbm = joblib.load(MODEL_DIR / "lgbm_baseline.pkl")
    lstm_f = json.load(open(MODEL_DIR / "feature_cols_lstm_temporal.json"))
    g = joblib.load(MODEL_DIR / "guardian_best.pkl"); gs = joblib.load(MODEL_DIR / "guardian_scaler.pkl")
    gf = json.load(open(MODEL_DIR / "guardian_feature_cols.json"))
    gst = [c for c in gf if c not in set(GUARDIAN_DYNAMIC_FEATURES)]
    from core.models import load_lstm
    lstm = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
    lsc = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
    trades = []; resized = 0; total_entries = 0

    for ci, coin in enumerate(coins):
        fp = LABEL_DIR / f"{coin}_features_v3.parquet"
        rp = LABEL_DIR / f"{coin}_regime_h1.parquet"
        if not fp.exists(): continue
        df = pd.read_parquet(fp).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]
        if rp.exists():
            reg = pd.read_parquet(rp)
            if "hmm_regime_enc" in df.columns: df = df.drop(columns=["hmm_regime_enc"])
            df = df.join(reg[["hmm_regime_enc"]], how="left")
            df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
        df = df[df["label"].astype(str).isin(LABEL_MAP)].copy()
        if len(df) < 500: continue
        df = add_pos(df, coin)
        n = len(df); X = np.zeros((n, len(lstm_f)))
        for i, col in enumerate(lstm_f):
            if col in df.columns: X[:, i] = df[col].ffill().fillna(0).values

        yp, cf = hierarchical_predict(None, lgbm, lstm, lsc, X, lstm_f, [], df,
                                       trend_alignment_enabled=False, regime_aware_alignment=True)
        sm = btu._last_size_mult.copy() if btu._last_size_mult is not None else np.ones(n)
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
            guardian_model=g, guardian_scaler=gs,
            X_guardian=Xg, guardian_exit_threshold=GUARDIAN_EXIT_THRESHOLD,
            guardian_min_hold_bars=2)

        for t in r.get("trades", []):
            bar_in = t.get("bar_in", 0)
            entry_sm = sm[bar_in] if bar_in < len(sm) else 1.0
            t["coin"] = coin; t["timestamp"] = df.index[bar_in]
            if entry_sm < 1.0:
                resized += 1
                t["net_pnl"] = t.get("net_pnl", 0) * entry_sm
            total_entries += 1
        trades.extend(r.get("trades", []))

        if (ci + 1) % 5 == 0:
            print(f"  ... {ci+1}/{len(coins)} coins done")

    return trades, resized, total_entries

def compute_scorecard(trades):
    n = len(trades); wins = [t for t in trades if t.get("net_pnl",0)>0]
    losses = [t for t in trades if t.get("net_pnl",0)<=0]
    pnl = sum(t.get("net_pnl",0) for t in trades)
    gw = sum(t["net_pnl"] for t in wins)
    gl = abs(sum(t["net_pnl"] for t in losses))
    lt = [t for t in trades if t.get("direction")=="LONG"]
    st = [t for t in trades if t.get("direction")=="SHORT"]
    sl_hits = sum(1 for t in trades if str(t.get("outcome","")).lower()=="loss")
    gx = sum(1 for t in trades if "guardian" in str(t.get("outcome","")).lower())
    gx_wr = sum(1 for t in trades if "guardian" in str(t.get("outcome","")).lower() and t.get("net_pnl",0)>0)/gx*100 if gx else 0

    # Monthly breakdown
    months = {}
    for t in trades:
        m = pd.Timestamp(t["timestamp"]).strftime("%Y-%m")
        months[m] = months.get(m, {"pnl":0,"trades":0,"wins":0})
        months[m]["pnl"] += t.get("net_pnl",0)
        months[m]["trades"] += 1
        if t.get("net_pnl",0) > 0: months[m]["wins"] += 1
    neg_m = sum(1 for d in months.values() if d["pnl"]<0)
    mpnl = [d["pnl"] for d in months.values()]

    # Yearly
    yearly = {}
    for m, d in months.items():
        y = m[:4]; yearly[y] = yearly.get(y, {"pnl":0,"trades":0})
        yearly[y]["pnl"] += d["pnl"]; yearly[y]["trades"] += d["trades"]

    # Max consecutive loss
    max_cl = cur = 0
    for t in sorted(trades, key=lambda x: str(x.get("timestamp",""))):
        if t.get("net_pnl",0) <= 0: cur += 1; max_cl = max(max_cl, cur)
        else: cur = 0

    holds = [t.get("bar_out",0)-t.get("bar_in",0) for t in trades if "bar_in" in t and "bar_out" in t]
    avg_hold = np.mean(holds) if holds else 0

    return dict(trades=n, wr=len(wins)/n*100, pnl=pnl, pf=gw/gl if gl else 0,
                lwr=len([t for t in lt if t.get("net_pnl",0)>0])/len(lt)*100 if lt else 0,
                swr=len([t for t in st if t.get("net_pnl",0)>0])/len(st)*100 if st else 0,
                long_n=len(lt), short_n=len(st), sl_hits=sl_hits, sl_pct=sl_hits/n*100,
                gx=gx, gx_wr=gx_wr, neg_m=neg_m, n_mo=len(months),
                max_cl=max_cl, avg_hold=avg_hold, yearly=yearly, months=months,
                mean_mpnl=np.mean(mpnl), std_mpnl=np.std(mpnl))

coins = TRAINING_COINS
print(f"\n{'='*65}")
print(f"  EXTENDED SCORECARD — 21 coins, Full Training Period")
print(f"  Comparing: A=BASELINE  B=HMM GATE  C=POS+HMM")
print(f"{'='*65}\n")

print("Running BASELINE (no pos, no HMM gate)...")
trades_a, _, entries_a = run_backtest(coins, False, False, "A")
sa = compute_scorecard(trades_a)

print("\nRunning HMM GATE (LSTM only TRENDING)...")
trades_b, _, entries_b = run_backtest(coins, False, True, "B")
sb = compute_scorecard(trades_b)

print("\nRunning POS+HMM (size mult + LSTM gate)...")
trades_c, resized_c, entries_c = run_backtest(coins, True, True, "C")
sc = compute_scorecard(trades_c)

print(f"\n{'='*70}")
print(f"  EXTENDED SCORECARD — 21 Coins, 2020-2025 Training")
print(f"  {'Metric':<22} {'A: BASELINE':>12} {'B: HMM GATE':>12} {'C: POS+HMM':>12}")
print(f"  {'-'*60}")
for k, label in [("trades","Total Trades"),("wr","Win Rate %"),("pnl","Net PnL $"),
                  ("pf","Profit Factor"),("lwr","LONG WR %"),("swr","SHORT WR %"),
                  ("sl_hits","SL Hits"),("sl_pct","SL Hit Rate %"),
                  ("gx_wr","Guardian Exit WR%"),("neg_m","Negative Months"),
                  ("max_cl","Max Cons Loss"),("avg_hold","Avg Hold Bars")]:
    av = sa[k]; bv = sb[k]; cv = sc[k]
    if isinstance(av, float):
        print(f"  {label:<22} {av:>12.1f} {bv:>12.1f} {cv:>12.1f}")
    else:
        print(f"  {label:<22} {av:>12,} {bv:>12,} {cv:>12,}")

print(f"\n  Yearly PnL:")
print(f"  {'Year':<8} {'A: BASELINE':>12} {'B: HMM GATE':>12} {'C: POS+HMM':>12}")
all_years = sorted(set(list(sa["yearly"].keys()) + list(sb["yearly"].keys()) + list(sc["yearly"].keys())))
for y in all_years:
    ay = sa["yearly"].get(y, {}).get("pnl", 0)
    by_ = sb["yearly"].get(y, {}).get("pnl", 0)
    cy = sc["yearly"].get(y, {}).get("pnl", 0)
    print(f"  {y:<8} {ay:>+12.0f} {by_:>+12.0f} {cy:>+12.0f}")

print(f"\n  Size changes: {resized_c} trades at 0.50x out of {entries_c} total ({resized_c/entries_c*100:.1f}%)")
print(f"{'='*70}")
