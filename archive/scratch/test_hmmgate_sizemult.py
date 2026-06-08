"""
A/B test: HMM Gate (LSTM only in TRENDING) + Size Multiplier (LS extreme -> 0.50x)
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

C = ROOT / "data" / "coinank"; OVERLAP = pd.Timestamp("2025-01-24", tz="UTC")

def add_pos(df, coin):
    oi_p = C / f"{coin}_oi.parquet"; lsp_p = C / f"{coin}_ls_position.parquet"
    lsa_p = C / f"{coin}_ls_account.parquet"
    if not oi_p.exists(): return df
    oi = pd.read_parquet(oi_p).sort_index()
    lsp = pd.read_parquet(lsp_p).sort_index() if lsp_p.exists() else None
    lsa = pd.read_parquet(lsa_p).sort_index() if lsa_p.exists() else None
    oi_t = None
    for col in [c for c in oi.columns if c.startswith("oi_")]:
        if oi_t is None: oi_t = oi[col].copy()
        else: oi_t = oi_t.fillna(0) + oi[col].fillna(0)
    daily = pd.DataFrame(index=oi_t.index)
    if lsp is not None and "top_trader_position_ls" in lsp.columns:
        ls = lsp["top_trader_position_ls"]
        lm = ls.rolling(20).mean(); ls_s = ls.rolling(20).std().clip(lower=1e-8)
        daily["ls_z20"] = (ls - lm) / ls_s
        daily["ls_extreme"] = (daily["ls_z20"].abs() > POSITIONING_LS_EXTREME_THR).astype(float)
    df["_d"] = pd.to_datetime(df.index.date, utc=True)
    daily["_d"] = pd.to_datetime(daily.index.date, utc=True)
    daily = daily.dropna(subset=["_d"]).set_index("_d")
    daily = daily[~daily.index.duplicated(keep="last")]
    for nc, sc in {"pos_extreme":"ls_extreme"}.items():
        if sc in daily.columns:
            df[nc] = daily[sc].reindex(df["_d"]).ffill().fillna(0).values
        else: df[nc] = 0.0
    return df.drop(columns=["_d"])

def run(coins, pos_enabled, hmm_gate_enabled):
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
    trades = []
    size_changes = 0
    for coin in coins:
        fp = LABEL_DIR / f"{coin}_features_v3.parquet"
        rp = LABEL_DIR / f"{coin}_regime_h1.parquet"
        if not fp.exists(): continue
        df = pd.read_parquet(fp).sort_index()
        df = df[(df.index >= OVERLAP) & (df.index < TRAIN_CUTOFF_DATE)]
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
        below = (yp != 1) & (cf < CONFIDENCE_THRESHOLD_ENTRY); yp[below] = 1

        # Get size multipliers per bar
        sm = btu._last_size_mult
        if sm is None: sm = np.ones(n)

        Xg = compute_guardian_static_array(df, gst)
        atr = df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(n)
        c = df["close"].values
        h = df["high"].values if "high" in df.columns else c
        l = df["low"].values if "low" in df.columns else c
        sh = df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan)
        sl = df["h4_swing_low"].values if "h4_swing_low" in df.columns else np.full(n, np.nan)

        # Apply size multiplier per entry bar
        for mod_val in [MODAL_PER_TRADE, MODAL_PER_TRADE]:
            pass

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
            t["coin"] = coin
            t["timestamp"] = df.index[bar_in]
            t["size_mult"] = entry_sm
            if entry_sm < 1.0:
                size_changes += 1
                # Adjust PnL for reduced size
                t["net_pnl"] = t.get("net_pnl", 0) * entry_sm
        trades.extend(r.get("trades", []))

    n = len(trades); wins = [t for t in trades if t.get("net_pnl",0)>0]
    pnl = sum(t.get("net_pnl",0) for t in trades)
    gw = sum(t["net_pnl"] for t in wins)
    gl = abs(sum(t["net_pnl"] for t in trades if t.get("net_pnl",0)<=0))
    lt = [t for t in trades if t.get("direction")=="LONG"]
    st = [t for t in trades if t.get("direction")=="SHORT"]
    return dict(trades=n, wr=len(wins)/n*100, pnl=pnl, pf=gw/gl if gl else 0,
                lwr=len([t for t in lt if t.get("net_pnl",0)>0])/len(lt)*100 if lt else 0,
                swr=len([t for t in st if t.get("net_pnl",0)>0])/len(st)*100 if st else 0,
                size_changes=size_changes, long_n=len(lt), short_n=len(st))

coins = TRAINING_COINS[:5]

print("Testing 3 configs...")
baseline = run(coins, False, False)     # A: no positioning, no HMM gate
hmm_only = run(coins, False, True)      # B: HMM gate only
pos_hmm  = run(coins, True,  True)      # C: HMM gate + size multiplier

print(f"""
{'='*70}
  POSITIONING + HMM GATE — Overlap Jan-Oct 2025, 5 coins
  {'Metric':<20} {'A: BASELINE':>12} {'B: HMM GATE':>12} {'C: SIZE+HMM':>12}
  {'-'*58}""")
for k, label in [("trades","Trades"),("wr","WR %"),("pnl","PnL $"),("pf","PF"),
                  ("lwr","LONG WR %"),("swr","SHORT WR %"),("size_changes","Size changes")]:
    av = baseline[k]; bv = hmm_only[k]; cv = pos_hmm[k]
    if isinstance(av, float):
        print(f"  {label:<20} {av:>12.1f} {bv:>12.1f} {cv:>12.1f}")
    else:
        print(f"  {label:<20} {av:>12,} {bv:>12,} {cv:>12,}")

# Check: are there EXTREME bars in the overlap?
from pathlib import Path as Pth
import pandas as pd
C = Pth("data/coinank")
oi_p = C / "BTCUSDT_oi.parquet"
lsp_p = C / "BTCUSDT_ls_position.parquet"
oi = pd.read_parquet(oi_p).sort_index()
lsp = pd.read_parquet(lsp_p).sort_index()
ls = lsp["top_trader_position_ls"]
lm = ls.rolling(20).mean(); ls_s = ls.rolling(20).std().clip(lower=1e-8)
ls_z20 = (ls - lm) / ls_s
extreme = (ls_z20.abs() > 2.0)
overlap_mask = (ls_z20.index >= pd.Timestamp("2025-01-24", tz="UTC")) & (ls_z20.index < pd.Timestamp("2025-11-01", tz="UTC"))
print(f"\n  LS extreme days in overlap: {extreme[overlap_mask].sum()} / {overlap_mask.sum()}")
print(f"  Size multiplier applied: {pos_hmm['size_changes']} trades at 0.50x")
print(f"{'='*70}")
