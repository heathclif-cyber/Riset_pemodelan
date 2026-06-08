"""A/B test: POSITIONING ENGINE ON vs OFF"""
import sys, json, joblib, numpy as np, pandas as pd
from pathlib import Path
ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))
import config
config.LSTM_CONFIRMATION_ENABLED = True
config.LSTM_FLAT_REVIEW_ENABLED = True
from config import *
from pipeline.backtest_utils import hierarchical_predict, compute_guardian_static_array
from core.evaluator import simulate_trades_swing
import pipeline.backtest_utils as btu

COINANK_DIR = ROOT / "data" / "coinank"

def add_pos(df, coin):
    oi_p = COINANK_DIR / f"{coin}_oi.parquet"
    lsp_p = COINANK_DIR / f"{coin}_ls_position.parquet"
    lsa_p = COINANK_DIR / f"{coin}_ls_account.parquet"
    if not oi_p.exists(): return df
    oi = pd.read_parquet(oi_p).sort_index()
    lsp = pd.read_parquet(lsp_p).sort_index() if lsp_p.exists() else None
    lsa = pd.read_parquet(lsa_p).sort_index() if lsa_p.exists() else None
    oi_cols = [c for c in oi.columns if c.startswith("oi_")]
    oi_t = None
    for col in oi_cols:
        if oi_t is None: oi_t = oi[col].copy()
        else: oi_t = oi_t.fillna(0) + oi[col].fillna(0)
    daily = pd.DataFrame(index=oi_t.index)
    oi_m = oi_t.rolling(20).mean(); oi_s = oi_t.rolling(20).std().clip(lower=1e-8)
    daily["oi_z20"] = (oi_t - oi_m) / oi_s
    daily["oi_d1"] = oi_t.pct_change(1)
    if lsp is not None and "top_trader_position_ls" in lsp.columns:
        ls = lsp["top_trader_position_ls"]
        daily["ls_pos_d7"] = ls.diff(7)
        ls_m = ls.rolling(20).mean(); ls_s = ls.rolling(20).std().clip(lower=1e-8)
        daily["ls_z20"] = (ls - ls_m) / ls_s
        daily["ls_extreme"] = (daily["ls_z20"].abs() > 2.0).astype(float)
    if lsa is not None and "top_trader_account_ls" in lsa.columns:
        daily["ls_acc"] = lsa["top_trader_account_ls"]
        if lsp is not None: daily["ls_pos"] = lsp["top_trader_position_ls"]
        if "ls_pos" in daily.columns:
            daily["smart_retail"] = daily["ls_pos"] - daily["ls_acc"]
    df["_date"] = pd.to_datetime(df.index.date, utc=True)
    daily["_date"] = pd.to_datetime(daily.index.date, utc=True)
    daily = daily.dropna(subset=["_date"]).set_index("_date")
    daily = daily[~daily.index.duplicated(keep="last")]
    for nc, sc in {"pos_oi_z20":"oi_z20","pos_ls_z20":"ls_z20","pos_ls_d7":"ls_pos_d7",
                    "pos_oi_d1":"oi_d1","pos_smart_retail":"smart_retail","pos_extreme":"ls_extreme"}.items():
        if sc in daily.columns:
            df[nc] = daily[sc].reindex(df["_date"]).ffill().fillna(0).values
        else: df[nc] = 0.0
    return df.drop(columns=["_date"])

def run(coins, pos_on):
    config.POSITIONING_ENGINE_ENABLED = pos_on
    btu.SMART_ENTRY_MODE = "disabled"
    lgbm = joblib.load(MODEL_DIR / "lgbm_baseline.pkl")
    lstm_f = json.load(open(MODEL_DIR / "feature_cols_lstm_temporal.json"))
    g = joblib.load(MODEL_DIR / "guardian_best.pkl")
    gs = joblib.load(MODEL_DIR / "guardian_scaler.pkl")
    gf = json.load(open(MODEL_DIR / "guardian_feature_cols.json"))
    gst = [c for c in gf if c not in set(GUARDIAN_DYNAMIC_FEATURES)]
    from core.models import load_lstm
    lstm = load_lstm(MODEL_DIR / "lstm_best.pt", device="cpu")
    lsc = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
    trades = []
    for coin in coins:
        fp = LABEL_DIR / f"{coin}_features_v3.parquet"
        rp = LABEL_DIR / f"{coin}_regime_h1.parquet"
        if not fp.exists(): continue
        df = pd.read_parquet(fp).sort_index(); df = df[df.index < TRAIN_CUTOFF_DATE]
        if rp.exists():
            reg = pd.read_parquet(rp)
            if "hmm_regime_enc" in df.columns: df = df.drop(columns=["hmm_regime_enc"])
            df = df.join(reg[["hmm_regime_enc"]], how="left")
            df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
        mask = df["label"].astype(str).isin(LABEL_MAP); df = df[mask].copy()
        if len(df) < 500: continue
        df = add_pos(df, coin)
        n = len(df); X = np.zeros((n, len(lstm_f)))
        for i, col in enumerate(lstm_f):
            if col in df.columns: X[:, i] = df[col].ffill().fillna(0).values
        yp, cf = hierarchical_predict(None, lgbm, lstm, lsc, X, lstm_f, [], df,
                                       trend_alignment_enabled=False, regime_aware_alignment=True)
        below = (yp != 1) & (cf < CONFIDENCE_THRESHOLD_ENTRY); yp[below] = 1
        Xg = compute_guardian_static_array(df, gst)
        atr = df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(n)
        c = df["close"].values; h = df["high"].values if "high" in df.columns else c
        l = df["low"].values if "low" in df.columns else c
        sh = df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan)
        sl = df["h4_swing_low"].values if "h4_swing_low" in df.columns else np.full(n, np.nan)
        r = simulate_trades_swing(
            y_pred=yp, close=c, high=h, low=l, atr=atr,
            h4_swing_highs=sh, h4_swing_lows=sl,
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
            t["coin"] = coin; t["timestamp"] = df.index[t.get("bar_in", 0)]
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
                long_n=len(lt), short_n=len(st))

coins = TRAINING_COINS[:5]
print("Running BASELINE..."); b = run(coins, False)
print("Running POSITIONING..."); p = run(coins, True)

print(f"\n{'='*65}")
print(f"  A/B TEST: POSITIONING ENGINE")
print(f"  {'Metric':<20} {'BASELINE (OFF)':>15} {'POS (ON)':>15} {'Delta':>10}")
print(f"  {'-'*62}")
for k, label in [("trades","Trades"),("wr","WR%"),("pnl","PnL $"),("pf","PF"),
                  ("lwr","LONG WR%"),("swr","SHORT WR%")]:
    bv = b[k]; pv = p[k]
    if isinstance(bv, float):
        print(f"  {label:<20} {bv:>15.1f} {pv:>15.1f} {pv-bv:>+10.1f}")
    else:
        print(f"  {label:<20} {bv:>15,} {pv:>15,} {pv-bv:>+10,}")
print(f"  {'='*62}")
print(f"  POS adjustment applied in cascade STEP 5")
print(f"  Rules active: LS extreme, OI z-score, smart money, OI delta")
print(f"{'='*65}")
