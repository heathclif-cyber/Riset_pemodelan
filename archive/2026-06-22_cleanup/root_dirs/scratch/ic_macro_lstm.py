"""
scratch/ic_macro_lstm.py - IC test candidate LSTM features
Macro (free live) + OHLCV temporal dynamics vs TB label

Target: TB label (TP=2xATR, SL=1.5xATR, max_hold=36)
Method: Spearman IC, N_eff = N/24, standalone + marginal (Gram-Schmidt)
Threshold: |IC| >= 0.02, |t| >= 2.0
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import LABEL_DIR, TRAIN_CUTOFF_DATE
from core.features import triple_barrier_labeling

# ── Params ───────────────────────────────────────────────────────────────────
AUTOCORR_K    = 24       # H1 autocorrelation correction
MIN_IC        = 0.02
MIN_T         = 2.0
MIN_MARGINAL  = 0.01
TP_MULT       = 2.0
SL_MULT       = 1.5
MAX_HOLD      = 36
COINS         = ["BTCUSDT","ETHUSDT","SOLUSDT","BNBUSDT","XRPUSDT",
                 "DOGEUSDT","ADAUSDT","AVAXUSDT","LINKUSDT","DOTUSDT"]

# ── Load macro parquets ───────────────────────────────────────────────────────
MACRO_DIR = Path("data/macro")

def _norm_daily_index(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize index to UTC midnight, cast to datetime64[us, UTC]."""
    idx = pd.to_datetime(df.index)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    else:
        idx = idx.tz_convert("UTC")
    # normalize to midnight
    idx = idx.normalize()
    # cast to consistent precision
    idx = idx.astype("datetime64[us, UTC]")
    df = df.copy()
    df.index = idx
    df = df[~df.index.duplicated(keep="last")].sort_index()
    return df


def load_macro():
    dfs = {}

    # Fear & Greed — shift 1 day (known at end of day)
    p = MACRO_DIR / "fear_greed.parquet"
    if p.exists():
        dm = _norm_daily_index(pd.read_parquet(p)[["fear_greed_value"]])
        dfs["fg"] = dm.shift(1)  # T-1: no look-ahead

    # CoinGecko global (BTC dominance) — resample irregular timestamps to daily
    p = MACRO_DIR / "coingecko_global.parquet"
    if p.exists():
        dm = pd.read_parquet(p)
        if "btc_dominance" in dm.columns:
            dm = _norm_daily_index(dm[["btc_dominance"]])
            # resample to daily (fill gaps)
            dm = dm.resample("1D").last().ffill()
            dfs["cg"] = dm.shift(1)

    # ETF flow (yfinance proxy) — shift 1 day (flow known after US market close)
    p = MACRO_DIR / "etf_flow_btc.parquet"
    if p.exists():
        dm = pd.read_parquet(p)
        cols = [c for c in ["etf_total_change_usd","etf_gbtc_change_usd"] if c in dm.columns]
        if cols:
            dm = _norm_daily_index(dm[cols])
            dfs["etf"] = dm.shift(1)  # T-1: flow known after close

    # Macro cross-asset (SPY, VIX, QQQ, GLD, UUP, TLT) — shift 1 day
    p = MACRO_DIR / "macro_cross_asset.parquet"
    if p.exists():
        dm = pd.read_parquet(p)
        want = ["vix_close","vix_ret_1d","spy_ret_1d","spy_ret_5d",
                "qqq_ret_1d","qqq_ret_5d","gld_ret_1d","uup_ret_1d",
                "uup_ret_5d","tlt_ret_1d","tlt_ret_5d","hyg_ret_1d"]
        cols = [c for c in want if c in dm.columns]
        dm = _norm_daily_index(dm[cols])
        dfs["macro"] = dm.shift(1)  # T-1: US market closes before next Asia open

    # Combine — outer join on normalized daily UTC index
    parts = list(dfs.values())
    if not parts:
        return pd.DataFrame()
    out = parts[0]
    for p in parts[1:]:
        out = out.join(p, how="outer")
    return out.sort_index()


def derive_macro_features(macro_daily: pd.DataFrame) -> pd.DataFrame:
    """Add rolling trajectory features to macro daily data."""
    d = macro_daily.copy()

    # Fear & Greed trajectory
    if "fear_greed_value" in d.columns:
        d["fg_z20"]         = (d["fear_greed_value"] - d["fear_greed_value"].rolling(20).mean()) / (d["fear_greed_value"].rolling(20).std() + 1e-9)
        d["fg_7d_change"]   = d["fear_greed_value"].diff(7)
        d["fg_3d_change"]   = d["fear_greed_value"].diff(3)

    # BTC dominance trajectory
    if "btc_dominance" in d.columns:
        d["btc_dom_7d_change"] = d["btc_dominance"].diff(7)
        d["btc_dom_14d_change"]= d["btc_dominance"].diff(14)

    # VIX — ffill over weekends before rolling (trading days only in macro_daily)
    if "vix_close" in d.columns:
        vix_f = d["vix_close"].ffill()
        d["vix_z20"]        = (vix_f - vix_f.rolling(20, min_periods=10).mean()) / (vix_f.rolling(20, min_periods=10).std() + 1e-9)
        d["vix_5d_change"]  = vix_f.diff(5)
        d["vix_spike"]      = (d["vix_5d_change"] / (vix_f.shift(5) + 1e-9))

    # ETF flow trajectory — ffill over non-trading days
    for col in ["etf_total_change_usd","etf_gbtc_change_usd"]:
        if col in d.columns:
            short = col.replace("_change_usd","")
            e = d[col].ffill()
            d[f"{short}_5d_sum"]   = e.rolling(5, min_periods=3).sum()
            d[f"{short}_10d_sum"]  = e.rolling(10, min_periods=5).sum()
            d[f"{short}_mom"]      = d[f"{short}_5d_sum"] - d[f"{short}_5d_sum"].shift(5)
            d[f"{short}_z20"]      = (e - e.rolling(20, min_periods=10).mean()) / (e.rolling(20, min_periods=10).std() + 1e-9)

    # Macro cross-asset composites — ffill to cover weekends
    spy5 = d["spy_ret_5d"].ffill() if "spy_ret_5d" in d.columns else pd.Series(0, index=d.index)
    vix1 = d["vix_ret_1d"].ffill() if "vix_ret_1d" in d.columns else pd.Series(0, index=d.index)
    uup5 = d["uup_ret_5d"].ffill() if "uup_ret_5d" in d.columns else pd.Series(0, index=d.index)
    tlt5 = d["tlt_ret_5d"].ffill() if "tlt_ret_5d" in d.columns else pd.Series(0, index=d.index)
    d["risk_on_score"]  = spy5 - vix1
    d["dxy_headwind"]   = -uup5
    # Add individual ffilled versions for IC test
    if "spy_ret_5d" in d.columns:
        d["spy_ret_5d_ff"] = spy5
    if "tlt_ret_5d" in d.columns:
        d["tlt_ret_5d_ff"] = tlt5
    if "qqq_ret_5d" in d.columns:
        d["qqq_ret_5d_ff"] = d["qqq_ret_5d"].ffill()

    return d


def spearman_ic(x, y):
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 50:
        return 0.0, 0.0
    r, _ = stats.spearmanr(x[mask], y[mask])
    n_eff = mask.sum() / AUTOCORR_K
    t = r * np.sqrt(n_eff - 2) / (np.sqrt(1 - r**2) + 1e-12)
    return float(r), float(t)


def residualize(x, regressors):
    """Remove linear projection of regressors from x (Gram-Schmidt)."""
    r = x.copy()
    for reg in regressors:
        mask = ~(np.isnan(r) | np.isnan(reg))
        if mask.sum() < 50:
            continue
        beta = np.dot(r[mask], reg[mask]) / (np.dot(reg[mask], reg[mask]) + 1e-12)
        r = r - beta * reg
    return r


# ── Main ─────────────────────────────────────────────────────────────────────
print("Loading macro data...")
macro_daily = load_macro()
macro_daily = derive_macro_features(macro_daily)
print(f"  Macro daily rows: {len(macro_daily)}, cols: {len(macro_daily.columns)}")
print(f"  Cols: {list(macro_daily.columns)}")

# Candidate LSTM features = all macro columns + key OHLCV temporal dynamics
# (OHLCV temporal dynamics already in features_v3, pull them fresh)
OHLCV_TEMPORAL = [
    "rsi_slope_h4", "trend_accel_4h", "ofi_h4_delta", "cvd_slope_h4",
    "ema_21_slope_h4", "ema_50_slope_h4", "ofi_acceleration",
    "cvd_momentum_adv", "price_accel_1h", "vol_accel_3h",
    "relative_strength_momentum", "atr_percent_h4",
    "swing_momentum", "ofi_momentum_ratio",
]

# Accumulate all bars across coins
all_label, all_feats = [], {}

print(f"\nProcessing {len(COINS)} coins...")
cutoff = pd.Timestamp(TRAIN_CUTOFF_DATE)
if cutoff.tzinfo is None:
    cutoff = cutoff.tz_localize("UTC")

for coin in COINS:
    p = LABEL_DIR / f"{coin}_features_v3.parquet"
    if not p.exists():
        print(f"  {coin}: MISSING, skip")
        continue

    df = pd.read_parquet(p)
    df = df[df.index < cutoff].copy()
    if len(df) < 500:
        print(f"  {coin}: too few rows, skip")
        continue

    # TB label
    tb = triple_barrier_labeling(
        close=df["close"], high=df["high"], low=df["low"],
        atr_base=df["atr_14_h1"], tp_atr_mult=TP_MULT,
        sl_atr_mult=SL_MULT, max_hold=MAX_HOLD,
    )
    df["tb_label"] = tb.map({"SHORT": -1, "FLAT": 0, "LONG": 1})
    df = df.dropna(subset=["tb_label"])

    # Merge macro (forward-fill daily -> H1) — cast index to same precision
    idx_date = df.index.normalize().astype("datetime64[us, UTC]")
    macro_aligned = macro_daily.reindex(idx_date, method="ffill")
    macro_aligned.index = df.index

    # OHLCV temporal features (already in df)
    for feat in OHLCV_TEMPORAL:
        if feat not in all_feats:
            all_feats[feat] = []
        val = df[feat].values if feat in df.columns else np.full(len(df), np.nan)
        all_feats[feat].append(val)

    # Macro features
    for feat in macro_daily.columns:
        if feat not in all_feats:
            all_feats[feat] = []
        val = macro_aligned[feat].values if feat in macro_aligned.columns else np.full(len(df), np.nan)
        all_feats[feat].append(val)

    all_label.append(df["tb_label"].values)
    print(f"  {coin}: {len(df):,} bars")

# Concatenate
y = np.concatenate(all_label).astype(float)
feats = {k: np.concatenate(v) for k, v in all_feats.items()}
print(f"\nTotal bars: {len(y):,}  |  N_eff = {len(y)//AUTOCORR_K:,}")
print(f"TB label: LONG={( y==1).mean()*100:.1f}%  FLAT={(y==0).mean()*100:.1f}%  SHORT={(y==-1).mean()*100:.1f}%")

# ── IC test ───────────────────────────────────────────────────────────────────
print("\n--- STANDALONE IC ---")
results = []
for feat, arr in feats.items():
    ic, t = spearman_ic(arr, y)
    nn = np.isfinite(arr).mean() * 100
    results.append({"feature": feat, "ic": ic, "tstat": t, "nn_pct": nn})

results.sort(key=lambda x: abs(x["ic"]), reverse=True)

keep_feats = []
for r in results:
    verdict = "KEEP" if abs(r["ic"]) >= MIN_IC and abs(r["tstat"]) >= MIN_T else "WEAK"
    if abs(r["ic"]) >= MIN_IC and abs(r["tstat"]) >= MIN_T:
        keep_feats.append(r["feature"])
    print(f"  {r['feature']:<32} IC={r['ic']:+.4f}  t={r['tstat']:+.2f}  nn={r['nn_pct']:.0f}%  [{verdict}]")

# ── Marginal IC ───────────────────────────────────────────────────────────────
print(f"\n--- MARGINAL IC (Gram-Schmidt residualization) ---")
print(f"  Testing {len(keep_feats)} KEEP features for marginal contribution")

selected = []
selected_arrays = []
marginal_results = []

for feat in keep_feats:
    arr = feats[feat].astype(float)
    # Residualize against already-selected features
    arr_res = residualize(arr, selected_arrays)
    mic, mt = spearman_ic(arr_res, y)
    verdict = "KEEP" if abs(mic) >= MIN_MARGINAL else "REDUNDANT"
    marginal_results.append({"feature": feat, "standalone_ic": next(r["ic"] for r in results if r["feature"]==feat),
                              "marginal_ic": mic, "marginal_t": mt, "verdict": verdict})
    if abs(mic) >= MIN_MARGINAL:
        selected.append(feat)
        selected_arrays.append(arr_res)
    print(f"  {feat:<32} standalone={next(r['ic'] for r in results if r['feature']==feat):+.4f}  marginal={mic:+.4f}  t={mt:+.2f}  [{verdict}]")

print(f"\n=== FINAL SELECTED FEATURES ({len(selected)}) ===")
for i, f in enumerate(selected, 1):
    r = next(x for x in marginal_results if x["feature"] == f)
    print(f"  {i:2}. {f:<32} IC={r['standalone_ic']:+.4f}  marginalIC={r['marginal_ic']:+.4f}")

print("\nDone.")
