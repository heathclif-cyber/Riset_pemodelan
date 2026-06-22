"""
pipeline/07_holdout_livelike_v4.py — Holdout comparison: tb_v3 vs tb_v4 (+ coinank).

v4 menggunakan 22 fitur: 18 v3 + 4 coinank (ls_pos_zscore_20d,
smart_retail_div_delta_1d, oi_pct_1d, oi_price_div_1d).
Holdout Nov 2025 – Apr 2026 = penuh coinank coverage.
"""
import json, sys, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import joblib
from core.utils import setup_logger, ensure_utc_index
from config import *

logger = setup_logger("07_v4_holdout")

COINANK_DIR  = ROOT / "data" / "coinank"
SL_MULT      = TP_SL_FALLBACK_SL
MAX_HOLD     = MAX_HOLDING_BARS
MODAL        = MODAL_PER_TRADE
LEVERAGE     = LEVERAGE_SIM[0] if isinstance(LEVERAGE_SIM, list) else LEVERAGE_SIM
COST_RT      = (FEE_PER_SIDE + SLIPPAGE_PER_SIDE) * 2
REGIME_THRESH = {0: 0.45, 1: 0.50, 2: 0.50, 3: 0.45}

# ── Load models ────────────────────────────────────────────────────────────────
v3_model = joblib.load(MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v3" / "lgbm.pkl")
with open(MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v3" /
          "tb_lgbm_widyawardhana_v3_features.json") as f:
    v3_feats = json.load(f)

v4_model = joblib.load(MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v4" / "lgbm.pkl")
with open(MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v4" /
          "tb_lgbm_widyawardhana_v4_features.json") as f:
    v4_feats = json.load(f)

COINANK_FEAT_NAMES = [
    "ls_pos_zscore_20d", "smart_retail_div_delta_1d",
    "oi_pct_1d", "oi_price_div_1d",
]

available = [s for s in ALL_COINS
             if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()]

print(f"\n{'='*68}")
print(f"  HOLDOUT OOS — tb_v3 vs tb_v4 (+coinank) | Nov 2025 – Apr 2026")
print(f"  v3: 18 feats | v4: 22 feats (+4 coinank, full coverage holdout)")
print(f"  Exit: SL={SL_MULT}xATR | max_hold={MAX_HOLD}bar | NO TP")
print(f"  Coins: {len(available)} | ${MODAL}/trade {LEVERAGE}x")
print(f"{'='*68}\n")


def add_coinank_feats(df, sym):
    """Generate coinank features untuk holdout data."""
    idx = df.index
    oi_p  = COINANK_DIR / f"{sym}_oi.parquet"
    lsp_p = COINANK_DIR / f"{sym}_ls_position.parquet"
    lsa_p = COINANK_DIR / f"{sym}_ls_account.parquet"

    result = pd.DataFrame(index=idx)
    ls_pos = None

    if oi_p.exists():
        oi = pd.read_parquet(oi_p).sort_index()
        if "oi_binance" in oi.columns:
            oi_s = oi["oi_binance"].reindex(idx, method="ffill")
            result["oi_pct_1d"]       = oi_s.pct_change(24)
            result["oi_price_div_1d"] = oi_s.pct_change(24) - df["close"].pct_change(24)

    if lsp_p.exists():
        lsp = pd.read_parquet(lsp_p).sort_index()
        if "top_trader_position_ls" in lsp.columns:
            ls_pos = lsp["top_trader_position_ls"].reindex(idx, method="ffill")
            lp_mean = ls_pos.rolling(24*20, min_periods=100).mean()
            lp_std  = ls_pos.rolling(24*20, min_periods=100).std().clip(lower=1e-8)
            result["ls_pos_zscore_20d"] = (ls_pos - lp_mean) / lp_std

    if lsa_p.exists() and ls_pos is not None:
        lsa = pd.read_parquet(lsa_p).sort_index()
        if "top_trader_account_ls" in lsa.columns:
            ls_acc = lsa["top_trader_account_ls"].reindex(idx, method="ffill")
            result["smart_retail_div_delta_1d"] = (ls_pos - ls_acc).diff(24)

    return result


def simulate_livelike(yp, close, high, low, atr):
    n = len(yp); trades = []; i = 0
    while i < n:
        sig = yp[i]
        if sig == 1:
            i += 1; continue
        direction  = 1 if sig == 2 else -1
        entry      = close[i]
        sl_price   = entry - direction * SL_MULT * atr[i]
        exit_price = close[min(i + MAX_HOLD, n - 1)]
        exit_bar   = min(i + MAX_HOLD, n - 1)
        outcome    = "TIME"
        for j in range(i + 1, min(i + MAX_HOLD + 1, n)):
            if direction == 1 and low[j] <= sl_price:
                exit_price, exit_bar, outcome = sl_price, j, "SL"; break
            elif direction == -1 and high[j] >= sl_price:
                exit_price, exit_bar, outcome = sl_price, j, "SL"; break
        ret     = (exit_price - entry) / entry * direction
        net_pnl = ret * MODAL * LEVERAGE - COST_RT * MODAL * LEVERAGE
        trades.append({"direction": "LONG" if direction==1 else "SHORT",
                        "outcome": outcome, "net_pnl": net_pnl,
                        "bars_held": exit_bar - i})
        i = exit_bar + 1
    return trades


def new_agg():
    return {"trades":0,"wins":0,"pnl":0.0,"longs":0,"long_wins":0,
            "sl_hits":0,"bars_held":[]}

def update_agg(a, trades):
    for t in trades:
        a["trades"]+=1; a["pnl"]+=t["net_pnl"]; a["bars_held"].append(t["bars_held"])
        if t["net_pnl"]>0: a["wins"]+=1
        if t["outcome"]=="SL": a["sl_hits"]+=1
        if t["direction"]=="LONG":
            a["longs"]+=1
            if t["net_pnl"]>0: a["long_wins"]+=1

agg = {"v3": new_agg(), "v4": new_agg()}

for sym in available:
    df = pd.read_parquet(HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet")
    df = ensure_utc_index(df).sort_index()

    rp  = HOLDOUT_DIR / "labeled" / f"{sym}_regime_h1.parquet"
    hmm = np.full(len(df), 1, np.int32)
    if rp.exists():
        reg = pd.read_parquet(rp)
        if "hmm_regime_enc" in reg.columns:
            hmm = reg["hmm_regime_enc"].reindex(df.index, fill_value=1).values.astype(np.int32)

    mask = df["label"].isin({"SHORT":0,"FLAT":1,"LONG":2})
    df   = df[mask].copy()
    hmm  = hmm[mask.values]
    n    = len(df)

    close = df["close"].values.astype(np.float64)
    high  = df["high"].values.astype(np.float64)
    low   = df["low"].values.astype(np.float64)
    atr   = df["atr_14_h1"].values.astype(np.float64)

    # ── v3 predictions ─────────────────────────────────────────────────────
    X_v3 = np.zeros((n, len(v3_feats)), dtype=np.float64)
    for idx_f, c in enumerate(v3_feats):
        if c in df.columns:
            X_v3[:, idx_f] = df[c].ffill().fillna(0).values.astype(np.float64)
    p_v3    = v3_model.predict_proba(X_v3)
    conf_v3 = np.max(p_v3, axis=1)
    yp_v3   = np.argmax(p_v3, axis=1).astype(np.int32)
    for r, th in REGIME_THRESH.items():
        yp_v3[(hmm==r) & (yp_v3!=1) & (conf_v3<th)] = 1

    # ── v4 predictions (+ coinank features) ────────────────────────────────
    coinank_df = add_coinank_feats(df, sym)
    cn_avail   = coinank_df.notna().all(axis=1).sum()

    X_v4 = np.zeros((n, len(v4_feats)), dtype=np.float64)
    for idx_f, c in enumerate(v4_feats):
        if c in df.columns:
            X_v4[:, idx_f] = df[c].ffill().fillna(0).values.astype(np.float64)
        elif c in coinank_df.columns:
            X_v4[:, idx_f] = coinank_df[c].ffill().values.astype(np.float64)
    p_v4    = v4_model.predict_proba(X_v4)
    conf_v4 = np.max(p_v4, axis=1)
    yp_v4   = np.argmax(p_v4, axis=1).astype(np.int32)
    for r, th in REGIME_THRESH.items():
        yp_v4[(hmm==r) & (yp_v4!=1) & (conf_v4<th)] = 1

    update_agg(agg["v3"], simulate_livelike(yp_v3, close, high, low, atr))
    update_agg(agg["v4"], simulate_livelike(yp_v4, close, high, low, atr))

    logger.info(f"[{sym}] v3={agg['v3']['trades']} v4={agg['v4']['trades']} "
                f"| coinank_bars={cn_avail}/{n}")


def sc(a):
    t=a["trades"]; w=a["wins"]; p=a["pnl"]
    l=a["longs"]; lw=a["long_wins"]; sl=a["sl_hits"]; bh=a["bars_held"]
    return dict(trades=t, wr=w/max(t,1)*100,
                long_wr=lw/max(l,1)*100, short_wr=(w-lw)/max(t-l,1)*100,
                long_pct=l/max(t,1)*100, sl_rate=sl/max(t,1)*100,
                avg_hold=float(np.mean(bh)) if bh else 0,
                pnl=p, ppm=p/5, ppt=p/max(t,1))

s = {k: sc(agg[k]) for k in agg}
W = 18

print(f"\n{'='*62}")
print(f"  SCORECARD — tb_v3 vs tb_v4 | Nov 2025–Apr 2026 | 21 koin")
print(f"{'='*62}")
print(f"  {'Metrik':<22} {'tb_v3 (18 feat)':>{W}} {'tb_v4 (22 feat)':>{W}}")
print(f"  {'-'*60}")

rows = [
    ("Total Trades",   lambda x: f"{x['trades']:,}"),
    ("Trades/bulan",   lambda x: f"{x['trades']/5:.0f}"),
    ("Win Rate",       lambda x: f"{x['wr']:.1f}%"),
    ("  LONG WR",      lambda x: f"{x['long_wr']:.1f}% ({x['long_pct']:.0f}%)"),
    ("  SHORT WR",     lambda x: f"{x['short_wr']:.1f}%"),
    ("SL hit rate",    lambda x: f"{x['sl_rate']:.1f}%"),
    ("Avg hold (bar)", lambda x: f"{x['avg_hold']:.1f}"),
    ("Net PnL (5bln)", lambda x: f"${x['pnl']:+.0f}"),
    ("PnL/bulan",      lambda x: f"${x['ppm']:+.0f}"),
    ("PnL/trade",      lambda x: f"${x['ppt']:+.3f}"),
]
for label, fn in rows:
    row = f"  {label:<22} {fn(s['v3']):>{W}} {fn(s['v4']):>{W}}"
    if label == "Net PnL (5bln)":
        row += "  <--"
    print(row)

delta_pnl = s['v4']['pnl'] - s['v3']['pnl']
delta_wr  = s['v4']['wr']  - s['v3']['wr']
print(f"\n  Delta v4 vs v3: PnL {delta_pnl:+.0f}  WR {delta_wr:+.1f}pp")

out_path = MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v4" / "holdout_v3_vs_v4.json"
with open(out_path, "w") as f:
    json.dump({k: s[k] for k in s}, f, indent=2, default=float)
print(f"  Saved: {out_path}")
print(f"{'='*62}")
