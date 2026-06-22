"""
pipeline/07_tbshap_threshold_sweep.py
Threshold sweep untuk tb_fs_shap_v1 (61 feat)

Test: 0.38, 0.42, 0.45, 0.48, 0.50, 0.53, 0.55, 0.58, 0.60, 0.65
Exit: SL=1.5xATR | max_hold=36bar | Guardian=tb_guardian_widyawardhana_v2
"""
import json, sys, warnings, numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import joblib
from core.utils import ensure_utc_index
from config import (
    ALL_COINS, HOLDOUT_DIR, MODEL_DIR,
    TP_SL_FALLBACK_SL, MAX_HOLDING_BARS,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
)

THRESHOLDS = [0.38, 0.42, 0.45, 0.48, 0.50, 0.53, 0.55, 0.58, 0.60, 0.65]

SL_MULT  = TP_SL_FALLBACK_SL
MAX_HOLD = MAX_HOLDING_BARS
MODAL    = MODAL_PER_TRADE
LEVERAGE = LEVERAGE_SIM[0] if isinstance(LEVERAGE_SIM, list) else LEVERAGE_SIM
COST_RT  = (FEE_PER_SIDE + SLIPPAGE_PER_SIDE) * 2

GDN_MIN_HOLD = 2
GDN_EXIT_THR = 0.65
LONG, FLAT, SHORT = 2, 1, 0

DYNAMIC_NAMES = frozenset({
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct",
    "direction", "entry_price_ratio",
})


def compute_derived(df):
    import numpy as np
    hl   = (df["high"] - df["low"]).replace(0, np.nan)
    body = (df["close"] - df["open"]).abs()
    df   = df.copy()
    df["candle_body_ratio"]      = body / hl
    df["upper_wick_ratio"]       = (df["high"] - df[["open","close"]].max(axis=1)) / hl
    df["lower_wick_ratio"]       = (df[["open","close"]].min(axis=1) - df["low"]) / hl
    df["candle_range_atr_ratio"] = hl / df["atr_14_h1"].replace(0, np.nan)
    if "funding_rate" in df.columns:
        df["funding_rate_change_8h"]  = df["funding_rate"].diff(8)
        fr_mean = df["funding_rate"].rolling(480, min_periods=100).mean()
        fr_std  = df["funding_rate"].rolling(480, min_periods=100).std()
        df["funding_rate_zscore_20d"] = (df["funding_rate"] - fr_mean) / fr_std.replace(0, np.nan)
    if "open_interest" in df.columns:
        oi_prev = df["open_interest"].shift(24)
        df["oi_pct_1d"] = (df["open_interest"] - oi_prev) / oi_prev.abs().replace(0, np.nan)
    return df


print("Loading models ...")
shap_model = joblib.load(MODEL_DIR / "runs" / "tb_fs_shap_v1" / "lgbm.pkl")
with open(MODEL_DIR / "runs" / "tb_fs_shap_v1" / "fs_selected.json") as f:
    shap_feats = json.load(f)

tbg_model  = joblib.load(MODEL_DIR / "runs" / "tb_guardian_widyawardhana_v2" / "guardian.pkl")
tbg_scaler = joblib.load(MODEL_DIR / "runs" / "tb_guardian_widyawardhana_v2" / "guardian_scaler.pkl")
with open(MODEL_DIR / "runs" / "tb_guardian_widyawardhana_v2" / "tb_guardian_widyawardhana_v2_feature_cols.json") as f:
    tbg_all_feats = json.load(f)

tbg_static = [f for f in tbg_all_feats if f not in DYNAMIC_NAMES]
tbg_smap   = {name: i for i, name in enumerate(tbg_static)}
tbg_order  = [("static", tbg_smap[f]) if f in tbg_smap else ("dyn", f) for f in tbg_all_feats]

print(f"  SHAP model: {len(shap_feats)} features")
print(f"  Guardian  : {len(tbg_all_feats)} features")


def build_X(df, feat_list):
    n = len(df)
    X = np.zeros((n, len(feat_list)), dtype=np.float64)
    for idx, c in enumerate(feat_list):
        if c in df.columns:
            X[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)
    return X


def build_gdn_row(j, i, close, atr, direction, max_fav, X_st, feat_order):
    bars_held = j - i
    pnl_pct   = (close[j] - close[i]) / close[i] * direction
    atr_pct   = atr[i] / close[i] if close[i] > 0 else 0.01
    new_max   = max(max_fav, pnl_pct)
    dyn = {
        "bars_held_norm"        : bars_held / MAX_HOLD,
        "current_pnl_pct"       : pnl_pct,
        "current_pnl_atr"       : pnl_pct / atr_pct if atr_pct > 0 else 0.0,
        "max_favorable_pnl_pct" : new_max,
        "drawdown_from_peak_pct": (new_max - pnl_pct) / new_max if new_max > 0.001 else 0.0,
        "direction"             : float(direction),
        "entry_price_ratio"     : close[i] / close[j] if close[j] > 0 else 1.0,
    }
    row = np.zeros(len(feat_order), dtype=np.float64)
    for idx, (src, key) in enumerate(feat_order):
        row[idx] = X_st[j, key] if src == "static" else dyn.get(key, 0.0)
    return row, new_max


def simulate(yp, close, high, low, atr, X_st=None, use_gdn=False):
    n = len(yp); trades = []; i = 0
    while i < n:
        sig = yp[i]
        if sig == FLAT:
            i += 1; continue
        direction  = 1 if sig == LONG else -1
        entry      = close[i]
        sl_price   = entry - direction * SL_MULT * atr[i]
        max_fav    = 0.0
        exit_price = close[min(i + MAX_HOLD, n - 1)]
        exit_bar   = min(i + MAX_HOLD, n - 1)
        for j in range(i + 1, min(i + MAX_HOLD + 1, n)):
            sl_hit = (direction == 1 and low[j]  <= sl_price) or \
                     (direction == -1 and high[j] >= sl_price)
            if sl_hit:
                exit_price, exit_bar = sl_price, j; break
            if use_gdn and X_st is not None and (j - i) >= GDN_MIN_HOLD:
                row, max_fav = build_gdn_row(j, i, close, atr, direction, max_fav, X_st, tbg_order)
                prob = tbg_model.predict_proba(tbg_scaler.transform(row.reshape(1, -1)))[0]
                ep   = prob[2] if len(prob) > 2 else prob[1]
                if ep >= GDN_EXIT_THR:
                    exit_price, exit_bar = close[j], j; break
            else:
                max_fav = max(max_fav, (close[j] - entry) / entry * direction)
        net_pnl = (exit_price - entry) / entry * direction * MODAL * LEVERAGE \
                  - COST_RT * MODAL * LEVERAGE
        trades.append({"pnl": net_pnl, "win": net_pnl > 0})
        i = exit_bar + 1
    return trades


LM  = {0, 1, 2}
available = [s for s in ALL_COINS
             if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()]

# Pre-load all coin data
print(f"Loading {len(available)} coin holdout parquets ...")
coin_data = {}
for sym in available:
    df = __import__("pandas").read_parquet(HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet")
    df = ensure_utc_index(df).sort_index()
    df = compute_derived(df)
    mask = df["label"].notna()
    df   = df[mask].copy()
    X_shap = build_X(df, shap_feats)
    X_tbg  = build_X(df, tbg_static)
    p_shap = shap_model.predict_proba(X_shap)
    coin_data[sym] = {
        "p_shap": p_shap,
        "close" : df["close"].values.astype(np.float64),
        "high"  : df["high"].values.astype(np.float64),
        "low"   : df["low"].values.astype(np.float64),
        "atr"   : df["atr_14_h1"].values.astype(np.float64),
        "X_tbg" : X_tbg,
    }

print("Running threshold sweep ...\n")

results = {}
for thr in THRESHOLDS:
    agg_bare = {"t": 0, "w": 0, "pnl": 0.0}
    agg_gdn  = {"t": 0, "w": 0, "pnl": 0.0}

    for sym, d in coin_data.items():
        p = d["p_shap"]
        n = len(p)
        yp = np.ones(n, dtype=np.int32)
        yp[p[:, 2] >= thr]                        = LONG
        yp[(p[:, 0] >= thr) & (yp != LONG)]       = SHORT

        for agg, use_gdn in [(agg_bare, False), (agg_gdn, True)]:
            trades = simulate(yp, d["close"], d["high"], d["low"], d["atr"],
                              X_st=d["X_tbg"], use_gdn=use_gdn)
            for t in trades:
                agg["t"] += 1; agg["pnl"] += t["pnl"]
                if t["win"]: agg["w"] += 1

    results[str(thr)] = {
        "bare": {
            "thr": thr,
            "trades"  : agg_bare["t"],
            "wr"      : round(agg_bare["w"] / max(agg_bare["t"], 1) * 100, 1),
            "pnl"     : round(agg_bare["pnl"], 1),
            "ppt"     : round(agg_bare["pnl"] / max(agg_bare["t"], 1), 3),
        },
        "gdn": {
            "thr": thr,
            "trades"  : agg_gdn["t"],
            "wr"      : round(agg_gdn["w"] / max(agg_gdn["t"], 1) * 100, 1),
            "pnl"     : round(agg_gdn["pnl"], 1),
            "ppt"     : round(agg_gdn["pnl"] / max(agg_gdn["t"], 1), 3),
        },
    }


# ── Print ─────────────────────────────────────────────────────────────────────
W = 9
print(f"{'='*90}")
print(f"  TB-SHAP (61 feat) Threshold Sweep | Apr 2026-Jun 2026 (~2.5 bln) | 21 koin")
print(f"{'='*90}")
print(f"\n  --- BARE (no Guardian) ---")
hdr = f"  {'Thr':>6}" + f"{'Trades':>{W}}" + f"{'WR%':>{W}}" + f"{'PnL $':>{W+2}}" + f"{'$/trade':>{W+1}}"
print(hdr)
print(f"  {'-'*6}" + "-" * (W * 3 + 13))
for thr in THRESHOLDS:
    r = results[str(thr)]["bare"]
    print(f"  {thr:>6.2f}"
          f"  {r['trades']:>{W-2},}"
          f"  {r['wr']:>{W-2}.1f}%"
          f"  {r['pnl']:>+{W+1}.1f}"
          f"  {r['ppt']:>+{W}.3f}")

print(f"\n  --- WITH GUARDIAN (tb_guardian_widyawardhana_v2) ---")
print(hdr)
print(f"  {'-'*6}" + "-" * (W * 3 + 13))
for thr in THRESHOLDS:
    r = results[str(thr)]["gdn"]
    print(f"  {thr:>6.2f}"
          f"  {r['trades']:>{W-2},}"
          f"  {r['wr']:>{W-2}.1f}%"
          f"  {r['pnl']:>+{W+1}.1f}"
          f"  {r['ppt']:>+{W}.3f}")

best_bare = max(THRESHOLDS, key=lambda t: results[str(t)]["bare"]["pnl"])
best_gdn  = max(THRESHOLDS, key=lambda t: results[str(t)]["gdn"]["pnl"])
print(f"\n  Best bare PnL : thr={best_bare}  -> ${results[str(best_bare)]['bare']['pnl']:+.1f}")
print(f"  Best gdn  PnL : thr={best_gdn}   -> ${results[str(best_gdn)]['gdn']['pnl']:+.1f}")
print(f"{'='*90}")

out_path = MODEL_DIR / "runs" / "tb_fs_shap_v1" / "threshold_sweep.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\n  Saved -> {out_path}")
