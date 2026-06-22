"""Compute daily equity curves + Sharpe/Sortino/Calmar/MaxDD for all 4 variants."""
import json, sys, warnings
import numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import joblib
from core.utils import ensure_utc_index
from config import *

PROD = Path("D:/Apps-Dev/swint_tradev2/models")
LM = {"SHORT": 0, "FLAT": 1, "LONG": 2}
SL_MULT = 1.5
MAX_HOLD = 36
MODAL = 10
LEV = 5
COST = (FEE_PER_SIDE + SLIPPAGE_PER_SIDE) * 2

# ── Load all models ──────────────────────────────────────────────────────────
ic32 = joblib.load(MODEL_DIR / "runs/ic32_regime_v1/lgbm.pkl")
ic32_f = list(ic32.feature_name_)

tb = joblib.load(MODEL_DIR / "runs/tb_lgbm_widyawardhana_v3/lgbm.pkl")
with open(MODEL_DIR / "runs/tb_lgbm_widyawardhana_v3/tb_lgbm_widyawardhana_v3_features.json") as f:
    tb_f = json.load(f)

# ic32 Guardian
g_ic = joblib.load(PROD / "guardian_best.pkl")
g_ic_s = joblib.load(PROD / "guardian_scaler.pkl")
with open(PROD / "guardian_feature_cols.json") as f:
    g_ic_all = json.load(f)

# tb Guardian v2
g_tb = joblib.load(MODEL_DIR / "runs/tb_guardian_widyawardhana_v2/guardian.pkl")
g_tb_s = joblib.load(MODEL_DIR / "runs/tb_guardian_widyawardhana_v2/guardian_scaler.pkl")
with open(MODEL_DIR / "runs/tb_guardian_widyawardhana_v2/tb_guardian_widyawardhana_v2_feature_cols.json") as f:
    g_tb_all = json.load(f)

DYN_NAMES = {"bars_held_norm", "current_pnl_pct", "current_pnl_atr",
             "max_favorable_pnl_pct", "drawdown_from_peak_pct",
             "direction", "entry_price_ratio"}


def make_order(feats):
    st = [f for f in feats if f not in DYN_NAMES]
    sm = {n: i for i, n in enumerate(st)}
    order = [("static", sm[f]) if f in sm else ("dyn", f) for f in feats]
    return st, order


g_ic_st, g_ic_order = make_order(g_ic_all)
g_tb_st, g_tb_order = make_order(g_tb_all)


def build_row(j, i, close, atr, direction, max_fav, Xs, order):
    bh = j - i
    pnl = (close[j] - close[i]) / close[i] * direction
    ap = atr[i] / close[i] if close[i] > 0 else 0.01
    nm = max(max_fav, pnl)
    dv = {
        "bars_held_norm": bh / 36,
        "current_pnl_pct": pnl,
        "current_pnl_atr": pnl / ap if ap > 0 else 0.0,
        "max_favorable_pnl_pct": nm,
        "drawdown_from_peak_pct": (nm - pnl) / nm if nm > 0.001 else 0.0,
        "direction": float(direction),
        "entry_price_ratio": close[i] / close[j] if close[j] > 0 else 1.0,
    }
    row = np.zeros(len(order))
    for idx, (src, k) in enumerate(order):
        row[idx] = Xs[j, k] if src == "static" else dv.get(k, 0.0)
    return row, nm


def sim_one(yp, close, high, low, atr, Xs=None, order=None, g=None, gs=None):
    n = len(yp)
    trades = []
    i = 0
    while i < n:
        if yp[i] == 1:
            i += 1; continue
        direction = 1 if yp[i] == 2 else -1
        entry = close[i]
        sl_price = entry - direction * SL_MULT * atr[i]
        max_fav = 0.0
        exit_bar = min(i + MAX_HOLD, n - 1)
        exit_price = close[exit_bar]
        outcome = "TIME"
        for j in range(i + 1, min(i + MAX_HOLD + 1, n)):
            sl_hit = (direction == 1 and low[j] <= sl_price) or \
                     (direction == -1 and high[j] >= sl_price)
            if sl_hit:
                exit_price, exit_bar, outcome = sl_price, j, "SL"; break
            if g is not None and (j - i) >= 2:
                row, max_fav = build_row(j, i, close, atr, direction, max_fav, Xs, order)
                prob = g.predict_proba(gs.transform(row.reshape(1, -1)))[0]
                exit_prob = prob[2] if len(prob) > 2 else prob[1]
                if exit_prob >= 0.65:
                    exit_price, exit_bar, outcome = close[j], j, "GDN"; break
        ret = (exit_price - entry) / entry * direction
        net_pnl = ret * MODAL * LEV - COST * MODAL * LEV
        trades.append({"bar": exit_bar, "pnl": net_pnl})
        i = exit_bar + 1
    return trades


THR = {0: 0.45, 1: 0.50, 2: 0.50, 3: 0.45}
available = [s for s in ALL_COINS
             if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()]

all_daily = {k: [] for k in ["ic32", "ic32_g", "tb", "tb_g"]}

for sym in available:
    df = pd.read_parquet(HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet")
    df = ensure_utc_index(df).sort_index()

    rp = HOLDOUT_DIR / "labeled" / f"{sym}_regime_h1.parquet"
    hmm = np.full(len(df), 1, np.int32)
    if rp.exists():
        reg = pd.read_parquet(rp)
        if "hmm_regime_enc" in reg.columns:
            hmm = reg["hmm_regime_enc"].reindex(df.index, fill_value=1).values.astype(np.int32)

    mask = df["label"].isin(LM)
    df = df[mask].copy()
    hmm = hmm[mask.values]
    n = len(df)

    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    atr = df["atr_14_h1"].values.astype(np.float64)

    # ic32 preds
    Xi = np.zeros((n, len(ic32_f)))
    for idx, cn in enumerate(ic32_f):
        if cn in df.columns:
            Xi[:, idx] = df[cn].ffill().fillna(0).values.astype(np.float64)
        elif cn == "hmm_regime_enc":
            Xi[:, idx] = hmm.astype(np.float64)
    pi = ic32.predict_proba(Xi)
    ypi = np.ones(n, np.int32)
    ypi[pi[:, 2] >= 0.69] = 2
    ypi[(pi[:, 0] >= 0.59) & (ypi != 2)] = 0

    # ic32 Guardian static
    Xgi = np.zeros((n, len(g_ic_st)))
    for idx, cn in enumerate(g_ic_st):
        if cn in df.columns:
            Xgi[:, idx] = df[cn].ffill().fillna(0).values.astype(np.float64)
        elif cn == "hmm_regime_enc":
            Xgi[:, idx] = hmm.astype(np.float64)

    # tb preds
    Xt = np.zeros((n, len(tb_f)))
    for idx, cn in enumerate(tb_f):
        if cn in df.columns:
            Xt[:, idx] = df[cn].ffill().fillna(0).values.astype(np.float64)
    pt = tb.predict_proba(Xt)
    ct = np.max(pt, axis=1)
    ypt = np.argmax(pt, axis=1).astype(np.int32)
    for r, th in THR.items():
        ypt[(hmm == r) & (ypt != 1) & (ct < th)] = 1

    # tb Guardian static
    Xgt = np.zeros((n, len(g_tb_st)))
    for idx, cn in enumerate(g_tb_st):
        if cn in df.columns:
            Xgt[:, idx] = df[cn].ffill().fillna(0).values.astype(np.float64)

    tr_ic32 = sim_one(ypi, close, high, low, atr)
    tr_ic32g = sim_one(ypi, close, high, low, atr, Xgi, g_ic_order, g_ic, g_ic_s)
    tr_tb = sim_one(ypt, close, high, low, atr)
    tr_tbg = sim_one(ypt, close, high, low, atr, Xgt, g_tb_order, g_tb, g_tb_s)

    for t in tr_ic32:
        all_daily["ic32"].append({"date": df.index[t["bar"]].date(), "pnl": t["pnl"]})
    for t in tr_ic32g:
        all_daily["ic32_g"].append({"date": df.index[t["bar"]].date(), "pnl": t["pnl"]})
    for t in tr_tb:
        all_daily["tb"].append({"date": df.index[t["bar"]].date(), "pnl": t["pnl"]})
    for t in tr_tbg:
        all_daily["tb_g"].append({"date": df.index[t["bar"]].date(), "pnl": t["pnl"]})
    print(".", end="", flush=True)

print()


def risk_metrics(trades_list, label):
    if not trades_list:
        return {}
    df = pd.DataFrame(trades_list)
    daily = df.groupby("date")["pnl"].sum().sort_index()
    eq = daily.cumsum()
    rets = daily.values
    ann = np.sqrt(365)
    mu = np.mean(rets)
    sigma = np.std(rets)
    neg = rets[rets < 0]
    neg_std = np.std(neg) if len(neg) > 1 else sigma
    sharpe = mu / max(sigma, 1e-9) * ann
    sortino = mu / max(neg_std, 1e-9) * ann
    peak = np.maximum.accumulate(eq.values)
    dd = (eq.values - peak) / np.where(peak != 0, peak, 1)
    max_dd_pct = abs(dd.min()) * 100
    total_days = len(daily)
    ann_ret = mu * 365
    calmar = ann_ret / max(max_dd_pct / 100, 1e-9)
    return {
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "calmar": round(calmar, 2),
        "max_dd_pct": round(max_dd_pct, 1),
        "total_pnl": round(eq.values[-1], 2),
        "n_days": total_days,
    }


print(f"\n{'='*75}")
print(f"  RISK METRICS — Live-like OOS | Nov 2025–Apr 2026 | {len(available)} coins")
print(f"{'='*75}")
print(f"  {'Metrik':<22} {'ic32 bare':>12} {'ic32+Gdn':>12} {'tb bare':>12} {'tb+Gdn v2':>12}")
print(f"  {'-'*70}")

labels = ["ic32", "ic32_g", "tb", "tb_g"]
names = ["ic32 bare", "ic32+Gdn", "tb bare", "tb+Gdn v2"]
all_m = [risk_metrics(all_daily[k], k) for k in labels]

rows = ["sharpe", "sortino", "calmar", "max_dd_pct"]
row_names = {"sharpe": "Sharpe Ratio", "sortino": "Sortino Ratio",
             "calmar": "Calmar Ratio", "max_dd_pct": "Max Drawdown %"}
for rk in rows:
    vals = "".join(f" {m[rk]:>12.2f}" if rk != "max_dd_pct" else f" {m[rk]:>11.1f}%"
                   for m in all_m)
    print(f"  {row_names[rk]:<22}{vals}")

print(f"  {'-'*70}")
pnls = "".join(f" ${m['total_pnl']:>+10.0f}" for m in all_m)
print(f"  {'Total PnL':<22}{pnls}")
ndays = "".join(f" {m['n_days']:>11}" for m in all_m)
print(f"  {'Trading days':<22}{ndays}")
print(f"{'='*75}")
