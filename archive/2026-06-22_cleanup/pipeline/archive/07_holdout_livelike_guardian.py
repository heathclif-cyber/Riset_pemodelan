"""
pipeline/07_holdout_livelike_guardian.py
Evaluasi OOS live-like — 4 variants sekaligus:
  1. ic32_regime_v1         (bare)
  2. ic32_regime_v1         + Guardian clean_v2 (production)
  3. tb_widyawardhana_v3    (bare)
  Model 4 : tb_widyawardhana_v3    + Guardian v2 (Triple Barrier pool + FS)

Exit method identik semua variant:
  - SL = 1.5xATR hard stop
  - max_hold = 36 bar
  - NO TP
  Guardian boleh exit lebih awal dari SL/time-exit.
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

logger = setup_logger("07_livelike_gdn")

PROD = Path("D:/Apps-Dev/swint_tradev2/models")
LM   = {"SHORT": 0, "FLAT": 1, "LONG": 2}

SL_MULT  = TP_SL_FALLBACK_SL
MAX_HOLD = MAX_HOLDING_BARS
MODAL    = MODAL_PER_TRADE
LEVERAGE = LEVERAGE_SIM[0] if isinstance(LEVERAGE_SIM, list) else LEVERAGE_SIM
COST_RT  = (FEE_PER_SIDE + SLIPPAGE_PER_SIDE) * 2

IC32_THR_LONG  = LGBM_THRESHOLD_LONG
IC32_THR_SHORT = LGBM_THRESHOLD_SHORT
REGIME_THRESH  = {0: 0.45, 1: 0.50, 2: 0.50, 3: 0.45}

GDN_MIN_HOLD  = 2
GDN_EXIT_THR  = 0.65    # Guardian exit threshold

# ── Load models ────────────────────────────────────────────────────────────────
ic32_model = joblib.load(MODEL_DIR / "runs" / "ic32_regime_v1" / "lgbm.pkl")
ic32_feats = list(ic32_model.feature_name_)

tb_model = joblib.load(MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v3" / "lgbm.pkl")
with open(MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v3" / "tb_lgbm_widyawardhana_v3_features.json") as f:
    tb_feats = json.load(f)

# Guardian clean_v2 (ic32)
gdn_model  = joblib.load(PROD / "guardian_best.pkl")
gdn_scaler = joblib.load(PROD / "guardian_scaler.pkl")
with open(PROD / "guardian_feature_cols.json") as f:
    gdn_all_feats = json.load(f)
DYNAMIC_NAMES = frozenset({"bars_held_norm", "current_pnl_pct", "current_pnl_atr",
                              "max_favorable_pnl_pct", "drawdown_from_peak_pct",
                              "direction", "entry_price_ratio"})

def make_guardian_config(all_feat_list):
    """
    Given a Guardian feature list, return:
      static_cols : list of feature names from df
      feat_order  : list of (source, index_or_name) tuples
                     source='static' → index in static_cols
                     source='dyn'    → name in DYNAMIC_NAMES
    """
    static = [f for f in all_feat_list if f not in DYNAMIC_NAMES]
    static_map = {name: idx for idx, name in enumerate(static)}
    order = []
    for f in all_feat_list:
        if f in static_map:
            order.append(("static", static_map[f]))
        else:
            order.append(("dyn", f))
    return static, order

gdn_static, gdn_order = make_guardian_config(gdn_all_feats)

# Guardian widyawardhana_v2 (tb) — clean_v2 style + Triple Barrier pool + FS
tbg_model  = joblib.load(MODEL_DIR / "runs" / "tb_guardian_widyawardhana_v2" / "guardian.pkl")
tbg_scaler = joblib.load(MODEL_DIR / "runs" / "tb_guardian_widyawardhana_v2" / "guardian_scaler.pkl")
with open(MODEL_DIR / "runs" / "tb_guardian_widyawardhana_v2" / "tb_guardian_widyawardhana_v2_feature_cols.json") as f:
    tbg_all_feats = json.load(f)
tbg_static, tbg_order = make_guardian_config(tbg_all_feats)

available = [s for s in ALL_COINS
             if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()]

print(f"\n{'='*70}")
print(f"  HOLDOUT OOS — Live-like + Guardian | Nov 2025 – Apr 2026")
print(f"  Exit: SL={SL_MULT}xATR | max_hold={MAX_HOLD}bar | NO TP | Guardian=early exit")
print(f"  Coins: {len(available)} | ${MODAL}/trade {LEVERAGE}x")
print(f"{'='*70}\n")


def build_guardian_row(j, i, close, atr, direction, max_fav, X_static, feat_order):
    """
    Build complete Guardian feature vector matching trained column order.
    feat_order: list of ('static', idx) | ('dyn', name) tuples.
    """
    bars_held = j - i
    pnl_pct   = (close[j] - close[i]) / close[i] * direction
    atr_pct   = atr[i] / close[i] if close[i] > 0 else 0.01
    new_max   = max(max_fav, pnl_pct)

    dyn_vals = {
        "bars_held_norm"       : bars_held / MAX_HOLD,
        "current_pnl_pct"      : pnl_pct,
        "current_pnl_atr"      : pnl_pct / atr_pct if atr_pct > 0 else 0.0,
        "max_favorable_pnl_pct": new_max,
        "drawdown_from_peak_pct": (new_max - pnl_pct) / new_max if new_max > 0.001 else 0.0,
        "direction"            : float(direction),
        "entry_price_ratio"    : close[i] / close[j] if close[j] > 0 else 1.0,
    }

    row = np.zeros(len(feat_order), dtype=np.float64)
    for idx, (src, key) in enumerate(feat_order):
        if src == "static":
            row[idx] = X_static[j, key]
        else:
            row[idx] = dyn_vals.get(key, 0.0)
    return row, new_max


def simulate(yp, close, high, low, atr,
             guardian=None, feat_order=None,
             X_static=None, gdn_scaler=None):
    """
    Live-like simulation dengan optional Guardian.
    Builds feature vector per bar sesuai trained Guardian column order.
    """
    n      = len(yp)
    trades = []
    i      = 0
    while i < n:
        sig = yp[i]
        if sig == 1:
            i += 1; continue

        direction  = 1 if sig == 2 else -1
        entry      = close[i]
        sl_price   = entry - direction * SL_MULT * atr[i]
        max_fav    = 0.0
        exit_price = close[min(i + MAX_HOLD, n - 1)]
        exit_bar   = min(i + MAX_HOLD, n - 1)
        outcome    = "TIME_EXIT"

        for j in range(i + 1, min(i + MAX_HOLD + 1, n)):
            sl_hit = (direction == 1 and low[j] <= sl_price) or \
                     (direction == -1 and high[j] >= sl_price)
            if sl_hit:
                exit_price, exit_bar, outcome = sl_price, j, "SL"
                break

            bars_held = j - i

            if guardian is not None and bars_held >= GDN_MIN_HOLD:
                row, max_fav = build_guardian_row(
                    j, i, close, atr, direction, max_fav, X_static, feat_order)
                scaled = gdn_scaler.transform(row.reshape(1, -1))
                prob   = guardian.predict_proba(scaled)[0]
                # classes [0,1,2]: 0=HOLD, 1=PARTIAL, 2=FULL_EXIT
                exit_prob = prob[2] if len(prob) > 2 else prob[1]
                if exit_prob >= GDN_EXIT_THR:
                    exit_price, exit_bar, outcome = close[j], j, "GUARDIAN_EXIT"
                    break
            else:
                pnl_pct = (close[j] - entry) / entry * direction
                max_fav = max(max_fav, pnl_pct)

        ret     = (exit_price - entry) / entry * direction
        net_pnl = ret * MODAL * LEVERAGE - COST_RT * MODAL * LEVERAGE

        trades.append({
            "direction": "LONG" if direction == 1 else "SHORT",
            "outcome"  : outcome,
            "net_pnl"  : net_pnl,
            "bars_held": exit_bar - i,
        })
        i = exit_bar + 1

    return trades


def new_agg():
    return {"trades": 0, "wins": 0, "pnl": 0.0,
            "longs": 0, "long_wins": 0, "sl_hits": 0,
            "gdn_exits": 0, "bars_held": []}

agg = {k: new_agg() for k in ["ic32", "ic32_gdn", "tb", "tb_gdn"]}

for sym in available:
    df = pd.read_parquet(HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet")
    df = ensure_utc_index(df).sort_index()

    rp  = HOLDOUT_DIR / "labeled" / f"{sym}_regime_h1.parquet"
    hmm = np.full(len(df), 1, np.int32)
    if rp.exists():
        reg = pd.read_parquet(rp)
        if "hmm_regime_enc" in reg.columns:
            hmm = reg["hmm_regime_enc"].reindex(df.index, fill_value=1).values.astype(np.int32)

    mask = df["label"].isin(LM)
    df   = df[mask].copy()
    hmm  = hmm[mask.values]
    n    = len(df)

    close = df["close"].values.astype(np.float64)
    high  = df["high"].values.astype(np.float64)
    low   = df["low"].values.astype(np.float64)
    atr   = df["atr_14_h1"].values.astype(np.float64)

    # ── ic32 predictions ───────────────────────────────────────────────────
    X_ic = np.zeros((n, len(ic32_feats)), dtype=np.float64)
    for idx, c in enumerate(ic32_feats):
        if c in df.columns:
            X_ic[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)
        elif c == "hmm_regime_enc":
            X_ic[:, idx] = hmm.astype(np.float64)
    p_ic  = ic32_model.predict_proba(X_ic)
    yp_ic = np.ones(n, dtype=np.int32)
    yp_ic[p_ic[:, 2] >= IC32_THR_LONG] = 2
    yp_ic[(p_ic[:, 0] >= IC32_THR_SHORT) & (yp_ic != 2)] = 0

    # ── Guardian clean_v2 static features ─────────────────────────────────
    X_gdn = np.zeros((n, len(gdn_static)), dtype=np.float64)
    for idx, c in enumerate(gdn_static):
        if c in df.columns:
            X_gdn[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)
        elif c == "hmm_regime_enc":
            X_gdn[:, idx] = hmm.astype(np.float64)

    # ── tb predictions ─────────────────────────────────────────────────────
    X_tb = np.zeros((n, len(tb_feats)), dtype=np.float64)
    for idx, c in enumerate(tb_feats):
        if c in df.columns:
            X_tb[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)
    p_tb   = tb_model.predict_proba(X_tb)
    conf_tb = np.max(p_tb, axis=1)
    yp_tb  = np.argmax(p_tb, axis=1).astype(np.int32)
    for r, th in REGIME_THRESH.items():
        yp_tb[(hmm == r) & (yp_tb != 1) & (conf_tb < th)] = 1

    # tb Guardian static features (from Guardian feature list, not entry model)
    X_tbg = np.zeros((n, len(tbg_static)), dtype=np.float64)
    for idx, c in enumerate(tbg_static):
        if c in df.columns:
            X_tbg[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)

    # ── Simulate 4 variants ────────────────────────────────────────────────
    variants = [
        ("ic32",     yp_ic, None,       None,      None,       None),
        ("ic32_gdn", yp_ic, gdn_model,  gdn_order, X_gdn,      gdn_scaler),
        ("tb",       yp_tb, None,       None,      None,       None),
        ("tb_gdn",   yp_tb, tbg_model,  tbg_order, X_tbg,      tbg_scaler),
    ]

    for key, yp, gdn, f_order, X_st, scaler in variants:
        trades = simulate(yp, close, high, low, atr,
                          guardian=gdn, feat_order=f_order,
                          X_static=X_st, gdn_scaler=scaler)
        a = agg[key]
        for t in trades:
            a["trades"]    += 1
            a["pnl"]       += t["net_pnl"]
            a["bars_held"].append(t["bars_held"])
            if t["net_pnl"] > 0:
                a["wins"] += 1
            if t["outcome"] == "SL":
                a["sl_hits"] += 1
            if "GUARDIAN" in t["outcome"]:
                a["gdn_exits"] += 1
            if t["direction"] == "LONG":
                a["longs"] += 1
                if t["net_pnl"] > 0:
                    a["long_wins"] += 1

    logger.info(f"[{sym}] ic32={agg['ic32']['trades']} ic32+gdn={agg['ic32_gdn']['trades']} "
                f"tb={agg['tb']['trades']} tb+gdn={agg['tb_gdn']['trades']}")


# ── Scorecard ──────────────────────────────────────────────────────────────────
def sc(a):
    t = a["trades"]; w = a["wins"]; p = a["pnl"]
    l = a["longs"]; lw = a["long_wins"]
    sl = a["sl_hits"]; gd = a["gdn_exits"]
    bh = a["bars_held"]
    return dict(
        trades=t,
        wr=w / max(t, 1) * 100,
        long_wr=lw / max(l, 1) * 100,
        short_wr=(w - lw) / max(t - l, 1) * 100,
        long_pct=l / max(t, 1) * 100,
        sl_rate=sl / max(t, 1) * 100,
        gdn_rate=gd / max(t, 1) * 100,
        avg_hold=np.mean(bh) if bh else 0,
        pnl=p, ppm=p / 5, ppt=p / max(t, 1),
    )

s = {k: sc(agg[k]) for k in agg}

W = 16
LABELS = {
    "ic32":     "ic32 bare",
    "ic32_gdn": "ic32+Guardian",
    "tb":       "tb_wdyw bare",
    "tb_gdn":   "tb_wdyw+Guardian",
}

print(f"\n{'='*75}")
print(f"  SCORECARD — Live-like OOS | Nov 2025–Apr 2026 | 21 koin | $10/trade 5x")
print(f"  Exit: SL={SL_MULT}xATR + max_hold={MAX_HOLD}bar, no TP | Guardian=early exit")
print(f"{'='*75}")
hdr = f"  {'Metrik':<24}"
for k in agg:
    hdr += f" {LABELS[k]:>{W}}"
print(hdr)
print(f"  {'-'*71}")

rows = [
    ("Total Trades",   lambda x: f"{x['trades']:,}"),
    ("Trades/bulan",   lambda x: f"{x['trades']/5:.0f}"),
    ("Win Rate",       lambda x: f"{x['wr']:.1f}%"),
    ("  LONG WR",      lambda x: f"{x['long_wr']:.1f}% ({x['long_pct']:.0f}%)"),
    ("  SHORT WR",     lambda x: f"{x['short_wr']:.1f}%"),
    ("SL hit rate",    lambda x: f"{x['sl_rate']:.1f}%"),
    ("Guardian exits", lambda x: f"{x['gdn_rate']:.1f}%" if x['gdn_rate'] > 0 else "—"),
    ("Avg hold (bar)", lambda x: f"{x['avg_hold']:.1f}"),
    ("Net PnL (5bln)", lambda x: f"${x['pnl']:+.0f}"),
    ("PnL/bulan",      lambda x: f"${x['ppm']:+.0f}"),
    ("PnL/trade",      lambda x: f"${x['ppt']:+.3f}"),
]

for label, fn in rows:
    row = f"  {label:<24}"
    for k in agg:
        row += f" {fn(s[k]):>{W}}"
    if label == "Net PnL (5bln)":
        row += "  <--"
    print(row)

best_k = max(s, key=lambda k: s[k]["pnl"])
print(f"\n  Best: {LABELS[best_k]}  (PnL ${s[best_k]['pnl']:+.0f})")

# Save
out = {k: {**s[k], "label": LABELS[k]} for k in s}
out_path = MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v3" / "holdout_livelike_guardian.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2, default=float)
print(f"\n  Saved -> {out_path}")
print(f"{'='*75}")
