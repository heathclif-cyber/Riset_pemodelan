"""
pipeline/07_holdout_livelike_meta_guardian.py
4-way holdout comparison: tb bare vs tb+Guardian vs tb+meta vs tb+meta+Guardian.

Flow tb+meta+Guardian:
  1. LGBM tb_v3 predicts arah + probabilities
  2. Meta-model filter: skip jika p_win < META_THR
  3. Entry → Guardian monitor per bar → exit early jika exit_prob >= GDN_EXIT_THR

Exit: SL=1.5xATR hard stop | max_hold=36 bar | NO TP | Guardian=early exit
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

logger = setup_logger("07_meta_gdn")

PROD = Path("D:/Apps-Dev/swint_tradev2/models")

SL_MULT  = TP_SL_FALLBACK_SL
MAX_HOLD = MAX_HOLDING_BARS
MODAL    = MODAL_PER_TRADE
LEVERAGE = LEVERAGE_SIM[0] if isinstance(LEVERAGE_SIM, list) else LEVERAGE_SIM
COST_RT  = (FEE_PER_SIDE + SLIPPAGE_PER_SIDE) * 2

REGIME_THRESH = {0: 0.45, 1: 0.50, 2: 0.50, 3: 0.45}
GDN_MIN_HOLD  = 2
GDN_EXIT_THR  = 0.65
META_THR      = 0.45   # sweet spot dari threshold sweep sebelumnya

# ── Load models ────────────────────────────────────────────────────────────────
tb_model = joblib.load(MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v3" / "lgbm.pkl")
with open(MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v3" /
          "tb_lgbm_widyawardhana_v3_features.json") as f:
    tb_feats = json.load(f)

meta_model = joblib.load(MODEL_DIR / "runs" / "tb_meta_v1" / "meta_lgbm.pkl")
with open(MODEL_DIR / "runs" / "tb_meta_v1" / "tb_meta_v1_features.json") as f:
    meta_feats = json.load(f)

tbg_model  = joblib.load(MODEL_DIR / "runs" / "tb_guardian_widyawardhana_v2" / "guardian.pkl")
tbg_scaler = joblib.load(MODEL_DIR / "runs" / "tb_guardian_widyawardhana_v2" / "guardian_scaler.pkl")
with open(MODEL_DIR / "runs" / "tb_guardian_widyawardhana_v2" /
          "tb_guardian_widyawardhana_v2_feature_cols.json") as f:
    tbg_all_feats = json.load(f)

DYNAMIC_NAMES = frozenset({
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct",
    "direction", "entry_price_ratio",
})

tbg_static = [f for f in tbg_all_feats if f not in DYNAMIC_NAMES]
tbg_static_map = {name: idx for idx, name in enumerate(tbg_static)}
tbg_order = [("static", tbg_static_map[f]) if f not in DYNAMIC_NAMES
             else ("dyn", f) for f in tbg_all_feats]

available = [s for s in ALL_COINS
             if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()]

print(f"\n{'='*72}")
print(f"  HOLDOUT OOS — Meta + Guardian | Nov 2025 – Apr 2026")
print(f"  Variants: tb bare | tb+Guardian | tb+meta({META_THR}) | tb+meta+Guardian")
print(f"  Exit: SL={SL_MULT}xATR | max_hold={MAX_HOLD}bar | NO TP | Guardian early exit")
print(f"  Coins: {len(available)} | ${MODAL}/trade {LEVERAGE}x")
print(f"{'='*72}\n")


def build_gdn_row(j, i, close, atr, direction, max_fav, X_static, feat_order):
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
        row[idx] = X_static[j, key] if src == "static" else dyn.get(key, 0.0)
    return row, new_max


def simulate(yp, close, high, low, atr, X_tbg=None, use_guardian=False):
    n = len(yp); trades = []; i = 0
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
        outcome    = "TIME"

        for j in range(i + 1, min(i + MAX_HOLD + 1, n)):
            sl_hit = ((direction == 1 and low[j] <= sl_price) or
                      (direction == -1 and high[j] >= sl_price))
            if sl_hit:
                exit_price, exit_bar, outcome = sl_price, j, "SL"; break

            bars_held = j - i
            if use_guardian and X_tbg is not None and bars_held >= GDN_MIN_HOLD:
                row, max_fav = build_gdn_row(j, i, close, atr, direction,
                                             max_fav, X_tbg, tbg_order)
                scaled    = tbg_scaler.transform(row.reshape(1, -1))
                prob      = tbg_model.predict_proba(scaled)[0]
                exit_prob = prob[2] if len(prob) > 2 else prob[1]
                if exit_prob >= GDN_EXIT_THR:
                    exit_price, exit_bar, outcome = close[j], j, "GDN"; break
            else:
                pnl_pct = (close[j] - entry) / entry * direction
                max_fav = max(max_fav, pnl_pct)

        ret     = (exit_price - entry) / entry * direction
        net_pnl = ret * MODAL * LEVERAGE - COST_RT * MODAL * LEVERAGE
        trades.append({
            "direction": "LONG" if direction == 1 else "SHORT",
            "outcome": outcome, "net_pnl": net_pnl,
            "bars_held": exit_bar - i,
        })
        i = exit_bar + 1
    return trades


def new_agg():
    return {"trades": 0, "wins": 0, "pnl": 0.0,
            "longs": 0, "long_wins": 0, "sl_hits": 0,
            "gdn_exits": 0, "bars_held": []}


def update_agg(a, trades):
    for t in trades:
        a["trades"] += 1; a["pnl"] += t["net_pnl"]
        a["bars_held"].append(t["bars_held"])
        if t["net_pnl"] > 0: a["wins"] += 1
        if t["outcome"] == "SL": a["sl_hits"] += 1
        if t["outcome"] == "GDN": a["gdn_exits"] += 1
        if t["direction"] == "LONG":
            a["longs"] += 1
            if t["net_pnl"] > 0: a["long_wins"] += 1


agg = {k: new_agg() for k in ["tb", "tb_gdn", "tb_meta", "tb_meta_gdn"]}

for sym in available:
    df = pd.read_parquet(HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet")
    df = ensure_utc_index(df).sort_index()

    rp  = HOLDOUT_DIR / "labeled" / f"{sym}_regime_h1.parquet"
    hmm = np.full(len(df), 1, np.int32)
    if rp.exists():
        reg = pd.read_parquet(rp)
        if "hmm_regime_enc" in reg.columns:
            hmm = reg["hmm_regime_enc"].reindex(df.index, fill_value=1).values.astype(np.int32)

    mask = df["label"].isin({"SHORT": 0, "FLAT": 1, "LONG": 2})
    df   = df[mask].copy()
    hmm  = hmm[mask.values]
    n    = len(df)

    close = df["close"].values.astype(np.float64)
    high  = df["high"].values.astype(np.float64)
    low   = df["low"].values.astype(np.float64)
    atr   = df["atr_14_h1"].values.astype(np.float64)

    # ── LGBM tb_v3 predictions ─────────────────────────────────────────────
    X_tb = np.zeros((n, len(tb_feats)), dtype=np.float64)
    for idx, c in enumerate(tb_feats):
        if c in df.columns:
            X_tb[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)

    p_tb    = tb_model.predict_proba(X_tb).astype(np.float32)
    conf_tb = np.max(p_tb, axis=1)
    yp_tb   = np.argmax(p_tb, axis=1).astype(np.int32)
    for r, th in REGIME_THRESH.items():
        yp_tb[(hmm == r) & (yp_tb != 1) & (conf_tb < th)] = 1

    # ── Meta-model predictions ─────────────────────────────────────────────
    meta_cols = []
    for feat in meta_feats:
        if feat == "p_short":
            meta_cols.append(p_tb[:, 0])
        elif feat == "p_flat":
            meta_cols.append(p_tb[:, 1])
        elif feat == "p_long":
            meta_cols.append(p_tb[:, 2])
        elif feat == "confidence":
            meta_cols.append(conf_tb)
        elif feat == "direction":
            meta_cols.append((yp_tb == 2).astype(np.float32))
        elif feat in df.columns:
            meta_cols.append(df[feat].ffill().fillna(0).values.astype(np.float32))
        else:
            meta_cols.append(np.zeros(n, dtype=np.float32))

    X_meta = np.stack(meta_cols, axis=1)
    p_win  = meta_model.predict_proba(X_meta)[:, 1]

    # Apply meta filter: skip directional signals where p_win < META_THR
    yp_meta = yp_tb.copy()
    yp_meta[(yp_meta != 1) & (p_win < META_THR)] = 1

    # ── Guardian static features ───────────────────────────────────────────
    X_tbg = np.zeros((n, len(tbg_static)), dtype=np.float64)
    for idx, c in enumerate(tbg_static):
        if c in df.columns:
            X_tbg[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)

    # ── Simulate 4 variants ────────────────────────────────────────────────
    update_agg(agg["tb"],         simulate(yp_tb,   close, high, low, atr))
    update_agg(agg["tb_gdn"],     simulate(yp_tb,   close, high, low, atr, X_tbg, use_guardian=True))
    update_agg(agg["tb_meta"],    simulate(yp_meta, close, high, low, atr))
    update_agg(agg["tb_meta_gdn"],simulate(yp_meta, close, high, low, atr, X_tbg, use_guardian=True))

    logger.info(f"[{sym}] tb={agg['tb']['trades']} tb+gdn={agg['tb_gdn']['trades']} "
                f"tb+meta={agg['tb_meta']['trades']} tb+meta+gdn={agg['tb_meta_gdn']['trades']}")


# ── Scorecard ──────────────────────────────────────────────────────────────────
def sc(a):
    t = a["trades"]; w = a["wins"]; p = a["pnl"]
    l = a["longs"]; lw = a["long_wins"]
    sl = a["sl_hits"]; gd = a["gdn_exits"]; bh = a["bars_held"]
    short_wins = w - lw
    shorts = t - l
    return dict(
        trades=t, wr=w / max(t, 1) * 100,
        long_wr=lw / max(l, 1) * 100, short_wr=short_wins / max(shorts, 1) * 100,
        long_pct=l / max(t, 1) * 100,
        sl_rate=sl / max(t, 1) * 100, gdn_rate=gd / max(t, 1) * 100,
        avg_hold=float(np.mean(bh)) if bh else 0,
        pnl=p, ppm=p / 5, ppt=p / max(t, 1),
    )

s = {k: sc(agg[k]) for k in agg}

LABELS = {
    "tb":          f"tb bare",
    "tb_gdn":      f"tb+Guardian",
    "tb_meta":     f"tb+meta({META_THR})",
    "tb_meta_gdn": f"tb+meta+Guardian",
}
W = 16

print(f"\n{'='*76}")
print(f"  SCORECARD — 4-way Comparison | Nov 2025–Apr 2026 | 21 koin | $10/trade 5x")
print(f"{'='*76}")
hdr = f"  {'Metrik':<22}"
for k in agg:
    hdr += f" {LABELS[k]:>{W}}"
print(hdr)
print(f"  {'-'*73}")

rows = [
    ("Total Trades",    lambda x: f"{x['trades']:,}"),
    ("Trades/bulan",    lambda x: f"{x['trades']/5:.0f}"),
    ("Win Rate",        lambda x: f"{x['wr']:.1f}%"),
    ("  LONG WR",       lambda x: f"{x['long_wr']:.1f}% ({x['long_pct']:.0f}%)"),
    ("  SHORT WR",      lambda x: f"{x['short_wr']:.1f}%"),
    ("SL hit rate",     lambda x: f"{x['sl_rate']:.1f}%"),
    ("Guardian exits",  lambda x: f"{x['gdn_rate']:.1f}%" if x['gdn_rate'] > 0 else "—"),
    ("Avg hold (bar)",  lambda x: f"{x['avg_hold']:.1f}"),
    ("Net PnL (5bln)",  lambda x: f"${x['pnl']:+.0f}"),
    ("PnL/bulan",       lambda x: f"${x['ppm']:+.0f}"),
    ("PnL/trade",       lambda x: f"${x['ppt']:+.3f}"),
]

for label, fn in rows:
    row = f"  {label:<22}"
    for k in agg:
        row += f" {fn(s[k]):>{W}}"
    if label == "Net PnL (5bln)":
        row += "  <--"
    print(row)

best_k = max(s, key=lambda k: s[k]["pnl"])
best_ppt_k = max(s, key=lambda k: s[k]["ppt"])
print(f"\n  Best PnL     : {LABELS[best_k]}  (${s[best_k]['pnl']:+.0f})")
print(f"  Best PnL/trade: {LABELS[best_ppt_k]}  (${s[best_ppt_k]['ppt']:+.3f})")

# Save
out_path = MODEL_DIR / "runs" / "tb_meta_v1" / "holdout_meta_guardian_results.json"
with open(out_path, "w") as f:
    json.dump({k: {**s[k], "label": LABELS[k]} for k in s}, f, indent=2, default=float)
print(f"\n  Saved: {out_path}")
print(f"{'='*76}")
