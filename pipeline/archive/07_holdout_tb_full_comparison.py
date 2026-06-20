"""
pipeline/07_holdout_tb_full_comparison.py
Head-to-head comparison — 6 variants | Nov 2025-Apr 2026

  1. ic32 bare
  2. ic32 + Guardian (production clean_v2)
  3. TB bare          (tb_lgbm_widyawardhana_v3)
  4. TB + LSTM-C      (FLIP hard veto — argmax opposite blocks entry)
  5. TB + Guardian v2
  6. TB + LSTM-C + Guardian v2

Exit: SL=1.5xATR | max_hold=36bar | NO TP | Guardian=early exit
"""
import json, sys, warnings
import numpy as np
import pandas as pd
import torch
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import joblib
from core.models import load_lstm
from core.utils import setup_logger, ensure_utc_index
from config import *

logger = setup_logger("07_tb_full_cmp")

PROD = Path("D:/Apps-Dev/swint_tradev2/models")
LM   = {"SHORT": 0, "FLAT": 1, "LONG": 2}
SHORT, FLAT, LONG = 0, 1, 2

SL_MULT  = TP_SL_FALLBACK_SL
MAX_HOLD = MAX_HOLDING_BARS
MODAL    = MODAL_PER_TRADE
LEVERAGE = LEVERAGE_SIM[0] if isinstance(LEVERAGE_SIM, list) else LEVERAGE_SIM
COST_RT  = (FEE_PER_SIDE + SLIPPAGE_PER_SIDE) * 2

IC32_THR_LONG  = LGBM_THRESHOLD_LONG
IC32_THR_SHORT = LGBM_THRESHOLD_SHORT
REGIME_THRESH  = {0: 0.45, 1: 0.50, 2: 0.50, 3: 0.45}

GDN_MIN_HOLD = 2
GDN_EXIT_THR = 0.65
SEQ_LEN      = 16   # tb_lstm_widyawardhana_v1

DYNAMIC_NAMES = frozenset({"bars_held_norm", "current_pnl_pct", "current_pnl_atr",
                            "max_favorable_pnl_pct", "drawdown_from_peak_pct",
                            "direction", "entry_price_ratio"})

# ── Load models ─────────────────────────────────────────────────────────────────
ic32_model = joblib.load(MODEL_DIR / "runs" / "ic32_regime_v1" / "lgbm.pkl")
ic32_feats = list(ic32_model.feature_name_)

tb_model = joblib.load(MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v3" / "lgbm.pkl")
with open(MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v3" / "tb_lgbm_widyawardhana_v3_features.json") as f:
    tb_feats = json.load(f)

lstm_model  = load_lstm(MODEL_DIR / "runs" / "tb_lstm_widyawardhana_v1" / "lstm.pt", device="cpu")
lstm_scaler = joblib.load(MODEL_DIR / "runs" / "tb_lstm_widyawardhana_v1" / "lstm_scaler.pkl")
with open(MODEL_DIR / "runs" / "tb_lstm_widyawardhana_v1" / "tb_lstm_widyawardhana_v1_features.json") as f:
    lstm_feats = json.load(f)

# ic32 Guardian (production clean_v2)
gdn_model  = joblib.load(PROD / "guardian_best.pkl")
gdn_scaler = joblib.load(PROD / "guardian_scaler.pkl")
with open(PROD / "guardian_feature_cols.json") as f:
    gdn_all_feats = json.load(f)

# TB Guardian v2
tbg_model  = joblib.load(MODEL_DIR / "runs" / "tb_guardian_widyawardhana_v2" / "guardian.pkl")
tbg_scaler = joblib.load(MODEL_DIR / "runs" / "tb_guardian_widyawardhana_v2" / "guardian_scaler.pkl")
with open(MODEL_DIR / "runs" / "tb_guardian_widyawardhana_v2" / "tb_guardian_widyawardhana_v2_feature_cols.json") as f:
    tbg_all_feats = json.load(f)

logger.info(f"ic32({len(ic32_feats)}f) | tb({len(tb_feats)}f) | lstm({len(lstm_feats)}f,seq={SEQ_LEN}) "
            f"| ic32_gdn({len(gdn_all_feats)}f) | tb_gdn({len(tbg_all_feats)}f)")


def make_guardian_config(feat_list):
    static    = [f for f in feat_list if f not in DYNAMIC_NAMES]
    static_map = {name: i for i, name in enumerate(static)}
    order = [("static", static_map[f]) if f in static_map else ("dyn", f) for f in feat_list]
    return static, order

gdn_static, gdn_order   = make_guardian_config(gdn_all_feats)
tbg_static, tbg_order   = make_guardian_config(tbg_all_feats)


def lstm_predict_proba(X_raw):
    """Batch LSTM inference. X_raw: (n, n_lstm_feats). Returns (n, 3) softmax probs."""
    n, f  = X_raw.shape
    X_sc  = lstm_scaler.transform(X_raw.reshape(-1, f)).reshape(n, f).astype(np.float32)
    probs = np.full((n, 3), 1.0 / 3.0, dtype=np.float32)
    if n < SEQ_LEN:
        return probs
    seqs = np.stack([X_sc[i - SEQ_LEN + 1: i + 1] for i in range(SEQ_LEN - 1, n)])
    all_p = []
    with torch.no_grad():
        for b in range(0, len(seqs), 512):
            t = torch.from_numpy(seqs[b: b + 512])
            all_p.append(torch.softmax(lstm_model(t), dim=1).cpu().numpy())
    probs[SEQ_LEN - 1:] = np.concatenate(all_p, axis=0)
    return probs


def apply_flip_veto(yp_base, p_lstm):
    """Variant C — FLIP hard veto: skip if LSTM argmax is opposite direction."""
    yp = yp_base.copy()
    lstm_argmax = np.argmax(p_lstm, axis=1)
    sig_mask    = (yp != FLAT)
    idxs        = np.where(sig_mask)[0]
    lgbm_dir    = yp[idxs]
    opposite    = np.where(lgbm_dir == LONG, SHORT, LONG)
    skip        = (lstm_argmax[idxs] == opposite)
    yp[idxs[skip]] = FLAT
    return yp


def build_guardian_row(j, i, close, atr, direction, max_fav, X_static, feat_order):
    bars_held = j - i
    pnl_pct   = (close[j] - close[i]) / close[i] * direction
    atr_pct   = atr[i] / close[i] if close[i] > 0 else 0.01
    new_max   = max(max_fav, pnl_pct)
    dyn_vals  = {
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
        row[idx] = X_static[j, key] if src == "static" else dyn_vals.get(key, 0.0)
    return row, new_max


def simulate(yp, close, high, low, atr,
             guardian=None, feat_order=None, X_static=None, gdn_scaler=None):
    n = len(yp); trades = []; i = 0
    while i < n:
        sig = yp[i]
        if sig == FLAT: i += 1; continue
        direction  = 1 if sig == LONG else -1
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
                exit_price, exit_bar, outcome = sl_price, j, "SL"; break
            if guardian is not None and (j - i) >= GDN_MIN_HOLD:
                row, max_fav = build_guardian_row(
                    j, i, close, atr, direction, max_fav, X_static, feat_order)
                prob = guardian.predict_proba(gdn_scaler.transform(row.reshape(1, -1)))[0]
                if (prob[2] if len(prob) > 2 else prob[1]) >= GDN_EXIT_THR:
                    exit_price, exit_bar, outcome = close[j], j, "GUARDIAN_EXIT"; break
            else:
                max_fav = max(max_fav, (close[j] - entry) / entry * direction)

        net_pnl = (exit_price - entry) / entry * direction * MODAL * LEVERAGE \
                  - COST_RT * MODAL * LEVERAGE
        trades.append({
            "direction": "LONG" if direction == 1 else "SHORT",
            "outcome"  : outcome,
            "net_pnl"  : net_pnl,
            "bars_held": exit_bar - i,
        })
        i = exit_bar + 1
    return trades


available = [s for s in ALL_COINS
             if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()]

print(f"\n{'='*78}")
print(f"  HOLDOUT OOS — Full System Comparison | Nov 2025-Apr 2026")
print(f"  6 variants: ic32(bare/+Gdn) | TB(bare/+LSTM-C/+Gdn/+LSTM-C+Gdn)")
print(f"  Exit: SL={SL_MULT}xATR | max_hold={MAX_HOLD}bar | NO TP | Guardian=early exit")
print(f"  Coins: {len(available)} | ${MODAL}/trade {LEVERAGE}x")
print(f"{'='*78}\n")

keys = ["ic32", "ic32_gdn", "tb", "tb_lstm_c", "tb_gdn", "tb_lstm_c_gdn"]

def new_agg():
    return {"trades": 0, "wins": 0, "pnl": 0.0,
            "longs": 0, "long_wins": 0, "sl_hits": 0,
            "gdn_exits": 0, "bars_held": []}

agg = {k: new_agg() for k in keys}
decor = {"flip": 0, "consensus": 0, "neutral": 0, "total": 0}

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

    # ── ic32 predictions ──────────────────────────────────────────────────────
    X_ic = np.zeros((n, len(ic32_feats)), dtype=np.float64)
    for idx, c in enumerate(ic32_feats):
        if c == "hmm_regime_enc":
            X_ic[:, idx] = hmm.astype(np.float64)  # always use loaded regime
        elif c in df.columns:
            X_ic[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)
    p_ic  = ic32_model.predict_proba(X_ic)
    yp_ic = np.ones(n, dtype=np.int32)
    yp_ic[p_ic[:, 2] >= IC32_THR_LONG] = LONG
    yp_ic[(p_ic[:, 0] >= IC32_THR_SHORT) & (yp_ic != LONG)] = SHORT

    # ── ic32 Guardian static features ─────────────────────────────────────────
    X_gdn = np.zeros((n, len(gdn_static)), dtype=np.float64)
    for idx, c in enumerate(gdn_static):
        if c == "hmm_regime_enc":
            X_gdn[:, idx] = hmm.astype(np.float64)  # always use loaded regime
        elif c in df.columns:
            X_gdn[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)

    # ── TB LGBM predictions ───────────────────────────────────────────────────
    X_tb = np.zeros((n, len(tb_feats)), dtype=np.float64)
    for idx, c in enumerate(tb_feats):
        if c in df.columns:
            X_tb[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)
    p_tb    = tb_model.predict_proba(X_tb)
    conf_tb = np.max(p_tb, axis=1)
    yp_tb   = np.argmax(p_tb, axis=1).astype(np.int32)
    for r, th in REGIME_THRESH.items():
        yp_tb[(hmm == r) & (yp_tb != FLAT) & (conf_tb < th)] = FLAT

    # ── TB LSTM predictions ───────────────────────────────────────────────────
    X_lstm_raw = np.zeros((n, len(lstm_feats)), dtype=np.float32)
    for idx, c in enumerate(lstm_feats):
        if c in df.columns:
            X_lstm_raw[:, idx] = df[c].ffill().fillna(0).values.astype(np.float32)
    p_lstm = lstm_predict_proba(X_lstm_raw)

    # FLIP veto (Variant C)
    yp_tb_lstm_c = apply_flip_veto(yp_tb, p_lstm)

    # Decorrelation
    lstm_argmax = np.argmax(p_lstm, axis=1)
    sig         = (yp_tb != FLAT)
    sig_idxs    = np.where(sig)[0]
    lgbm_dir    = yp_tb[sig_idxs]
    opp_dir     = np.where(lgbm_dir == LONG, SHORT, LONG)
    decor["total"]    += len(sig_idxs)
    decor["flip"]     += int(np.sum(lstm_argmax[sig_idxs] == opp_dir))
    decor["consensus"]+= int(np.sum(lstm_argmax[sig_idxs] == lgbm_dir))
    decor["neutral"]  += int(np.sum(lstm_argmax[sig_idxs] == FLAT))

    # ── TB Guardian static features ───────────────────────────────────────────
    X_tbg = np.zeros((n, len(tbg_static)), dtype=np.float64)
    for idx, c in enumerate(tbg_static):
        if c in df.columns:
            X_tbg[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)

    # ── Simulate 6 variants ───────────────────────────────────────────────────
    variants = [
        ("ic32",         yp_ic,          None,      None,      None,  None),
        ("ic32_gdn",     yp_ic,          gdn_model, gdn_order, X_gdn, gdn_scaler),
        ("tb",           yp_tb,          None,      None,      None,  None),
        ("tb_lstm_c",    yp_tb_lstm_c,   None,      None,      None,  None),
        ("tb_gdn",       yp_tb,          tbg_model, tbg_order, X_tbg, tbg_scaler),
        ("tb_lstm_c_gdn",yp_tb_lstm_c,   tbg_model, tbg_order, X_tbg, tbg_scaler),
    ]
    for key, yp, gdn, f_order, X_st, scaler in variants:
        trades = simulate(yp, close, high, low, atr,
                          guardian=gdn, feat_order=f_order,
                          X_static=X_st, gdn_scaler=scaler)
        a = agg[key]
        for t in trades:
            a["trades"] += 1; a["pnl"] += t["net_pnl"]
            a["bars_held"].append(t["bars_held"])
            if t["net_pnl"] > 0: a["wins"] += 1
            if t["outcome"] == "SL": a["sl_hits"] += 1
            if "GUARDIAN" in t["outcome"]: a["gdn_exits"] += 1
            if t["direction"] == "LONG":
                a["longs"] += 1
                if t["net_pnl"] > 0: a["long_wins"] += 1

    logger.info(
        f"[{sym:>14s}]"
        f" ic32={agg['ic32']['trades']}"
        f" ic32+gdn={agg['ic32_gdn']['trades']}"
        f" tb={agg['tb']['trades']}"
        f" tb+lstm={agg['tb_lstm_c']['trades']}"
        f" tb+gdn={agg['tb_gdn']['trades']}"
        f" tb+lstm+gdn={agg['tb_lstm_c_gdn']['trades']}"
    )


# ── Decorrelation report ──────────────────────────────────────────────────────
tot = max(decor["total"], 1)
print(f"\n--- DECORRELATION (TB LSTM vs LGBM signal bars) ---")
print(f"  Total LGBM signals : {decor['total']:,}")
print(f"  LSTM FLIP          : {decor['flip']:,}  ({decor['flip']/tot*100:.1f}%)  <- variant C/D skip")
print(f"  LSTM CONSENSUS     : {decor['consensus']:,}  ({decor['consensus']/tot*100:.1f}%)")
print(f"  LSTM FLAT/neutral  : {decor['neutral']:,}  ({decor['neutral']/tot*100:.1f}%)")


# ── Scorecard ─────────────────────────────────────────────────────────────────
def sc(a):
    t = a["trades"]; w = a["wins"]; p = a["pnl"]
    l = a["longs"]; lw = a["long_wins"]
    bh = a["bars_held"]
    wins_list = []  # compute PF from pnl is not straightforward, track wins/losses
    return dict(
        trades   = t,
        wr       = w / max(t, 1) * 100,
        long_wr  = lw / max(l, 1) * 100,
        short_wr = (w - lw) / max(t - l, 1) * 100,
        long_pct = l / max(t, 1) * 100,
        sl_rate  = a["sl_hits"] / max(t, 1) * 100,
        gdn_rate = a["gdn_exits"] / max(t, 1) * 100,
        avg_hold = np.mean(bh) if bh else 0,
        pnl      = p,
        ppm      = p / 5,
        ppt      = p / max(t, 1),
    )

s = {k: sc(agg[k]) for k in keys}

LABELS = {
    "ic32"         : "ic32 bare",
    "ic32_gdn"     : "ic32+Guardian",
    "tb"           : "TB bare",
    "tb_lstm_c"    : "TB+LSTM-C",
    "tb_gdn"       : "TB+Guardian",
    "tb_lstm_c_gdn": "TB+LSTM-C+Gdn",
}

W = 14
print(f"\n{'='*90}")
print(f"  SCORECARD — Full System Comparison | Nov 2025-Apr 2026 | 21 koin | $10/trade 5x")
print(f"{'='*90}")
hdr = f"  {'Metrik':<22}" + "".join(f"{LABELS[k]:>{W}}" for k in keys)
print(hdr)
print(f"  {'-'*22}" + "-" * (W * len(keys)))

rows = [
    ("Trades",          lambda x: f"{x['trades']:,}"),
    ("Trades/bulan",    lambda x: f"{x['trades']/5:.0f}"),
    ("Win Rate %",      lambda x: f"{x['wr']:.1f}%"),
    ("  LONG WR %",     lambda x: f"{x['long_wr']:.1f}%({x['long_pct']:.0f}%)"),
    ("  SHORT WR %",    lambda x: f"{x['short_wr']:.1f}%"),
    ("SL hit rate",     lambda x: f"{x['sl_rate']:.1f}%"),
    ("Guardian exit",   lambda x: f"{x['gdn_rate']:.1f}%" if x['gdn_rate'] > 0 else "—"),
    ("Avg hold (bar)",  lambda x: f"{x['avg_hold']:.1f}"),
    ("Net PnL $",       lambda x: f"${x['pnl']:+.0f}"),
    ("PnL/bulan $",     lambda x: f"${x['ppm']:+.0f}"),
    ("PnL/trade $",     lambda x: f"${x['ppt']:+.3f}"),
]

for label, fn in rows:
    row = f"  {label:<22}"
    for k in keys:
        row += f"{fn(s[k]):>{W}}"
    if "Net PnL" in label:
        row += "  <-- BEST?"
    print(row)

best_k = max(s, key=lambda k: s[k]["pnl"])
print(f"\n  Best PnL  : {LABELS[best_k]}  = ${s[best_k]['pnl']:+.0f}")
best_wk = max(s, key=lambda k: s[k]["wr"])
print(f"  Best WR   : {LABELS[best_wk]}  = {s[best_wk]['wr']:.1f}%")

# ── Save ──────────────────────────────────────────────────────────────────────
out = {k: {m: round(float(v), 4) if isinstance(v, (float, np.floating)) else v
           for m, v in s[k].items()}
       for k in keys}
out["_meta"] = {
    "description": "6-variant full comparison — ic32 vs TB, bare vs LSTM-C vs Guardian",
    "lstm_model": "tb_lstm_widyawardhana_v1",
    "lstm_cv_f1": 0.3622,
    "lstm_variant": "FLIP hard veto (Variant C — argmax opposite blocks entry)",
    "guardian_tb": "tb_guardian_widyawardhana_v2",
    "guardian_ic32": "ic32_guardian_clean_v2 (production)",
    "period": "Nov2025-Apr2026",
    "coins": len(available),
    "decorrelation": {k: round(v / tot * 100, 1) for k, v in decor.items() if k != "total"},
}
out_path = MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v3" / "holdout_full_comparison.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved -> {out_path}")
print(f"{'='*90}")
