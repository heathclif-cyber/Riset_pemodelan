"""
pipeline/07_holdout_livelike_lstm.py
Soft LSTM ensemble holdout + decorrelation analysis.

Variants:
  1. tb bare             (baseline)
  2. tb + Guardian v2
  3. tb + LSTM soft      (alpha=0.3 blending)
  4. tb + Guardian + LSTM soft

Decorrelation: % agreement LGBM vs LSTM pada signal bars
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

logger = setup_logger("07_lstm_ensemble")

LM       = {"SHORT": 0, "FLAT": 1, "LONG": 2}
SL_MULT  = TP_SL_FALLBACK_SL
MAX_HOLD = MAX_HOLDING_BARS
MODAL    = MODAL_PER_TRADE
LEVERAGE = LEVERAGE_SIM[0] if isinstance(LEVERAGE_SIM, list) else LEVERAGE_SIM
COST_RT  = (FEE_PER_SIDE + SLIPPAGE_PER_SIDE) * 2

REGIME_THRESH = {0: 0.45, 1: 0.50, 2: 0.50, 3: 0.45}
GDN_MIN_HOLD  = 2
GDN_EXIT_THR  = 0.65
LSTM_ALPHA    = 0.3    # LSTM weight in soft ensemble
SEQ_LEN       = 48

DYNAMIC_NAMES = frozenset({"bars_held_norm", "current_pnl_pct", "current_pnl_atr",
                            "max_favorable_pnl_pct", "drawdown_from_peak_pct",
                            "direction", "entry_price_ratio"})

# ── Load models ────────────────────────────────────────────────────────────────
tb_model = joblib.load(MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v3" / "lgbm.pkl")
with open(MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v3" / "tb_lgbm_widyawardhana_v3_features.json") as f:
    tb_feats = json.load(f)

tbg_model  = joblib.load(MODEL_DIR / "runs" / "tb_guardian_widyawardhana_v2" / "guardian.pkl")
tbg_scaler = joblib.load(MODEL_DIR / "runs" / "tb_guardian_widyawardhana_v2" / "guardian_scaler.pkl")
with open(MODEL_DIR / "runs" / "tb_guardian_widyawardhana_v2" / "tb_guardian_widyawardhana_v2_feature_cols.json") as f:
    tbg_all_feats = json.load(f)

lstm_model  = load_lstm(MODEL_DIR / "runs" / "tb_lstm_v1" / "lstm.pt", device="cpu")
lstm_scaler = joblib.load(MODEL_DIR / "runs" / "tb_lstm_v1" / "lstm_scaler.pkl")
with open(MODEL_DIR / "runs" / "tb_lstm_v1" / "tb_lstm_v1_features.json") as f:
    lstm_feats = json.load(f)

logger.info(f"Models loaded: tb_lgbm({len(tb_feats)}feat) | guardian({len(tbg_all_feats)}feat) | lstm({len(lstm_feats)}feat,seq={SEQ_LEN})")


def make_guardian_config(all_feat_list):
    static    = [f for f in all_feat_list if f not in DYNAMIC_NAMES]
    static_map = {name: idx for idx, name in enumerate(static)}
    order = [("static", static_map[f]) if f in static_map else ("dyn", f)
             for f in all_feat_list]
    return static, order

tbg_static, tbg_order = make_guardian_config(tbg_all_feats)


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


def lstm_predict_proba(X_raw):
    """
    Batch LSTM inference. X_raw: (n, n_lstm_feats). Returns (n, 3) softmax probs.
    First SEQ_LEN-1 bars get uniform prior (no complete sequence yet).
    """
    n, f  = X_raw.shape
    X_sc  = lstm_scaler.transform(X_raw.reshape(-1, f)).reshape(n, f).astype(np.float32)
    probs = np.full((n, 3), 1.0 / 3.0, dtype=np.float32)
    if n < SEQ_LEN:
        return probs

    seqs  = np.stack([X_sc[i - SEQ_LEN + 1: i + 1] for i in range(SEQ_LEN - 1, n)])
    all_p = []
    with torch.no_grad():
        for b in range(0, len(seqs), 512):
            t  = torch.from_numpy(seqs[b: b + 512])
            lg = lstm_model(t)
            all_p.append(torch.softmax(lg, dim=1).cpu().numpy())
    probs[SEQ_LEN - 1:] = np.concatenate(all_p, axis=0)
    return probs


def simulate(yp, close, high, low, atr,
             guardian=None, feat_order=None, X_static=None, gdn_scaler=None):
    n = len(yp); trades = []; i = 0
    while i < n:
        sig = yp[i]
        if sig == 1: i += 1; continue
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
                exit_price, exit_bar, outcome = sl_price, j, "SL"; break

            if guardian is not None and (j - i) >= GDN_MIN_HOLD:
                row, max_fav = build_guardian_row(
                    j, i, close, atr, direction, max_fav, X_static, feat_order)
                scaled   = gdn_scaler.transform(row.reshape(1, -1))
                prob     = guardian.predict_proba(scaled)[0]
                exit_p   = prob[2] if len(prob) > 2 else prob[1]
                if exit_p >= GDN_EXIT_THR:
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

print(f"\n{'='*72}")
print(f"  HOLDOUT OOS — TB LGBM + LSTM Soft Ensemble | Nov 2025 - Apr 2026")
print(f"  Exit: SL={SL_MULT}xATR | max_hold={MAX_HOLD}bar | NO TP")
print(f"  LSTM alpha={LSTM_ALPHA} | seq={SEQ_LEN}")
print(f"  Coins: {len(available)} | ${MODAL}/trade {LEVERAGE}x")
print(f"{'='*72}\n")

def new_agg():
    return {"trades": 0, "wins": 0, "pnl": 0.0,
            "longs": 0, "long_wins": 0, "sl_hits": 0,
            "gdn_exits": 0, "bars_held": []}

agg = {k: new_agg() for k in ["tb", "tb_gdn", "tb_lstm", "tb_gdn_lstm"]}

# Decorrelation tracking
decor = {"agree": 0, "total": 0, "agree_dir": 0, "total_dir": 0}

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

    # ── LGBM TB predictions ────────────────────────────────────────────────
    X_tb = np.zeros((n, len(tb_feats)), dtype=np.float64)
    for idx, c in enumerate(tb_feats):
        if c in df.columns:
            X_tb[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)
    p_tb    = tb_model.predict_proba(X_tb)
    conf_tb = np.max(p_tb, axis=1)
    yp_tb   = np.argmax(p_tb, axis=1).astype(np.int32)
    for r, th in REGIME_THRESH.items():
        yp_tb[(hmm == r) & (yp_tb != 1) & (conf_tb < th)] = 1

    # ── LSTM predictions ───────────────────────────────────────────────────
    avail_lstm = [c for c in lstm_feats if c in df.columns]
    X_lstm_raw = np.zeros((n, len(avail_lstm)), dtype=np.float32)
    for idx, c in enumerate(avail_lstm):
        X_lstm_raw[:, idx] = df[c].ffill().fillna(0).values.astype(np.float32)
    p_lstm = lstm_predict_proba(X_lstm_raw)

    # ── Soft ensemble ──────────────────────────────────────────────────────
    # Pad LGBM probs to (n,3) if needed
    p_lgbm_full = p_tb if p_tb.shape[1] == 3 else np.column_stack([
        p_tb[:, 0], np.zeros(n), p_tb[:, 1]])
    p_combined  = (1 - LSTM_ALPHA) * p_lgbm_full + LSTM_ALPHA * p_lstm
    conf_comb   = np.max(p_combined, axis=1)
    yp_comb     = np.argmax(p_combined, axis=1).astype(np.int32)
    for r, th in REGIME_THRESH.items():
        yp_comb[(hmm == r) & (yp_comb != 1) & (conf_comb < th)] = 1

    # ── Decorrelation ──────────────────────────────────────────────────────
    lstm_argmax = np.argmax(p_lstm, axis=1)
    sig_bars    = (yp_tb != 1)                  # bars where LGBM has signal
    both_dir    = sig_bars & (lstm_argmax != 1)  # both have directional opinion
    decor["agree"]    += np.sum(yp_tb[sig_bars] == lstm_argmax[sig_bars])
    decor["total"]    += sig_bars.sum()
    decor["agree_dir"] += np.sum(yp_tb[both_dir] == lstm_argmax[both_dir])
    decor["total_dir"] += both_dir.sum()

    # ── Guardian static features ───────────────────────────────────────────
    X_tbg = np.zeros((n, len(tbg_static)), dtype=np.float64)
    for idx, c in enumerate(tbg_static):
        if c in df.columns:
            X_tbg[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)

    # ── Simulate 4 variants ────────────────────────────────────────────────
    variants = [
        ("tb",          yp_tb,   None,      None,      None,  None),
        ("tb_gdn",      yp_tb,   tbg_model, tbg_order, X_tbg, tbg_scaler),
        ("tb_lstm",     yp_comb, None,      None,      None,  None),
        ("tb_gdn_lstm", yp_comb, tbg_model, tbg_order, X_tbg, tbg_scaler),
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

    logger.info(f"[{sym}] tb={agg['tb']['trades']} gdn={agg['tb_gdn']['trades']} "
                f"lstm={agg['tb_lstm']['trades']} gdn+lstm={agg['tb_gdn_lstm']['trades']}")


# ── Decorrelation Report ──────────────────────────────────────────────────────
agree_all = decor["agree"] / max(decor["total"], 1) * 100
agree_dir = decor["agree_dir"] / max(decor["total_dir"], 1) * 100
print(f"\n--- DECORRELATION ANALYSIS ---")
print(f"LGBM signal bars  : {decor['total']:,}")
print(f"Both directional  : {decor['total_dir']:,}")
print(f"Agreement (all)   : {agree_all:.1f}%  (target <70% for value)")
print(f"Agreement (dir)   : {agree_dir:.1f}%  (when both non-FLAT)")


# ── Scorecard ─────────────────────────────────────────────────────────────────
def sc(a):
    t = a["trades"]; w = a["wins"]; p = a["pnl"]
    l = a["longs"];  lw = a["long_wins"]
    sl = a["sl_hits"]; gd = a["gdn_exits"]; bh = a["bars_held"]
    wr  = w / max(t, 1) * 100
    return dict(trades=t, wr=wr,
                long_wr=lw / max(l, 1) * 100,
                short_wr=(w - lw) / max(t - l, 1) * 100,
                long_pct=l / max(t, 1) * 100,
                sl_rate=sl / max(t, 1) * 100,
                gdn_rate=gd / max(t, 1) * 100,
                avg_hold=np.mean(bh) if bh else 0,
                pnl=p, ppm=p / 5, ppt=p / max(t, 1))

s = {k: sc(agg[k]) for k in agg}

LABELS = {
    "tb":          "tb bare",
    "tb_gdn":      "tb+Guardian",
    "tb_lstm":     "tb+LSTM soft",
    "tb_gdn_lstm": "tb+Gdn+LSTM",
}

keys = ["tb", "tb_gdn", "tb_lstm", "tb_gdn_lstm"]
W = 16
print(f"\n{'='*72}")
print(f"  SCORECARD - TB LGBM + LSTM Soft Ensemble | Nov 2025-Apr 2026")
print(f"  LSTM alpha={LSTM_ALPHA} | seq={SEQ_LEN} | CV F1={0.3862}")
print(f"{'='*72}")
hdr = f"  {'Metrik':<22}" + "".join(f"{LABELS[k]:>{W}}" for k in keys)
print(hdr)
print(f"  {'-'*22}" + "-" * (W * len(keys)))

rows = [
    ("Trades",      "trades",    "{:>6.0f}"),
    ("Win Rate %",  "wr",        "{:>6.1f}%"),
    ("Long WR %",   "long_wr",   "{:>6.1f}%"),
    ("Short WR %",  "short_wr",  "{:>6.1f}%"),
    ("Long %",      "long_pct",  "{:>6.1f}%"),
    ("SL Rate %",   "sl_rate",   "{:>6.1f}%"),
    ("Gdn Rate %",  "gdn_rate",  "{:>6.1f}%"),
    ("Avg Hold",    "avg_hold",  "{:>6.1f}"),
    ("Net PnL $",   "pnl",       "{:>+6.0f}"),
    ("PnL/month $", "ppm",       "{:>+6.0f}"),
    ("PnL/trade $", "ppt",       "{:>+6.2f}"),
]

for label, key, fmt in rows:
    row = f"  {label:<22}"
    for k in keys:
        val = s[k][key]
        row += f"{fmt.format(val):>{W}}"
    print(row)

print(f"{'='*72}")
print(f"\n  Decorrelation: {agree_all:.1f}% agreement (all) | {agree_dir:.1f}% (directional)")

# ── Save ─────────────────────────────────────────────────────────────────────
result = {
    "run": "tb_lstm_soft_ensemble",
    "lstm_alpha": LSTM_ALPHA,
    "lstm_cv_f1": 0.3862,
    "decorrelation": {
        "agree_pct_all": round(agree_all, 2),
        "agree_pct_dir": round(agree_dir, 2),
        "total_signal_bars": int(decor["total"]),
    },
    "scorecard": {k: {m: round(v, 4) if isinstance(v, float) else v
                      for m, v in s[k].items() if m != "bars_held"}
                  for k in keys},
}
out = MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v3" / "holdout_lstm_ensemble.json"
import json as _json
with open(out, "w") as f:
    _json.dump(result, f, indent=2)
print(f"\n  Saved -> {out}")
