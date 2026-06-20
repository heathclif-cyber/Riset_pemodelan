"""
pipeline/07_ic32_full_cascade.py
Simulasi ic32 full production cascade pada holdout Apr-Jun 2026.

Stack:
  LGBM  : ic32_regime_v1 (33 feat, thr_long=0.69, thr_short=0.59)
  LSTM  : swint_tradev2/models/lstm_best.pt (11 temporal feat, seq=32)
           hard_consensus: LGBM gates → LSTM adjusts confidence
  Guardian: ic32_guardian_clean_v2 (40 feat, min_hold=2, exit_thr=0.65)

Cascade logic (inference.py hard_consensus):
  Stage 1 — LGBM gate:
    p_long  >= 0.69 → LONG signal
    p_short >= 0.59 → SHORT signal
    else           → flat_review: if LSTM directional >= 0.70 → override

  Stage 2 — LSTM confidence adjustment:
    agree   : conf += 0.05
    neutral : conf -= 0.05
    opposite: conf -= min(0.08, max_safe_pen)  [capped jika LGBM kuat]

  Stage 3 — Entry gate: adjusted conf >= 0.59 → masuk trade

  Exit   : SL=1.5xATR | max_hold=36 | Guardian (min_hold=2, thr=0.65)
"""
import json, sys, warnings, numpy as np
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import joblib, torch
from core.models import load_lstm
from core.utils import ensure_utc_index
from config import (
    ALL_COINS, HOLDOUT_DIR, MODEL_DIR,
    TP_SL_FALLBACK_SL, MAX_HOLDING_BARS,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
)

PROD = Path("D:/Apps-Dev/swint_tradev2/models")

# ── Cascade parameters (dari CLAUDE.md + inference.py defaults) ────────────────
LGBM_THR_LONG   = 0.69
LGBM_THR_SHORT  = 0.59
CONF_ENTRY_THR  = 0.59
AGREE_BOOST     = 0.05
NEUTRAL_PEN     = 0.05
OPPOSITE_PEN    = 0.08
NO_VETO_THR     = 0.50
FLAT_REVIEW     = True
DIR_REVIEW_THR  = 0.35
LSTM_OVERRIDE   = 0.70

GDN_MIN_HOLD    = 2
GDN_EXIT_THR    = 0.65
SL_MULT         = TP_SL_FALLBACK_SL
MAX_HOLD        = MAX_HOLDING_BARS
MODAL           = MODAL_PER_TRADE
LEVERAGE        = LEVERAGE_SIM[0] if isinstance(LEVERAGE_SIM, list) else LEVERAGE_SIM
COST_RT         = (FEE_PER_SIDE + SLIPPAGE_PER_SIDE) * 2

LONG, FLAT, SHORT = 2, 1, 0
SEQ_LEN = 32

DYNAMIC_NAMES = frozenset({
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct",
    "direction", "entry_price_ratio",
})

# ── Load models ────────────────────────────────────────────────────────────────
print("Loading models ...")

lgbm_model  = joblib.load(MODEL_DIR / "runs" / "ic32_regime_v1" / "lgbm.pkl")
lgbm_feats  = list(lgbm_model.feature_name_)

lstm_model  = load_lstm(PROD / "lstm_best.pt", device="cpu")
lstm_scaler = joblib.load(PROD / "lstm_scaler.pkl")
with open(PROD / "feature_cols_lstm_temporal.json") as f:
    lstm_feats = json.load(f)

gdn_model   = joblib.load(PROD / "guardian_best.pkl")
gdn_scaler  = joblib.load(PROD / "guardian_scaler.pkl")
with open(PROD / "guardian_feature_cols.json") as f:
    gdn_all_feats = json.load(f)

gdn_static  = [f for f in gdn_all_feats if f not in DYNAMIC_NAMES]
gdn_smap    = {name: i for i, name in enumerate(gdn_static)}
gdn_order   = [("static", gdn_smap[f]) if f in gdn_smap else ("dyn", f) for f in gdn_all_feats]

print(f"  LGBM  : {len(lgbm_feats)} feat")
print(f"  LSTM  : {len(lstm_feats)} feat, seq={SEQ_LEN}")
print(f"  Guard : {len(gdn_all_feats)} feat (static={len(gdn_static)})")


def build_X_lgbm(df, hmm):
    n = len(df)
    X = np.zeros((n, len(lgbm_feats)), dtype=np.float64)
    for idx, c in enumerate(lgbm_feats):
        if c == "hmm_regime_enc":
            X[:, idx] = hmm.astype(np.float64)
        elif c in df.columns:
            X[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)
    return X


def build_lstm_seq(df):
    n = len(df)
    X_raw = np.zeros((n, len(lstm_feats)), dtype=np.float32)
    for idx, c in enumerate(lstm_feats):
        if c in df.columns:
            X_raw[:, idx] = df[c].ffill().fillna(0).values.astype(np.float32)
    X_sc = lstm_scaler.transform(X_raw.reshape(-1, len(lstm_feats))).reshape(n, len(lstm_feats)).astype(np.float32)
    # Return per-bar proba matrix (n, 3)
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


def cascade_signals(p_lgbm, p_lstm):
    """Apply hard_consensus cascade, return (yp, conf_adj)."""
    n = len(p_lgbm)
    yp       = np.ones(n, dtype=np.int32)   # FLAT default
    conf_adj = np.zeros(n, dtype=np.float32)

    for i in range(n):
        pl = p_lgbm[i]
        ps = p_lstm[i]

        # Stage 1: LGBM gate
        if pl[LONG] >= LGBM_THR_LONG:
            sig  = LONG;  lgbm_conf = float(pl[LONG])
        elif pl[SHORT] >= LGBM_THR_SHORT:
            sig  = SHORT; lgbm_conf = float(pl[SHORT])
        else:
            # Flat review
            if FLAT_REVIEW:
                dir_score = max(float(pl[LONG]), float(pl[SHORT]))
                if dir_score >= DIR_REVIEW_THR:
                    lstm_idx  = int(np.argmax(ps))
                    lstm_conf = float(ps[lstm_idx])
                    if lstm_idx != FLAT and lstm_conf >= LSTM_OVERRIDE:
                        sig = lstm_idx; lgbm_conf = lstm_conf
                    else:
                        continue   # stays FLAT
                else:
                    continue
            else:
                continue

        # Stage 2: LSTM confidence adjustment
        lstm_idx  = int(np.argmax(ps))
        base_conf = lgbm_conf

        if lstm_idx == sig:
            adj = AGREE_BOOST
        elif lstm_idx == FLAT:
            adj = -NEUTRAL_PEN
        else:
            # opposite — cap penalty if LGBM strong
            if base_conf > NO_VETO_THR:
                other = [j for j in range(3) if j != sig]
                o = max(float(pl[j]) for j in other)
                f = sum(float(pl[j]) for j in other) - o
                tot_other = o + f
                if tot_other > 0 and o < base_conf:
                    max_safe = float((base_conf - o) * tot_other / (2 * o + f)) - 0.01
                    max_safe = max(0.0, max_safe)
                else:
                    max_safe = 0.0
                adj = -min(OPPOSITE_PEN, max_safe)
            else:
                adj = -OPPOSITE_PEN

        final_conf = base_conf + adj

        # Stage 3: entry gate
        if final_conf >= CONF_ENTRY_THR:
            yp[i]       = sig
            conf_adj[i] = final_conf

    return yp, conf_adj


def build_gdn_row(j, i, close, atr, direction, max_fav, X_st):
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
    row = np.zeros(len(gdn_order), dtype=np.float64)
    for idx, (src, key) in enumerate(gdn_order):
        row[idx] = X_st[j, key] if src == "static" else dyn.get(key, 0.0)
    return row, new_max


def simulate(yp, close, high, low, atr, X_gdn):
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
        outcome    = "TIME_EXIT"
        for j in range(i + 1, min(i + MAX_HOLD + 1, n)):
            sl_hit = (direction == 1 and low[j] <= sl_price) or \
                     (direction == -1 and high[j] >= sl_price)
            if sl_hit:
                exit_price, exit_bar, outcome = sl_price, j, "SL"; break
            if (j - i) >= GDN_MIN_HOLD:
                row, max_fav = build_gdn_row(j, i, close, atr, direction, max_fav, X_gdn)
                prob = gdn_model.predict_proba(gdn_scaler.transform(row.reshape(1, -1)))[0]
                ep   = prob[2] if len(prob) > 2 else prob[1]
                if ep >= GDN_EXIT_THR:
                    exit_price, exit_bar, outcome = close[j], j, "GUARDIAN"; break
            else:
                max_fav = max(max_fav, (close[j] - entry) / entry * direction)
        net_pnl = (exit_price - entry) / entry * direction * MODAL * LEVERAGE \
                  - COST_RT * MODAL * LEVERAGE
        trades.append({
            "dir"    : direction,
            "outcome": outcome,
            "pnl"    : net_pnl,
            "hold"   : exit_bar - i,
        })
        i = exit_bar + 1
    return trades


# ── Run ────────────────────────────────────────────────────────────────────────
available = [s for s in ALL_COINS
             if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()]

import pandas as pd
agg = {"t": 0, "w": 0, "pnl": 0.0,
       "l": 0, "lw": 0, "sl": 0, "gdn": 0, "holds": []}
cascade_stats = {"agree": 0, "neutral": 0, "opposite": 0,
                 "flat_review": 0, "override": 0, "total_lgbm_sig": 0}

print(f"\nRunning full cascade on {len(available)} coins ...")

for sym in available:
    df = pd.read_parquet(HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet")
    df = ensure_utc_index(df).sort_index()

    # HMM regime
    rp  = HOLDOUT_DIR / "labeled" / f"{sym}_regime_h1.parquet"
    hmm = np.full(len(df), 1, np.int32)
    if rp.exists():
        reg = pd.read_parquet(rp)
        if "hmm_regime_enc" in reg.columns:
            hmm = reg["hmm_regime_enc"].reindex(df.index, fill_value=1).values.astype(np.int32)

    n = len(df)
    close = df["close"].values.astype(np.float64)
    high  = df["high"].values.astype(np.float64)
    low   = df["low"].values.astype(np.float64)
    atr   = df["atr_14_h1"].values.astype(np.float64)

    X_lgbm  = build_X_lgbm(df, hmm)
    p_lgbm  = lgbm_model.predict_proba(X_lgbm)
    p_lstm  = build_lstm_seq(df)

    yp, conf_adj = cascade_signals(p_lgbm, p_lstm)

    # Guardian static features
    X_gdn = np.zeros((n, len(gdn_static)), dtype=np.float64)
    for idx, c in enumerate(gdn_static):
        if c in df.columns:
            X_gdn[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)
        elif c == "hmm_regime_enc":
            X_gdn[:, idx] = hmm.astype(np.float64)

    # Cascade stats per coin
    lgbm_sig = (p_lgbm[:, LONG] >= LGBM_THR_LONG) | (p_lgbm[:, SHORT] >= LGBM_THR_SHORT)
    cascade_stats["total_lgbm_sig"] += int(lgbm_sig.sum())
    sig_mask = (yp != FLAT)
    for i in np.where(sig_mask)[0]:
        lstm_idx = int(np.argmax(p_lstm[i]))
        if lstm_idx == yp[i]:  cascade_stats["agree"] += 1
        elif lstm_idx == FLAT: cascade_stats["neutral"] += 1
        else:                  cascade_stats["opposite"] += 1

    trades = simulate(yp, close, high, low, atr, X_gdn)
    for t in trades:
        agg["t"] += 1; agg["pnl"] += t["pnl"]; agg["holds"].append(t["hold"])
        if t["pnl"] > 0: agg["w"] += 1
        if t["dir"] == 1:
            agg["l"] += 1
            if t["pnl"] > 0: agg["lw"] += 1
        if t["outcome"] == "SL":       agg["sl"]  += 1
        if t["outcome"] == "GUARDIAN": agg["gdn"] += 1

    print(f"  [{sym:>14s}] trades so far: {agg['t']:,}")

# ── Scorecard ─────────────────────────────────────────────────────────────────
T   = agg["t"]; W = agg["w"]; P = agg["pnl"]
L   = agg["l"]; LW = agg["lw"]
BULAN = 2.5

print(f"\n{'='*65}")
print(f"  ic32 FULL CASCADE | Apr 2026-Jun 2026 (~2.5 bln) | 21 koin")
print(f"  Stack: LGBM(0.69/0.59) + LSTM-adj + Guardian(min=2, thr=0.65)")
print(f"{'='*65}")
print(f"  Trades total       : {T:,}")
print(f"  Trades/bulan       : {T/BULAN:.0f}")
print(f"  Win Rate           : {W/max(T,1)*100:.1f}%")
print(f"  LONG WR            : {LW/max(L,1)*100:.1f}% ({L/max(T,1)*100:.0f}% long)")
print(f"  SHORT WR           : {(W-LW)/max(T-L,1)*100:.1f}%")
print(f"  SL hit rate        : {agg['sl']/max(T,1)*100:.1f}%")
print(f"  Guardian exit rate : {agg['gdn']/max(T,1)*100:.1f}%")
print(f"  Avg hold (bars)    : {np.mean(agg['holds']):.1f}")
print(f"  Net PnL            : ${P:+.1f}")
print(f"  PnL/bulan          : ${P/BULAN:+.1f}")
print(f"  PnL/trade          : ${P/max(T,1):+.3f}")
print(f"  {'-'*50}")
print(f"  LSTM cascade stats:")
print(f"    LGBM signals     : {cascade_stats['total_lgbm_sig']:,}")
print(f"    agree            : {cascade_stats['agree']:,}")
print(f"    neutral          : {cascade_stats['neutral']:,}")
print(f"    opposite         : {cascade_stats['opposite']:,}")
print(f"{'='*65}")

# ── Compare ───────────────────────────────────────────────────────────────────
print(f"\n  PERBANDINGAN (Apr-Jun 2026, 21 koin, SL-only sim):")
print(f"  {'Variant':<24} {'Trades':>8} {'T/bln':>7} {'WR%':>7} {'PnL':>8} {'$/trade':>9}")
print(f"  {'-'*65}")
rows = [
    ("ic32 bare",       712,  285,  41.7, 185,  0.260),
    ("ic32 + Guardian", 843,  337,  47.9, 257,  0.305),
    ("ic32 FULL CASCADE", T,  round(T/BULAN), round(W/max(T,1)*100,1), round(P,0), round(P/max(T,1),3)),
    ("TB+Gdn (thr=0.42)", 1322, 529, 46.1, 301,  0.227),
    ("TB-SHAP+Gdn (0.58)", 1688, 675, 48.7, 523, 0.310),
]
for lbl, trades, tpm, wr, pnl, ppt in rows:
    marker = " <--" if lbl == "ic32 FULL CASCADE" else ""
    print(f"  {lbl:<24} {trades:>8,} {tpm:>7} {wr:>6.1f}% ${pnl:>+7.0f}  ${ppt:>+.3f}{marker}")
print()

# Save
out = {
    "variant": "ic32_full_cascade",
    "period": "Apr2026-Jun2026",
    "stack": {
        "lgbm": "ic32_regime_v1",
        "lstm": "lstm_best.pt (11 feat, seq=32, hard_consensus)",
        "guardian": "ic32_guardian_clean_v2 (40 feat, min_hold=2, thr=0.65)",
    },
    "thresholds": {
        "lgbm_long": LGBM_THR_LONG, "lgbm_short": LGBM_THR_SHORT,
        "conf_entry": CONF_ENTRY_THR,
    },
    "results": {
        "trades": T, "win_rate": round(W/max(T,1)*100,2),
        "pnl": round(P,2), "pnl_per_month": round(P/BULAN,2),
        "pnl_per_trade": round(P/max(T,1),4),
        "sl_rate": round(agg["sl"]/max(T,1)*100,2),
        "gdn_rate": round(agg["gdn"]/max(T,1)*100,2),
        "avg_hold": round(float(np.mean(agg["holds"])),2),
    },
    "cascade_stats": cascade_stats,
}
out_path = MODEL_DIR / "runs" / "ic32_regime_v1" / "holdout_full_cascade.json"
out_path.parent.mkdir(parents=True, exist_ok=True)
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"  Saved -> {out_path}")
