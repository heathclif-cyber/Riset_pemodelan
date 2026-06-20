"""
pipeline/07_holdout_tb_lstm_widyawardhana.py
Holdout OOS Nov 2025 - Apr 2026: Cascade comparison 4 variant

Variant A: LGBM only (baseline)
Variant B: FLIP veto soft  -- skip jika P_lstm(opposite) > FLIP_THR_SOFT (0.40)
Variant C: FLIP veto hard  -- skip jika lstm_argmax == opposite direction (berapapun P)
Variant D: Consensus only  -- hanya enter jika lstm_argmax == lgbm direction

Exit: SL=1.5xATR + max_hold=36bar, NO TP (live-like eval)
Output: models/runs/tb_lgbm_widyawardhana_v3/holdout_tb_lstm_cascade.json
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
from config import (
    HOLDOUT_DIR, MODEL_DIR, ALL_COINS,
    TP_SL_FALLBACK_SL, MAX_HOLDING_BARS,
    MODAL_PER_TRADE, LEVERAGE_SIM,
    FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
)

logger = setup_logger("07_holdout_tb_lstm_wid")

# ── Config ─────────────────────────────────────────────────────────────────────
LGBM_RUN      = "tb_lgbm_widyawardhana_v3"
LSTM_RUN      = "tb_lstm_widyawardhana_v1"
SEQ_LEN       = 16
SL_MULT       = TP_SL_FALLBACK_SL          # 1.5
MAX_HOLD      = MAX_HOLDING_BARS           # 36
MODAL         = MODAL_PER_TRADE            # 10.0
LEVERAGE      = LEVERAGE_SIM[0] if isinstance(LEVERAGE_SIM, list) else LEVERAGE_SIM
COST_RT       = (FEE_PER_SIDE + SLIPPAGE_PER_SIDE) * 2  # round-trip cost
# Regime-aware threshold (matching 07_holdout_livelike.py)
# TRENDING (0,3) → 0.45, RANGING (1,2) → 0.50
REGIME_THRESH = {0: 0.45, 1: 0.50, 2: 0.50, 3: 0.45}
FLIP_THR_SOFT = 0.40   # Variant B: skip jika P(opposite) > threshold ini

SHORT, FLAT, LONG = 0, 1, 2

VARIANTS = ["A", "B", "C", "D"]
VARIANT_LABELS = {
    "A": "LGBM only",
    "B": "FLIP veto soft (P>0.40)",
    "C": "FLIP veto hard (argmax)",
    "D": "Consensus only",
}

# ── Load models ────────────────────────────────────────────────────────────────
lgbm_path = MODEL_DIR / "runs" / LGBM_RUN / "lgbm.pkl"
lstm_path  = MODEL_DIR / "runs" / LSTM_RUN / "lstm.pt"

if not lstm_path.exists():
    raise FileNotFoundError(
        f"LSTM model belum ada: {lstm_path}\n"
        f"Tunggu training selesai dulu: python pipeline/05_train_lstm_widyawardhana_v1.py"
    )

lgbm_model  = joblib.load(lgbm_path)
with open(MODEL_DIR / "runs" / LGBM_RUN / f"{LGBM_RUN}_features.json") as f:
    lgbm_feats = json.load(f)

lstm_model  = load_lstm(lstm_path, device="cpu")
lstm_scaler = joblib.load(MODEL_DIR / "runs" / LSTM_RUN / "lstm_scaler.pkl")
with open(MODEL_DIR / "runs" / LSTM_RUN / f"{LSTM_RUN}_features.json") as f:
    lstm_feats = json.load(f)

# Load CV F1 dari meta jika ada
try:
    with open(MODEL_DIR / "runs" / LSTM_RUN / f"{LSTM_RUN}_meta.json") as f:
        lstm_cv_f1 = json.load(f).get("cv_f1_mean", None)
except Exception:
    lstm_cv_f1 = None

logger.info(f"LGBM : {LGBM_RUN} ({len(lgbm_feats)} feat)")
logger.info(f"LSTM : {LSTM_RUN} ({len(lstm_feats)} feat, seq={SEQ_LEN}, cv_f1={lstm_cv_f1})")


# ── LSTM batch inference ───────────────────────────────────────────────────────
def lstm_predict_proba(X_raw: np.ndarray) -> np.ndarray:
    """
    Batch inference. X_raw: (n, n_feats).
    Returns (n, 3) softmax probs [SHORT, FLAT, LONG].
    Baris pertama SEQ_LEN-1 mendapat uniform prior (sequence belum penuh).
    """
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


# ── Entry filter per variant ───────────────────────────────────────────────────
def apply_entry_filter(yp_lgbm: np.ndarray, p_lstm: np.ndarray, variant: str) -> np.ndarray:
    """
    Terapkan LSTM filter ke sinyal LGBM.
    yp_lgbm: (n,) int array [0=SHORT, 1=FLAT, 2=LONG]
    p_lstm  : (n, 3) softmax probs
    Returns: filtered yp (1=FLAT berarti skip)
    """
    yp          = yp_lgbm.copy()
    lstm_argmax = np.argmax(p_lstm, axis=1)

    if variant == "A":
        return yp   # no filter

    sig_mask = (yp != FLAT)
    if not sig_mask.any():
        return yp

    idxs     = np.where(sig_mask)[0]
    lgbm_dir = yp[idxs]
    opposite = np.where(lgbm_dir == LONG, SHORT, LONG)
    p_opp    = p_lstm[idxs, opposite]
    lstm_prd = lstm_argmax[idxs]

    if variant == "B":
        # Skip jika LSTM confident pada arah berlawanan (P > FLIP_THR_SOFT)
        skip = (lstm_prd == opposite) & (p_opp > FLIP_THR_SOFT)

    elif variant == "C":
        # Skip jika argmax LSTM = opposite direction (berapapun probabilitas)
        skip = (lstm_prd == opposite)

    elif variant == "D":
        # Hanya enter jika LSTM setuju arah (consensus)
        skip = (lstm_prd != lgbm_dir)

    else:
        raise ValueError(f"Unknown variant: {variant}")

    yp[idxs[skip]] = FLAT
    return yp


# ── Live-like simulation ───────────────────────────────────────────────────────
def simulate_livelike(yp, close, high, low, atr):
    """SL exit + time exit, no TP. Sequential (no overlapping trades)."""
    n = len(yp); trades = []; i = 0
    while i < n:
        if yp[i] == FLAT:
            i += 1; continue

        direction  = 1 if yp[i] == LONG else -1
        entry      = close[i]
        sl_price   = entry - direction * SL_MULT * atr[i]
        exit_price = close[min(i + MAX_HOLD, n - 1)]
        exit_bar   = min(i + MAX_HOLD, n - 1)
        outcome    = "TIME_EXIT"

        for j in range(i + 1, min(i + MAX_HOLD + 1, n)):
            hit = (direction == 1  and low[j]  <= sl_price) or \
                  (direction == -1 and high[j] >= sl_price)
            if hit:
                exit_price, exit_bar, outcome = sl_price, j, "SL"; break

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


# ── Scorecard helper ───────────────────────────────────────────────────────────
def scorecard(trades):
    if not trades:
        return dict(trades=0, wr=0.0, long_wr=0.0, short_wr=0.0,
                    pnl=0.0, ppt=0.0, ppm=0.0, pf=0.0,
                    sl_rate=0.0, avg_hold=0.0, max_dd=0.0)
    arr    = np.array([t["net_pnl"] for t in trades])
    wins   = arr > 0
    longs  = [t for t in trades if t["direction"] == "LONG"]
    shorts = [t for t in trades if t["direction"] == "SHORT"]
    sls    = sum(1 for t in trades if t["outcome"] == "SL")
    holds  = [t["bars_held"] for t in trades]
    gw     = arr[wins].sum()
    gl     = abs(arr[~wins].sum())
    pf     = gw / gl if gl > 0 else float("inf")
    lw     = sum(1 for t in longs  if t["net_pnl"] > 0)
    sw     = sum(1 for t in shorts if t["net_pnl"] > 0)

    # Max drawdown dari equity curve
    equity = np.cumsum(arr)
    peak   = np.maximum.accumulate(equity)
    max_dd = float((peak - equity).max()) if len(equity) else 0.0

    return dict(
        trades   = len(trades),
        wr       = float(wins.mean() * 100),
        long_wr  = lw / max(len(longs),  1) * 100,
        short_wr = sw / max(len(shorts), 1) * 100,
        long_pct = len(longs) / max(len(trades), 1) * 100,
        pnl      = float(arr.sum()),
        ppt      = float(arr.mean()),
        ppm      = float(arr.sum() / 5),
        pf       = float(pf),
        sl_rate  = sls / len(trades) * 100,
        avg_hold = float(np.mean(holds)),
        max_dd   = max_dd,
    )


# ── Main loop ─────────────────────────────────────────────────────────────────
available = [s for s in ALL_COINS
             if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()]

print(f"\n{'='*72}")
print(f"  HOLDOUT OOS -- TB LGBM + LSTM Cascade | Nov 2025-Apr 2026")
print(f"  LGBM: {LGBM_RUN} | LSTM: {LSTM_RUN} (cv_f1={lstm_cv_f1})")
print(f"  Variants: {', '.join(f'{v}={VARIANT_LABELS[v]}' for v in VARIANTS)}")
print(f"  Exit: SL={SL_MULT}xATR + max_hold={MAX_HOLD}bar | NO TP")
print(f"  Coins: {len(available)} | ${MODAL}/trade {LEVERAGE}x")
print(f"{'='*72}\n")

all_trades = {v: [] for v in VARIANTS}
decor = {"flip": 0, "consensus": 0, "neutral": 0, "total_signals": 0}

for sym in available:
    df = pd.read_parquet(HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet")
    df = ensure_utc_index(df).sort_index()

    # Filter ke baris berlabel valid (hindari forward-looking NaN di ujung data)
    # Load HMM regime (sama seperti 07_holdout_livelike.py)
    rp  = HOLDOUT_DIR / "labeled" / f"{sym}_regime_h1.parquet"
    hmm = np.full(len(df), 1, np.int32)
    if rp.exists():
        reg = pd.read_parquet(rp)
        if "hmm_regime_enc" in reg.columns:
            hmm = reg["hmm_regime_enc"].reindex(df.index, fill_value=1).values.astype(np.int32)

    if "label" in df.columns:
        mask = df["label"].isin(["SHORT", "FLAT", "LONG", 0, 1, 2])
        df   = df[mask].copy()
        hmm  = hmm[mask.values]
    n = len(df)
    if n < SEQ_LEN + MAX_HOLD:
        logger.warning(f"{sym}: skip (n={n} terlalu kecil)"); continue

    close = df["close"].values.astype(np.float64)
    high  = df["high"].values.astype(np.float64)
    low   = df["low"].values.astype(np.float64)
    atr   = df["atr_14_h1"].values.astype(np.float64)
    atr   = np.where(atr <= 0, np.nanmedian(atr[atr > 0]) if (atr > 0).any() else 1e-4, atr)

    # ── LGBM inference ────────────────────────────────────────────────────────
    avail_lgbm = [c for c in lgbm_feats if c in df.columns]
    X_lgbm = np.zeros((n, len(lgbm_feats)), dtype=np.float64)
    for idx, c in enumerate(lgbm_feats):
        if c in df.columns:
            X_lgbm[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)

    p_lgbm  = lgbm_model.predict_proba(X_lgbm)
    conf    = np.max(p_lgbm, axis=1)
    yp_base = np.argmax(p_lgbm, axis=1).astype(np.int32)
    # Regime-aware threshold (matching 07_holdout_livelike.py)
    for r, th in REGIME_THRESH.items():
        yp_base[(hmm == r) & (yp_base != FLAT) & (conf < th)] = FLAT

    # ── LSTM inference ────────────────────────────────────────────────────────
    avail_lstm = [c for c in lstm_feats if c in df.columns]
    missing    = set(lstm_feats) - set(avail_lstm)
    if missing:
        logger.warning(f"{sym}: LSTM missing {len(missing)} feat: {list(missing)[:3]}...")
    X_lstm = np.zeros((n, len(lstm_feats)), dtype=np.float32)
    for idx, c in enumerate(lstm_feats):
        if c in df.columns:
            X_lstm[:, idx] = df[c].ffill().fillna(0).values.astype(np.float32)
    p_lstm = lstm_predict_proba(X_lstm)

    # ── Decorrelation stats ───────────────────────────────────────────────────
    sig_mask    = (yp_base != FLAT)
    n_sig       = int(sig_mask.sum())
    if n_sig > 0:
        lstm_argmax = np.argmax(p_lstm, axis=1)
        opp         = np.where(yp_base == LONG, SHORT, LONG)
        decor["flip"]          += int(((lstm_argmax == opp) & sig_mask).sum())
        decor["consensus"]     += int(((lstm_argmax == yp_base) & sig_mask).sum())
        decor["neutral"]       += int(((lstm_argmax == FLAT) & sig_mask).sum())
        decor["total_signals"] += n_sig

    # ── Simulate each variant ─────────────────────────────────────────────────
    coin_counts = {}
    for v in VARIANTS:
        yp_v   = apply_entry_filter(yp_base, p_lstm, v)
        trades = simulate_livelike(yp_v, close, high, low, atr)
        all_trades[v].extend(trades)
        coin_counts[v] = len(trades)

    logger.info(f"[{sym:>14}] " + " | ".join(f"{v}:{coin_counts[v]}t" for v in VARIANTS))


# ── Decorrelation report ───────────────────────────────────────────────────────
tot = max(decor["total_signals"], 1)
print(f"\n--- DECORRELATION (LGBM signal bars) ---")
print(f"Total LGBM signals : {decor['total_signals']:,}")
print(f"LSTM FLIP          : {decor['flip']:,}      ({decor['flip']/tot*100:.1f}%)  <- var C/D skip ini")
print(f"LSTM CONSENSUS     : {decor['consensus']:,}  ({decor['consensus']/tot*100:.1f}%)  <- var D masuk ini")
print(f"LSTM FLAT/neutral  : {decor['neutral']:,}  ({decor['neutral']/tot*100:.1f}%)  <- var D skip ini")


# ── Scorecard ─────────────────────────────────────────────────────────────────
sc = {v: scorecard(all_trades[v]) for v in VARIANTS}

W = 22
print(f"\n{'='*78}")
print(f"  SCORECARD -- TB LGBM + LSTM Cascade | Nov 2025-Apr 2026")
print(f"  LSTM cv_f1={lstm_cv_f1} | FLIP_THR_SOFT={FLIP_THR_SOFT}")
print(f"{'='*78}")
print(f"  {'Metrik':<22}" + "".join(f"{VARIANT_LABELS[v]:>{W}}" for v in VARIANTS))
print(f"  {'-'*22}" + "-" * (W * len(VARIANTS)))

rows = [
    ("Trades",        "trades",    "{:>7.0f}"),
    ("Win Rate %",    "wr",        "{:>7.1f}%"),
    ("Long WR %",     "long_wr",   "{:>7.1f}%"),
    ("Short WR %",    "short_wr",  "{:>7.1f}%"),
    ("Long %",        "long_pct",  "{:>7.1f}%"),
    ("Net PnL $",     "pnl",       "{:>+8.0f}"),
    ("PnL/month $",   "ppm",       "{:>+8.0f}"),
    ("PnL/trade $",   "ppt",       "{:>+8.2f}"),
    ("Profit Factor", "pf",        "{:>7.2f}"),
    ("SL Rate %",     "sl_rate",   "{:>7.1f}%"),
    ("Avg Hold bar",  "avg_hold",  "{:>7.1f}"),
    ("Max DD $",      "max_dd",    "{:>8.0f}"),
]

for label, key, fmt in rows:
    row = f"  {label:<22}"
    for v in VARIANTS:
        val = sc[v][key]
        row += f"{fmt.format(val):>{W}}"
    print(row)

print(f"{'='*78}")
print(f"\n  Decorrelation: {decor['flip']/tot*100:.1f}% FLIP | "
      f"{decor['consensus']/tot*100:.1f}% CONSENSUS | "
      f"{decor['neutral']/tot*100:.1f}% FLAT")

# ── Highlight best variant ────────────────────────────────────────────────────
best_pnl = max(VARIANTS, key=lambda v: sc[v]["pnl"])
best_pf  = max(VARIANTS, key=lambda v: sc[v]["pf"])
best_wr  = max(VARIANTS, key=lambda v: sc[v]["wr"])
print(f"\n  Best PnL       : Variant {best_pnl} ({VARIANT_LABELS[best_pnl]}) = ${sc[best_pnl]['pnl']:+.0f}")
print(f"  Best PF        : Variant {best_pf}  ({VARIANT_LABELS[best_pf]})  = {sc[best_pf]['pf']:.2f}")
print(f"  Best WR        : Variant {best_wr}  ({VARIANT_LABELS[best_wr]})  = {sc[best_wr]['wr']:.1f}%")

# ── Save ──────────────────────────────────────────────────────────────────────
result = {
    "run"          : "holdout_tb_lstm_widyawardhana_cascade",
    "lgbm_model"   : LGBM_RUN,
    "lstm_model"   : LSTM_RUN,
    "lstm_cv_f1"   : lstm_cv_f1,
    "period"       : "Nov 2025 - Apr 2026",
    "flip_thr_soft": FLIP_THR_SOFT,
    "variant_labels": VARIANT_LABELS,
    "decorrelation": {
        "total_signals" : decor["total_signals"],
        "flip_count"    : decor["flip"],
        "flip_pct"      : round(decor["flip"]   / tot * 100, 2),
        "consensus_count": decor["consensus"],
        "consensus_pct" : round(decor["consensus"] / tot * 100, 2),
        "neutral_count" : decor["neutral"],
        "neutral_pct"   : round(decor["neutral"] / tot * 100, 2),
    },
    "scorecard": {
        v: {k: round(val, 4) if isinstance(val, float) else val
            for k, val in sc[v].items()}
        for v in VARIANTS
    },
    "best": {
        "pnl_variant": best_pnl,
        "pf_variant" : best_pf,
        "wr_variant" : best_wr,
    }
}

out_path = MODEL_DIR / "runs" / LGBM_RUN / "holdout_tb_lstm_cascade.json"
with open(out_path, "w") as f:
    json.dump(result, f, indent=2)

print(f"\n  Saved -> {out_path}")
print(f"{'='*78}")
