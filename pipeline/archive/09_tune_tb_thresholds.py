"""
pipeline/09_tune_tb_thresholds.py
Threshold sweep: TB LGBM x Guardian exit thr x Guardian min_hold

288 kombinasi = 8 LGBM configs x 9 Guardian thresholds x 4 min_hold values

Fixed (best dari 08_tune_tb_combination.py):
  LSTM mode      : soft_mul
  LSTM regime    : skip_trending (apply hanya di RANGING)

Sweep:
  LGBM threshold : 8 fine-grained HMM configs (trending_thr x ranging_thr)
  Guardian thr   : 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75
  Guardian hold  : min_hold = 1, 2, 3, 4 bars
"""

import json, sys, warnings, itertools
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
    ALL_COINS, HOLDOUT_DIR, MODEL_DIR,
    TP_SL_FALLBACK_SL, MAX_HOLDING_BARS,
    MODAL_PER_TRADE, LEVERAGE_SIM,
    FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
)

logger = setup_logger("09_tune_thr")

SHORT, FLAT, LONG = 0, 1, 2
SL_MULT  = TP_SL_FALLBACK_SL
MAX_HOLD = MAX_HOLDING_BARS
MODAL    = MODAL_PER_TRADE
LEVERAGE = LEVERAGE_SIM[0] if isinstance(LEVERAGE_SIM, list) else LEVERAGE_SIM
COST_RT  = (FEE_PER_SIDE + SLIPPAGE_PER_SIDE) * 2

DYNAMIC_NAMES = frozenset({
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct",
    "direction", "entry_price_ratio",
})

# ── Load models ────────────────────────────────────────────────────────────────
tb_model = joblib.load(MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v3" / "lgbm.pkl")
with open(MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v3" / "tb_lgbm_widyawardhana_v3_features.json") as f:
    tb_feats = json.load(f)

lstm_model  = load_lstm(MODEL_DIR / "runs" / "tb_lstm_widyawardhana_v1" / "lstm.pt", device="cpu")
lstm_scaler = joblib.load(MODEL_DIR / "runs" / "tb_lstm_widyawardhana_v1" / "lstm_scaler.pkl")
with open(MODEL_DIR / "runs" / "tb_lstm_widyawardhana_v1" / "tb_lstm_widyawardhana_v1_features.json") as f:
    lstm_feats = json.load(f)

tbg_model  = joblib.load(MODEL_DIR / "runs" / "tb_guardian_widyawardhana_v2" / "guardian.pkl")
tbg_scaler = joblib.load(MODEL_DIR / "runs" / "tb_guardian_widyawardhana_v2" / "guardian_scaler.pkl")
with open(MODEL_DIR / "runs" / "tb_guardian_widyawardhana_v2" / "tb_guardian_widyawardhana_v2_feature_cols.json") as f:
    tbg_all_feats = json.load(f)

tbg_static    = [f for f in tbg_all_feats if f not in DYNAMIC_NAMES]
tbg_static_map = {name: i for i, name in enumerate(tbg_static)}
tbg_order      = [
    ("static", tbg_static_map[f]) if f in tbg_static_map else ("dyn", f)
    for f in tbg_all_feats
]

SEQ_LEN = 16
logger.info(f"Models loaded | tb({len(tb_feats)}f) lstm({len(lstm_feats)}f) gdn({len(tbg_all_feats)}f)")

# ── Sweep dimensions ───────────────────────────────────────────────────────────

# State mapping: 0=TRENDING_DOWN, 1=RANGING_LOW_VOL, 2=RANGING_HIGH_VOL, 3=TRENDING_UP
# Format: (trending_thr, ranging_thr)
LGBM_CONFIGS = {
    "T040_R055": {0: 0.40, 1: 0.55, 2: 0.55, 3: 0.40},
    "T041_R053": {0: 0.41, 1: 0.53, 2: 0.53, 3: 0.41},
    "T042_R050": {0: 0.42, 1: 0.50, 2: 0.50, 3: 0.42},
    "T042_R052": {0: 0.42, 1: 0.52, 2: 0.52, 3: 0.42},  # best dari sweep 08
    "T042_R054": {0: 0.42, 1: 0.54, 2: 0.54, 3: 0.42},
    "T044_R052": {0: 0.44, 1: 0.52, 2: 0.52, 3: 0.44},
    "T045_R050": {0: 0.45, 1: 0.50, 2: 0.50, 3: 0.45},  # hmm_current baseline
    "T045_R052": {0: 0.45, 1: 0.52, 2: 0.52, 3: 0.45},
}

GDN_THR_SWEEP   = [0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75]
GDN_HOLD_SWEEP  = [1, 2, 3, 4]

# Fixed LSTM config (best dari sweep 08)
LSTM_SOFT_ALPHA  = 0.5   # soft_mul exponent: scale = (p_agree/0.333)^alpha
LSTM_REGIME_APPLY = {0: False, 1: True, 2: True, 3: False}  # skip_trending


# ── LSTM inference ─────────────────────────────────────────────────────────────
def lstm_predict_proba(X_raw: np.ndarray) -> np.ndarray:
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


# ── Entry filter ───────────────────────────────────────────────────────────────
def apply_entry_filter(p_lgbm, p_lstm, hmm, hmm_config):
    """HMM-aware LGBM threshold + soft_mul LSTM (skip in TRENDING)."""
    n    = len(hmm)
    conf = np.max(p_lgbm, axis=1)
    yp   = np.ones(n, dtype=np.int32)

    for state, thr in hmm_config.items():
        mask = (hmm == state)
        yp[mask & (p_lgbm[:, 2] >= thr)] = LONG
        yp[mask & (p_lgbm[:, 0] >= thr) & (yp != LONG)] = SHORT

    # soft_mul LSTM — apply hanya di RANGING (states 1, 2)
    sig_idx = np.where(yp != FLAT)[0]
    for i in sig_idx:
        state = int(hmm[i])
        if not LSTM_REGIME_APPLY.get(state, True):
            continue
        lgbm_dir = yp[i]
        p_agree  = p_lstm[i, lgbm_dir]
        scale    = (p_agree / 0.3333) ** LSTM_SOFT_ALPHA
        if conf[i] * scale < hmm_config[state]:
            yp[i] = FLAT

    return yp


# ── Pre-build static column indices (computed once at module load) ─────────────
_STATIC_COLS = {key: idx for idx, (src, key) in enumerate(tbg_order) if src == "static"}
_DYN_COLS    = {key: idx for idx, (src, key) in enumerate(tbg_order) if src == "dyn"}
_N_FEAT      = len(tbg_order)


def _build_guardian_batch(j_arr, i, close, atr, direction, X_tbg):
    """
    Build Guardian feature matrix for all bars in j_arr at once.
    Returns ndarray (len(j_arr), n_features).
    Batch approach: one predict_proba call per trade instead of per bar.
    """
    k     = len(j_arr)
    rows  = np.zeros((k, _N_FEAT), dtype=np.float64)
    entry = close[i]
    atr_i = atr[i]
    atr_pct = atr_i / entry if entry > 0 else 0.01

    # Static features — index directly into X_tbg
    for col_name, feat_idx in _STATIC_COLS.items():
        rows[:, feat_idx] = X_tbg[j_arr, col_name]

    # Dynamic features — vectorized
    pnl   = (close[j_arr] - entry) / entry * direction          # (k,)
    bars  = (j_arr - i).astype(np.float64)                      # (k,)
    # max_fav: running max of pnl from bar i+1 to each j
    all_j = np.arange(i + 1, j_arr[-1] + 1)
    all_r = (close[all_j] - entry) / entry * direction
    all_mx = np.maximum.accumulate(np.concatenate([[0.0], all_r]))  # (len(all_j)+1,)
    # max_fav at j = all_mx[j - i]
    max_fav = all_mx[j_arr - i]                                  # (k,)
    dd      = np.where(max_fav > 0.001, (max_fav - pnl) / max_fav, 0.0)

    for col_name, feat_idx in _DYN_COLS.items():
        if col_name == "bars_held_norm":
            rows[:, feat_idx] = bars / MAX_HOLD
        elif col_name == "current_pnl_pct":
            rows[:, feat_idx] = pnl
        elif col_name == "current_pnl_atr":
            rows[:, feat_idx] = pnl / atr_pct if atr_pct > 0 else 0.0
        elif col_name == "max_favorable_pnl_pct":
            rows[:, feat_idx] = max_fav
        elif col_name == "drawdown_from_peak_pct":
            rows[:, feat_idx] = dd
        elif col_name == "direction":
            rows[:, feat_idx] = float(direction)
        elif col_name == "entry_price_ratio":
            rows[:, feat_idx] = entry / np.where(close[j_arr] > 0, close[j_arr], 1.0)

    return rows


# ── Simulate (batch Guardian per trade) ────────────────────────────────────────
def simulate(yp, close, high, low, atr, X_tbg, gdn_thr, gdn_min_hold):
    n = len(yp); trades = []; i = 0
    exit_col = 2 if tbg_model.n_classes_ > 2 else 1
    while i < n:
        sig = yp[i]
        if sig == FLAT:
            i += 1; continue
        direction  = 1 if sig == LONG else -1
        entry      = close[i]
        sl_price   = entry - direction * SL_MULT * atr[i]
        exit_price = close[min(i + MAX_HOLD, n - 1)]
        exit_bar   = min(i + MAX_HOLD, n - 1)
        outcome    = "TIME_EXIT"

        # Find SL bar first (cheap, no model call)
        sl_bar = None
        for j in range(i + 1, min(i + MAX_HOLD + 1, n)):
            if (direction == 1 and low[j] <= sl_price) or \
               (direction == -1 and high[j] >= sl_price):
                sl_bar = j
                break

        # Guardian: batch predict all bars from min_hold to SL (or max_hold)
        gdn_end  = sl_bar if sl_bar is not None else min(i + MAX_HOLD + 1, n)
        gdn_start = i + gdn_min_hold
        gdn_bar  = None

        if gdn_start < gdn_end:
            j_arr = np.arange(gdn_start, gdn_end, dtype=np.int64)
            rows  = _build_guardian_batch(j_arr, i, close, atr, direction, X_tbg)
            probs = tbg_model.predict_proba(tbg_scaler.transform(rows))
            hits  = np.where(probs[:, exit_col] >= gdn_thr)[0]
            if len(hits):
                gdn_bar = int(j_arr[hits[0]])

        # Resolve exit priority: SL < Guardian < Time
        if sl_bar is not None and (gdn_bar is None or sl_bar <= gdn_bar):
            exit_price, exit_bar, outcome = sl_price, sl_bar, "SL"
        elif gdn_bar is not None:
            exit_price, exit_bar, outcome = close[gdn_bar], gdn_bar, "GUARDIAN_EXIT"

        net_pnl = (exit_price - entry) / entry * direction * MODAL * LEVERAGE \
                  - COST_RT * MODAL * LEVERAGE
        trades.append({
            "win" : net_pnl > 0,
            "pnl" : net_pnl,
            "bars": exit_bar - i,
            "long": sig == LONG,
            "sl"  : outcome == "SL",
            "gdn" : "GUARDIAN" in outcome,
        })
        i = exit_bar + 1
    return trades


# ── Load all coins once ────────────────────────────────────────────────────────
def load_all_coins():
    coins = {}
    for sym in ALL_COINS:
        fpath = HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet"
        if not fpath.exists():
            continue
        df  = pd.read_parquet(fpath)
        df  = ensure_utc_index(df).sort_index()

        rp  = HOLDOUT_DIR / "labeled" / f"{sym}_regime_h1.parquet"
        hmm = np.full(len(df), 1, np.int32)
        if rp.exists():
            reg = pd.read_parquet(rp)
            if "hmm_regime_enc" in reg.columns:
                hmm = reg["hmm_regime_enc"].reindex(df.index, fill_value=1).values.astype(np.int32)

        LM   = {"SHORT": 0, "FLAT": 1, "LONG": 2}
        mask = df["label"].isin(LM)
        df   = df[mask].copy()
        hmm  = hmm[mask.values]
        n    = len(df)

        close = df["close"].values.astype(np.float64)
        high  = df["high"].values.astype(np.float64)
        low   = df["low"].values.astype(np.float64)
        atr   = df["atr_14_h1"].values.astype(np.float64)

        X_tb = np.zeros((n, len(tb_feats)), dtype=np.float64)
        for idx, c in enumerate(tb_feats):
            if c in df.columns:
                X_tb[:, idx] = df[c].ffill().fillna(0).values
        p_lgbm = tb_model.predict_proba(X_tb)

        X_lstm = np.zeros((n, len(lstm_feats)), dtype=np.float32)
        for idx, c in enumerate(lstm_feats):
            if c in df.columns:
                X_lstm[:, idx] = df[c].ffill().fillna(0).values.astype(np.float32)
        p_lstm = lstm_predict_proba(X_lstm)

        X_tbg = np.zeros((n, len(tbg_static)), dtype=np.float64)
        for idx, c in enumerate(tbg_static):
            if c in df.columns:
                X_tbg[:, idx] = df[c].ffill().fillna(0).values

        coins[sym] = dict(close=close, high=high, low=low, atr=atr,
                          p_lgbm=p_lgbm, p_lstm=p_lstm, X_tbg=X_tbg, hmm=hmm, n=n)
    return coins


# ── Main sweep ─────────────────────────────────────────────────────────────────
def run_sweep(coin_data):
    combos = list(itertools.product(
        LGBM_CONFIGS.keys(),
        GDN_THR_SWEEP,
        GDN_HOLD_SWEEP,
    ))
    logger.info(f"Total combinations: {len(combos)} "
                f"({len(LGBM_CONFIGS)} LGBM x {len(GDN_THR_SWEEP)} GDN_THR x {len(GDN_HOLD_SWEEP)} MIN_HOLD)")

    results = []
    for ci, (lgbm_name, gdn_thr, gdn_hold) in enumerate(combos):
        hmm_config = LGBM_CONFIGS[lgbm_name]

        agg = {"trades": 0, "wins": 0, "pnl": 0.0,
               "longs": 0, "long_wins": 0, "sl": 0, "gdn": 0, "bars": []}

        for sym, d in coin_data.items():
            yp     = apply_entry_filter(d["p_lgbm"], d["p_lstm"], d["hmm"], hmm_config)
            trades = simulate(yp, d["close"], d["high"], d["low"], d["atr"],
                              d["X_tbg"], gdn_thr=gdn_thr, gdn_min_hold=gdn_hold)
            for t in trades:
                agg["trades"] += 1
                agg["pnl"]    += t["pnl"]
                agg["bars"].append(t["bars"])
                if t["win"]:       agg["wins"]      += 1
                if t["long"]:      agg["longs"]     += 1
                if t["long"] and t["win"]: agg["long_wins"] += 1
                if t["sl"]:        agg["sl"]        += 1
                if t["gdn"]:       agg["gdn"]       += 1

        n      = agg["trades"]
        wr     = agg["wins"]  / n * 100 if n > 0 else 0.0
        lwr    = agg["long_wins"] / agg["longs"] * 100 if agg["longs"] > 0 else 0.0
        sl_r   = agg["sl"]    / n * 100 if n > 0 else 0.0
        gdn_r  = agg["gdn"]   / n * 100 if n > 0 else 0.0
        avg_h  = float(np.mean(agg["bars"])) if agg["bars"] else 0.0

        results.append({
            "lgbm_config" : lgbm_name,
            "gdn_thr"     : gdn_thr,
            "gdn_min_hold": gdn_hold,
            "trades"      : n,
            "wr_pct"      : round(wr, 1),
            "long_wr_pct" : round(lwr, 1),
            "long_pct"    : round(agg["longs"] / n * 100, 1) if n > 0 else 0.0,
            "sl_pct"      : round(sl_r, 1),
            "gdn_pct"     : round(gdn_r, 1),
            "avg_hold"    : round(avg_h, 1),
            "net_pnl"     : round(agg["pnl"], 2),
            "pnl_per_trade": round(agg["pnl"] / n, 3) if n > 0 else 0.0,
        })

        if (ci + 1) % 30 == 0:
            best_so_far = max(results, key=lambda x: x["net_pnl"])
            logger.info(f"  [{ci+1}/{len(combos)}] {lgbm_name}|gdn={gdn_thr}|hold={gdn_hold}"
                        f" -> t={n} WR={wr:.1f}% PnL=${agg['pnl']:.0f}"
                        f"  [best so far: ${best_so_far['net_pnl']:.0f}]")

    return results


def print_results(results):
    df = pd.DataFrame(results).sort_values("net_pnl", ascending=False)

    print(f"\n{'='*115}")
    print(f"  TB THRESHOLD SWEEP | Apr-Jun 2026 | 21 koin | $10/trade 5x")
    print(f"  Fixed: LSTM=soft_mul|skip_trending | {len(results)} combos")
    print(f"  8 LGBM configs x 9 Guardian thr x 4 min_hold")
    print(f"{'='*115}")
    print(f"  {'Rank':>4}  {'LGBM Config':<14} {'GDN_Thr':>7} {'MinHold':>7}"
          f"  {'Trades':>6} {'WR%':>5} {'LONG%':>6} {'SL%':>5} {'GDN%':>5}"
          f"  {'AvgHold':>7} {'PnL':>8} {'$/trade':>8}")
    print(f"  {'-'*111}")
    for rank, row in enumerate(df.head(20).itertuples(), 1):
        print(f"  {rank:>4}  {row.lgbm_config:<14} {row.gdn_thr:>7.2f} {row.gdn_min_hold:>7}"
              f"  {row.trades:>6} {row.wr_pct:>5.1f} {row.long_pct:>6.1f}"
              f" {row.sl_pct:>5.1f} {row.gdn_pct:>5.1f}  {row.avg_hold:>7.1f}"
              f" {row.net_pnl:>8.0f} {row.pnl_per_trade:>8.3f}")

    print(f"\n  {'--- Bottom 5 ---':^111}")
    for rank, row in enumerate(df.tail(5).itertuples(), len(df) - 4):
        print(f"  {rank:>4}  {row.lgbm_config:<14} {row.gdn_thr:>7.2f} {row.gdn_min_hold:>7}"
              f"  {row.trades:>6} {row.wr_pct:>5.1f} {row.long_pct:>6.1f}"
              f" {row.sl_pct:>5.1f} {row.gdn_pct:>5.1f}  {row.avg_hold:>7.1f}"
              f" {row.net_pnl:>8.0f} {row.pnl_per_trade:>8.3f}")

    # Summary per dimension
    print(f"\n  --- PnL avg per LGBM config ---")
    for name, grp in df.groupby("lgbm_config"):
        print(f"    {name:<14}  avg=${grp['net_pnl'].mean():>7.0f}  "
              f"max=${grp['net_pnl'].max():>7.0f}  trades_avg={grp['trades'].mean():>5.0f}")

    print(f"\n  --- PnL avg per Guardian threshold ---")
    for thr, grp in df.groupby("gdn_thr"):
        print(f"    gdn_thr={thr:.2f}  avg=${grp['net_pnl'].mean():>7.0f}  "
              f"max=${grp['net_pnl'].max():>7.0f}  trades_avg={grp['trades'].mean():>5.0f}")

    print(f"\n  --- PnL avg per min_hold ---")
    for h, grp in df.groupby("gdn_min_hold"):
        print(f"    min_hold={h}  avg=${grp['net_pnl'].mean():>7.0f}  "
              f"max=${grp['net_pnl'].max():>7.0f}  trades_avg={grp['trades'].mean():>5.0f}")

    best = df.iloc[0]
    print(f"\n  Best PnL  : {best['lgbm_config']} | gdn_thr={best['gdn_thr']}"
          f" | min_hold={best['gdn_min_hold']} = ${best['net_pnl']:.0f}"
          f"  ({best['trades']} trades, WR={best['wr_pct']}%)")
    best_wr = df.sort_values("wr_pct", ascending=False).iloc[0]
    print(f"  Best WR   : {best_wr['lgbm_config']} | gdn_thr={best_wr['gdn_thr']}"
          f" | min_hold={best_wr['gdn_min_hold']} = {best_wr['wr_pct']}%"
          f"  (${best_wr['net_pnl']:.0f})")
    best_ppt = df.sort_values("pnl_per_trade", ascending=False).iloc[0]
    print(f"  Best $/t  : {best_ppt['lgbm_config']} | gdn_thr={best_ppt['gdn_thr']}"
          f" | min_hold={best_ppt['gdn_min_hold']} = ${best_ppt['pnl_per_trade']:.3f}"
          f"  ({best_ppt['trades']} trades)")
    print(f"{'='*115}\n")
    return df


if __name__ == "__main__":
    coin_data = load_all_coins()
    logger.info(f"Loaded {len(coin_data)} coins")
    results = run_sweep(coin_data)
    df      = print_results(results)

    out = MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v3" / "tune_threshold_results.json"
    df.to_json(out, orient="records", indent=2)
    logger.info(f"Saved -> {out}")
