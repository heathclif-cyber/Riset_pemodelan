"""
pipeline/10_eval_guardian_comparison.py
Holdout Apr-Jun 2026 — Guardian MFE vs Guardian Swing H4

Fixed: T042_R052 | LSTM soft_mul | skip_trending
Sweep: Guardian threshold 0.45-0.75 (step 0.05) x min_hold 1-4
Both Guardian models compared side-by-side.
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

logger = setup_logger("10_guardian_cmp")

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

# Fixed config: optimal dari sweep 08+09
HMM_CONFIG      = {0: 0.42, 1: 0.52, 2: 0.52, 3: 0.42}   # T042_R052
LSTM_SOFT_ALPHA = 0.5
LSTM_REGIME_APPLY = {0: False, 1: True, 2: True, 3: False}  # skip_trending
SEQ_LEN = 16

GDN_THR_SWEEP  = [0.45, 0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.68, 0.70, 0.72, 0.75]
GDN_HOLD_SWEEP = [1, 2, 3, 4]

# ── Load entry + LSTM models ───────────────────────────────────────────────────
tb_model = joblib.load(MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v3" / "lgbm.pkl")
with open(MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v3" / "tb_lgbm_widyawardhana_v3_features.json") as f:
    tb_feats = json.load(f)

lstm_model  = load_lstm(MODEL_DIR / "runs" / "tb_lstm_widyawardhana_v1" / "lstm.pt", device="cpu")
lstm_scaler = joblib.load(MODEL_DIR / "runs" / "tb_lstm_widyawardhana_v1" / "lstm_scaler.pkl")
with open(MODEL_DIR / "runs" / "tb_lstm_widyawardhana_v1" / "tb_lstm_widyawardhana_v1_features.json") as f:
    lstm_feats = json.load(f)

logger.info(f"tb({len(tb_feats)}f) lstm({len(lstm_feats)}f)")


def _build_tbg_order(feature_cols):
    static = [f for f in feature_cols if f not in DYNAMIC_NAMES]
    static_map = {name: i for i, name in enumerate(static)}
    return [
        ("static", static_map[f]) if f in static_map else ("dyn", f)
        for f in feature_cols
    ], static_map


def load_guardian(run_name):
    run_dir = MODEL_DIR / "runs" / run_name
    model   = joblib.load(run_dir / "guardian.pkl")
    scaler  = joblib.load(run_dir / "guardian_scaler.pkl")
    fc_path = list(run_dir.glob("*feature_cols.json"))[0]
    with open(fc_path) as f:
        feat_cols = json.load(f)
    tbg_order, _ = _build_tbg_order(feat_cols)
    static_cols = {key: idx for idx, (src, key) in enumerate(tbg_order) if src == "static"}
    dyn_cols    = {key: idx for idx, (src, key) in enumerate(tbg_order) if src == "dyn"}
    n_feat      = len(tbg_order)
    return model, scaler, static_cols, dyn_cols, n_feat


# Load both Guardian models
GDN_MFE  = load_guardian("tb_guardian_widyawardhana_v2")
GDN_SWH4 = load_guardian("tb_guardian_swing_h4_v1")
logger.info(f"Guardians loaded: MFE({GDN_MFE[4]}f) SwingH4({GDN_SWH4[4]}f)")


# ── LSTM inference ─────────────────────────────────────────────────────────────
def lstm_predict_proba(X_raw):
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


def apply_entry_filter(p_lgbm, p_lstm, hmm):
    n  = len(hmm)
    yp = np.ones(n, dtype=np.int32)
    conf = np.max(p_lgbm, axis=1)
    for state, thr in HMM_CONFIG.items():
        mask = (hmm == state)
        yp[mask & (p_lgbm[:, 2] >= thr)] = LONG
        yp[mask & (p_lgbm[:, 0] >= thr) & (yp != LONG)] = SHORT
    sig_idx = np.where(yp != FLAT)[0]
    for i in sig_idx:
        state = int(hmm[i])
        if not LSTM_REGIME_APPLY.get(state, True):
            continue
        lgbm_dir = yp[i]
        p_agree  = p_lstm[i, lgbm_dir]
        scale    = (p_agree / 0.3333) ** LSTM_SOFT_ALPHA
        if conf[i] * scale < HMM_CONFIG[state]:
            yp[i] = FLAT
    return yp


def _build_guardian_batch(j_arr, i, close, atr, direction, X_tbg, static_cols, dyn_cols, n_feat):
    k     = len(j_arr)
    rows  = np.zeros((k, n_feat), dtype=np.float64)
    entry = close[i]
    atr_i = atr[i]
    atr_pct = atr_i / entry if entry > 0 else 0.01

    for col_name, feat_idx in static_cols.items():
        rows[:, feat_idx] = X_tbg[j_arr, col_name]

    pnl     = (close[j_arr] - entry) / entry * direction
    bars    = (j_arr - i).astype(np.float64)
    all_j   = np.arange(i + 1, j_arr[-1] + 1)
    all_r   = (close[all_j] - entry) / entry * direction
    all_mx  = np.maximum.accumulate(np.concatenate([[0.0], all_r]))
    max_fav = all_mx[j_arr - i]
    dd      = np.where(max_fav > 0.001, (max_fav - pnl) / max_fav, 0.0)

    for col_name, feat_idx in dyn_cols.items():
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


def simulate(yp, close, high, low, atr, X_tbg, gdn_model, gdn_scaler,
             static_cols, dyn_cols, n_feat, gdn_thr, gdn_min_hold):
    n = len(yp); trades = []; i = 0
    exit_col = 2 if gdn_model.n_classes_ > 2 else 1
    while i < n:
        sig = yp[i]
        if sig == FLAT:
            i += 1; continue
        direction = 1 if sig == LONG else -1
        entry     = close[i]
        sl_price  = entry - direction * SL_MULT * atr[i]
        exit_price = close[min(i + MAX_HOLD, n - 1)]
        exit_bar   = min(i + MAX_HOLD, n - 1)
        outcome    = "TIME_EXIT"

        sl_bar = None
        for j in range(i + 1, min(i + MAX_HOLD + 1, n)):
            if (direction == 1 and low[j] <= sl_price) or \
               (direction == -1 and high[j] >= sl_price):
                sl_bar = j; break

        gdn_end   = sl_bar if sl_bar is not None else min(i + MAX_HOLD + 1, n)
        gdn_start = i + gdn_min_hold
        gdn_bar   = None

        if gdn_start < gdn_end:
            j_arr = np.arange(gdn_start, gdn_end, dtype=np.int64)
            rows  = _build_guardian_batch(j_arr, i, close, atr, direction, X_tbg,
                                          static_cols, dyn_cols, n_feat)
            probs = gdn_model.predict_proba(gdn_scaler.transform(rows))
            hits  = np.where(probs[:, exit_col] >= gdn_thr)[0]
            if len(hits):
                gdn_bar = int(j_arr[hits[0]])

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


# ── Load holdout coins ─────────────────────────────────────────────────────────
def load_coin(sym):
    p = HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df = ensure_utc_index(df).sort_index()
    if len(df) < 50:
        return None
    rp = HOLDOUT_DIR / "labeled" / f"{sym}_regime_h1.parquet"
    hmm = np.full(len(df), 1, np.int32)
    if rp.exists():
        reg = pd.read_parquet(rp)
        if "hmm_regime_enc" in reg.columns:
            hmm = reg["hmm_regime_enc"].reindex(df.index, fill_value=1).values.astype(np.int32)
    return df, hmm


coins_data = {}
for sym in ALL_COINS:
    res = load_coin(sym)
    if res is not None:
        coins_data[sym] = res
logger.info(f"Loaded {len(coins_data)} coins")


# ── Pre-compute per-coin signals + static Guardian feature matrix ──────────────
def precompute_coin(sym, df, hmm):
    n = len(df)
    close = df["close"].values.astype(np.float64)
    high  = df["high"].values.astype(np.float64) if "high" in df.columns else close
    low   = df["low"].values.astype(np.float64)  if "low"  in df.columns else close
    atr   = df["atr_14_h1"].values.astype(np.float64)

    # LGBM
    X_tb = np.zeros((n, len(tb_feats)), dtype=np.float64)
    for idx, c in enumerate(tb_feats):
        if c in df.columns:
            X_tb[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)
    p_lgbm = tb_model.predict_proba(X_tb)

    # LSTM
    X_lstm = np.zeros((n, len(lstm_feats)), dtype=np.float64)
    for idx, c in enumerate(lstm_feats):
        if c in df.columns:
            X_lstm[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)
    p_lstm = lstm_predict_proba(X_lstm)

    # Entry signal
    yp = apply_entry_filter(p_lgbm, p_lstm, hmm)

    # Guardian static features = same as TB_FEATS (same matrix, same order)
    # Both MFE and SwingH4 Guardian were trained with X_static = X_tb
    X_tbg = X_tb

    return yp, close, high, low, atr, X_tbg


# Pre-compute coin data — both Guardians share same X_tbg (TB_FEATS)
mfe_model, mfe_scaler, mfe_static, mfe_dyn, mfe_nfeat = GDN_MFE
sh4_model, sh4_scaler, sh4_static, sh4_dyn, sh4_nfeat = GDN_SWH4

logger.info("Precomputing coin signals...")
coin_cache = {}
for sym, (df, hmm) in coins_data.items():
    yp, close, high, low, atr, X_tbg = precompute_coin(sym, df, hmm)
    coin_cache[sym] = (yp, close, high, low, atr, X_tbg)


# ── Sweep ──────────────────────────────────────────────────────────────────────
def run_sweep(gdn_model, gdn_scaler, static_cols, dyn_cols, n_feat, label):
    results = []
    combos  = list(itertools.product(GDN_THR_SWEEP, GDN_HOLD_SWEEP))
    total   = len(combos)

    for ci, (gdn_thr, min_hold) in enumerate(combos, 1):
        all_trades = []
        for sym, (yp, close, high, low, atr, X_tbg) in coin_cache.items():
            t = simulate(yp, close, high, low, atr, X_tbg,
                         gdn_model, gdn_scaler, static_cols, dyn_cols, n_feat,
                         gdn_thr, min_hold)
            all_trades.extend(t)

        if not all_trades:
            continue
        t_arr   = all_trades
        n_t     = len(t_arr)
        wins    = sum(1 for t in t_arr if t["win"])
        pnl     = sum(t["pnl"] for t in t_arr)
        wr      = wins / n_t * 100
        avg_b   = np.mean([t["bars"] for t in t_arr])
        gdn_pct = sum(1 for t in t_arr if t["gdn"]) / n_t * 100
        sl_pct  = sum(1 for t in t_arr if t["sl"])  / n_t * 100
        long_pct = sum(1 for t in t_arr if t["long"]) / n_t * 100
        results.append({
            "model"    : label,
            "gdn_thr"  : gdn_thr,
            "min_hold" : min_hold,
            "trades"   : n_t,
            "wr"       : round(wr, 1),
            "long_pct" : round(long_pct, 1),
            "sl_pct"   : round(sl_pct, 1),
            "gdn_pct"  : round(gdn_pct, 1),
            "avg_hold" : round(avg_b, 1),
            "pnl"      : round(pnl, 0),
            "pnl_pt"   : round(pnl / n_t, 3),
        })
        if ci % 10 == 0 or ci == total:
            best = max(results, key=lambda x: x["pnl"])
            logger.info(f"  [{label}] [{ci}/{total}] thr={gdn_thr} hold={min_hold} "
                        f"-> t={n_t} WR={wr:.1f}% PnL=${pnl:.0f}  "
                        f"[best: ${best['pnl']:.0f}]")
    return results


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"\n{'='*100}")
    print(f"  GUARDIAN COMPARISON | Apr-Jun 2026 | 21 koin | T042_R052 | LSTM soft_mul skip_trending")
    print(f"  {len(GDN_THR_SWEEP)} thresholds x {len(GDN_HOLD_SWEEP)} min_hold = {len(GDN_THR_SWEEP)*len(GDN_HOLD_SWEEP)} combos per Guardian")
    print(f"{'='*100}\n")

    print("=== MFE Guardian (heuristic labels) ===")
    res_mfe  = run_sweep(mfe_model, mfe_scaler, mfe_static, mfe_dyn, mfe_nfeat, "MFE")

    print("\n=== Swing H4 Guardian (swing-based labels) ===")
    res_sh4  = run_sweep(sh4_model, sh4_scaler, sh4_static, sh4_dyn, sh4_nfeat, "SwingH4")

    all_res = res_mfe + res_sh4
    all_res.sort(key=lambda x: x["pnl"], reverse=True)

    # ── Print comparison ────────────────────────────────────────────────────────
    print(f"\n{'='*105}")
    print(f"  GUARDIAN COMPARISON RESULTS | Top 30 (both models combined)")
    print(f"{'='*105}")
    print(f"  {'Model':<10} {'GDN_Thr':>7} {'Hold':>5} {'Trades':>7} {'WR%':>6} {'LONG%':>6} "
          f"{'SL%':>5} {'GDN%':>5} {'Hold':>6}  {'PnL':>8}  {'$/t':>7}")
    print(f"  {'-'*103}")
    for r in all_res[:30]:
        print(f"  {r['model']:<10} {r['gdn_thr']:>7.2f} {r['min_hold']:>5} {r['trades']:>7} "
              f"{r['wr']:>6.1f} {r['long_pct']:>6.1f} {r['sl_pct']:>5.1f} {r['gdn_pct']:>5.1f} "
              f"{r['avg_hold']:>6.1f}  ${r['pnl']:>7.0f}  ${r['pnl_pt']:>6.3f}")

    # Per-model summary
    for label, res in [("MFE", res_mfe), ("SwingH4", res_sh4)]:
        best_pnl = max(res, key=lambda x: x["pnl"])
        best_wr  = max(res, key=lambda x: x["wr"])
        print(f"\n  [{label}] Best PnL : thr={best_pnl['gdn_thr']} hold={best_pnl['min_hold']} "
              f"= ${best_pnl['pnl']:.0f}  ({best_pnl['trades']} trades, WR={best_pnl['wr']:.1f}%)")
        print(f"  [{label}] Best WR  : thr={best_wr['gdn_thr']} hold={best_wr['min_hold']} "
              f"= {best_wr['wr']:.1f}%  (${best_wr['pnl']:.0f})")

    # Per-threshold averages
    print(f"\n  --- Avg PnL per threshold (across all min_hold) ---")
    print(f"  {'Thr':>6}  {'MFE avg':>10}  {'SwingH4 avg':>13}  {'Delta':>8}")
    for thr in GDN_THR_SWEEP:
        mfe_avg  = np.mean([r["pnl"] for r in res_mfe  if r["gdn_thr"] == thr])
        sh4_avg  = np.mean([r["pnl"] for r in res_sh4  if r["gdn_thr"] == thr])
        delta    = sh4_avg - mfe_avg
        flag     = "  <-- SwingH4 wins" if delta > 5 else ("  <-- MFE wins" if delta < -5 else "")
        print(f"  {thr:>6.2f}  ${mfe_avg:>8.0f}   ${sh4_avg:>10.0f}   ${delta:>+7.0f}{flag}")

    # Save
    out_path = MODEL_DIR / "runs" / "tb_guardian_swing_h4_v1" / "guardian_comparison_results.json"
    with open(out_path, "w") as f:
        json.dump(all_res, f, indent=2)
    print(f"\n  Saved -> {out_path}")
    print(f"{'='*105}")
