"""
pipeline/11_tune_full_stack.py
Full-stack tuning sweep: HMM/LGBM × LSTM × Guardian (MFE & SwingH4)

Dimensi:
  LGBM configs  : 6  (HMM-aware threshold combos)
  LSTM modes    : 4  (none, soft_mul, flip_p40, flip_hard)
  LSTM regime   : 2  (uniform, skip_trending)
  Guardian model: 2  (MFE, SwingH4)
  Guardian thr  : 7  (0.45–0.70)
  min_hold      : 2  (1, 2)
  ─────────────────────────────
  Total         : 6×4×2×2×7×2 = 1,344 kombinasi

Optimisasi: pre-compute p_lgbm, p_lstm, X_tbg sekali per koin.
Sweep hanya pada simulasi Guardian (batch predict, ~0.02s/koin).

Output: models/runs/tb_lgbm_widyawardhana_v3/tune_full_stack_results.json
"""
import json, sys, warnings, itertools
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime

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

logger = setup_logger("11_full_stack")

SHORT, FLAT, LONG = 0, 1, 2
SL_MULT  = TP_SL_FALLBACK_SL
MAX_HOLD = MAX_HOLDING_BARS
MODAL    = MODAL_PER_TRADE
LEVERAGE = LEVERAGE_SIM[0] if isinstance(LEVERAGE_SIM, list) else LEVERAGE_SIM
COST_RT  = (FEE_PER_SIDE + SLIPPAGE_PER_SIDE) * 2
SEQ_LEN  = 16

DYNAMIC_NAMES = frozenset({
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct",
    "direction", "entry_price_ratio",
})

# ── Sweep dimensions ──────────────────────────────────────────────────────────
LGBM_CONFIGS = {
    "T040_R055": {0: 0.40, 1: 0.55, 2: 0.55, 3: 0.40},
    "T042_R050": {0: 0.42, 1: 0.50, 2: 0.50, 3: 0.42},
    "T042_R052": {0: 0.42, 1: 0.52, 2: 0.52, 3: 0.42},
    "T042_R054": {0: 0.42, 1: 0.54, 2: 0.54, 3: 0.42},
    "T044_R052": {0: 0.44, 1: 0.52, 2: 0.52, 3: 0.44},
    "T045_R052": {0: 0.45, 1: 0.52, 2: 0.52, 3: 0.45},
}
# none=pakai LGBM saja; soft_mul=scale confidence; flip_p40=hard veto <40%; flip_hard=veto opposite
LSTM_MODES   = ["none", "soft_mul", "flip_p40", "flip_hard"]
LSTM_REGIMES = {
    "uniform":      {0: True,  1: True,  2: True,  3: True},
    "skip_trending":{0: False, 1: True,  2: True,  3: False},
}
GUARDIANS = {
    "MFE":     ("tb_guardian_widyawardhana_v2",  "MFE"),
    "SwingH4": ("tb_guardian_swing_h4_v1",       "SwingH4"),
}
GDN_THR_SWEEP  = [0.45, 0.50, 0.55, 0.58, 0.62, 0.65, 0.70]
GDN_HOLD_SWEEP = [1, 2]

LSTM_SOFT_ALPHA = 0.5

# ── Load models ───────────────────────────────────────────────────────────────
tb_model = joblib.load(MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v3" / "lgbm.pkl")
with open(MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v3" / "tb_lgbm_widyawardhana_v3_features.json") as f:
    tb_feats = json.load(f)

lstm_model  = load_lstm(MODEL_DIR / "runs" / "tb_lstm_widyawardhana_v1" / "lstm.pt", device="cpu")
lstm_scaler = joblib.load(MODEL_DIR / "runs" / "tb_lstm_widyawardhana_v1" / "lstm_scaler.pkl")
with open(MODEL_DIR / "runs" / "tb_lstm_widyawardhana_v1" / "tb_lstm_widyawardhana_v1_features.json") as f:
    lstm_feats = json.load(f)

logger.info(f"LGBM({len(tb_feats)}f) LSTM({len(lstm_feats)}f)")


def load_guardian(run_name):
    run_dir = MODEL_DIR / "runs" / run_name
    model   = joblib.load(run_dir / "guardian.pkl")
    scaler  = joblib.load(run_dir / "guardian_scaler.pkl")
    fc_path = list(run_dir.glob("*feature_cols.json"))[0]
    with open(fc_path) as f:
        feat_cols = json.load(f)
    static = [c for c in feat_cols if c not in DYNAMIC_NAMES]
    static_map = {name: i for i, name in enumerate(static)}
    order = [("static", static_map[f]) if f in static_map else ("dyn", f) for f in feat_cols]
    sc = {key: idx for idx, (src, key) in enumerate(order) if src == "static"}
    dc = {key: idx for idx, (src, key) in enumerate(order) if src == "dyn"}
    return model, scaler, sc, dc, len(order)


gdn_models = {k: load_guardian(v[0]) for k, v in GUARDIANS.items()}
logger.info(f"Guardians: {', '.join(f'{k}({gdn_models[k][4]}f)' for k in gdn_models)}")


# ── LSTM inference ────────────────────────────────────────────────────────────
def lstm_predict_proba(X_raw):
    n, f  = X_raw.shape
    X_sc  = lstm_scaler.transform(X_raw.reshape(-1, f)).reshape(n, f).astype(np.float32)
    probs = np.full((n, 3), 1.0 / 3.0, dtype=np.float32)
    if n < SEQ_LEN:
        return probs
    seqs  = np.stack([X_sc[i - SEQ_LEN + 1: i + 1] for i in range(SEQ_LEN - 1, n)])
    parts = []
    with torch.no_grad():
        for b in range(0, len(seqs), 512):
            t = torch.from_numpy(seqs[b: b + 512])
            parts.append(torch.softmax(lstm_model(t), dim=1).cpu().numpy())
    probs[SEQ_LEN - 1:] = np.concatenate(parts)
    return probs


def apply_entry_filter(p_lgbm, p_lstm, hmm, lgbm_cfg, lstm_mode, lstm_regime_map):
    n    = len(hmm)
    conf = np.max(p_lgbm, axis=1)
    yp   = np.ones(n, dtype=np.int32)

    for state, thr in lgbm_cfg.items():
        mask = (hmm == state)
        yp[mask & (p_lgbm[:, LONG]  >= thr)] = LONG
        yp[mask & (p_lgbm[:, SHORT] >= thr) & (yp != LONG)] = SHORT

    if lstm_mode == "none":
        return yp

    sig_idx = np.where(yp != FLAT)[0]
    for i in sig_idx:
        state = int(hmm[i])
        if not lstm_regime_map.get(state, True):
            continue
        thr = lgbm_cfg[state]
        lgbm_dir = yp[i]

        if lstm_mode == "soft_mul":
            p_agree = p_lstm[i, lgbm_dir]
            scale   = (p_agree / 0.3333) ** LSTM_SOFT_ALPHA
            if conf[i] * scale < thr:
                yp[i] = FLAT

        elif lstm_mode == "flip_p40":
            p_opp = p_lstm[i, SHORT if lgbm_dir == LONG else LONG]
            if p_opp > 0.40:
                yp[i] = FLAT

        elif lstm_mode == "flip_hard":
            if np.argmax(p_lstm[i]) not in (lgbm_dir, FLAT):
                yp[i] = FLAT

    return yp


# ── Guardian batch builder ────────────────────────────────────────────────────
def _build_guardian_batch(j_arr, i, close, atr, direction, X_tbg, sc, dc, n_feat):
    k    = len(j_arr)
    rows = np.zeros((k, n_feat), dtype=np.float64)
    entry = close[i]; atr_pct = atr[i] / entry if entry > 0 else 0.01

    for col_name, feat_idx in sc.items():
        rows[:, feat_idx] = X_tbg[j_arr, col_name]

    pnl   = (close[j_arr] - entry) / entry * direction
    bars  = (j_arr - i).astype(np.float64)
    all_j = np.arange(i + 1, j_arr[-1] + 1)
    all_r = (close[all_j] - entry) / entry * direction
    all_mx = np.maximum.accumulate(np.concatenate([[0.0], all_r]))
    max_fav = all_mx[j_arr - i]
    dd      = np.where(max_fav > 0.001, (max_fav - pnl) / max_fav, 0.0)

    for col_name, feat_idx in dc.items():
        if col_name == "bars_held_norm":           rows[:, feat_idx] = bars / MAX_HOLD
        elif col_name == "current_pnl_pct":        rows[:, feat_idx] = pnl
        elif col_name == "current_pnl_atr":        rows[:, feat_idx] = pnl / atr_pct if atr_pct else 0
        elif col_name == "max_favorable_pnl_pct":  rows[:, feat_idx] = max_fav
        elif col_name == "drawdown_from_peak_pct": rows[:, feat_idx] = dd
        elif col_name == "direction":              rows[:, feat_idx] = float(direction)
        elif col_name == "entry_price_ratio":
            rows[:, feat_idx] = entry / np.where(close[j_arr] > 0, close[j_arr], 1.0)
    return rows


def simulate(yp, close, high, low, atr, X_tbg, gdn_mdl, gdn_scl, sc, dc, n_feat, gdn_thr, min_hold):
    n = len(yp); trades = []; i = 0
    exit_col = 2 if gdn_mdl.n_classes_ > 2 else 1
    while i < n:
        sig = yp[i]
        if sig == FLAT: i += 1; continue
        direction  = 1 if sig == LONG else -1
        entry      = close[i]
        sl_price   = entry - direction * SL_MULT * atr[i]
        exit_bar   = min(i + MAX_HOLD, n - 1)
        outcome    = "TIME_EXIT"

        sl_bar = None
        for j in range(i + 1, min(i + MAX_HOLD + 1, n)):
            if (direction == 1 and low[j] <= sl_price) or \
               (direction == -1 and high[j] >= sl_price):
                sl_bar = j; break

        gdn_end   = sl_bar if sl_bar is not None else min(i + MAX_HOLD + 1, n)
        gdn_start = i + min_hold
        gdn_bar   = None
        if gdn_start < gdn_end:
            j_arr = np.arange(gdn_start, gdn_end, dtype=np.int64)
            rows  = _build_guardian_batch(j_arr, i, close, atr, direction, X_tbg, sc, dc, n_feat)
            probs = gdn_mdl.predict_proba(gdn_scl.transform(rows))
            hits  = np.where(probs[:, exit_col] >= gdn_thr)[0]
            if len(hits): gdn_bar = int(j_arr[hits[0]])

        if sl_bar is not None and (gdn_bar is None or sl_bar <= gdn_bar):
            exit_price, exit_bar, outcome = sl_price, sl_bar, "SL"
        elif gdn_bar is not None:
            exit_price, exit_bar, outcome = close[gdn_bar], gdn_bar, "GUARDIAN_EXIT"
        else:
            exit_price = close[exit_bar]

        net_pnl = (exit_price - entry) / entry * direction * MODAL * LEVERAGE \
                  - COST_RT * MODAL * LEVERAGE
        trades.append({
            "win": net_pnl > 0, "pnl": net_pnl, "bars": exit_bar - i,
            "long": sig == LONG, "sl": outcome == "SL", "gdn": "GUARDIAN" in outcome,
        })
        i = exit_bar + 1
    return trades


def summarize(trades):
    if not trades: return None
    n = len(trades)
    return {
        "trades"  : n,
        "wr"      : round(sum(t["win"] for t in trades) / n * 100, 1),
        "long_pct": round(sum(t["long"] for t in trades) / n * 100, 1),
        "sl_pct"  : round(sum(t["sl"]   for t in trades) / n * 100, 1),
        "gdn_pct" : round(sum(t["gdn"]  for t in trades) / n * 100, 1),
        "avg_hold": round(np.mean([t["bars"] for t in trades]), 1),
        "pnl"     : round(sum(t["pnl"] for t in trades), 0),
        "pnl_pt"  : round(sum(t["pnl"] for t in trades) / n, 3),
    }


# ── Load holdout data ─────────────────────────────────────────────────────────
logger.info("Loading holdout coins...")
raw_coins = {}
for sym in ALL_COINS:
    p = HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet"
    if not p.exists(): continue
    df = pd.read_parquet(p)
    df = ensure_utc_index(df).sort_index()
    if len(df) < 50: continue
    rp = HOLDOUT_DIR / "labeled" / f"{sym}_regime_h1.parquet"
    hmm = np.full(len(df), 1, np.int32)
    if rp.exists():
        reg = pd.read_parquet(rp)
        if "hmm_regime_enc" in reg.columns:
            hmm = reg["hmm_regime_enc"].reindex(df.index, fill_value=1).values.astype(np.int32)
    raw_coins[sym] = (df, hmm)
logger.info(f"Loaded {len(raw_coins)} coins")

# Pre-compute LGBM + LSTM probabilities and Guardian feature matrix per coin
logger.info("Pre-computing LGBM/LSTM probabilities per coin...")
coin_probs = {}
for sym, (df, hmm) in raw_coins.items():
    n = len(df)
    close = df["close"].values.astype(np.float64)
    high  = df["high"].values.astype(np.float64) if "high" in df.columns else close
    low   = df["low"].values.astype(np.float64)  if "low"  in df.columns else close
    atr   = df["atr_14_h1"].values.astype(np.float64)

    X_tb = np.zeros((n, len(tb_feats)), dtype=np.float64)
    for idx, c in enumerate(tb_feats):
        if c in df.columns:
            X_tb[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)
    p_lgbm = tb_model.predict_proba(X_tb)

    X_lstm = np.zeros((n, len(lstm_feats)), dtype=np.float64)
    for idx, c in enumerate(lstm_feats):
        if c in df.columns:
            X_lstm[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)
    p_lstm = lstm_predict_proba(X_lstm)

    coin_probs[sym] = (p_lgbm, p_lstm, hmm, close, high, low, atr, X_tb)

logger.info(f"Pre-compute done. Starting sweep...")


# ── Main sweep ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    all_results = []
    total_signal_combos = len(LGBM_CONFIGS) * len(LSTM_MODES) * len(LSTM_REGIMES)
    total_gdn_combos    = len(GUARDIANS) * len(GDN_THR_SWEEP) * len(GDN_HOLD_SWEEP)
    total_combos        = total_signal_combos * total_gdn_combos

    print(f"\n{'='*110}")
    print(f"  TB WIDYAWARDHANA — FULL STACK SWEEP | Apr-Jun 2026 | 21 koin | $10/trade 5x")
    print(f"  {len(LGBM_CONFIGS)} LGBM  x  {len(LSTM_MODES)} LSTM  x  {len(LSTM_REGIMES)} REGIME  "
          f"x  {len(GUARDIANS)} Guardian  x  {len(GDN_THR_SWEEP)} thr  x  {len(GDN_HOLD_SWEEP)} hold"
          f"  =  {total_combos} kombinasi")
    print(f"{'='*110}\n")

    combo_n = 0
    for lgbm_name, lgbm_cfg in LGBM_CONFIGS.items():
        for lstm_mode in LSTM_MODES:
            for regime_name, lstm_regime_map in LSTM_REGIMES.items():

                # Build signals for all coins under this LGBM×LSTM×regime config
                coin_signals = {}
                for sym, (p_lgbm, p_lstm, hmm, close, high, low, atr, X_tbg) in coin_probs.items():
                    yp = apply_entry_filter(p_lgbm, p_lstm, hmm, lgbm_cfg, lstm_mode, lstm_regime_map)
                    coin_signals[sym] = (yp, close, high, low, atr, X_tbg)

                # Sweep Guardian combos for this signal
                for gdn_key, (gdn_run, _) in GUARDIANS.items():
                    gdn_mdl, gdn_scl, sc, dc, n_feat = gdn_models[gdn_key]

                    for gdn_thr in GDN_THR_SWEEP:
                        for min_hold in GDN_HOLD_SWEEP:
                            combo_n += 1
                            all_trades = []
                            for sym, (yp, close, high, low, atr, X_tbg) in coin_signals.items():
                                t = simulate(yp, close, high, low, atr, X_tbg,
                                             gdn_mdl, gdn_scl, sc, dc, n_feat,
                                             gdn_thr, min_hold)
                                all_trades.extend(t)

                            s = summarize(all_trades)
                            if s is None: continue
                            s.update({
                                "lgbm"      : lgbm_name,
                                "lstm"      : lstm_mode,
                                "regime"    : regime_name,
                                "guardian"  : gdn_key,
                                "gdn_thr"   : gdn_thr,
                                "min_hold"  : min_hold,
                            })
                            all_results.append(s)

                            if combo_n % 100 == 0:
                                best = max(all_results, key=lambda x: x["pnl"])
                                logger.info(
                                    f"  [{combo_n}/{total_combos}] "
                                    f"{lgbm_name}|{lstm_mode}|{regime_name}|"
                                    f"{gdn_key}|thr={gdn_thr}|h={min_hold} "
                                    f"-> t={s['trades']} WR={s['wr']}% PnL=${s['pnl']:.0f}  "
                                    f"[best: ${best['pnl']:.0f}]"
                                )

    # ── Sort & print ──────────────────────────────────────────────────────────
    all_results.sort(key=lambda x: x["pnl"], reverse=True)

    print(f"\n{'='*115}")
    print(f"  FULL STACK SWEEP RESULTS | Top 30")
    print(f"{'='*115}")
    print(f"  {'LGBM':<12} {'LSTM':<10} {'REGIME':<14} {'GDN':<10} {'THR':>5} {'H':>2}"
          f"  {'T':>6} {'WR%':>6} {'L%':>5} {'SL%':>5} {'GDN%':>5} {'Hld':>5}  {'PnL':>8}  {'$/t':>7}")
    print(f"  {'-'*113}")
    for r in all_results[:30]:
        print(f"  {r['lgbm']:<12} {r['lstm']:<10} {r['regime']:<14} {r['guardian']:<10} "
              f"{r['gdn_thr']:>5.2f} {r['min_hold']:>2}"
              f"  {r['trades']:>6} {r['wr']:>6.1f} {r['long_pct']:>5.1f} {r['sl_pct']:>5.1f} "
              f"{r['gdn_pct']:>5.1f} {r['avg_hold']:>5.1f}  ${r['pnl']:>7.0f}  ${r['pnl_pt']:>6.3f}")

    print(f"\n  --- Bottom 5 ---")
    for r in all_results[-5:]:
        print(f"  {r['lgbm']:<12} {r['lstm']:<10} {r['regime']:<14} {r['guardian']:<10} "
              f"{r['gdn_thr']:>5.2f} {r['min_hold']:>2}  ${r['pnl']:>7.0f}")

    # ── Per-dimension averages ────────────────────────────────────────────────
    def dim_avg(key, vals):
        print(f"\n  --- Avg PnL per {key} ---")
        rows = []
        for v in vals:
            sub = [r for r in all_results if r[key] == v]
            if sub:
                rows.append((v, np.mean([r["pnl"] for r in sub]),
                               max(r["pnl"] for r in sub),
                               np.mean([r["wr"] for r in sub])))
        rows.sort(key=lambda x: x[1], reverse=True)
        for v, avg, mx, wr in rows:
            print(f"    {str(v):<20}  avg=${avg:>7.0f}  max=${mx:>7.0f}  avg_wr={wr:.1f}%")

    dim_avg("lgbm",     list(LGBM_CONFIGS.keys()))
    dim_avg("lstm",     LSTM_MODES)
    dim_avg("regime",   list(LSTM_REGIMES.keys()))
    dim_avg("guardian", list(GUARDIANS.keys()))
    dim_avg("gdn_thr",  GDN_THR_SWEEP)
    dim_avg("min_hold", GDN_HOLD_SWEEP)

    # ── Best per guardian model ───────────────────────────────────────────────
    print(f"\n  --- Best config per Guardian model ---")
    for gdn_key in GUARDIANS:
        sub = [r for r in all_results if r["guardian"] == gdn_key]
        if not sub: continue
        best_pnl = max(sub, key=lambda x: x["pnl"])
        best_wr  = max(sub, key=lambda x: x["wr"])
        print(f"\n  [{gdn_key}] Best PnL: {best_pnl['lgbm']}|{best_pnl['lstm']}|"
              f"{best_pnl['regime']}|thr={best_pnl['gdn_thr']}|h={best_pnl['min_hold']} "
              f"= ${best_pnl['pnl']:.0f} ({best_pnl['trades']} trades, WR={best_pnl['wr']:.1f}%)")
        print(f"  [{gdn_key}] Best WR : {best_wr['lgbm']}|{best_wr['lstm']}|"
              f"{best_wr['regime']}|thr={best_wr['gdn_thr']}|h={best_wr['min_hold']} "
              f"= {best_wr['wr']:.1f}% (${best_wr['pnl']:.0f})")

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v3" / "tune_full_stack_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "meta": {
                "n_combos": total_combos, "n_results": len(all_results),
                "run_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "holdout": "Apr-Jun 2026",
                "lgbm_model": "tb_lgbm_widyawardhana_v3 (purge=36)",
                "guardians": list(GUARDIANS.keys()),
            },
            "results": all_results,
        }, f, indent=2)
    logger.info(f"Saved -> {out_path}")
    print(f"\n  Saved -> {out_path}")
    print(f"{'='*115}")
