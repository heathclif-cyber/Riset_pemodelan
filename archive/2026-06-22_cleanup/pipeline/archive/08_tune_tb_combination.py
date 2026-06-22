"""
pipeline/08_tune_tb_combination.py
Grid sweep: TB LGBM x LSTM filter mode x HMM regime threshold config

84 kombinasi = 4 HMM configs x 7 LSTM modes x 3 regime-conditionality

Models (TB system only, ic32 TIDAK disentuh):
  LGBM     : tb_lgbm_widyawardhana_v3
  LSTM     : tb_lstm_widyawardhana_v1
  HMM      : regime dari holdout labeled/{coin}_regime_h1.parquet
  Guardian : tb_guardian_widyawardhana_v2 (threshold fixed 0.65)

Exit: SL=1.5xATR | max_hold=36bar | NO TP | Guardian early exit
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

logger = setup_logger("08_tune_tb")

SHORT, FLAT, LONG = 0, 1, 2
SL_MULT  = TP_SL_FALLBACK_SL
MAX_HOLD = MAX_HOLDING_BARS
MODAL    = MODAL_PER_TRADE
LEVERAGE = LEVERAGE_SIM[0] if isinstance(LEVERAGE_SIM, list) else LEVERAGE_SIM
COST_RT  = (FEE_PER_SIDE + SLIPPAGE_PER_SIDE) * 2

GDN_THR     = 0.65
GDN_MIN_HOLD = 2

DYNAMIC_NAMES = frozenset({
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct",
    "direction", "entry_price_ratio",
})

# ── Load TB models ─────────────────────────────────────────────────────────────
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

tbg_static = [f for f in tbg_all_feats if f not in DYNAMIC_NAMES]
tbg_static_map = {name: i for i, name in enumerate(tbg_static)}
tbg_order = [
    ("static", tbg_static_map[f]) if f in tbg_static_map else ("dyn", f)
    for f in tbg_all_feats
]

SEQ_LEN = 16

logger.info(f"tb_lgbm({len(tb_feats)}f) | tb_lstm({len(lstm_feats)}f,seq={SEQ_LEN}) | tb_gdn({len(tbg_all_feats)}f)")

# ── Sweep configurations ───────────────────────────────────────────────────────

# Dim 1: HMM regime -> LGBM threshold per state
# State: 0=TRENDING_DOWN, 1=RANGING_LOW_VOL, 2=RANGING_HIGH_VOL, 3=TRENDING_UP
HMM_CONFIGS = {
    "hmm_flat":    {0: 0.50, 1: 0.50, 2: 0.50, 3: 0.50},
    "hmm_current": {0: 0.45, 1: 0.50, 2: 0.50, 3: 0.45},
    "hmm_v2":      {0: 0.42, 1: 0.52, 2: 0.52, 3: 0.42},
    "hmm_v3":      {0: 0.40, 1: 0.55, 2: 0.55, 3: 0.40},
}

# Dim 2: LSTM entry filter mode
LSTM_MODES = [
    "none",       # no LSTM filtering
    "flip_hard",  # veto if LSTM argmax == opposite (current)
    "flip_p40",   # veto only if p_opposite >= 0.40
    "flip_p45",   # veto only if p_opposite >= 0.45
    "flip_p50",   # veto only if p_opposite >= 0.50
    "consensus",  # only enter if LSTM argmax == LGBM direction
    "soft_mul",   # multiply LGBM confidence by LSTM alignment factor
]

# Dim 3: LSTM regime-conditionality
# Which HMM states apply LSTM filter (True=apply, False=skip filter)
REGIME_APPLY = {
    "uniform":       {0: True,  1: True,  2: True,  3: True },
    "skip_trending": {0: False, 1: True,  2: True,  3: False},
    "skip_ranging":  {0: True,  1: False, 2: False, 3: True },
}


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
def apply_entry_filter(
    yp_base: np.ndarray,
    p_lgbm: np.ndarray,
    p_lstm: np.ndarray,
    hmm: np.ndarray,
    hmm_config: dict,
    lstm_mode: str,
    regime_apply: dict,
) -> np.ndarray:
    """
    Apply HMM threshold + LSTM filter to raw LGBM predictions.
    Returns yp array with FLAT for filtered-out bars.
    """
    n = len(yp_base)
    conf = np.max(p_lgbm, axis=1)

    # Step 1: HMM-aware threshold on LGBM confidence
    yp = np.ones(n, dtype=np.int32)  # FLAT baseline
    for state, thr in hmm_config.items():
        mask = (hmm == state)
        # LONG: p_lgbm[:,2] >= thr
        yp[mask & (p_lgbm[:, 2] >= thr)] = LONG
        # SHORT: p_lgbm[:,0] >= thr AND not already LONG
        yp[mask & (p_lgbm[:, 0] >= thr) & (yp != LONG)] = SHORT

    if lstm_mode == "none":
        return yp

    lstm_argmax = np.argmax(p_lstm, axis=1)

    # Step 2: Per-bar LSTM filter (respects regime-conditionality)
    sig_idx = np.where(yp != FLAT)[0]
    for i in sig_idx:
        state  = int(hmm[i])
        apply  = regime_apply.get(state, True)
        if not apply:
            continue  # skip LSTM in this regime

        lgbm_dir = yp[i]
        opp_dir  = SHORT if lgbm_dir == LONG else LONG
        p_opp    = p_lstm[i, opp_dir]
        p_agree  = p_lstm[i, lgbm_dir]

        if lstm_mode == "flip_hard":
            if lstm_argmax[i] == opp_dir:
                yp[i] = FLAT

        elif lstm_mode.startswith("flip_p"):
            thr = float(lstm_mode.split("flip_p")[1]) / 100.0
            if p_opp >= thr:
                yp[i] = FLAT

        elif lstm_mode == "consensus":
            if lstm_argmax[i] != lgbm_dir:
                yp[i] = FLAT

        elif lstm_mode == "soft_mul":
            # Adjust LGBM confidence by LSTM alignment
            # mult_factor = p_agree / 0.333 (ratio to random baseline)
            # adj_conf = conf * sqrt(mult_factor) — moderate effect
            mult = (p_agree / 0.3333) ** 0.5
            adj_conf = conf[i] * mult
            thr = hmm_config[state]
            if adj_conf < thr:
                yp[i] = FLAT

    return yp


# ── Guardian helper ────────────────────────────────────────────────────────────
def build_guardian_row(j, i, close, atr, direction, max_fav, X_static):
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
    row = np.zeros(len(tbg_order), dtype=np.float64)
    for idx, (src, key) in enumerate(tbg_order):
        row[idx] = X_static[j, key] if src == "static" else dyn_vals.get(key, 0.0)
    return row, new_max


# ── Simulate ───────────────────────────────────────────────────────────────────
def simulate(yp, close, high, low, atr, X_tbg):
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
                exit_price, exit_bar, outcome = sl_price, j, "SL"
                break
            if (j - i) >= GDN_MIN_HOLD:
                row, max_fav = build_guardian_row(j, i, close, atr, direction, max_fav, X_tbg)
                prob = tbg_model.predict_proba(tbg_scaler.transform(row.reshape(1, -1)))[0]
                if (prob[2] if len(prob) > 2 else prob[1]) >= GDN_THR:
                    exit_price, exit_bar, outcome = close[j], j, "GUARDIAN_EXIT"
                    break
            else:
                max_fav = max(max_fav, (close[j] - entry) / entry * direction)

        net_pnl = (exit_price - entry) / entry * direction * MODAL * LEVERAGE \
                  - COST_RT * MODAL * LEVERAGE
        trades.append({
            "win"    : net_pnl > 0,
            "pnl"    : net_pnl,
            "bars"   : exit_bar - i,
            "is_long": sig == LONG,
            "sl"     : outcome == "SL",
            "gdn"    : "GUARDIAN" in outcome,
        })
        i = exit_bar + 1
    return trades


# ── Per-coin data loading ──────────────────────────────────────────────────────
def load_coin(sym):
    fpath = HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet"
    if not fpath.exists():
        return None
    df = pd.read_parquet(fpath)
    df = ensure_utc_index(df).sort_index()

    # HMM regime
    rp  = HOLDOUT_DIR / "labeled" / f"{sym}_regime_h1.parquet"
    hmm = np.full(len(df), 1, np.int32)
    if rp.exists():
        reg = pd.read_parquet(rp)
        if "hmm_regime_enc" in reg.columns:
            hmm = reg["hmm_regime_enc"].reindex(df.index, fill_value=1).values.astype(np.int32)

    from config import LABEL_MAP
    LM = {"SHORT": 0, "FLAT": 1, "LONG": 2}
    mask = df["label"].isin(LM)
    df   = df[mask].copy()
    hmm  = hmm[mask.values]
    n    = len(df)

    close = df["close"].values.astype(np.float64)
    high  = df["high"].values.astype(np.float64)
    low   = df["low"].values.astype(np.float64)
    atr   = df["atr_14_h1"].values.astype(np.float64)

    # LGBM probs
    X_tb = np.zeros((n, len(tb_feats)), dtype=np.float64)
    for idx, c in enumerate(tb_feats):
        if c in df.columns:
            X_tb[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)
    p_lgbm = tb_model.predict_proba(X_tb)

    # LSTM probs
    X_lstm = np.zeros((n, len(lstm_feats)), dtype=np.float32)
    for idx, c in enumerate(lstm_feats):
        if c in df.columns:
            X_lstm[:, idx] = df[c].ffill().fillna(0).values.astype(np.float32)
    p_lstm = lstm_predict_proba(X_lstm)

    # Guardian static features
    X_tbg = np.zeros((n, len(tbg_static)), dtype=np.float64)
    for idx, c in enumerate(tbg_static):
        if c in df.columns:
            X_tbg[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)

    return dict(
        close=close, high=high, low=low, atr=atr,
        p_lgbm=p_lgbm, p_lstm=p_lstm, X_tbg=X_tbg, hmm=hmm, n=n,
    )


# ── Main sweep ────────────────────────────────────────────────────────────────
def run_sweep():
    available = [s for s in ALL_COINS
                 if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()]
    logger.info(f"Loading {len(available)} coins...")
    coin_data = {}
    for sym in available:
        d = load_coin(sym)
        if d is not None:
            coin_data[sym] = d
    logger.info(f"Loaded {len(coin_data)} coins | Running sweep...")

    combos = list(itertools.product(
        HMM_CONFIGS.keys(),
        LSTM_MODES,
        REGIME_APPLY.keys(),
    ))
    logger.info(f"Total combinations: {len(combos)}")

    results = []
    for ci, (hmm_name, lstm_mode, regime_name) in enumerate(combos):
        hmm_config   = HMM_CONFIGS[hmm_name]
        regime_apply = REGIME_APPLY[regime_name]

        agg = {"trades": 0, "wins": 0, "pnl": 0.0, "longs": 0, "long_wins": 0,
               "sl": 0, "gdn": 0, "bars": []}

        for sym, d in coin_data.items():
            yp = apply_entry_filter(
                np.ones(d["n"], dtype=np.int32),
                d["p_lgbm"], d["p_lstm"], d["hmm"],
                hmm_config, lstm_mode, regime_apply,
            )
            trades = simulate(yp, d["close"], d["high"], d["low"], d["atr"], d["X_tbg"])
            for t in trades:
                agg["trades"] += 1
                agg["pnl"]    += t["pnl"]
                agg["bars"].append(t["bars"])
                if t["win"]:  agg["wins"] += 1
                if t["is_long"]: agg["longs"] += 1
                if t["is_long"] and t["win"]: agg["long_wins"] += 1
                if t["sl"]:   agg["sl"] += 1
                if t["gdn"]:  agg["gdn"] += 1

        n  = agg["trades"]
        wr = agg["wins"] / n * 100 if n > 0 else 0.0
        lwr = agg["long_wins"] / agg["longs"] * 100 if agg["longs"] > 0 else 0.0
        sl_r = agg["sl"] / n * 100 if n > 0 else 0.0
        gdn_r = agg["gdn"] / n * 100 if n > 0 else 0.0
        avg_hold = np.mean(agg["bars"]) if agg["bars"] else 0.0

        results.append({
            "hmm_config"  : hmm_name,
            "lstm_mode"   : lstm_mode,
            "regime_apply": regime_name,
            "trades"      : n,
            "wr_pct"      : round(wr, 1),
            "long_wr_pct" : round(lwr, 1),
            "long_pct"    : round(agg["longs"] / n * 100, 1) if n > 0 else 0.0,
            "sl_pct"      : round(sl_r, 1),
            "gdn_pct"     : round(gdn_r, 1),
            "avg_hold"    : round(avg_hold, 1),
            "net_pnl"     : round(agg["pnl"], 2),
            "pnl_per_trade": round(agg["pnl"] / n, 3) if n > 0 else 0.0,
        })

        if (ci + 1) % 10 == 0:
            logger.info(f"  [{ci+1}/{len(combos)}] {hmm_name}|{lstm_mode}|{regime_name}"
                        f" -> trades={n} WR={wr:.1f}% PnL=${agg['pnl']:.0f}")

    return results


def print_results(results):
    df = pd.DataFrame(results).sort_values("net_pnl", ascending=False)

    print(f"\n{'='*110}")
    print(f"  TB COMBINATION SWEEP | Apr-Jun 2026 | 21 koin | $10/trade 5x | Guardian fixed thr=0.65")
    print(f"  84 combos: 4 HMM configs x 7 LSTM modes x 3 regime-conditionality")
    print(f"{'='*110}")
    print(f"  {'Rank':>4}  {'HMM Config':<14} {'LSTM Mode':<12} {'Regime Apply':<16}"
          f"  {'Trades':>6} {'WR%':>5} {'LONG%':>6} {'SL%':>5} {'GDN%':>5}"
          f"  {'AvgHold':>7} {'PnL':>8} {'$/trade':>8}")
    print(f"  {'-'*106}")
    for rank, row in enumerate(df.head(20).itertuples(), 1):
        print(f"  {rank:>4}  {row.hmm_config:<14} {row.lstm_mode:<12} {row.regime_apply:<16}"
              f"  {row.trades:>6} {row.wr_pct:>5.1f} {row.long_pct:>6.1f} {row.sl_pct:>5.1f}"
              f" {row.gdn_pct:>5.1f}  {row.avg_hold:>7.1f} {row.net_pnl:>8.0f} {row.pnl_per_trade:>8.3f}")

    print(f"\n  {'--- Bottom 5 ---':^106}")
    for rank, row in enumerate(df.tail(5).itertuples(), len(df) - 4):
        print(f"  {rank:>4}  {row.hmm_config:<14} {row.lstm_mode:<12} {row.regime_apply:<16}"
              f"  {row.trades:>6} {row.wr_pct:>5.1f} {row.long_pct:>6.1f} {row.sl_pct:>5.1f}"
              f" {row.gdn_pct:>5.1f}  {row.avg_hold:>7.1f} {row.net_pnl:>8.0f} {row.pnl_per_trade:>8.3f}")

    print(f"\n  Best PnL : {df.iloc[0]['hmm_config']} | {df.iloc[0]['lstm_mode']}"
          f" | {df.iloc[0]['regime_apply']} = ${df.iloc[0]['net_pnl']:.0f}")
    best_wr = df.sort_values("wr_pct", ascending=False).iloc[0]
    print(f"  Best WR  : {best_wr['hmm_config']} | {best_wr['lstm_mode']}"
          f" | {best_wr['regime_apply']} = {best_wr['wr_pct']:.1f}%")
    best_ppt = df.sort_values("pnl_per_trade", ascending=False).iloc[0]
    print(f"  Best $/t : {best_ppt['hmm_config']} | {best_ppt['lstm_mode']}"
          f" | {best_ppt['regime_apply']} = ${best_ppt['pnl_per_trade']:.3f}")
    print(f"{'='*110}\n")

    return df


if __name__ == "__main__":
    results = run_sweep()
    df = print_results(results)

    out_path = MODEL_DIR / "runs" / "tb_lgbm_widyawardhana_v3" / "tune_combination_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out_path, orient="records", indent=2)
    logger.info(f"Saved -> {out_path}")
