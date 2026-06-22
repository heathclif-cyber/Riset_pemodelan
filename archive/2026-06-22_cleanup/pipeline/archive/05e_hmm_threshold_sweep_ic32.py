"""
pipeline/05e_hmm_threshold_sweep_ic32.py -- HMM Per-State Threshold Sweep (ic32 Simons)

Pendekatan Simons untuk ic32_regime_v1:
  - TIDAK menambah filter (breadth gate, hard block, dll)
  - Satu lever: HMM per-state [thr_long, thr_short] dari OOF
  - Baseline = production fixed thr_long=0.69, thr_short=0.59

Prerequisite:
  models/runs/ic32_regime_v1/oof_predictions.parquet

Jalankan:
  python pipeline/05e_hmm_threshold_sweep_ic32.py
"""
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    ALL_COINS, TRAIN_CUTOFF_DATE, LABEL_DIR, MODEL_DIR,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL, MAX_HOLDING_BARS,
)
from core.evaluator import simulate_trades_swing
from core.utils import ensure_utc_index

RUN_DIR = MODEL_DIR / "runs" / "ic32_regime_v1"
OOF_PATH = RUN_DIR / "oof_predictions.parquet"
OUT_PATH = RUN_DIR / "hmm_threshold_ic32_simons.json"

# Production baseline (ic32_regime_v1 cascade)
BASE_TL = 0.69
BASE_TS = 0.59

HMM_NAMES = {0: "TRENDING_DN", 1: "RANGING_LOW", 2: "RANGING_HIGH", 3: "TRENDING_UP"}

# Sweep grids — focused around production + Simons asymmetric fix for SHORT bias
LONG_GRID = [0.55, 0.58, 0.60, 0.62, 0.65, 0.69, 0.72]
SHORT_GRID = [0.55, 0.58, 0.59, 0.62, 0.65, 0.69, 0.72]
SYM_GRID = [0.50, 0.55, 0.58, 0.60, 0.62, 0.65, 0.69]


def preload_coin_data(oof_pred_df: pd.DataFrame) -> list:
    coin_data = []
    for sym in ALL_COINS:
        path = LABEL_DIR / f"{sym}_features_v3.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = ensure_utc_index(df).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]
        if df.empty:
            continue

        rp = LABEL_DIR / f"{sym}_regime_h1.parquet"
        if rp.exists() and "hmm_regime_enc" not in df.columns:
            try:
                reg = pd.read_parquet(rp)
                cols = [c for c in ["hmm_regime_enc"] if c in reg.columns]
                if cols:
                    df = df.join(reg[cols], how="left")
            except Exception:
                pass

        sym_oof = oof_pred_df[oof_pred_df["coin"] == sym]
        sym_oof = sym_oof[sym_oof["has_oof"] == True][["p0", "p2"]]
        proba = sym_oof.reindex(df.index)
        has_oof = proba["p0"].notna()

        df_oof = df[has_oof].copy()
        n = len(df_oof)
        if n < 30:
            continue

        coin_data.append({
            "sym": sym,
            "n": n,
            "p0": proba["p0"][has_oof].values.astype(np.float32),
            "p2": proba["p2"][has_oof].values.astype(np.float32),
            "hmm": df_oof["hmm_regime_enc"].fillna(-1).values.astype(np.int8)
            if "hmm_regime_enc" in df_oof.columns else np.full(n, -1, np.int8),
            "close": df_oof["close"].values.astype(np.float64),
            "high": df_oof["high"].values.astype(np.float64),
            "low": df_oof["low"].values.astype(np.float64),
            "atr": df_oof["atr_14_h1"].values.astype(np.float64),
            "h4_sh": df_oof["h4_swing_high"].values.astype(np.float64)
            if "h4_swing_high" in df_oof.columns else np.full(n, np.nan),
            "h4_sl": df_oof["h4_swing_low"].values.astype(np.float64)
            if "h4_swing_low" in df_oof.columns else np.full(n, np.nan),
        })
    return coin_data


def simulate_with_hmm_thr(coin_data: list, thr_cfg: dict,
                          regime_size_mult: dict | None = None) -> dict:
    """Vectorized HMM gate + optional Simons regime sizing (bet, not block)."""
    default_tl, default_ts = thr_cfg.get(-1, (BASE_TL, BASE_TS))
    regime_size_mult = regime_size_mult or {}

    agg_trades = []
    n_long_sig = n_short_sig = 0

    for cd in coin_data:
        n, p0, p2, hmm = cd["n"], cd["p0"], cd["p2"], cd["hmm"]

        tl_arr = np.full(n, default_tl, dtype=np.float32)
        ts_arr = np.full(n, default_ts, dtype=np.float32)
        for state, (tl, ts) in thr_cfg.items():
            if state == -1:
                continue
            mask = hmm == state
            tl_arr[mask] = tl
            ts_arr[mask] = ts

        long_mask = p2 >= tl_arr
        short_mask = (p0 >= ts_arr) & ~long_mask
        n_long_sig += int(long_mask.sum())
        n_short_sig += int(short_mask.sum())

        y_pred = np.ones(n, dtype=np.int32)
        y_pred[long_mask] = 2
        y_pred[short_mask] = 0

        if (y_pred != 1).sum() == 0:
            continue

        modal_arr = np.full(n, MODAL_PER_TRADE, dtype=np.float64)
        for state, mult in regime_size_mult.items():
            modal_arr[hmm == state] = MODAL_PER_TRADE * mult

        # simulate_trades_swing uses scalar modal — approximate with mean modal on entries
        entry_mask = y_pred != 1
        eff_modal = float(np.mean(modal_arr[entry_mask])) if entry_mask.any() else MODAL_PER_TRADE

        result = simulate_trades_swing(
            y_pred=y_pred,
            close=cd["close"], high=cd["high"], low=cd["low"], atr=cd["atr"],
            h4_swing_highs=cd["h4_sh"], h4_swing_lows=cd["h4_sl"],
            modal=eff_modal, leverage=LEVERAGE_SIM[0],
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            max_hold=MAX_HOLDING_BARS,
            min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
            max_sl_atr=SWING_LABEL_MAX_SL,
            tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
            guardian_enabled=False,
        )
        agg_trades.extend(result.get("trades", []))

    if not agg_trades:
        return {"n": 0, "wr": 0, "pnl": 0, "ppt": 0, "pf": 0, "sl_pct": 0,
                "long_sig": 0, "short_sig": 0, "short_long_ratio": 0}

    nt = len(agg_trades)
    wins = sum(1 for t in agg_trades if t["outcome"] == "WIN")
    sl_hit = sum(1 for t in agg_trades if t["outcome"] == "LOSS")
    gpnl = sum(t["net_pnl"] for t in agg_trades if t["net_pnl"] > 0)
    lloss = sum(abs(t["net_pnl"]) for t in agg_trades if t["net_pnl"] < 0)
    tpnl = sum(t["net_pnl"] for t in agg_trades)
    pf = gpnl / lloss if lloss > 0 else float("inf")
    ratio = n_short_sig / max(n_long_sig, 1)
    return {
        "n": nt, "wr": wins / nt * 100, "pnl": tpnl, "ppt": tpnl / nt, "pf": pf,
        "sl_pct": sl_hit / nt * 100, "long_sig": n_long_sig, "short_sig": n_short_sig,
        "short_long_ratio": ratio,
    }


def _cfg_uniform(tl: float, ts: float | None = None) -> dict:
    ts = ts if ts is not None else tl
    return {s: (tl, ts) for s in range(4)} | {-1: (tl, ts)}


def _cfg_prod() -> dict:
    return _cfg_uniform(BASE_TL, BASE_TS)


def _row(label, r, delta_ppt=None, mark=""):
    pf = f"{r['pf']:.3f}" if r["pf"] != float("inf") else " INF"
    d = f"  dPPT={delta_ppt:>+7.4f}" if delta_ppt is not None else ""
    print(
        f"  {label:<32} n={r['n']:>6,} WR={r['wr']:>5.1f}% "
        f"PnL=${r['pnl']:>9.1f} PPT={r['ppt']:>+7.4f} PF={pf} "
        f"S/L={r['short_long_ratio']:.2f}{d}{mark}"
    )


def main():
    if not OOF_PATH.exists():
        raise FileNotFoundError(f"Missing {OOF_PATH} — run 04_train_lgbm_ic32_genuine_oof.py first")

    print("=" * 88)
    print("  HMM THRESHOLD SWEEP ic32 — Simons (per-state thr, no extra filters)")
    print("=" * 88)

    oof_df = pd.read_parquet(OOF_PATH)
    coin_data = preload_coin_data(oof_df)
    print(f"  Coins loaded: {len(coin_data)}")

    base = simulate_with_hmm_thr(coin_data, _cfg_prod())
    print("\n  BASELINE (production fixed 0.69/0.59 all states):")
    _row("BASELINE", base)

    # Phase 1: per-state symmetric (others at production baseline)
    print("\n" + "=" * 88)
    print("  PHASE 1 — Symmetric sweep per HMM state (others = production 0.69/0.59)")
    print("=" * 88)
    best_sym = {}
    for state in range(4):
        print(f"\n  S{state} {HMM_NAMES[state]}:")
        best_ppt, best_thr, best_r = -999, BASE_TL, None
        for thr in SYM_GRID:
            cfg = _cfg_prod()
            cfg[state] = (thr, thr)
            r = simulate_with_hmm_thr(coin_data, cfg)
            tag = ""
            if r["ppt"] > best_ppt and r["n"] >= base["n"] * 0.05:
                best_ppt, best_thr, best_r = r["ppt"], thr, r
                tag = " <<BEST"
            pf = f"{r['pf']:.3f}" if r["pf"] != float("inf") else " INF"
            print(f"    thr={thr:.2f} n={r['n']:>6,} WR={r['wr']:>5.1f}% "
                  f"PPT={r['ppt']:>+7.4f} PF={pf} S/L={r['short_long_ratio']:.2f}{tag}")
        best_sym[state] = (best_thr, best_r)

    # Phase 2: S1 direction-aware (RANGING dominant — fix SHORT bias)
    print("\n" + "=" * 88)
    print("  PHASE 2 — S1 RANGING_LOW direction-aware [thr_long x thr_short]")
    print("=" * 88)
    s0 = best_sym[0][0]
    s2 = best_sym[2][0]
    s3 = best_sym[3][0]
    p2_best = {"cfg": None, "r": None, "ppt": -999}
    p2_results = []

    for tl in LONG_GRID:
        for ts in SHORT_GRID:
            if ts < tl - 0.15:
                continue
            cfg = {
                0: (s0, s0), 1: (tl, ts), 2: (s2, s2), 3: (s3, s3), -1: (BASE_TL, BASE_TS),
            }
            r = simulate_with_hmm_thr(coin_data, cfg)
            p2_results.append({"tl": tl, "ts": ts, **r})
            if r["ppt"] > p2_best["ppt"] and r["n"] >= base["n"] * 0.05:
                p2_best = {"cfg": (tl, ts), "r": r, "ppt": r["ppt"]}

    p2_sorted = sorted(p2_results, key=lambda x: -x["ppt"])[:12]
    print(f"  {'tl':>5} {'ts':>5}  {'n':>6}  {'WR%':>5}  {'PPT':>7}  {'PF':>5}  S/L")
    print("  " + "-" * 60)
    for row in p2_sorted:
        pf = f"{row['pf']:.3f}" if row["pf"] != float("inf") else " INF"
        print(f"  {row['tl']:>5.2f} {row['ts']:>5.2f}  {row['n']:>6,}  "
              f"{row['wr']:>5.1f}  {row['ppt']:>+7.4f}  {pf:>5}  {row['short_long_ratio']:.2f}")

    # Phase 3: candidate configs
    print("\n" + "=" * 88)
    print("  PHASE 3 — Candidate configs vs baseline")
    print("=" * 88)

    s1_tl, s1_ts = p2_best["cfg"] if p2_best["cfg"] else (best_sym[1][0], best_sym[1][0])
    candidates = {
        "A: production (0.69/0.59)": _cfg_prod(),
        "B: sym-best all states": {s: (best_sym[s][0], best_sym[s][0]) for s in range(4)}
            | {-1: (BASE_TL, BASE_TS)},
        "C: sym-best + S1 dir-aware": {
            0: (s0, s0), 1: (s1_tl, s1_ts), 2: (s2, s2), 3: (s3, s3), -1: (BASE_TL, BASE_TS),
        },
        "D: balanced 0.65/0.65 all": _cfg_uniform(0.65, 0.65),
        "E: fix SHORT bias S1=(0.65,0.72)": {
            0: (s0, s0), 1: (0.65, 0.72), 2: (s2, s2), 3: (s3, s3), -1: (BASE_TL, BASE_TS),
        },
        "F: Config-B style (tb ref)": {
            0: (0.55, 0.55), 1: (0.55, 0.55), 2: (0.50, 0.50),
            3: (0.45, 0.50), -1: (0.45, 0.45),
        },
    }

    phase3 = {}
    best_name, best_r, best_ppt = "A", base, base["ppt"]
    for name, cfg in candidates.items():
        r = simulate_with_hmm_thr(coin_data, cfg)
        phase3[name] = r
        mark = ""
        if r["ppt"] > best_ppt and r["n"] >= base["n"] * 0.05:
            best_ppt, best_name, best_r = r["ppt"], name, r
            mark = " <<BEST"
        _row(name, r, r["ppt"] - base["ppt"], mark)

    # Phase 4: HMM threshold + regime sizing (Simons: change bet not signal)
    print("\n" + "=" * 88)
    print("  PHASE 4 — Best HMM thr + regime sizing (TRENDING 0.5x)")
    print("=" * 88)
    best_cfg = candidates.get(best_name, _cfg_prod())
    size_mult = {0: 0.50, 1: 1.00, 2: 0.80, 3: 0.50}
    r_size = simulate_with_hmm_thr(coin_data, best_cfg, regime_size_mult=size_mult)
    _row(f"{best_name} + sizing", r_size, r_size["ppt"] - base["ppt"])

    out = {
        "created": str(datetime.now()),
        "methodology": "simons_hmm_per_state_threshold_oof",
        "run": "ic32_regime_v1",
        "baseline_production": {"thr_long": BASE_TL, "thr_short": BASE_TS},
        "baseline_stats": base,
        "phase1_best_symmetric": {
            str(s): {"thr": best_sym[s][0], "stats": best_sym[s][1]} for s in range(4)
        },
        "phase2_s1_best": {"thr_long": s1_tl, "thr_short": s1_ts, "stats": p2_best["r"]},
        "phase3_candidates": {k: v for k, v in phase3.items()},
        "best_config_name": best_name,
        "best_config": {str(k): list(v) for k, v in candidates[best_name].items()},
        "best_stats": best_r,
        "phase4_sizing": r_size,
        "delta_ppt_vs_baseline": best_r["ppt"] - base["ppt"],
        "note": "Signal-only OOF (no Guardian/LSTM/FLIP). Simons lever #1: HMM threshold.",
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"\n  BEST: {best_name}")
    _row("WINNER", best_r, best_r["ppt"] - base["ppt"])
    print(f"\n  Saved: {OUT_PATH}")
    print("=" * 88)


if __name__ == "__main__":
    main()