"""
pipeline/05e_hmm_threshold_sweep.py -- HMM Per-State Threshold Sweep (v2, vectorized)

Tujuan:
  Cari threshold optimal per HMM state via OOF simulation. Murni OOF --
  tidak menyentuh holdout Apr-Jun 2026.

Perubahan v2 vs v1:
  - Vectorized numpy signal generation (ganti per-bar Python loop) → ~50x lebih cepat
  - Phase 2: hanya sweep S3 direction-aware (Phase 1 konfirmasi S0 symmetric optimal)
  - Phase 3: bukan grid exhaustive -- test 8 config kandidat spesifik
  - Total simulasi: ~100 vs ~1000+ di v1

State mapping:
  0 = TRENDING_DOWN, 1 = RANGING_LOW_VOL, 2 = RANGING_HIGH_VOL, 3 = TRENDING_UP
"""

import sys, warnings, json
import numpy as np
import pandas as pd
from pathlib import Path

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

LGBM_RUN = "tb_lgbm_genuine_v2"
RUN_DIR  = MODEL_DIR / "runs" / LGBM_RUN

HMM_NAMES = {0: "TRENDING_DN", 1: "RANGING_LOW", 2: "RANGING_HIGH", 3: "TRENDING_UP"}

SYM_GRID = [0.38, 0.40, 0.42, 0.44, 0.45, 0.47, 0.50, 0.52, 0.55, 0.58, 0.60]
P2_S3_LONG  = [0.38, 0.40, 0.42, 0.44, 0.45]
P2_S3_SHORT = [0.44, 0.45, 0.47, 0.50, 0.52, 0.55]


# ─── Pre-load all coin data once ─────────────────────────────────────────────

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

        for feat, shifts in [("ret_7d", 672), ("ret_14d", 1344), ("ret_30d", 2880)]:
            if feat not in df.columns:
                df[feat] = np.log(df["close"] / df["close"].shift(shifts).replace(0, np.nan))
        for feat, col in [("dist_pdh", "PDH"), ("dist_pdl", "PDL")]:
            if feat not in df.columns and col in df.columns:
                df[feat] = df["close"] / df[col].replace(0, np.nan) - 1

        sym_oof = oof_pred_df[oof_pred_df["coin"] == sym]
        sym_oof = sym_oof[sym_oof["has_oof"] == True][["p0", "p2"]]
        proba   = sym_oof.reindex(df.index)
        has_oof = proba["p0"].notna()

        df_oof = df[has_oof].copy()
        n = len(df_oof)
        if n < 30:
            continue

        coin_data.append({
            "sym":   sym,
            "n":     n,
            "p0":    proba["p0"][has_oof].values.astype(np.float32),
            "p2":    proba["p2"][has_oof].values.astype(np.float32),
            "hmm":   df_oof["hmm_regime_enc"].fillna(-1).values.astype(np.int8)
                     if "hmm_regime_enc" in df_oof.columns else np.full(n, -1, np.int8),
            "close": df_oof["close"].values.astype(np.float64),
            "high":  df_oof["high"].values.astype(np.float64),
            "low":   df_oof["low"].values.astype(np.float64),
            "atr":   df_oof["atr_14_h1"].values.astype(np.float64),
            "h4_sh": df_oof["h4_swing_high"].values.astype(np.float64)
                     if "h4_swing_high" in df_oof.columns else np.full(n, np.nan),
            "h4_sl": df_oof["h4_swing_low"].values.astype(np.float64)
                     if "h4_swing_low" in df_oof.columns else np.full(n, np.nan),
        })

    return coin_data


# ─── Vectorized simulate ─────────────────────────────────────────────────────

def simulate_with_hmm_thr(coin_data: list, thr_cfg: dict) -> dict:
    """
    thr_cfg = {state_int: (thr_long, thr_short), ...}
    state -1 = fallback untuk unknown HMM.

    Signal generation di-vectorize dengan numpy -- tidak ada per-bar Python loop.
    """
    default_tl, default_ts = thr_cfg.get(-1, (0.45, 0.45))

    agg_trades = []
    for cd in coin_data:
        n   = cd["n"]
        p0  = cd["p0"]
        p2  = cd["p2"]
        hmm = cd["hmm"]

        # Vectorized: build threshold arrays sesuai HMM state per bar
        tl_arr = np.full(n, default_tl, dtype=np.float32)
        ts_arr = np.full(n, default_ts, dtype=np.float32)
        for state, (tl, ts) in thr_cfg.items():
            if state == -1:
                continue
            mask = hmm == state
            tl_arr[mask] = tl
            ts_arr[mask] = ts

        # Apply thresholds — LONG takes priority over SHORT
        long_mask  = p2 >= tl_arr
        short_mask = (p0 >= ts_arr) & ~long_mask
        y_pred = np.ones(n, dtype=np.int32)   # default FLAT
        y_pred[long_mask]  = 2
        y_pred[short_mask] = 0

        if (y_pred != 1).sum() == 0:
            continue

        result = simulate_trades_swing(
            y_pred=y_pred,
            close=cd["close"], high=cd["high"], low=cd["low"], atr=cd["atr"],
            h4_swing_highs=cd["h4_sh"], h4_swing_lows=cd["h4_sl"],
            modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            max_hold=MAX_HOLDING_BARS,
            min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
            max_sl_atr=SWING_LABEL_MAX_SL,
            tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
            guardian_enabled=False,
        )
        agg_trades.extend(result.get("trades", []))

    if not agg_trades:
        return {"n": 0, "wr": 0, "pnl": 0, "ppt": 0, "pf": 0, "sl_pct": 0}

    n      = len(agg_trades)
    wins   = sum(1 for t in agg_trades if t["outcome"] == "WIN")
    sl_hit = sum(1 for t in agg_trades if t["outcome"] == "LOSS")
    gpnl   = sum(t["net_pnl"] for t in agg_trades if t["net_pnl"] > 0)
    lloss  = sum(abs(t["net_pnl"]) for t in agg_trades if t["net_pnl"] < 0)
    tpnl   = sum(t["net_pnl"] for t in agg_trades)
    pf     = gpnl / lloss if lloss > 0 else float("inf")
    return {"n": n, "wr": wins / n * 100, "pnl": tpnl,
            "ppt": tpnl / n, "pf": pf, "sl_pct": sl_hit / n * 100}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _row(label, r, delta_ppt=None, mark=""):
    pf = f"{r['pf']:.3f}" if r["pf"] != float("inf") else "  INF"
    d  = f"  delta={delta_ppt:>+7.4f}" if delta_ppt is not None else ""
    print(f"  {label:<26} n={r['n']:>6,}  WR={r['wr']:>5.1f}%  "
          f"PnL=${r['pnl']:>9.1f}  PPT={r['ppt']:>+7.4f}  PF={pf}{d}{mark}")


def _cfg_uniform(tl, ts=None):
    ts = ts if ts is not None else tl
    return {s: (tl, ts) for s in range(4)} | {-1: (tl, ts)}


def _cfg_from(sym_best, overrides: dict) -> dict:
    """Build config: semua state pakai sym_best, override dengan dict."""
    cfg = {s: (sym_best[s], sym_best[s]) for s in range(4)}
    cfg[-1] = (0.45, 0.45)
    cfg.update(overrides)
    return cfg


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    SEP  = "=" * 80
    SEP2 = "-" * 80

    oof_pred_df = pd.read_parquet(RUN_DIR / "oof_predictions.parquet")
    print(f"\n{SEP}")
    print(f"  HMM THRESHOLD SWEEP v2 (vectorized) -- tb_lgbm_genuine_v2")
    print(f"{SEP}\n")
    print("Pre-loading coin data...")
    coin_data = preload_coin_data(oof_pred_df)
    print(f"  Loaded {len(coin_data)} coins")

    # ── Baseline ─────────────────────────────────────────────────────────────
    base = simulate_with_hmm_thr(coin_data, _cfg_uniform(0.45))
    print(f"\n{SEP}")
    print("  BASELINE:")
    _row("BASELINE (0.45/0.45)", base)

    # ── Phase 1: Per-state symmetric sweep ───────────────────────────────────
    print(f"\n{SEP}")
    print("  PHASE 1 -- SYMMETRIC SWEEP PER STATE")
    print(f"  (state yang di-sweep berubah, state lain tetap 0.45)")
    print(SEP)

    best_sym = {}   # state -> (thr, result)

    for state in range(4):
        print(f"\n  State {state} ({HMM_NAMES[state]}):")
        print(f"  {'Thr':<6} {'N':>6}  {'WR%':>5}  {'PnL':>9}  {'PPT':>7}  {'PF':>5}  SL%")
        print(f"  {SEP2}")

        best_ppt = -999
        best_r, best_thr = None, 0.45

        for thr in SYM_GRID:
            cfg = _cfg_uniform(0.45)
            cfg[state] = (thr, thr)
            r = simulate_with_hmm_thr(coin_data, cfg)
            is_base = abs(thr - 0.45) < 0.001
            tag = " [BASE]" if is_base else ""
            if r["ppt"] > best_ppt and r["n"] >= base["n"] * 0.10:
                best_ppt, best_r, best_thr = r["ppt"], r, thr
                tag = " <<BEST"
            pf = f"{r['pf']:.3f}" if r["pf"] != float("inf") else "  INF"
            print(f"  {thr:.2f}   n={r['n']:>6,}  WR={r['wr']:>5.1f}%  "
                  f"PnL=${r['pnl']:>9.1f}  PPT={r['ppt']:>+7.4f}  PF={pf}  "
                  f"SL={r['sl_pct']:.1f}%{tag}")

        best_sym[state] = (best_thr, best_r)
        print(f"\n  => S{state} best: thr={best_thr:.2f}  "
              f"PPT={best_ppt:+.4f}  delta={best_ppt-base['ppt']:+.4f}")

    print(f"\n{SEP}")
    print(f"  PHASE 1 SUMMARY:")
    print(f"  {'State':<22} {'BestThr':>7}  {'PPT_base':>8}  {'PPT_best':>8}  {'Delta':>8}  N")
    print(SEP2)
    for s in range(4):
        thr, r = best_sym[s]
        print(f"  S{s} {HMM_NAMES[s]:<18} {thr:>7.2f}  "
              f"{base['ppt']:>+8.4f}  {r['ppt']:>+8.4f}  "
              f"{r['ppt']-base['ppt']:>+8.4f}  {r['n']:,}")

    # ── Phase 2: S3 direction-aware sweep ────────────────────────────────────
    # S3 = TRENDING_UP: long lebih agresif (with-trend), short lebih konservatif
    # S0 dilewati karena Phase 1 konsisten menunjukkan symmetric optimal
    print(f"\n{SEP}")
    print(f"  PHASE 2 -- S3 DIRECTION-AWARE SWEEP")
    print(f"  (S0/S1/S2 pakai best symmetric Phase 1, hanya S3 yang di-sweep)")
    print(SEP)

    s0_t = best_sym[0][0]
    s1_t = best_sym[1][0]
    s2_t = best_sym[2][0]

    print(f"\n  Fixed: S0={s0_t:.2f}, S1={s1_t:.2f}, S2={s2_t:.2f}  | sweep S3 thr_L x thr_S")
    print(f"  {'thr_L':>6} x {'thr_S':<6}  {'N':>6}  {'WR%':>5}  {'PPT':>7}  {'PF':>5}  delta")
    print(f"  {SEP2}")

    p2_best_ppt = -999
    p2_best_cfg = (0.45, 0.45)
    p2_best_r   = None

    for tl in P2_S3_LONG:
        for ts in P2_S3_SHORT:
            cfg = {
                0:  (s0_t, s0_t),
                1:  (s1_t, s1_t),
                2:  (s2_t, s2_t),
                3:  (tl, ts),
                -1: (0.45, 0.45),
            }
            r    = simulate_with_hmm_thr(coin_data, cfg)
            delta = r["ppt"] - base["ppt"]
            is_base = abs(tl - 0.45) < 0.001 and abs(ts - 0.45) < 0.001
            tag = " [BASE]" if is_base else ""
            if r["ppt"] > p2_best_ppt and r["n"] >= base["n"] * 0.10:
                p2_best_ppt, p2_best_cfg, p2_best_r = r["ppt"], (tl, ts), r
                tag = " <<BEST"
            pf = f"{r['pf']:.3f}" if r["pf"] != float("inf") else " INF"
            print(f"  {tl:.2f} x {ts:.2f}    "
                  f"n={r['n']:>6,}  WR={r['wr']:>5.1f}%  "
                  f"PPT={r['ppt']:>+7.4f}  PF={pf}  "
                  f"delta={delta:>+7.4f}{tag}")

    print(f"\n  => P2 best S3: thr_L={p2_best_cfg[0]:.2f}  thr_S={p2_best_cfg[1]:.2f}  "
          f"PPT={p2_best_ppt:+.4f}  delta={p2_best_ppt-base['ppt']:+.4f}")

    # ── Phase 3: Specific candidate configs ──────────────────────────────────
    print(f"\n{SEP}")
    print(f"  PHASE 3 -- VALIDASI CONFIG KANDIDAT (8 config spesifik)")
    print(SEP)

    s3_dir = p2_best_cfg  # (thr_long, thr_short)

    candidates = {
        "BASE (0.45/0.45 all)":       _cfg_uniform(0.45),
        "A: Sym-Best all states":     _cfg_from(
                                          {s: best_sym[s][0] for s in range(4)}, {}),
        "B: Sym+S3-dir":              _cfg_from(
                                          {s: best_sym[s][0] for s in range(4)},
                                          {3: (s3_dir[0], s3_dir[1])}),
        "C: Sym+S3-dir+S3long-0.42": _cfg_from(
                                          {s: best_sym[s][0] for s in range(4)},
                                          {3: (0.42, s3_dir[1])}),
        "D: S1=0.58, rest sym":       _cfg_from(
                                          {s: best_sym[s][0] for s in range(4)},
                                          {1: (0.58, 0.58)}),
        "E: Conservative (0.50 all)": _cfg_uniform(0.50),
        "F: S0=0.55,S1=0.55,S2=0.47,S3=dir": _cfg_from(
                                          {0: 0.55, 1: 0.55, 2: 0.47, 3: 0.45},
                                          {3: (s3_dir[0], s3_dir[1])}),
        "G: Sym+S3=(0.40,sym_s)":    _cfg_from(
                                          {s: best_sym[s][0] for s in range(4)},
                                          {3: (0.40, best_sym[3][0])}),
    }

    print(f"\n  {'Config':<30} {'N':>6}  {'WR%':>5}  {'PnL':>10}  {'PPT':>7}  {'PF':>5}  delta")
    print(f"  {SEP2}")

    p3_results = {}
    for name, cfg in candidates.items():
        r = simulate_with_hmm_thr(coin_data, cfg)
        d = r["ppt"] - base["ppt"]
        p3_results[name] = r
        pf = f"{r['pf']:.3f}" if r["pf"] != float("inf") else " INF"
        print(f"  {name:<30} n={r['n']:>6,}  WR={r['wr']:>5.1f}%  "
              f"PnL=${r['pnl']:>9.1f}  PPT={r['ppt']:>+7.4f}  PF={pf}  "
              f"delta={d:>+7.4f}")

    # ── Final Recommendation ──────────────────────────────────────────────────
    best_name = max(p3_results, key=lambda k: p3_results[k]["ppt"])
    r_best    = p3_results[best_name]
    cfg_best  = candidates[best_name]

    print(f"\n{SEP}")
    print(f"  REKOMENDASI FINAL")
    print(SEP)
    print(f"\n  Config terbaik: {best_name}")
    print(f"\n  Per-state threshold:")
    for s in range(4):
        tl, ts = cfg_best.get(s, (0.45, 0.45))
        dir_tag = "" if abs(tl - ts) < 0.001 else f"  (dir-aware: L={tl:.2f} vs S={ts:.2f})"
        print(f"    S{s} {HMM_NAMES[s]:<14}: thr_long={tl:.2f}  thr_short={ts:.2f}{dir_tag}")

    print(f"\n  Perbandingan:")
    print(f"    Baseline : n={base['n']:,}  WR={base['wr']:.1f}%  PPT=${base['ppt']:+.4f}  "
          f"PF={base['pf']:.3f}  PnL=${base['pnl']:.1f}")
    print(f"    Best HMM : n={r_best['n']:,}  WR={r_best['wr']:.1f}%  PPT=${r_best['ppt']:+.4f}  "
          f"PF={r_best['pf']:.3f}  PnL=${r_best['pnl']:.1f}")
    d_ppt = r_best["ppt"] - base["ppt"]
    d_pnl = r_best["pnl"] - base["pnl"]
    d_wr  = r_best["wr"]  - base["wr"]
    print(f"    Delta    : n={r_best['n']-base['n']:+,}  WR={d_wr:+.1f}pp  "
          f"PPT=${d_ppt:+.4f}  PnL=${d_pnl:+.1f}")

    # ── Save ─────────────────────────────────────────────────────────────────
    out = {
        "created":      str(pd.Timestamp.now()),
        "lgbm_run":     LGBM_RUN,
        "phase1_best_symmetric": {str(s): {"thr": best_sym[s][0],
                                           "ppt": best_sym[s][1]["ppt"]} for s in range(4)},
        "phase2_s3_best": {"thr_long": s3_dir[0], "thr_short": s3_dir[1],
                           "ppt": p2_best_ppt},
        "best_config_name": best_name,
        "best_config": {str(k): list(v) for k, v in cfg_best.items()},
        "baseline_stats": base,
        "best_stats": r_best,
        "phase3_all": {
            name: {"ppt": r["ppt"], "wr": r["wr"], "n": r["n"],
                   "pnl": r["pnl"], "pf": r["pf"]}
            for name, r in p3_results.items()
        },
    }
    out_path = RUN_DIR / "hmm_threshold_best.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")
    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()
