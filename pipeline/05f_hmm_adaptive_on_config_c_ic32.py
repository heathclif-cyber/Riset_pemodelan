"""
pipeline/05f_hmm_adaptive_on_config_c_ic32.py — HMM Adaptive Threshold on Config C

Extends frozen Simons Config C with per-bar threshold adaptation:
  base thr[state] + offset(hmm_regime_enc, h4_trend, direction)

Phase A: H4-trend adaptive offsets (mirror FLIP logic at threshold layer)
Phase B: Per-state direction-aware sweep (S0/S2/S3) around Config C values

Prerequisite:
  models/runs/ic32_regime_v1/hmm_threshold_ic32_simons.json
  models/runs/ic32_regime_v1/oof_predictions.parquet

Jalankan:
  python pipeline/05f_hmm_adaptive_on_config_c_ic32.py
"""
import json
import sys
import warnings
from datetime import datetime
from itertools import product
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
CONFIG_C_PATH = RUN_DIR / "hmm_threshold_ic32_simons.json"
OUT_PATH = RUN_DIR / "hmm_adaptive_config_c_ic32.json"

HMM_NAMES = {0: "TRENDING_DN", 1: "RANGING_LOW", 2: "RANGING_HIGH", 3: "TRENDING_UP"}
RANGING = {1, 2}
TRENDING = {0, 3}


def load_config_c() -> dict:
    with open(CONFIG_C_PATH, encoding="utf-8") as f:
        raw = json.load(f)["best_config"]
    return {int(k): (float(v[0]), float(v[1])) for k, v in raw.items()}


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

        h4t = (
            df_oof["h4_trend"].fillna(0).values.astype(np.float32)
            if "h4_trend" in df_oof.columns
            else np.zeros(n, dtype=np.float32)
        )

        coin_data.append({
            "sym": sym,
            "n": n,
            "p0": proba["p0"][has_oof].values.astype(np.float32),
            "p2": proba["p2"][has_oof].values.astype(np.float32),
            "hmm": df_oof["hmm_regime_enc"].fillna(-1).values.astype(np.int8)
            if "hmm_regime_enc" in df_oof.columns else np.full(n, -1, np.int8),
            "h4": h4t,
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


def _build_thr_arrays(hmm: np.ndarray, h4: np.ndarray, base_cfg: dict,
                      offsets: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Per-bar thr_long / thr_short from base Config C + optional H4 adaptive offsets."""
    n = len(hmm)
    default_tl, default_ts = base_cfg[-1]
    tl_arr = np.full(n, default_tl, dtype=np.float32)
    ts_arr = np.full(n, default_ts, dtype=np.float32)

    for state, (tl, ts) in base_cfg.items():
        if state == -1:
            continue
        mask = hmm == state
        tl_arr[mask] = tl
        ts_arr[mask] = ts

    if not offsets:
        return tl_arr, ts_arr

    rng_wt = float(offsets.get("ranging_wt_tight", 0.0))
    rng_ct = float(offsets.get("ranging_ct_ease", 0.0))
    trd_wt = float(offsets.get("trending_wt_ease", 0.0))
    trd_ct = float(offsets.get("trending_ct_tight", 0.0))

    for i in range(n):
        state = int(hmm[i])
        h = float(h4[i])
        if abs(h) < 1e-9:
            continue

        tl, ts = float(tl_arr[i]), float(ts_arr[i])

        if state in RANGING:
            # Swing mode: counter-trend easier, with-trend harder
            if h > 0:
                tl -= rng_ct   # LONG counter uptrend (SHORT-bias fix area)
                ts += rng_wt   # SHORT with downtrend... h4>0: SHORT is counter
                ts -= rng_ct   # SHORT counter uptrend easier
            else:
                ts -= rng_ct   # SHORT counter (with downtrend) easier
                tl += rng_wt   # LONG with downtrend harder
                tl -= rng_ct   # LONG counter downtrend easier
        elif state in TRENDING:
            # Momentum mode: with-trend easier, counter-trend harder
            if state == 3:  # TRENDING_UP
                if h > 0:
                    tl -= trd_wt
                    ts += trd_ct
                else:
                    tl += trd_ct
                    ts -= trd_wt
            else:  # TRENDING_DOWN (S0)
                if h < 0:
                    ts -= trd_wt
                    tl += trd_ct
                else:
                    ts += trd_ct
                    tl -= trd_wt

        tl_arr[i] = np.clip(tl, 0.35, 0.85)
        ts_arr[i] = np.clip(ts, 0.35, 0.85)

    return tl_arr, ts_arr


def simulate(coin_data: list, base_cfg: dict, offsets: dict | None = None) -> dict:
    agg_trades = []
    n_long_sig = n_short_sig = 0

    for cd in coin_data:
        n, p0, p2, hmm, h4 = cd["n"], cd["p0"], cd["p2"], cd["hmm"], cd["h4"]
        tl_arr, ts_arr = _build_thr_arrays(hmm, h4, base_cfg, offsets)

        long_mask = p2 >= tl_arr
        short_mask = (p0 >= ts_arr) & ~long_mask
        n_long_sig += int(long_mask.sum())
        n_short_sig += int(short_mask.sum())

        y_pred = np.ones(n, dtype=np.int32)
        y_pred[long_mask] = 2
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
        return {"n": 0, "wr": 0, "pnl": 0, "ppt": 0, "pf": 0,
                "long_sig": 0, "short_sig": 0, "short_long_ratio": 0}

    nt = len(agg_trades)
    wins = sum(1 for t in agg_trades if t["outcome"] == "WIN")
    gpnl = sum(t["net_pnl"] for t in agg_trades if t["net_pnl"] > 0)
    lloss = sum(abs(t["net_pnl"]) for t in agg_trades if t["net_pnl"] < 0)
    tpnl = sum(t["net_pnl"] for t in agg_trades)
    pf = gpnl / lloss if lloss > 0 else float("inf")
    ratio = n_short_sig / max(n_long_sig, 1)
    return {
        "n": nt, "wr": wins / nt * 100, "pnl": tpnl, "ppt": tpnl / nt, "pf": pf,
        "long_sig": n_long_sig, "short_sig": n_short_sig, "short_long_ratio": ratio,
    }


def _row(label, r, delta_ppt=None, mark=""):
    pf = f"{r['pf']:.3f}" if r["pf"] != float("inf") else " INF"
    d = f"  dPPT={delta_ppt:>+7.4f}" if delta_ppt is not None else ""
    print(
        f"  {label:<40} n={r['n']:>6,} WR={r['wr']:>5.1f}% "
        f"PnL=${r['pnl']:>9.1f} PPT={r['ppt']:>+7.4f} PF={pf} "
        f"S/L={r['short_long_ratio']:.2f}{d}{mark}"
    )


def main():
    if not OOF_PATH.exists():
        raise FileNotFoundError(f"Missing {OOF_PATH}")
    if not CONFIG_C_PATH.exists():
        raise FileNotFoundError(f"Missing {CONFIG_C_PATH} — run 05e first")

    base_cfg = load_config_c()
    oof_df = pd.read_parquet(OOF_PATH)
    coin_data = preload_coin_data(oof_df)
    print("=" * 92)
    print("  HMM ADAPTIVE THRESHOLD on Config C — ic32 OOF signal-only")
    print("=" * 92)
    print(f"  Coins: {len(coin_data)}")
    print(f"  Base Config C: {base_cfg}")

    static = simulate(coin_data, base_cfg, offsets=None)
    print("\n  BASELINE Config C (static per-state):")
    _row("Config C static", static)

    # ── Phase A: H4-trend adaptive offsets ─────────────────────────────────────
    print("\n" + "=" * 92)
    print("  PHASE A — H4-trend adaptive offsets on Config C base")
    print("  RANGING: wt_tight / ct_ease | TRENDING: wt_ease / ct_tight")
    print("=" * 92)

    grid_rng_wt = [0.0, 0.05, 0.07]
    grid_rng_ct = [0.0, 0.05, 0.07]
    grid_trd_wt = [0.0, 0.05]
    grid_trd_ct = [0.0, 0.05, 0.07]

    phase_a_results = []
    best_a = {"ppt": -999, "offsets": None, "r": None}
    combos = list(product(grid_rng_wt, grid_rng_ct, grid_trd_wt, grid_trd_ct))
    total_a = len(combos) - 1

    for idx, (rng_wt, rng_ct, trd_wt, trd_ct) in enumerate(combos, 1):
        if rng_wt == 0 and rng_ct == 0 and trd_wt == 0 and trd_ct == 0:
            continue
        offsets = {
            "ranging_wt_tight": rng_wt,
            "ranging_ct_ease": rng_ct,
            "trending_wt_ease": trd_wt,
            "trending_ct_tight": trd_ct,
        }
        r = simulate(coin_data, base_cfg, offsets)
        entry = {"offsets": offsets, **r}
        phase_a_results.append(entry)
        if r["ppt"] > best_a["ppt"] and r["n"] >= static["n"] * 0.40:
            best_a = {"ppt": r["ppt"], "offsets": offsets, "r": r}
        if idx % 10 == 0 or idx == total_a:
            print(f"  Phase A progress: {idx}/{total_a}", flush=True)

    phase_a_sorted = sorted(phase_a_results, key=lambda x: -x["ppt"])[:15]
    print(f"\n  Top 15 adaptive offset combos (of {len(phase_a_results)}):")
    print(f"  {'rng_wt':>6} {'rng_ct':>6} {'trd_wt':>6} {'trd_ct':>6}  "
          f"{'n':>6}  {'WR%':>5}  {'PPT':>7}  {'PF':>5}  S/L")
    print("  " + "-" * 72)
    for row in phase_a_sorted:
        o = row["offsets"]
        pf = f"{row['pf']:.3f}" if row["pf"] != float("inf") else " INF"
        print(
            f"  {o['ranging_wt_tight']:>6.2f} {o['ranging_ct_ease']:>6.2f} "
            f"{o['trending_wt_ease']:>6.2f} {o['trending_ct_tight']:>6.2f}  "
            f"{row['n']:>6,}  {row['wr']:>5.1f}  {row['ppt']:>+7.4f}  {pf:>5}  "
            f"{row['short_long_ratio']:.2f}"
        )

    print("\n  Phase A best:")
    _row("A-best adaptive", best_a["r"], best_a["r"]["ppt"] - static["ppt"], " <<BEST_A")
    print(f"    offsets: {best_a['offsets']}")

    # ── Phase B: Per-state direction-aware around Config C ───────────────────
    print("\n" + "=" * 92)
    print("  PHASE B — Direction-aware per state (S0/S2/S3 sweep, S1 fixed at C)")
    print("=" * 92)

    s1 = base_cfg[1]
    dir_grid = [0.62, 0.65, 0.68, 0.72]
    phase_b_results = []
    best_b = {"ppt": -999, "cfg": None, "r": None, "label": ""}

    sweep_specs = [
        ("S0 dir-aware", 0, base_cfg[0][0]),
        ("S2 dir-aware", 2, base_cfg[2][0]),
        ("S3 dir-aware", 3, base_cfg[3][0]),
    ]

    for label, state, sym_base in sweep_specs:
        print(f"\n  {label} (sym base={sym_base:.2f}, S1 fixed {s1}):")
        local_best = {"ppt": -999}
        for tl in dir_grid:
            for ts in dir_grid:
                if ts < tl - 0.12:
                    continue
                cfg = dict(base_cfg)
                cfg[state] = (tl, ts)
                r = simulate(coin_data, cfg, offsets=None)
                tag = ""
                if r["ppt"] > local_best["ppt"] and r["n"] >= static["n"] * 0.40:
                    local_best = {"ppt": r["ppt"], "tl": tl, "ts": ts, "r": r}
                    tag = " *"
                if abs(tl - sym_base) < 0.001 and abs(ts - sym_base) < 0.001:
                    tag = " [sym]"
                pf = f"{r['pf']:.3f}" if r["pf"] != float("inf") else " INF"
                pass  # only print local best per state below

        if local_best.get("r"):
            entry = {
                "label": label,
                "state": state,
                "thr_long": local_best["tl"],
                "thr_short": local_best["ts"],
                **local_best["r"],
            }
            phase_b_results.append(entry)
            cfg_b = dict(base_cfg)
            cfg_b[state] = (local_best["tl"], local_best["ts"])
            if local_best["ppt"] > best_b["ppt"]:
                best_b = {
                    "ppt": local_best["ppt"],
                    "cfg": cfg_b,
                    "r": local_best["r"],
                    "label": label,
                }
            print(f"    => best {label}: L={local_best['tl']:.2f} S={local_best['ts']:.2f} "
                  f"PPT={local_best['ppt']:+.4f}")

    # Combined: apply all 3 state bests at once
    cfg_combined = dict(base_cfg)
    for entry in phase_b_results:
        cfg_combined[entry["state"]] = (entry["thr_long"], entry["thr_short"])
    r_combined = simulate(coin_data, cfg_combined, offsets=None)

    # ── Phase C: Combined adaptive + dir-aware ────────────────────────────────
    print("\n" + "=" * 92)
    print("  PHASE C — Candidate validation")
    print("=" * 92)

    r_a_static_cfg = simulate(coin_data, base_cfg, offsets=best_a["offsets"])
    candidates = {
        "C-static (Config C)": (base_cfg, None),
        "A-best H4 adaptive": (base_cfg, best_a["offsets"]),
        f"B-best {best_b['label']}": (best_b["cfg"], None) if best_b["cfg"] else (base_cfg, None),
        "B-combined dir-aware": (cfg_combined, None),
        "C-combined: dir + H4 adaptive": (cfg_combined, best_a["offsets"]),
    }

    phase_c = {}
    winner_name, winner_r, winner_ppt = "C-static (Config C)", static, static["ppt"]
    winner_cfg, winner_offsets = base_cfg, None

    for name, (cfg, offsets) in candidates.items():
        r = simulate(coin_data, cfg, offsets)
        phase_c[name] = r
        mark = ""
        if r["ppt"] > winner_ppt and r["n"] >= static["n"] * 0.35:
            winner_ppt, winner_name, winner_r = r["ppt"], name, r
            winner_cfg, winner_offsets = cfg, offsets
            mark = " <<WINNER"
        _row(name, r, r["ppt"] - static["ppt"], mark)

    out = {
        "created": str(datetime.now()),
        "methodology": "hmm_adaptive_threshold_on_config_c",
        "run": "ic32_regime_v1",
        "base_config_c": {str(k): list(v) for k, v in base_cfg.items()},
        "config_c_static_stats": static,
        "phase_a_best_offsets": best_a["offsets"],
        "phase_a_best_stats": best_a["r"],
        "phase_a_top15": [
            {"offsets": x["offsets"], "n": x["n"], "wr": x["wr"], "ppt": x["ppt"],
             "pf": x["pf"], "short_long_ratio": x["short_long_ratio"]}
            for x in phase_a_sorted
        ],
        "phase_b_per_state": phase_b_results,
        "phase_b_combined_cfg": {str(k): list(v) for k, v in cfg_combined.items()},
        "phase_b_combined_stats": r_combined,
        "phase_c_candidates": {k: v for k, v in phase_c.items()},
        "winner_name": winner_name,
        "winner_cfg": {str(k): list(v) for k, v in winner_cfg.items()},
        "winner_offsets": winner_offsets,
        "winner_stats": winner_r,
        "delta_ppt_vs_config_c": winner_r["ppt"] - static["ppt"],
        "note": "Signal-only OOF. Next: full-stack eval for winner via 08d script.",
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)

    print(f"\n  WINNER: {winner_name}")
    _row("FINAL", winner_r, winner_r["ppt"] - static["ppt"])
    print(f"\n  Saved: {OUT_PATH}")
    print("=" * 92)


if __name__ == "__main__":
    main()