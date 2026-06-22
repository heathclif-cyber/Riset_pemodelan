"""
pipeline/05g_hmm_promote_dynsize_sweep.py

1. Side-by-side OOF eval: HMM Config B (deploy) vs Config D (sweep winner)
   Stages: HMM-only | HMM+Guardian | HMM+Guardian+DynSize
2. DynSize re-sweep on OOF 36f stack with frozen Config B + Guardian v2

Genuine: OOF only, holdout NOT touched.
"""
import json
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from pathlib import Path

import joblib
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
    GUARDIAN_ACTIVATION_ATR,
)
from core.evaluator import simulate_trades_swing
from core.utils import ensure_utc_index

LGBM_RUN = "tb_lgbm_genuine_v2"
GUARDIAN_RUN = "tb_guardian_genuine_v2_hmm_v2"
LGBM_DIR = MODEL_DIR / "runs" / LGBM_RUN
GUARDIAN_DIR = MODEL_DIR / "runs" / GUARDIAN_RUN
INFERENCE_CFG = MODEL_DIR / "inference_config.json"

HMM_CONFIG_B = {
    0: (0.55, 0.55),
    1: (0.55, 0.55),
    2: (0.50, 0.50),
    3: (0.45, 0.50),
    -1: (0.45, 0.45),
}

DYNAMIC_FEATS = [
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct",
    "direction", "entry_price_ratio",
]

DEFAULT_DYNSIZE = {
    "conf_window": 0.10,
    "conf_max_mult": 0.5,
    "regime_mult": {0: 0.75, 1: 1.0, 2: 1.0, "3_long": 1.5, "3_short": 0.75, -1: 0.80},
    "clamp_min": 0.5,
    "clamp_max": 2.0,
}

DYNSIZE_GRID = [
    {"name": "current_deploy", **DEFAULT_DYNSIZE},
    {"name": "cw_0.08", "conf_window": 0.08, "conf_max_mult": 0.5,
     "regime_mult": DEFAULT_DYNSIZE["regime_mult"], "clamp_min": 0.5, "clamp_max": 2.0},
    {"name": "cw_0.12", "conf_window": 0.12, "conf_max_mult": 0.5,
     "regime_mult": DEFAULT_DYNSIZE["regime_mult"], "clamp_min": 0.5, "clamp_max": 2.0},
    {"name": "cm_0.40", "conf_window": 0.10, "conf_max_mult": 0.4,
     "regime_mult": DEFAULT_DYNSIZE["regime_mult"], "clamp_min": 0.5, "clamp_max": 2.0},
    {"name": "cm_0.60", "conf_window": 0.10, "conf_max_mult": 0.6,
     "regime_mult": DEFAULT_DYNSIZE["regime_mult"], "clamp_min": 0.5, "clamp_max": 2.0},
    {"name": "s3l_1.25", "conf_window": 0.10, "conf_max_mult": 0.5,
     "regime_mult": {0: 0.75, 1: 1.0, 2: 1.0, "3_long": 1.25, "3_short": 0.75, -1: 0.80},
     "clamp_min": 0.5, "clamp_max": 2.0},
    {"name": "s3l_1.75", "conf_window": 0.10, "conf_max_mult": 0.5,
     "regime_mult": {0: 0.75, 1: 1.0, 2: 1.0, "3_long": 1.75, "3_short": 0.75, -1: 0.80},
     "clamp_min": 0.5, "clamp_max": 2.0},
    {"name": "s3s_0.50", "conf_window": 0.10, "conf_max_mult": 0.5,
     "regime_mult": {0: 0.75, 1: 1.0, 2: 1.0, "3_long": 1.5, "3_short": 0.50, -1: 0.80},
     "clamp_min": 0.5, "clamp_max": 2.0},
    {"name": "s3s_1.00", "conf_window": 0.10, "conf_max_mult": 0.5,
     "regime_mult": {0: 0.75, 1: 1.0, 2: 1.0, "3_long": 1.5, "3_short": 1.00, -1: 0.80},
     "clamp_min": 0.5, "clamp_max": 2.0},
    {"name": "clamp_1.5", "conf_window": 0.10, "conf_max_mult": 0.5,
     "regime_mult": DEFAULT_DYNSIZE["regime_mult"], "clamp_min": 0.5, "clamp_max": 1.5},
    {"name": "clamp_2.5", "conf_window": 0.10, "conf_max_mult": 0.5,
     "regime_mult": DEFAULT_DYNSIZE["regime_mult"], "clamp_min": 0.5, "clamp_max": 2.5},
    {"name": "combo_best_cw_cm", "conf_window": 0.08, "conf_max_mult": 0.6,
     "regime_mult": {0: 0.75, 1: 1.0, 2: 1.0, "3_long": 1.75, "3_short": 0.75, -1: 0.80},
     "clamp_min": 0.5, "clamp_max": 2.0},
]


def load_hmm_config_d() -> dict:
    path = LGBM_DIR / "hmm_threshold_best.json"
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {int(k): tuple(v) for k, v in data["best_config"].items()}


def load_guardian_params() -> dict:
    with open(INFERENCE_CFG, encoding="utf-8") as f:
        inf = json.load(f)
    g = inf.get("guardian", {})
    return {
        "exit_threshold": float(g.get("exit_threshold", 0.55)),
        "min_hold_bars": int(g.get("min_hold_bars", 2)),
    }


def _apply_hmm_thr(p0, p2, hmm_enc, hmm_cfg):
    n = len(p0)
    default_tl, default_ts = hmm_cfg[-1]
    tl_arr = np.full(n, default_tl, dtype=np.float32)
    ts_arr = np.full(n, default_ts, dtype=np.float32)
    for state, (tl, ts) in hmm_cfg.items():
        if state == -1:
            continue
        mask = hmm_enc == state
        tl_arr[mask] = tl
        ts_arr[mask] = ts
    long_mask = p2 >= tl_arr
    short_mask = (p0 >= ts_arr) & ~long_mask
    y = np.ones(n, dtype=np.int32)
    y[long_mask] = 2
    y[short_mask] = 0
    return y, tl_arr, ts_arr


def _compute_dynamic_modal(p0, p2, hmm_enc, y_pred, base_modal, ds_cfg, tl_arr, ts_arr):
    n = len(p0)
    rm = ds_cfg["regime_mult"]
    long_mask = y_pred == 2
    short_mask = y_pred == 0
    conf = np.where(long_mask, p2, np.where(short_mask, p0, 0.0)).astype(np.float32)
    thr = np.where(long_mask, tl_arr, ts_arr)
    cw = ds_cfg["conf_window"]
    cm = ds_cfg["conf_max_mult"]
    c_mult = 1.0 + np.clip((conf - thr) / cw, 0.0, cm)
    r_mult = np.full(n, rm[-1], dtype=np.float64)
    r_mult[hmm_enc == 0] = rm[0]
    r_mult[hmm_enc == 1] = rm[1]
    r_mult[hmm_enc == 2] = rm[2]
    r_mult[(hmm_enc == 3) & long_mask] = rm.get("3_long", 1.5)
    r_mult[(hmm_enc == 3) & short_mask] = rm.get("3_short", 0.75)
    total_mult = np.clip(r_mult * c_mult, ds_cfg["clamp_min"], ds_cfg["clamp_max"])
    modal_arr = (base_modal * total_mult).astype(np.float32)
    modal_arr[y_pred == 1] = base_modal
    return modal_arr


def _summarize(trades, base_modal=MODAL_PER_TRADE):
    if not trades:
        return {"n": 0, "wr": 0, "pnl": 0, "ppt": 0, "pf": 0,
                "sl_pct": 0, "avg_modal": base_modal, "ppt_norm": 0}
    n = len(trades)
    wins = sum(1 for t in trades if t["net_pnl"] > 0)
    sl_hit = sum(1 for t in trades if t["outcome"] == "LOSS")
    gpnl = sum(t["net_pnl"] for t in trades if t["net_pnl"] > 0)
    lloss = sum(abs(t["net_pnl"]) for t in trades if t["net_pnl"] < 0)
    tpnl = sum(t["net_pnl"] for t in trades)
    pf = gpnl / lloss if lloss > 0 else float("inf")
    modals = [t.get("modal_used", base_modal) for t in trades]
    avg_modal = float(np.mean(modals))
    ppt_norm = (tpnl / n) * (base_modal / avg_modal) if avg_modal > 0 else 0.0
    return {
        "n": n, "wr": wins / n * 100, "pnl": tpnl,
        "ppt": tpnl / n, "pf": pf, "sl_pct": sl_hit / n * 100,
        "avg_modal": avg_modal, "ppt_norm": ppt_norm,
    }


def preload_coins(oof_pred_df, g_static_feats):
    coins = []
    for sym in ALL_COINS:
        path = LABEL_DIR / f"{sym}_features_v3.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = ensure_utc_index(df).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]
        if df.empty:
            continue
        sym_oof = oof_pred_df[oof_pred_df["coin"] == sym]
        sym_oof = sym_oof[sym_oof["has_oof"] == True][["p0", "p2"]]
        proba = sym_oof.reindex(df.index)
        has_oof = proba["p0"].notna()
        df_oof = df[has_oof].copy()
        n = len(df_oof)
        if n < 30:
            continue
        X_grd = np.zeros((n, len(g_static_feats)), dtype=np.float64)
        for idx, col in enumerate(g_static_feats):
            if col in df_oof.columns:
                X_grd[:, idx] = df_oof[col].ffill().fillna(0).values.astype(np.float64)
        coins.append({
            "sym": sym,
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
            "X_grd": X_grd,
        })
    return coins


def eval_hmm_config(coins, hmm_cfg, g_model, g_scaler, g_params, ds_cfg):
    all_hmm, all_full, all_dyn = [], [], []
    common_base = dict(
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        max_hold=MAX_HOLDING_BARS,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
    )
    for c in coins:
        y_hmm, tl_arr, ts_arr = _apply_hmm_thr(c["p0"], c["p2"], c["hmm"], hmm_cfg)
        kw = dict(
            close=c["close"], high=c["high"], low=c["low"], atr=c["atr"],
            h4_swing_highs=c["h4_sh"], h4_swing_lows=c["h4_sl"],
            **common_base,
        )
        r_hmm = simulate_trades_swing(y_pred=y_hmm, guardian_enabled=False, **kw)
        r_full = simulate_trades_swing(
            y_pred=y_hmm,
            guardian_enabled=True,
            guardian_model=g_model, guardian_scaler=g_scaler,
            X_guardian=c["X_grd"],
            guardian_exit_threshold=g_params["exit_threshold"],
            guardian_min_hold_bars=g_params["min_hold_bars"],
            guardian_activation_atr=GUARDIAN_ACTIVATION_ATR,
            **kw,
        )
        modal_arr = _compute_dynamic_modal(
            c["p0"], c["p2"], c["hmm"], y_hmm, MODAL_PER_TRADE,
            ds_cfg, tl_arr, ts_arr,
        )
        r_dyn = simulate_trades_swing(
            y_pred=y_hmm,
            guardian_enabled=True,
            guardian_model=g_model, guardian_scaler=g_scaler,
            X_guardian=c["X_grd"],
            guardian_exit_threshold=g_params["exit_threshold"],
            guardian_min_hold_bars=g_params["min_hold_bars"],
            guardian_activation_atr=GUARDIAN_ACTIVATION_ATR,
            modal_arr=modal_arr,
            **kw,
        )
        all_hmm.extend(r_hmm.get("trades", []))
        all_full.extend(r_full.get("trades", []))
        all_dyn.extend(r_dyn.get("trades", []))
    return {
        "hmm": _summarize(all_hmm),
        "full": _summarize(all_full),
        "dyn": _summarize(all_dyn),
    }


def decide_hmm_promote(res_b, res_d):
    """Keep B unless D wins full pipeline with acceptable trade count."""
    b_dyn = res_b["dyn"]
    d_dyn = res_d["dyn"]
    trade_ratio = d_dyn["n"] / b_dyn["n"] if b_dyn["n"] else 0
    promote_d = (
        d_dyn["ppt_norm"] > b_dyn["ppt_norm"]
        and d_dyn["pf"] >= b_dyn["pf"] * 0.98
        and d_dyn["wr"] >= b_dyn["wr"] - 1.0
        and trade_ratio >= 0.80
    )
    reason = []
    if promote_d:
        reason.append("Config D wins DYN ppt_norm with acceptable PF/WR/trades")
    else:
        if d_dyn["ppt_norm"] <= b_dyn["ppt_norm"]:
            reason.append(f"D ppt_norm {d_dyn['ppt_norm']:.4f} <= B {b_dyn['ppt_norm']:.4f}")
        if d_dyn["pf"] < b_dyn["pf"] * 0.98:
            reason.append(f"D PF {d_dyn['pf']:.3f} < B PF {b_dyn['pf']:.3f}")
        if trade_ratio < 0.80:
            reason.append(f"D trades {d_dyn['n']:,} < 80% of B {b_dyn['n']:,}")
        if d_dyn["wr"] < b_dyn["wr"] - 1.0:
            reason.append(f"D WR {d_dyn['wr']:.1f}% below B {b_dyn['wr']:.1f}%")
    return promote_d, reason


def update_inference_dynsize(best_ds):
    with open(INFERENCE_CFG, encoding="utf-8") as f:
        cfg = json.load(f)
    rm = best_ds["regime_mult"]
    cfg["sizing"]["dynamic"]["conf_window"] = best_ds["conf_window"]
    cfg["sizing"]["dynamic"]["conf_max_mult"] = best_ds["conf_max_mult"]
    cfg["sizing"]["dynamic"]["clamp_min"] = best_ds["clamp_min"]
    cfg["sizing"]["dynamic"]["clamp_max"] = best_ds["clamp_max"]
    cfg["sizing"]["dynamic"]["regime_mult"] = {
        "0": rm[0], "1": rm[1], "2": rm[2],
        "3_long": rm["3_long"], "3_short": rm["3_short"], "-1": rm[-1],
    }
    note = cfg["sizing"].get("note", "")
    cfg["sizing"]["note"] = (
        f"OOF re-sweep 36f {datetime.now().strftime('%Y-%m-%d')}: "
        f"winner={best_ds.get('name', 'custom')}. " + note[:120]
    )
    cfg["_snapshot_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(INFERENCE_CFG, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    return cfg


def print_row(label, s):
    pf = f"{s['pf']:.3f}" if s["pf"] != float("inf") else "INF"
    print(f"  {label:<28} {s['n']:>7,}  {s['wr']:>5.1f}%  "
          f"${s['pnl']:>9.0f}  {s['ppt']:>+7.4f}  {s['ppt_norm']:>+8.4f}  "
          f"{pf:>5}  ${s['avg_modal']:>6.1f}")


def main():
    SEP = "=" * 78
    hmm_d = load_hmm_config_d()
    g_params = load_guardian_params()

    print(f"\n{SEP}")
    print("  05g: HMM Config B vs D + DynSize OOF Re-sweep (36f stack)")
    print(f"  Genuine OOF only | holdout NOT touched")
    print(SEP)

    g_model = joblib.load(GUARDIAN_DIR / "guardian.pkl")
    g_scaler = joblib.load(GUARDIAN_DIR / "guardian_scaler.pkl")
    with open(GUARDIAN_DIR / "guardian_features.json", encoding="utf-8") as f:
        g_all_feats = json.load(f)
    g_static_feats = [f for f in g_all_feats if f not in DYNAMIC_FEATS]

    oof_pred_df = pd.read_parquet(LGBM_DIR / "oof_predictions.parquet")
    print("  Preloading coin data...")
    coins = preload_coins(oof_pred_df, g_static_feats)
    print(f"  Coins loaded: {len(coins)}")

    # ── Part 1: HMM B vs D ───────────────────────────────────────────────────
    print(f"\n{SEP}")
    print("  PART 1: HMM Config B (deploy) vs Config D (sweep winner)")
    print(SEP)
    print(f"\n  {'Stage':<28} {'N':>7}  {'WR%':>5}  {'PnL':>10}  {'PPT':>7}  "
          f"{'PPT_norm':>8}  {'PF':>5}  {'Avg$':>6}")

    res_b = eval_hmm_config(coins, HMM_CONFIG_B, g_model, g_scaler, g_params, DEFAULT_DYNSIZE)
    res_d = eval_hmm_config(coins, hmm_d, g_model, g_scaler, g_params, DEFAULT_DYNSIZE)

    for stage in ["hmm", "full", "dyn"]:
        print(f"\n  --- Config B | {stage.upper()} ---")
        print_row(stage, res_b[stage])
    for stage in ["hmm", "full", "dyn"]:
        print(f"\n  --- Config D | {stage.upper()} ---")
        print_row(stage, res_d[stage])

    promote_d, reasons = decide_hmm_promote(res_b, res_d)
    hmm_decision = "PROMOTE_D" if promote_d else "KEEP_B"
    print(f"\n  HMM DECISION: {hmm_decision}")
    for r in reasons:
        print(f"    - {r}")
    if promote_d:
        print("    ACTION: update inference_config HMM + retrain Guardian on new entry mix")
    else:
        print("    ACTION: keep Config B in inference_config (Guardian already aligned)")

    hmm_cfg_final = hmm_d if promote_d else HMM_CONFIG_B

    # ── Part 2: DynSize sweep (on chosen HMM config) ─────────────────────────
    print(f"\n{SEP}")
    print(f"  PART 2: DynSize re-sweep ({len(DYNSIZE_GRID)} configs, HMM={hmm_decision})")
    print(SEP)

    sweep_rows = []
    for ds in DYNSIZE_GRID:
        res = eval_hmm_config(coins, hmm_cfg_final, g_model, g_scaler, g_params, ds)
        s = res["dyn"]
        sweep_rows.append({
            "name": ds["name"],
            "config": ds,
            **s,
        })
        print(f"  {ds['name']:<18} N={s['n']:>6,}  WR={s['wr']:>5.1f}%  "
              f"PPT_norm={s['ppt_norm']:>+7.4f}  PF={s['pf']:.3f}  "
              f"AvgModal=${s['avg_modal']:.2f}")

    sweep_df = pd.DataFrame(sweep_rows).sort_values("ppt_norm", ascending=False)
    best = sweep_df.iloc[0]
    current = sweep_df[sweep_df["name"] == "current_deploy"].iloc[0]
    delta_ppt = best["ppt_norm"] - current["ppt_norm"]

    print(f"\n  DynSize WINNER: {best['name']}")
    print(f"    PPT_norm={best['ppt_norm']:+.4f}  PF={best['pf']:.3f}  "
          f"vs current {current['ppt_norm']:+.4f} (delta {delta_ppt:+.4f})")

    update_ds = delta_ppt >= 0.002 and best["pf"] >= current["pf"] * 0.99
    if update_ds:
        update_inference_dynsize(best["config"])
        print(f"  DynSize UPDATED in inference_config.json")
    else:
        print(f"  DynSize KEEP current_deploy (delta {delta_ppt:+.4f} below 0.002 gate)")

    # ── Save artifact ────────────────────────────────────────────────────────
    out = {
        "created": datetime.now().isoformat(),
        "methodology": "oof_genuine_hmm_promote_dynsize_sweep",
        "holdout_used": False,
        "lgbm_run": LGBM_RUN,
        "guardian_run": GUARDIAN_RUN,
        "hmm_config_b": {str(k): list(v) for k, v in HMM_CONFIG_B.items()},
        "hmm_config_d": {str(k): list(v) for k, v in hmm_d.items()},
        "hmm_decision": hmm_decision,
        "hmm_promote_reasons": reasons,
        "config_b_results": res_b,
        "config_d_results": res_d,
        "dynsize_sweep": sweep_rows,
        "dynsize_winner": best["name"],
        "dynsize_updated": update_ds,
        "dynsize_best_config": best["config"] if update_ds else DEFAULT_DYNSIZE,
    }
    out_path = LGBM_DIR / "hmm_promote_dynsize_sweep.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")
    print(f"\n{SEP}\n")


if __name__ == "__main__":
    main()