"""
pipeline/05i_lstm_score_fusion_eval.py

OOF eval: LSTM boost/penalty pada skor LGBM (bukan complement entry terpisah).

Stack frozen: HMM Config B + Guardian v2 + DynSize cm_0.60.
Sweep fusion params pada OOF LGBM + OOF LSTM (tb_lstm_genuine_v2).

Genuine: OOF only, holdout NOT touched.
"""
import itertools
import json
import sys
import warnings
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
from core.cascade_utils import apply_lstm_conf_fusion_post, apply_lstm_proba_fusion_pre
from core.evaluator import simulate_trades_swing
from core.utils import ensure_utc_index

LGBM_RUN = "tb_lgbm_genuine_v2"
LSTM_RUN = "tb_lstm_genuine_v2"
GUARDIAN_RUN = "tb_guardian_genuine_v2_hmm_v2"
LGBM_DIR = MODEL_DIR / "runs" / LGBM_RUN
LSTM_DIR = MODEL_DIR / "runs" / LSTM_RUN
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

DYNSIZE_CFG = {
    "conf_window": 0.10,
    "conf_max_mult": 0.6,
    "regime_mult": {0: 0.75, 1: 1.0, 2: 1.0, "3_long": 1.5, "3_short": 0.75, -1: 0.80},
    "clamp_min": 0.5,
    "clamp_max": 2.0,
}

# Sweep grid (OOF only — holdout sealed)
FUSION_MODES = ["pre_hmm", "post_hmm"]
GATE_MODES = ["all_oof", "vol_spike2"]
AGREE_BOOST = [0.04, 0.06, 0.08, 0.10]
NEUTRAL_PEN = [0.04, 0.06, 0.08]
OPPOSITE_PEN = [0.06, 0.10, 0.14]

PPT_NORM_GATE = 0.002
MIN_TRADE_RATIO = 0.80


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
    conf = np.where(long_mask, p2, np.where(short_mask, p0, 0.0)).astype(np.float32)
    return y, conf, tl_arr, ts_arr


def _compute_dynamic_modal(p0, p2, hmm_enc, y_pred, base_modal, ds_cfg, tl_arr, ts_arr):
    n = len(p0)
    rm = ds_cfg["regime_mult"]
    long_mask = y_pred == 2
    short_mask = y_pred == 0
    conf = np.where(long_mask, p2, np.where(short_mask, p0, 0.0)).astype(np.float32)
    thr = np.where(long_mask, tl_arr, ts_arr)
    c_mult = 1.0 + np.clip((conf - thr) / ds_cfg["conf_window"], 0.0, ds_cfg["conf_max_mult"])
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


def _gate_mask(vol_spike: np.ndarray, lstm_valid: np.ndarray, gate_mode: str) -> np.ndarray:
    if gate_mode == "all_oof":
        return lstm_valid
    if gate_mode == "vol_spike2":
        return lstm_valid & (vol_spike >= 2.0)
    return lstm_valid


def preload_coins(lgbm_oof: pd.DataFrame, lstm_oof: pd.DataFrame, g_static_feats: list) -> list:
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

        sym_lgbm = lgbm_oof[lgbm_oof["coin"] == sym]
        sym_lgbm = sym_lgbm[sym_lgbm["has_oof"] == True][["p0", "p2"]]
        proba = sym_lgbm.reindex(df.index)
        has_oof = proba["p0"].notna()
        if has_oof.sum() < 30:
            continue

        sym_lstm = lstm_oof[lstm_oof["coin"] == sym][["p0", "p1", "p2", "vol_spike", "has_oof"]]
        lstm_aligned = sym_lstm.reindex(df.index[has_oof])
        lstm_p = lstm_aligned[["p0", "p1", "p2"]].values.astype(np.float32)
        lstm_valid = lstm_aligned["has_oof"].fillna(False).values.astype(bool)
        vol_spike = lstm_aligned["vol_spike"].fillna(-99).values.astype(np.float32)

        df_oof = df[has_oof].copy()
        n = len(df_oof)
        X_grd = np.zeros((n, len(g_static_feats)), dtype=np.float64)
        for idx, col in enumerate(g_static_feats):
            if col in df_oof.columns:
                X_grd[:, idx] = df_oof[col].ffill().fillna(0).values.astype(np.float64)

        if "log_ret_4" in df_oof.columns:
            ret4 = df_oof["log_ret_4"].fillna(0).values.astype(np.float32)
        else:
            ret4 = df_oof["close"].pct_change(4).fillna(0).values.astype(np.float32)

        coins.append({
            "sym": sym,
            "ts": df_oof.index,
            "p0": proba["p0"][has_oof].values.astype(np.float32),
            "p2": proba["p2"][has_oof].values.astype(np.float32),
            "ret4": ret4,
            "hmm": df_oof["hmm_regime_enc"].fillna(-1).values.astype(np.int8)
            if "hmm_regime_enc" in df_oof.columns else np.full(n, -1, np.int8),
            "lstm_p": lstm_p,
            "lstm_valid": lstm_valid,
            "vol_spike": vol_spike,
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


def build_signal_panel(coins: list, cfg: dict) -> pd.DataFrame:
    """Cross-sectional panel for rally diagnostics."""
    rows = []
    for c in coins:
        p0, p2 = c["p0"].copy(), c["p2"].copy()
        active = _gate_mask(c["vol_spike"], c["lstm_valid"], cfg.get("gate", "all_oof"))

        if cfg.get("fusion") == "baseline":
            y, _, _, _ = _apply_hmm_thr(p0, p2, c["hmm"], HMM_CONFIG_B)
        elif cfg["mode"] == "pre_hmm":
            p0, p2 = apply_lstm_proba_fusion_pre(
                p0, p2, c["lstm_p"],
                agree_boost=cfg["agree_boost"],
                neutral_pen=cfg["neutral_pen"],
                opposite_pen=cfg["opposite_pen"],
                active_mask=active,
            )
            y, _, _, _ = _apply_hmm_thr(p0, p2, c["hmm"], HMM_CONFIG_B)
        else:
            y, conf, tl, ts = _apply_hmm_thr(p0, p2, c["hmm"], HMM_CONFIG_B)
            y, _ = apply_lstm_conf_fusion_post(
                y, conf, c["lstm_p"], tl, ts,
                agree_boost=cfg["agree_boost"],
                neutral_pen=cfg["neutral_pen"],
                opposite_pen=cfg["opposite_pen"],
                active_mask=active,
            )

        for i, ts in enumerate(c["ts"]):
            rows.append({
                "ts": ts, "coin": c["sym"],
                "p0": float(c["p0"][i]), "p2": float(c["p2"][i]),
                "ret4": float(c["ret4"][i]),
                "y": int(y[i]),
            })
    return pd.DataFrame(rows)


def rally_metrics(panel: pd.DataFrame, baseline_panel: pd.DataFrame) -> dict:
    if panel.empty:
        return {"rally_bars": 0, "entry_unlock": 0, "extra_long": 0}

    pivot_p2 = panel.pivot_table(index="ts", columns="coin", values="p2", aggfunc="first")
    pivot_ret = panel.pivot_table(index="ts", columns="coin", values="ret4", aggfunc="first")
    frac_up = (pivot_ret > 0).mean(axis=1)
    max_p2 = pivot_p2.max(axis=1)
    rally_ts = frac_up.index[(frac_up >= 0.8) & (max_p2 < 0.45)]

    sub = panel[panel["ts"].isin(rally_ts)]
    base_sub = baseline_panel[baseline_panel["ts"].isin(rally_ts)]
    if sub.empty:
        return {"rally_bars": len(rally_ts), "entry_unlock": 0, "extra_long": 0}

    merged = sub.merge(
        base_sub[["ts", "coin", "y"]].rename(columns={"y": "y_base"}),
        on=["ts", "coin"], how="left",
    )
    extra_long = int(((merged["y"] == 2) & (merged["y_base"] != 2)).sum())
    return {
        "rally_bars": int(len(rally_ts)),
        "entry_unlock": extra_long,
        "extra_long": extra_long,
    }


def eval_config(coins: list, cfg: dict, g_model, g_scaler, g_params) -> tuple[list, dict]:
    all_trades = []
    signal_rows = []

    common_base = dict(
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        max_hold=MAX_HOLDING_BARS,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
    )

    for c in coins:
        p0, p2 = c["p0"].copy(), c["p2"].copy()
        active = _gate_mask(c["vol_spike"], c["lstm_valid"], cfg.get("gate", "all_oof"))

        if cfg.get("fusion") == "baseline":
            y, conf, tl_arr, ts_arr = _apply_hmm_thr(p0, p2, c["hmm"], HMM_CONFIG_B)
        elif cfg["mode"] == "pre_hmm":
            p0, p2 = apply_lstm_proba_fusion_pre(
                p0, p2, c["lstm_p"],
                agree_boost=cfg["agree_boost"],
                neutral_pen=cfg["neutral_pen"],
                opposite_pen=cfg["opposite_pen"],
                active_mask=active,
            )
            y, conf, tl_arr, ts_arr = _apply_hmm_thr(p0, p2, c["hmm"], HMM_CONFIG_B)
        else:
            y, conf, tl_arr, ts_arr = _apply_hmm_thr(p0, p2, c["hmm"], HMM_CONFIG_B)
            y, conf = apply_lstm_conf_fusion_post(
                y, conf, c["lstm_p"], tl_arr, ts_arr,
                agree_boost=cfg["agree_boost"],
                neutral_pen=cfg["neutral_pen"],
                opposite_pen=cfg["opposite_pen"],
                active_mask=active,
            )

        kw = dict(
            close=c["close"], high=c["high"], low=c["low"], atr=c["atr"],
            h4_swing_highs=c["h4_sh"], h4_swing_lows=c["h4_sl"],
            **common_base,
        )
        p0_eff = np.zeros_like(p0)
        p2_eff = np.zeros_like(p2)
        p0_eff[y == 0] = conf[y == 0]
        p2_eff[y == 2] = conf[y == 2]
        if cfg.get("fusion") != "baseline" and cfg.get("mode") == "pre_hmm":
            p0_eff, p2_eff = p0, p2
        modal_arr = _compute_dynamic_modal(
            p0_eff, p2_eff, c["hmm"], y, MODAL_PER_TRADE, DYNSIZE_CFG, tl_arr, ts_arr,
        )
        rep = simulate_trades_swing(
            y_pred=y,
            guardian_enabled=True,
            guardian_model=g_model, guardian_scaler=g_scaler,
            X_guardian=c["X_grd"],
            guardian_exit_threshold=g_params["exit_threshold"],
            guardian_min_hold_bars=g_params["min_hold_bars"],
            guardian_activation_atr=GUARDIAN_ACTIVATION_ATR,
            modal_arr=modal_arr,
            **kw,
        )
        all_trades.extend(rep.get("trades", []))

    return all_trades, {}


def config_label(cfg: dict) -> str:
    if cfg.get("fusion") == "baseline":
        return "baseline_no_lstm"
    return (
        f"{cfg['mode']}_{cfg['gate']}"
        f"_b{int(cfg['agree_boost']*100)}"
        f"_n{int(cfg['neutral_pen']*100)}"
        f"_o{int(cfg['opposite_pen']*100)}"
    )


def build_sweep_configs() -> list[dict]:
    configs = [{"fusion": "baseline", "label": "baseline_no_lstm"}]
    for mode, gate, ab, np_, op in itertools.product(
        FUSION_MODES, GATE_MODES, AGREE_BOOST, NEUTRAL_PEN, OPPOSITE_PEN,
    ):
        cfg = {
            "fusion": "lstm",
            "mode": mode,
            "gate": gate,
            "agree_boost": ab,
            "neutral_pen": np_,
            "opposite_pen": op,
        }
        cfg["label"] = config_label(cfg)
        configs.append(cfg)
    return configs


def passes_gate(variant: dict, baseline: dict) -> bool:
    trade_ratio = variant["n"] / baseline["n"] if baseline["n"] else 0
    return (
        variant["ppt_norm"] >= baseline["ppt_norm"] + PPT_NORM_GATE
        and trade_ratio >= MIN_TRADE_RATIO
        and variant["pf"] >= baseline["pf"] * 0.98
    )


def main():
    SEP = "=" * 78
    g_params = load_guardian_params()
    sweep_configs = build_sweep_configs()

    print(f"\n{SEP}")
    print("  05i: LSTM Score Fusion OOF Eval (boost/penalty on LGBM scores)")
    print(f"  Stack: HMM-B + Guardian + DynSize cm_0.60 | holdout NOT touched")
    print(f"  LSTM: {LSTM_RUN} | configs: {len(sweep_configs)}")
    print(SEP)

    g_model = joblib.load(GUARDIAN_DIR / "guardian.pkl")
    g_scaler = joblib.load(GUARDIAN_DIR / "guardian_scaler.pkl")
    with open(GUARDIAN_DIR / "guardian_features.json", encoding="utf-8") as f:
        g_all_feats = json.load(f)
    g_static_feats = [f for f in g_all_feats if f not in DYNAMIC_FEATS]

    lgbm_oof = pd.read_parquet(LGBM_DIR / "oof_predictions.parquet")
    lstm_oof = pd.read_parquet(LSTM_DIR / "oof_lstm_predictions.parquet")

    print("  Preloading coins...")
    coins = preload_coins(lgbm_oof, lstm_oof, g_static_feats)
    lstm_bars = sum(int(c["lstm_valid"].sum()) for c in coins)
    print(f"  Coins: {len(coins)} | LSTM OOF bars aligned: {lstm_bars:,}")

    results = []
    baseline_summary = None
    baseline_panel = None

    for i, cfg in enumerate(sweep_configs):
        trades, _ = eval_config(coins, cfg, g_model, g_scaler, g_params)
        sm = _summarize(trades)
        row = {"label": cfg["label"], **cfg, **sm}

        if cfg.get("fusion") == "baseline":
            baseline_summary = sm
            baseline_panel = build_signal_panel(coins, cfg)
            row["rally"] = rally_metrics(baseline_panel, baseline_panel)
        else:
            panel = build_signal_panel(coins, cfg)
            row["rally"] = rally_metrics(panel, baseline_panel)

        if baseline_summary and cfg.get("fusion") != "baseline":
            row["delta_ppt_norm"] = sm["ppt_norm"] - baseline_summary["ppt_norm"]
            row["delta_n"] = sm["n"] - baseline_summary["n"]
            row["passes_gate"] = passes_gate(sm, baseline_summary)

        results.append(row)
        if (i + 1) % 20 == 0 or i == 0:
            print(f"  [{i+1}/{len(sweep_configs)}] {cfg['label']}: "
                  f"N={sm['n']:,} PPT_norm={sm['ppt_norm']:+.4f} PF={sm['pf']:.3f}")

    results_sorted = sorted(
        [r for r in results if r.get("fusion") != "baseline"],
        key=lambda x: x["ppt_norm"],
        reverse=True,
    )
    winners = [r for r in results_sorted if r.get("passes_gate")]

    print(f"\n{SEP}")
    print("  BASELINE (no LSTM fusion)")
    print(SEP)
    b = baseline_summary
    print(f"  N={b['n']:,}  WR={b['wr']:.1f}%  PPT_norm={b['ppt_norm']:+.4f}  PF={b['pf']:.3f}")

    print(f"\n{SEP}")
    print("  TOP 15 BY PPT_norm (DYN stack)")
    print(SEP)
    print(f"  {'Label':<42} {'N':>7}  {'dN':>6}  {'PPT_norm':>8}  {'dPPT':>7}  {'PF':>5}  "
          f"{'Rally+':>6}  {'PASS':>4}")
    for r in results_sorted[:15]:
        pf = f"{r['pf']:.3f}" if r["pf"] != float("inf") else "  INF"
        pas = "Y" if r.get("passes_gate") else "N"
        print(f"  {r['label']:<42} {r['n']:>7,}  {r.get('delta_n', 0):>+6,}  "
              f"{r['ppt_norm']:>+8.4f}  {r.get('delta_ppt_norm', 0):>+7.4f}  {pf}  "
              f"{r['rally'].get('extra_long', 0):>6,}  {pas:>4}")

    print(f"\n  Winners (PPT_norm gate + trade count): {len(winners)}")
    if winners:
        w = winners[0]
        print(f"  BEST: {w['label']}")
        print(f"    agree_boost={w['agree_boost']} neutral_pen={w['neutral_pen']} "
              f"opposite_pen={w['opposite_pen']}")
        print(f"    mode={w['mode']} gate={w['gate']}")
        print(f"    PPT_norm={w['ppt_norm']:+.4f} (delta {w['delta_ppt_norm']:+.4f})")
        print(f"    N={w['n']:,} (delta {w['delta_n']:+,}) rally_unlock={w['rally']['extra_long']:,}")

    out = {
        "methodology": "oof_lstm_score_fusion_sweep",
        "holdout_used": False,
        "created": datetime.now().isoformat(),
        "lgbm_run": LGBM_RUN,
        "lstm_run": LSTM_RUN,
        "guardian_run": GUARDIAN_RUN,
        "baseline": baseline_summary,
        "ppt_norm_gate": PPT_NORM_GATE,
        "min_trade_ratio": MIN_TRADE_RATIO,
        "n_configs": len(sweep_configs),
        "n_winners": len(winners),
        "best": winners[0] if winners else None,
        "top15": results_sorted[:15],
        "all_results": results,
    }
    out_path = LGBM_DIR / "lstm_score_fusion_sweep.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n  Saved: {out_path}")
    print(f"{SEP}\n")


if __name__ == "__main__":
    main()