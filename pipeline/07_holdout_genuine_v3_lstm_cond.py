"""
pipeline/07_holdout_genuine_v3_lstm_cond.py — Holdout Evaluation (SEKALI SAJA)

Stack frozen (inference_config.json):
  LGBM 36f+dow -> HMM Config B -> LSTM conditional_momentum (Ref) -> Guardian v2 -> DynSize cm_0.60

Period : Apr 1 – Jun 30, 2026 (data tersedia s/d ~Jun 13)
Training cutoff : TRAIN_CUTOFF_DATE = 2026-04-01 (model dilatih Jan 2020 – Mar 2026)

PENTING:
  - First look — tidak tune parameter dari hasil ini
  - Set HOLDOUT_EVALUATED = True setelah dijalankan
  - Beda dari 07_holdout_genuine_v2.py (34f, tanpa LSTM, STALE)

Usage:
  python pipeline/07_holdout_genuine_v3_lstm_cond.py
"""

HOLDOUT_EVALUATED = True

if HOLDOUT_EVALUATED:
    raise RuntimeError(
        "Holdout genuine_v3_lstm_cond sudah dievaluasi!\n"
        "Jangan jalankan ulang — kontaminasi holdout.\n"
        "Butuh periode baru: buat v4 dengan OOS Jul–Sep 2026."
    )

import json
import sys
import warnings
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    ALL_COINS, OOS_START, OOS_END, TRAIN_CUTOFF_DATE,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP,
    SWING_LABEL_MAX_SL, TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    GUARDIAN_ACTIVATION_ATR, LSTM_SEQ_LEN, MODEL_DIR, HOLDOUT_DIR, REPORT_DIR,
)
from core.cascade_utils import apply_conditional_momentum_fusion_pre, _lstm_dominant_direction
from core.evaluator import simulate_trades_swing
from core.models import load_lstm
from core.utils import ensure_utc_index
from pipeline.lstm_fusion_shared import (
    DYNAMIC_FEATS, DYNSIZE_CFG, FLAT, LONG, SHORT,
    apply_hmm_thr, build_y_pred, compute_dynamic_modal,
    load_guardian_params, load_hmm_cfg, summarize_trades,
)

LSTM_RUN = "tb_lstm_genuine_v2"
LSTM_DIR = MODEL_DIR / "runs" / LSTM_RUN
INFERENCE_CFG = MODEL_DIR / "inference_config.json"

REF_FUSION = {
    "fusion": "lstm",
    "mode": "conditional_momentum",
    "bull_thr": 0.38,
    "bear_thr": 0.50,
    "boost": 0.10,
    "opposite_pen": 0.14,
    "near_miss_gap": 0.03,
    "vol_thr": 2.0,
    "proportional": True,
    "enable_boost": True,
    "enable_penalty": True,
    "label": "ref_lstm_cond_o14",
}

BASELINE_CFG = {"fusion": "baseline", "label": "baseline_no_lstm"}
VOL_THR = 2.0
RALLY_FRAC = 0.8


def load_promoted_models():
    lgbm = joblib.load(MODEL_DIR / "lgbm_baseline.pkl")
    with open(MODEL_DIR / "feature_cols_v2.json", encoding="utf-8") as f:
        lgbm_feats = json.load(f)

    guard = joblib.load(MODEL_DIR / "guardian_best.pkl")
    scaler = joblib.load(MODEL_DIR / "guardian_scaler.pkl")
    with open(MODEL_DIR / "guardian_feature_cols.json", encoding="utf-8") as f:
        guard_feats = json.load(f)
    g_static = [f for f in guard_feats if f not in DYNAMIC_FEATS]

    with open(LSTM_DIR / "lstm_v4_selected_features.json", encoding="utf-8") as f:
        lstm_feats = json.load(f)
    lstm_model = load_lstm(LSTM_DIR / "lstm_momentum.pt", device="cpu")
    lstm_scaler = joblib.load(LSTM_DIR / "lstm_momentum_scaler.pkl")

    with open(INFERENCE_CFG, encoding="utf-8") as f:
        inf = json.load(f)
    mom = inf.get("cascade", {}).get("lstm_momentum", REF_FUSION)

    return {
        "lgbm": lgbm, "lgbm_feats": lgbm_feats,
        "guard": guard, "guard_scaler": scaler, "g_static": g_static,
        "lstm_model": lstm_model, "lstm_scaler": lstm_scaler, "lstm_feats": lstm_feats,
        "fusion_cfg": {
            "fusion": "lstm", "mode": "conditional_momentum",
            "bull_thr": float(mom.get("bull_thr", 0.38)),
            "bear_thr": float(mom.get("bear_thr", 0.50)),
            "boost": float(mom.get("boost", 0.10)),
            "opposite_pen": float(mom.get("opposite_pen", 0.14)),
            "near_miss_gap": float(mom.get("near_miss_gap", 0.03)),
            "vol_thr": float(mom.get("vol_thr", 2.0)),
            "proportional": bool(mom.get("proportional", True)),
            "enable_boost": bool(mom.get("enable_boost", True)),
            "enable_penalty": bool(mom.get("enable_penalty", True)),
            "label": "ref_lstm_cond_from_inference_config",
        },
    }


def lstm_predict_proba(X_raw: np.ndarray, lstm_model, lstm_scaler, seq_len: int) -> np.ndarray:
    n, f = X_raw.shape
    probs = np.full((n, 3), 1.0 / 3.0, dtype=np.float32)
    if n < seq_len:
        return probs
    X_sc = lstm_scaler.transform(X_raw.reshape(-1, f)).reshape(n, f).astype(np.float32)
    seqs = np.stack([X_sc[i - seq_len + 1: i + 1] for i in range(seq_len - 1, n)])
    chunks = []
    with torch.no_grad():
        for b in range(0, len(seqs), 512):
            t = torch.from_numpy(seqs[b: b + 512])
            lg = lstm_model(t)
            chunks.append(torch.softmax(lg, dim=1).cpu().numpy())
    probs[seq_len - 1:] = np.concatenate(chunks, axis=0)
    return probs


def load_holdout_coin(sym: str, models: dict) -> dict | None:
    path = HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet"
    if not path.exists():
        return None

    df = pd.read_parquet(path)
    df = ensure_utc_index(df).sort_index()
    df = df[(df.index >= OOS_START) & (df.index < OOS_END)]
    if len(df) < LSTM_SEQ_LEN + 10:
        return None

    n = len(df)
    lgbm_feats = models["lgbm_feats"]
    X_lgbm = np.zeros((n, len(lgbm_feats)), dtype=np.float64)
    for i, col in enumerate(lgbm_feats):
        if col in df.columns:
            X_lgbm[:, i] = df[col].ffill().fillna(0).values
    proba = models["lgbm"].predict_proba(X_lgbm)
    p0 = proba[:, 0].astype(np.float32)
    p2 = proba[:, 2].astype(np.float32)

    lstm_feats = models["lstm_feats"]
    X_lstm = np.zeros((n, len(lstm_feats)), dtype=np.float64)
    for i, col in enumerate(lstm_feats):
        if col in df.columns:
            X_lstm[:, i] = df[col].ffill().fillna(0).values
    lstm_p = lstm_predict_proba(X_lstm, models["lstm_model"], models["lstm_scaler"], LSTM_SEQ_LEN)
    lstm_valid = np.isfinite(lstm_p).all(axis=1)
    vol_spike = (
        df["vol_spike_zscore"].fillna(-99).values.astype(np.float32)
        if "vol_spike_zscore" in df.columns else np.full(n, -99.0, np.float32)
    )

    if "log_ret_4" in df.columns:
        ret4 = df["log_ret_4"].fillna(0).values.astype(np.float32)
    else:
        ret4 = df["close"].pct_change(4).fillna(0).values.astype(np.float32)

    g_static = models["g_static"]
    X_grd = np.zeros((n, len(g_static)), dtype=np.float64)
    for i, col in enumerate(g_static):
        if col in df.columns:
            X_grd[:, i] = df[col].ffill().fillna(0).values

    return {
        "sym": sym,
        "ts": df.index,
        "p0": p0, "p2": p2,
        "hmm": df["hmm_regime_enc"].fillna(-1).values.astype(np.int8)
        if "hmm_regime_enc" in df.columns else np.full(n, -1, np.int8),
        "lstm_p": lstm_p,
        "lstm_valid": lstm_valid,
        "vol_spike": vol_spike,
        "ret4": ret4,
        "close": df["close"].values.astype(np.float64),
        "high": df["high"].values.astype(np.float64),
        "low": df["low"].values.astype(np.float64),
        "atr": df["atr_14_h1"].values.astype(np.float64),
        "h4_sh": df["h4_swing_high"].values.astype(np.float64)
        if "h4_swing_high" in df.columns else np.full(n, np.nan),
        "h4_sl": df["h4_swing_low"].values.astype(np.float64)
        if "h4_swing_low" in df.columns else np.full(n, np.nan),
        "X_grd": X_grd,
    }


def attach_cross_section(coins: list) -> None:
    rows = []
    for c in coins:
        for i, ts in enumerate(c["ts"]):
            rows.append({"ts": ts, "coin": c["sym"], "ret4": float(c["ret4"][i])})
    panel = pd.DataFrame(rows)
    pivot = panel.pivot_table(index="ts", columns="coin", values="ret4", aggfunc="first")
    frac_up = (pivot > 0).mean(axis=1)
    for c in coins:
        c["frac_up"] = frac_up.reindex(c["ts"]).fillna(0.5).values.astype(np.float32)


def eval_stack(coins: list, cfg: dict, hmm_cfg: dict, g_model, g_scaler, g_params: dict) -> dict:
    all_trades = []
    common = dict(
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        max_hold=MAX_HOLDING_BARS,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
    )
    for c in coins:
        y = build_y_pred(c, cfg, hmm_cfg)
        p0, p2 = c["p0"].copy(), c["p2"].copy()
        if cfg.get("mode") == "conditional_momentum":
            _, _, tl, ts = apply_hmm_thr(p0, p2, c["hmm"], hmm_cfg)
            p0, p2 = apply_conditional_momentum_fusion_pre(
                p0, p2, c["lstm_p"], tl, ts, c["vol_spike"],
                vol_thr=cfg.get("vol_thr", 2.0),
                bull_thr=cfg["bull_thr"], bear_thr=cfg["bear_thr"],
                near_miss_gap=cfg["near_miss_gap"],
                boost=cfg["boost"], opposite_pen=cfg["opposite_pen"],
                enable_boost=cfg.get("enable_boost", True),
                enable_penalty=cfg.get("enable_penalty", True),
                lstm_valid=c["lstm_valid"],
                proportional=cfg.get("proportional", True),
            )
            _, conf, tl, ts = apply_hmm_thr(p0, p2, c["hmm"], hmm_cfg)
        else:
            _, conf, tl, ts = apply_hmm_thr(p0, p2, c["hmm"], hmm_cfg)

        modal_arr = compute_dynamic_modal(p0, p2, c["hmm"], y, MODAL_PER_TRADE, DYNSIZE_CFG, tl, ts)
        rep = simulate_trades_swing(
            y_pred=y, guardian_enabled=True,
            guardian_model=g_model, guardian_scaler=g_scaler, X_guardian=c["X_grd"],
            guardian_exit_threshold=g_params["exit_threshold"],
            guardian_min_hold_bars=g_params["min_hold_bars"],
            guardian_activation_atr=GUARDIAN_ACTIVATION_ATR,
            modal_arr=modal_arr,
            close=c["close"], high=c["high"], low=c["low"], atr=c["atr"],
            h4_swing_highs=c["h4_sh"], h4_swing_lows=c["h4_sl"],
            **common,
        )
        for t in rep.get("trades", []):
            bi = t.get("bar_in", 0)
            t2 = dict(t)
            t2["sym"] = c["sym"]
            t2["direction"] = int(y[bi]) if bi < len(y) else FLAT
            t2["vol_spike"] = float(c["vol_spike"][bi]) if bi < len(c["vol_spike"]) else 0.0
            t2["frac_up"] = float(c["frac_up"][bi]) if bi < len(c["frac_up"]) else 0.5
            all_trades.append(t2)

    port = summarize_trades(all_trades)
    long_t = summarize_trades([t for t in all_trades if t["direction"] == LONG])
    short_t = summarize_trades([t for t in all_trades if t["direction"] == SHORT])
    pump_short = summarize_trades([
        t for t in all_trades
        if t["direction"] == SHORT and t["vol_spike"] >= VOL_THR and t["frac_up"] >= RALLY_FRAC
    ])
    rally_short = summarize_trades([
        t for t in all_trades if t["direction"] == SHORT and t["frac_up"] >= RALLY_FRAC
    ])
    return {
        "portfolio": port,
        "all_long": long_t,
        "all_short": short_t,
        "pump_short": pump_short,
        "rally_short": rally_short,
        "trades": all_trades,
    }


def signal_stats(coins: list, cfg: dict, y_base: dict, hmm_cfg: dict) -> dict:
    boost = pen = 0
    for c in coins:
        sym = c["sym"]
        y = build_y_pred(c, cfg, hmm_cfg)
        yb = y_base[sym]
        mom = c["vol_spike"] >= VOL_THR
        for i in range(len(y)):
            if not mom[i]:
                continue
            if yb[i] == FLAT and y[i] != FLAT:
                boost += 1
            if yb[i] != FLAT and y[i] == FLAT:
                pen += 1
    return {"boost_unlock": boost, "penalty_block": pen}


def hmm_state_dist(coins: list) -> dict:
    counts = defaultdict(int)
    for c in coins:
        for st in c["hmm"]:
            counts[int(st)] += 1
    return dict(counts)


def main():
    sep = "=" * 78
    print(f"\n{sep}")
    print("  HOLDOUT GENUINE V3 — 36f LGBM + HMM B + LSTM Ref + Guardian v2 + DynSize")
    print(f"  Training cutoff : < {TRAIN_CUTOFF_DATE.date()} (sampai Mar 2026)")
    print(f"  Holdout period  : {OOS_START.date()} -> {OOS_END.date()}")
    print(sep)

    models = load_promoted_models()
    hmm_cfg = load_hmm_cfg()
    g_params = load_guardian_params()
    fusion_cfg = models["fusion_cfg"]

    print(f"  LGBM features : {len(models['lgbm_feats'])}")
    print(f"  LSTM features : {len(models['lstm_feats'])} | seq={LSTM_SEQ_LEN}")
    print(f"  Fusion        : {fusion_cfg['label']}")
    print(f"    opp_pen={fusion_cfg['opposite_pen']} bull={fusion_cfg['bull_thr']} bear={fusion_cfg['bear_thr']}")
    print(f"  Guardian      : exit_thr={g_params['exit_threshold']} min_hold={g_params['min_hold_bars']}\n")

    coins = []
    for sym in ALL_COINS:
        c = load_holdout_coin(sym, models)
        if c is not None:
            coins.append(c)
    attach_cross_section(coins)
    print(f"  Coins evaluated: {len(coins)}")

    if not coins:
        raise RuntimeError("No holdout coin data found under data/holdout-test/labeled/")

    y_base = {c["sym"]: build_y_pred(c, BASELINE_CFG, hmm_cfg) for c in coins}
    sig_lstm = signal_stats(coins, fusion_cfg, y_base, hmm_cfg)

    res_base = eval_stack(coins, BASELINE_CFG, hmm_cfg, models["guard"], models["guard_scaler"], g_params)
    res_lstm = eval_stack(coins, fusion_cfg, hmm_cfg, models["guard"], models["guard_scaler"], g_params)

    pb, pl = res_base["portfolio"], res_lstm["portfolio"]
    print(f"\n{sep}")
    print("  SCORECARD HOLDOUT")
    print(sep)
    print(f"  {'Stack':<28} {'N':>6} {'WR%':>6} {'PnL':>10} {'PPT':>8} {'PPT_n':>8} {'PF':>6} {'AvgM':>6}")
    print(f"  {'-'*78}")
    for name, s in [
        ("baseline_no_lstm", pb),
        ("ref_lstm_cond", pl),
    ]:
        pf = f"{s['pf']:.3f}" if s["pf"] != float("inf") else "   inf"
        print(f"  {name:<28} {s['n']:>6,} {s['wr']:>6.1f} ${s['pnl']:>9.2f} "
              f"{s['ppt']:>+8.4f} {s['ppt_norm']:>+8.4f} {pf:>6} ${s['avg_modal']:>5.1f}")

    print(f"\n  LSTM signal: boost_unlock={sig_lstm['boost_unlock']} penalty_block={sig_lstm['penalty_block']}")
    print(f"  Delta LSTM vs base: WR {pl['wr']-pb['wr']:+.1f}pp  "
          f"PPT_n {pl['ppt_norm']-pb['ppt_norm']:+.4f}  PnL ${pl['pnl']-pb['pnl']:+.2f}")

    print(f"\n  SLICE pump/SHORT (vol>={VOL_THR}, frac_up>={RALLY_FRAC})")
    for label, key in [("baseline", "pump_short"), ("ref_lstm", "pump_short")]:
        s = res_base[key] if label == "baseline" else res_lstm[key]
        pf = f"{s['pf']:.2f}" if s["pf"] != float("inf") else "inf"
        print(f"    {label:<12} n={s['n']:>4} WR={s['wr']:.1f}% PF={pf} PPT_n={s['ppt_norm']:+.4f}")

    states = hmm_state_dist(coins)
    state_names = {0: "TRENDING_DOWN", 1: "RANGING_LOW", 2: "RANGING_HIGH", 3: "TRENDING_UP", -1: "unknown"}
    total_bars = sum(states.values())
    print(f"\n  HMM state distribution ({total_bars:,} bars):")
    for st in sorted(states):
        cnt = states[st]
        print(f"    S{st} {state_names.get(st, '?'):<16} {cnt:>6,} ({cnt/total_bars*100:.1f}%)")

    today = datetime.now().strftime("%Y-%m-%d")
    rdir = REPORT_DIR / "experiments"
    rdir.mkdir(parents=True, exist_ok=True)
    md_path = rdir / f"{today}_genuine_v3_lstm_cond_holdout.md"
    json_path = rdir / f"{today}_genuine_v3_lstm_cond_holdout.json"

    def _fmt(s):
        pf = s["pf"] if s["pf"] != float("inf") else None
        return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items() if k != "trades"}

    payload = {
        "created": datetime.now().isoformat(),
        "script": "07_holdout_genuine_v3_lstm_cond.py",
        "model_version": "tb_genuine_v2_dynsize_lstm_cond",
        "train_cutoff": str(TRAIN_CUTOFF_DATE.date()),
        "holdout_start": str(OOS_START.date()),
        "holdout_end": str(OOS_END.date()),
        "n_coins": len(coins),
        "fusion_cfg": fusion_cfg,
        "lstm_signal": sig_lstm,
        "hmm_state_dist": states,
        "baseline": {k: _fmt(v) if isinstance(v, dict) else v for k, v in res_base.items() if k != "trades"},
        "ref_lstm": {k: _fmt(v) if isinstance(v, dict) else v for k, v in res_lstm.items() if k != "trades"},
        "delta_port_ppt_norm": round(pl["ppt_norm"] - pb["ppt_norm"], 6),
        "methodology": "first_look_frozen_config_no_holdout_tuning",
        "supersedes": "2026-06-15_genuine_v2_holdout.md (34f, no LSTM — STALE)",
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    md = f"""# Holdout Evaluation — Genuine v3 LSTM Cond

**Date**: {today}
**Stack**: LGBM 36f+dow + HMM Config B + LSTM conditional_momentum (Ref) + Guardian v2 + DynSize cm_0.60
**Training**: 2020-01-01 – 2026-03-31 (`TRAIN_CUTOFF_DATE={TRAIN_CUTOFF_DATE.date()}`)
**Holdout**: {OOS_START.date()} – {OOS_END.date()} ({len(coins)} koin)

## Metodologi
- Model dilatih pada data **sebelum** {TRAIN_CUTOFF_DATE.date()} — tidak ada leakage holdout
- Semua parameter frozen dari `inference_config.json` (OOF winner 05j)
- **First look** — tidak ada tuning berdasarkan Apr–Jun 2026
- Menggantikan holdout v2 (34f, tanpa LSTM) yang STALE

## Scorecard Portfolio

| Stack | N | WR% | PnL | PPT | PPT_norm | PF | AvgModal |
|-------|--:|----:|----:|----:|---------:|---:|---------:|
| baseline_no_lstm | {pb['n']:,} | {pb['wr']:.1f}% | ${pb['pnl']:.2f} | {pb['ppt']:+.4f} | {pb['ppt_norm']:+.4f} | {pb['pf']:.3f} | ${pb['avg_modal']:.1f} |
| **ref_lstm_cond** | **{pl['n']:,}** | **{pl['wr']:.1f}%** | **${pl['pnl']:.2f}** | **{pl['ppt']:+.4f}** | **{pl['ppt_norm']:+.4f}** | **{pl['pf']:.3f}** | **${pl['avg_modal']:.1f}** |

LSTM signal: boost_unlock={sig_lstm['boost_unlock']}, penalty_block={sig_lstm['penalty_block']}
Delta LSTM vs baseline: PPT_norm {pl['ppt_norm']-pb['ppt_norm']:+.4f}, PnL ${pl['pnl']-pb['pnl']:+.2f}

## Slice pump/SHORT (vol>={VOL_THR}, frac_up>={RALLY_FRAC})

| Stack | N | WR% | PF | PPT_norm |
|-------|--:|----:|---:|---------:|
| baseline | {res_base['pump_short']['n']} | {res_base['pump_short']['wr']:.1f}% | {res_base['pump_short']['pf']:.2f} | {res_base['pump_short']['ppt_norm']:+.4f} |
| ref_lstm | {res_lstm['pump_short']['n']} | {res_lstm['pump_short']['wr']:.1f}% | {res_lstm['pump_short']['pf']:.2f} | {res_lstm['pump_short']['ppt_norm']:+.4f} |

## HMM State Distribution
{chr(10).join(f'- S{st} {state_names.get(int(st),"?")}: {cnt:,} bars ({cnt/total_bars*100:.1f}%)' for st, cnt in sorted(states.items()))}

## Catatan
- Modal base ${MODAL_PER_TRADE}, leverage {LEVERAGE_SIM[0]}x
- Set `HOLDOUT_EVALUATED = True` di script setelah review
- Jangan re-run — gunakan holdout baru (Jul–Sep 2026) untuk eval berikutnya

*Generated by 07_holdout_genuine_v3_lstm_cond.py*
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md)

    print(f"\n  Report : {md_path}")
    print(f"  JSON   : {json_path}")
    print(f"\n  INGAT: Set HOLDOUT_EVALUATED = True di baris 18 setelah review.\n")


if __name__ == "__main__":
    main()