"""
pipeline/17_full_stack_tune_fb_v2.py
Entry-stack tuning ONLY: LGBM flatboost_v2 + HMM adaptive + LSTM soft veto
(NO Guardian — interaksi entry saja)

Holdout: Apr 2026 – Jun 2026 (21 koin)
Grid (~91 configs):
  HMM thr_T : [0.48, 0.50, 0.52]
  HMM thr_R : [0.55, 0.60]
  short_off : [0.05, 0.07]
  LSTM      : off | soft veto (scope all/trending × opposite_pen)

Usage:
  python pipeline/17_full_stack_tune_fb_v2.py
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
    ALL_COINS, HOLDOUT_DIR, MODEL_DIR, LABEL_MAP,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL, MAX_HOLDING_BARS,
)
from core.evaluator import full_trading_report
from core.models import load_lstm
from core.utils import setup_logger, ensure_utc_index
from pipeline.backtest_utils import get_lstm_proba

logger = setup_logger("17_full_stack_tune")

FB_RUN = "tb_lgbm_flatboost_v2"
HOLDOUT_START = "2026-04-01"
PERIOD_MONTHS = 2.5
LM = LABEL_MAP if isinstance(LABEL_MAP, dict) else {"SHORT": 0, "FLAT": 1, "LONG": 2}
TRENDING = {0, 3}

# Current production baseline
PROD = {
    "thr_T": 0.50, "thr_R": 0.55, "short_off": 0.05,
    "lstm": True, "lstm_scope": "all", "opposite_pen": 0.08,
    "agree_boost": 0.05, "neutral_pen": 0.05, "no_veto_thr": 0.50,
}

THR_T_VALS = [0.48, 0.50, 0.52]
THR_R_VALS = [0.55, 0.60]
SHORT_OFFS = [0.05, 0.07]
LSTM_SCOPES = ["all", "trending"]
OPPOSITE_PENS = [0.06, 0.08, 0.10]
AGREE_BOOST = 0.05
NEUTRAL_PEN = 0.05
NO_VETO_THR = 0.50

IC32_BENCH = {"trades": 936, "wr": 62.07, "pnl": 207.22, "pnl_per_trade": 0.2214, "pf": 2.54}


def hmm_entry(proba_fb, hmm, thr_t, thr_r, short_off):
    n = len(proba_fb)
    yp = np.ones(n, dtype=np.int32)
    conf = np.zeros(n, dtype=np.float64)
    for i in range(n):
        is_t = hmm[i] in TRENDING
        thr_l = thr_t if is_t else thr_r
        thr_s = thr_l + short_off
        p = proba_fb[i]
        if p[2] >= thr_l and p[2] >= p[0]:
            yp[i] = 2
            conf[i] = float(p[2])
        elif p[0] >= thr_s and p[0] > p[2]:
            yp[i] = 0
            conf[i] = float(p[0])
    return yp, conf


def apply_lstm_soft(yp, conf, proba_lstm, hmm, thr_t, thr_r, short_off,
                    scope, opposite_pen, agree_boost, neutral_pen, no_veto_thr):
    yp_out = yp.copy()
    conf_out = conf.copy()
    for i in range(len(yp_out)):
        if yp_out[i] == 1:
            continue
        if scope == "trending" and hmm[i] not in TRENDING:
            continue
        sig = int(yp_out[i])
        adj = float(conf_out[i])
        li = int(np.argmax(proba_lstm[i]))
        if li == sig:
            adj += agree_boost
        elif li == 1:
            adj -= neutral_pen
        else:
            adj -= opposite_pen
        adj = float(np.clip(adj, 0.0, 1.0))
        is_t = hmm[i] in TRENDING
        thr_l = thr_t if is_t else thr_r
        thr_s = thr_l + short_off
        thr = thr_l if sig == 2 else thr_s
        if adj < thr:
            yp_out[i] = 1
            conf_out[i] = 0.0
        else:
            conf_out[i] = adj
    return yp_out, conf_out


def run_stack(coin_data, yp_map, conf_map):
    agg = {"trades": 0, "wins": 0, "pnl": 0.0, "gp": 0.0, "gl": 0.0}
    for sym, d in coin_data.items():
        rep = full_trading_report(
            y_pred=yp_map[sym], y_actual=d["yt"],
            atr=d["atr"], close=d["close"], high=d["high"], low=d["low"],
            h4_swing_highs=d["h4_sh"], h4_swing_lows=d["h4_sl"],
            index=d["index"], modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
            max_sl_atr=SWING_LABEL_MAX_SL,
            tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
            max_hold=MAX_HOLDING_BARS, symbol=sym, confidence=conf_map[sym],
            guardian_enabled=False,
            trailing_stop_enabled=False, h4_trend=d["h4_tr"],
        )
        lev = rep.get("lev5x", rep)
        trades = lev.get("trades", [])
        agg["trades"] += len(trades)
        for t in trades:
            pnl = float(t.get("net_pnl", 0))
            if pnl > 0:
                agg["wins"] += 1
                agg["gp"] += pnl
            else:
                agg["gl"] += abs(pnl)
            agg["pnl"] += pnl
    t, p = agg["trades"], agg["pnl"]
    pf = agg["gp"] / max(agg["gl"], 1e-9)
    return {
        "trades": t,
        "wr": round(agg["wins"] / max(t, 1) * 100, 2),
        "pf": round(pf, 3),
        "pnl": round(p, 2),
        "pnl_per_trade": round(p / max(t, 1), 4),
        "pnl_per_month": round(p / PERIOD_MONTHS, 2),
    }


def config_label(cfg):
    if not cfg["lstm"]:
        return f"T{int(cfg['thr_T']*100)}_R{int(cfg['thr_R']*100)}_s{int(cfg['short_off']*100)}_lgbm"
    return (
        f"T{int(cfg['thr_T']*100)}_R{int(cfg['thr_R']*100)}_s{int(cfg['short_off']*100)}"
        f"_{cfg['lstm_scope'][0]}_opp{int(cfg['opposite_pen']*100)}"
    )


def build_configs():
    configs = []
    for thr_t, thr_r, short_off in itertools.product(THR_T_VALS, THR_R_VALS, SHORT_OFFS):
        configs.append({
            "thr_T": thr_t, "thr_R": thr_r, "short_off": short_off,
            "lstm": False, "lstm_scope": None, "opposite_pen": 0.0,
            "agree_boost": 0.0, "neutral_pen": 0.0, "no_veto_thr": 0.0,
        })
        for scope, opp in itertools.product(LSTM_SCOPES, OPPOSITE_PENS):
            configs.append({
                "thr_T": thr_t, "thr_R": thr_r, "short_off": short_off,
                "lstm": True, "lstm_scope": scope, "opposite_pen": opp,
                "agree_boost": AGREE_BOOST, "neutral_pen": NEUTRAL_PEN,
                "no_veto_thr": NO_VETO_THR,
            })
    prod_cfg = {**PROD, "label": "PROD_current"}
    return configs, prod_cfg


def main():
    fb_model = joblib.load(MODEL_DIR / "runs" / FB_RUN / "lgbm.pkl")
    with open(MODEL_DIR / "runs" / FB_RUN / f"{FB_RUN}_features.json", encoding="utf-8") as f:
        fb_feats = json.load(f)

    lstm_model = load_lstm(MODEL_DIR / "lstm_best.pt")
    lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
    with open(MODEL_DIR / "feature_cols_lstm_temporal.json", encoding="utf-8") as f:
        lstm_feats = json.load(f)

    available = sorted(
        s for s in ALL_COINS
        if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()
    )

    print(f"\nPre-loading {len(available)} coins...")
    coin_data = {}
    for sym in available:
        df = ensure_utc_index(pd.read_parquet(HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet"))
        df = df[df.index >= HOLDOUT_START].sort_index()
        if len(df) < 30:
            continue

        rp = HOLDOUT_DIR / "labeled" / f"{sym}_regime_h1.parquet"
        hmm = np.full(len(df), 1, np.int32)
        if rp.exists():
            reg = pd.read_parquet(rp)
            if "hmm_regime_enc" in reg.columns:
                hmm = reg["hmm_regime_enc"].reindex(df.index, fill_value=1).values.astype(np.int32)

        mask = df["label"].astype(str).isin(LM)
        df = df[mask].copy()
        hmm = hmm[mask.values]
        n = len(df)

        X_fb = np.zeros((n, len(fb_feats)), dtype=np.float64)
        for i, c in enumerate(fb_feats):
            if c in df.columns:
                X_fb[:, i] = df[c].ffill().fillna(0).values.astype(np.float64)
        proba_fb = fb_model.predict_proba(X_fb)

        X_lstm = np.zeros((n, len(lstm_feats)), dtype=np.float64)
        for i, c in enumerate(lstm_feats):
            if c in df.columns:
                X_lstm[:, i] = df[c].ffill().fillna(0).values.astype(np.float64)
        proba_lstm = get_lstm_proba(lstm_model, lstm_scaler, X_lstm, n)

        coin_data[sym] = {
            "proba_fb": proba_fb,
            "proba_lstm": proba_lstm,
            "hmm": hmm,
            "close": df["close"].values,
            "high": df["high"].values,
            "low": df["low"].values,
            "atr": df["atr_14_h1"].values,
            "h4_sh": df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan),
            "h4_sl": df["h4_swing_low"].values if "h4_swing_low" in df.columns else np.full(n, np.nan),
            "h4_tr": df["h4_trend"].values if "h4_trend" in df.columns else None,
            "yt": np.array([LM.get(str(v), 1) for v in df["label"].values], dtype=np.int32),
            "index": df.index,
        }
    print(f"Loaded {len(coin_data)} coins.\n")

    configs, prod_cfg = build_configs()
    all_cfgs = configs + [prod_cfg]
    print(f"Running {len(all_cfgs)} configs (entry only, Guardian OFF)...\n")

    results = []
    for idx, cfg in enumerate(all_cfgs):
        label = cfg.get("label") or config_label(cfg)
        yp_map, conf_map = {}, {}
        for sym, d in coin_data.items():
            yp, conf = hmm_entry(
                d["proba_fb"], d["hmm"],
                cfg["thr_T"], cfg["thr_R"], cfg["short_off"],
            )
            if cfg["lstm"]:
                yp, conf = apply_lstm_soft(
                    yp, conf, d["proba_lstm"], d["hmm"],
                    cfg["thr_T"], cfg["thr_R"], cfg["short_off"],
                    cfg["lstm_scope"], cfg["opposite_pen"],
                    cfg["agree_boost"], cfg["neutral_pen"], cfg["no_veto_thr"],
                )
            yp_map[sym] = yp
            conf_map[sym] = conf

        sc = run_stack(coin_data, yp_map, conf_map)
        row = {
            "label": label,
            "thr_T": cfg["thr_T"],
            "thr_R": cfg["thr_R"],
            "short_off": cfg["short_off"],
            "lstm": cfg["lstm"],
            "lstm_scope": cfg.get("lstm_scope"),
            "opposite_pen": cfg.get("opposite_pen"),
            **sc,
            "vs_ic32_pnl": round(sc["pnl"] - IC32_BENCH["pnl"], 2),
        }
        results.append(row)
        if (idx + 1) % 10 == 0 or idx == len(all_cfgs) - 1:
            print(
                f"  [{idx+1}/{len(all_cfgs)}] {label:<32} "
                f"{sc['trades']:>4} tr | WR {sc['wr']:>5.1f}% | PF {sc['pf']:.2f} | "
                f"${sc['pnl']:>+7.0f} | ${sc['pnl_per_trade']:+.3f}/tr"
            )

    df = pd.DataFrame(results).sort_values("pnl", ascending=False)

    print(f"\n{'='*110}")
    print(f"  ENTRY STACK TUNE — LGBM + HMM + LSTM (Guardian OFF)")
    print(f"  Holdout {HOLDOUT_START} – Jun 2026 | ic32 entry-only ref: {IC32_BENCH['trades']} tr")
    print(f"{'='*110}")
    print(f"  {'Label':<32} {'Trades':>6} {'WR':>6} {'PF':>6} {'PnL':>9} {'$/tr':>7} {'vs_ic32':>8}")
    print("-" * 110)
    for r in df.head(20).to_dict("records"):
        mark = " *" if r["label"] == "PROD_current" else ""
        print(
            f"  {r['label']:<32} {int(r['trades']):>6,} {r['wr']:>5.1f}% "
            f"{r['pf']:>6.2f} ${r['pnl']:>+8.0f} ${r['pnl_per_trade']:>+6.3f} "
            f"{r['vs_ic32_pnl']:>+8.0f}{mark}"
        )

    prod_row = df[df["label"] == "PROD_current"]
    if not prod_row.empty:
        pr = prod_row.iloc[0]
        rank = int(df.index.get_loc(prod_row.index[0]) + 1)
        print(f"\n  PROD_current rank: #{rank}/{len(df)} by PnL (${pr['pnl']:+.0f})")

    best = df.iloc[0]
    print(f"\n  BEST: {best['label']} | {int(best['trades'])} tr | WR {best['wr']:.1f}% | "
          f"PF {best['pf']:.2f} | PnL ${best['pnl']:+.0f} | ${best['pnl_per_trade']:+.3f}/tr")

    # Top by PF with min trades
    df_pf = df[df["trades"] >= 400].sort_values("pf", ascending=False)
    if not df_pf.empty:
        bp = df_pf.iloc[0]
        print(f"  BEST PF (>=400 tr): {bp['label']} | PF {bp['pf']:.2f} | PnL ${bp['pnl']:+.0f}")

    out = {
        "evaluated_at": datetime.now().isoformat(),
        "period": f"{HOLDOUT_START} to 2026-06-13",
        "scope": "entry_only_lgbm_hmm_lstm",
        "guardian_enabled": False,
        "grid": {
            "thr_T": THR_T_VALS, "thr_R": THR_R_VALS, "short_off": SHORT_OFFS,
            "lstm_scopes": LSTM_SCOPES, "opposite_pens": OPPOSITE_PENS,
        },
        "production_baseline": PROD,
        "ic32_benchmark": IC32_BENCH,
        "n_configs": len(results),
        "best_by_pnl": best.to_dict(),
        "results": df.to_dict("records"),
    }
    out_path = MODEL_DIR / "runs" / FB_RUN / "entry_stack_tune_lgbm_hmm_lstm.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved -> {out_path}")
    print(f"{'='*110}\n")


if __name__ == "__main__":
    main()