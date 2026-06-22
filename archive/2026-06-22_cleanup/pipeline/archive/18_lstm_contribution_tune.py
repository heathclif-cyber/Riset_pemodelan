"""
pipeline/18_lstm_contribution_tune.py
LSTM contribution sweep — cari mode di luar soft_veto standar.

Scope: LGBM flatboost_v2 + HMM adaptive + LSTM (Guardian OFF)
Holdout: Apr–Jun 2026

Mode LSTM (belum/sedikit di flatboost_v2):
  - lgbm_only          baseline per HMM
  - soft_veto          prod-like (opp/agree/neutral)
  - boost_only         hanya boost saat agree, tanpa penalty
  - opp_gated          penalty opposite hanya jika lstm_opp >= min_opp
  - dual_gate          cascade_utils parallel gate
  - dual_dominant      cascade_utils dominant LSTM
  - lstm_confirm       filter: lstm argmax == lgbm & dom >= min
  - conf_multiplier    conf *= blend saat agree/disagree
  - candidate_unlock   HMM thr -delta, keep jika lstm confirm
  - hard_consensus     cascade_utils style (boost/neu/opp)

Pass "LSTM contributes":
  pnl > baseline (same HMM) AND trade_drop <= 20%

Usage:
  python pipeline/18_lstm_contribution_tune.py
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

logger = setup_logger("18_lstm_tune")

FB_RUN = "tb_lgbm_flatboost_v2"
HOLDOUT_START = "2026-04-01"
PERIOD_MONTHS = 2.5
LM = LABEL_MAP if isinstance(LABEL_MAP, dict) else {"SHORT": 0, "FLAT": 1, "LONG": 2}
SHORT, FLAT, LONG = 0, 1, 2
TRENDING = {0, 3}
MAX_TRADE_DROP = 20.0

HMM_BASES = [
    {"label": "T48_R55_s5", "thr_T": 0.48, "thr_R": 0.55, "short_off": 0.05},
    {"label": "T50_R55_s5", "thr_T": 0.50, "thr_R": 0.55, "short_off": 0.05},
    {"label": "T52_R55_s5", "thr_T": 0.52, "thr_R": 0.55, "short_off": 0.05},
]


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


def _thr_for_bar(hmm_i, sig, thr_t, thr_r, short_off):
    is_t = hmm_i in TRENDING
    thr_l = thr_t if is_t else thr_r
    thr_s = thr_l + short_off
    return thr_l if sig == LONG else thr_s


def _scope_ok(scope, hmm_i):
    return scope != "trending" or hmm_i in TRENDING


def regate(yp, conf, hmm, thr_t, thr_r, short_off):
    yp_o, conf_o = yp.copy(), conf.copy()
    for i in range(len(yp_o)):
        if yp_o[i] == FLAT:
            continue
        sig = int(yp_o[i])
        thr = _thr_for_bar(hmm[i], sig, thr_t, thr_r, short_off)
        if conf_o[i] < thr:
            yp_o[i] = FLAT
            conf_o[i] = 0.0
    return yp_o, conf_o


def apply_mode(mode, yp, conf, proba_lstm, hmm, thr_t, thr_r, short_off, params):
    yp_o, conf_o = yp.copy(), conf.copy()
    scope = params.get("scope", "all")

    if mode == "lgbm_only":
        return yp_o, conf_o

    if mode == "soft_veto":
        for i in range(len(yp_o)):
            if yp_o[i] == FLAT or not _scope_ok(scope, hmm[i]):
                continue
            sig = int(yp_o[i])
            adj = float(conf_o[i])
            li = int(np.argmax(proba_lstm[i]))
            if li == sig:
                adj += params["agree_boost"]
            elif li == FLAT:
                adj -= params["neutral_pen"]
            else:
                adj -= params["opposite_pen"]
            conf_o[i] = float(np.clip(adj, 0.0, 1.0))
        return regate(yp_o, conf_o, hmm, thr_t, thr_r, short_off)

    if mode == "boost_only":
        for i in range(len(yp_o)):
            if yp_o[i] == FLAT or not _scope_ok(scope, hmm[i]):
                continue
            if int(np.argmax(proba_lstm[i])) == int(yp_o[i]):
                conf_o[i] = float(np.clip(conf_o[i] + params["agree_boost"], 0.0, 1.0))
        return regate(yp_o, conf_o, hmm, thr_t, thr_r, short_off)

    if mode == "opp_gated":
        for i in range(len(yp_o)):
            if yp_o[i] == FLAT or not _scope_ok(scope, hmm[i]):
                continue
            sig = int(yp_o[i])
            li = int(np.argmax(proba_lstm[i]))
            adj = float(conf_o[i])
            if li == sig:
                adj += params.get("agree_boost", 0.0)
            elif li != FLAT:
                opp_c = float(proba_lstm[i][0] if sig == LONG else proba_lstm[i][2])
                if opp_c >= params["min_opp"]:
                    adj -= params["opposite_pen"]
            conf_o[i] = float(np.clip(adj, 0.0, 1.0))
        return regate(yp_o, conf_o, hmm, thr_t, thr_r, short_off)

    if mode == "dual_gate":
        lstm_gate = params["lstm_gate"]
        for i in range(len(yp_o)):
            if yp_o[i] == FLAT:
                continue
            sig = int(yp_o[i])
            li = int(np.argmax(proba_lstm[i]))
            lc = float(proba_lstm[i][li])
            if li != sig or lc < lstm_gate:
                yp_o[i] = FLAT
                conf_o[i] = 0.0
            else:
                conf_o[i] = (conf_o[i] + lc) / 2.0
        return yp_o, conf_o

    if mode == "dual_dominant":
        lstm_dom_thr = params["lstm_dom_thr"]
        for i in range(len(yp_o)):
            if yp_o[i] == FLAT:
                continue
            sig = int(yp_o[i])
            lstm_l, lstm_s = float(proba_lstm[i][2]), float(proba_lstm[i][0])
            if lstm_l > lstm_s:
                ld, dom = LONG, lstm_l
            elif lstm_s > lstm_l:
                ld, dom = SHORT, lstm_s
            else:
                yp_o[i] = FLAT
                conf_o[i] = 0.0
                continue
            if ld != sig or dom < lstm_dom_thr:
                yp_o[i] = FLAT
                conf_o[i] = 0.0
            else:
                conf_o[i] = (conf_o[i] + dom) / 2.0
        return yp_o, conf_o

    if mode == "lstm_confirm":
        lstm_min = params["lstm_min"]
        for i in range(len(yp_o)):
            if yp_o[i] == FLAT:
                continue
            sig = int(yp_o[i])
            lstm_l, lstm_s = float(proba_lstm[i][2]), float(proba_lstm[i][0])
            dom = max(lstm_l, lstm_s)
            ld = LONG if lstm_l > lstm_s else (SHORT if lstm_s > lstm_l else FLAT)
            if ld != sig or dom < lstm_min:
                yp_o[i] = FLAT
                conf_o[i] = 0.0
        return yp_o, conf_o

    if mode == "conf_multiplier":
        lam = params["lam"]
        for i in range(len(yp_o)):
            if yp_o[i] == FLAT or not _scope_ok(scope, hmm[i]):
                continue
            sig = int(yp_o[i])
            lstm_l, lstm_s = float(proba_lstm[i][2]), float(proba_lstm[i][0])
            dom = lstm_l if sig == LONG else lstm_s
            mult = 1.0 + lam * (dom - 0.33)
            mult = float(np.clip(mult, 0.70, 1.30))
            conf_o[i] = float(np.clip(conf_o[i] * mult, 0.0, 1.0))
        return regate(yp_o, conf_o, hmm, thr_t, thr_r, short_off)

    if mode == "candidate_unlock":
        # Re-scan with lower HMM thr, keep only lstm-confirmed
        delta = params["thr_delta"]
        lstm_min = params["lstm_min"]
        n = len(yp_o)
        yp_n = np.ones(n, dtype=np.int32)
        conf_n = np.zeros(n, dtype=np.float64)
        for i in range(n):
            is_t = hmm[i] in TRENDING
            thr_l = (thr_t - delta) if is_t else (thr_r - delta)
            thr_s = thr_l + short_off
            pf = params.get("proba_fb")
            if pf is None:
                continue
            p = pf[i]
            sig = FLAT
            if p[2] >= thr_l and p[2] >= p[0]:
                sig, c = LONG, float(p[2])
            elif p[0] >= thr_s and p[0] > p[2]:
                sig, c = SHORT, float(p[0])
            if sig == FLAT:
                continue
            lstm_l, lstm_s = float(proba_lstm[i][2]), float(proba_lstm[i][0])
            dom = lstm_l if sig == LONG else lstm_s
            ld = LONG if lstm_l > lstm_s else (SHORT if lstm_s > lstm_l else FLAT)
            if ld == sig and dom >= lstm_min:
                yp_n[i] = sig
                conf_n[i] = c
        return yp_n, conf_n

    if mode == "hard_consensus":
        for i in range(len(yp_o)):
            if yp_o[i] == FLAT:
                continue
            sig = int(yp_o[i])
            adj = float(conf_o[i])
            li = int(np.argmax(proba_lstm[i]))
            if li == sig:
                adj += params["agree_boost"]
            elif li == FLAT:
                adj -= params["neutral_pen"]
            else:
                adj -= params["opposite_pen"]
            conf_o[i] = float(np.clip(adj, 0.0, 1.0))
        thr_entry = params.get("thr_entry", thr_t)
        for i in range(len(yp_o)):
            if yp_o[i] != FLAT and conf_o[i] < thr_entry:
                yp_o[i] = FLAT
                conf_o[i] = 0.0
        return yp_o, conf_o

    return yp_o, conf_o


def build_lstm_configs():
    cfgs = [{"mode": "lgbm_only", "params": {}, "tag": "lgbm_only"}]

    for scope, opp, agr, neu in itertools.product(
        ["all", "trending"],
        [0.03, 0.05, 0.08, 0.12],
        [0.03, 0.05, 0.08],
        [0.0, 0.03, 0.05],
    ):
        cfgs.append({
            "mode": "soft_veto",
            "params": {"scope": scope, "opposite_pen": opp, "agree_boost": agr, "neutral_pen": neu},
            "tag": f"soft_{scope[0]}_o{int(opp*100)}_a{int(agr*100)}_n{int(neu*100)}",
        })

    for scope, agr in itertools.product(["all", "trending"], [0.03, 0.05, 0.08, 0.10, 0.15]):
        cfgs.append({
            "mode": "boost_only",
            "params": {"scope": scope, "agree_boost": agr},
            "tag": f"boost_{scope[0]}_a{int(agr*100)}",
        })

    for min_opp, opp, agr in itertools.product(
        [0.60, 0.65, 0.70, 0.75],
        [0.05, 0.08, 0.12],
        [0.0, 0.05],
    ):
        cfgs.append({
            "mode": "opp_gated",
            "params": {"min_opp": min_opp, "opposite_pen": opp, "agree_boost": agr, "scope": "all"},
            "tag": f"oppG_m{int(min_opp*100)}_p{int(opp*100)}_a{int(agr*100)}",
        })

    for lg in [0.32, 0.35, 0.38, 0.40, 0.45, 0.50]:
        cfgs.append({"mode": "dual_gate", "params": {"lstm_gate": lg}, "tag": f"dualG_{int(lg*100)}"})

    for ld in [0.28, 0.32, 0.35, 0.38, 0.42]:
        cfgs.append({"mode": "dual_dominant", "params": {"lstm_dom_thr": ld}, "tag": f"dualD_{int(ld*100)}"})

    for lm in [0.28, 0.32, 0.35, 0.38, 0.42, 0.45]:
        cfgs.append({"mode": "lstm_confirm", "params": {"lstm_min": lm}, "tag": f"confirm_{int(lm*100)}"})

    for lam in [0.15, 0.25, 0.40, 0.60]:
        for scope in ["all", "trending"]:
            cfgs.append({
                "mode": "conf_multiplier",
                "params": {"lam": lam, "scope": scope},
                "tag": f"mult_{scope[0]}_l{int(lam*100)}",
            })

    for delta, lm in itertools.product([0.01, 0.02, 0.03], [0.32, 0.35, 0.38, 0.40]):
        cfgs.append({
            "mode": "candidate_unlock",
            "params": {"thr_delta": delta, "lstm_min": lm},
            "tag": f"unlock_d{int(delta*100)}_m{int(lm*100)}",
        })

    for opp, agr, neu in itertools.product([0.05, 0.10, 0.20, 0.40], [0.03, 0.05, 0.08], [0.03, 0.08]):
        cfgs.append({
            "mode": "hard_consensus",
            "params": {"opposite_pen": opp, "agree_boost": agr, "neutral_pen": neu, "thr_entry": 0.50},
            "tag": f"hc_o{int(opp*100)}_a{int(agr*100)}_n{int(neu*100)}",
        })

    return cfgs


def eval_config(coin_data, yp_map, conf_map):
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
            guardian_enabled=False, trailing_stop_enabled=False, h4_trend=d["h4_tr"],
        )
        lev = rep.get("lev5x", rep)
        for t in lev.get("trades", []):
            pnl = float(t.get("net_pnl", 0))
            agg["trades"] += 1
            agg["pnl"] += pnl
            if pnl > 0:
                agg["wins"] += 1
                agg["gp"] += pnl
            else:
                agg["gl"] += abs(pnl)
    t, p = agg["trades"], agg["pnl"]
    pf = agg["gp"] / max(agg["gl"], 1e-9)
    return {
        "trades": t,
        "wr": round(agg["wins"] / max(t, 1) * 100, 2),
        "pf": round(pf, 3),
        "pnl": round(p, 2),
        "pnl_per_trade": round(p / max(t, 1), 4),
    }


def main():
    fb_model = joblib.load(MODEL_DIR / "runs" / FB_RUN / "lgbm.pkl")
    with open(MODEL_DIR / "runs" / FB_RUN / f"{FB_RUN}_features.json", encoding="utf-8") as f:
        fb_feats = json.load(f)
    lstm_model = load_lstm(MODEL_DIR / "lstm_best.pt")
    lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
    with open(MODEL_DIR / "feature_cols_lstm_temporal.json", encoding="utf-8") as f:
        lstm_feats = json.load(f)

    coin_data = {}
    for sym in ALL_COINS:
        path = HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet"
        if not path.exists():
            continue
        df = ensure_utc_index(pd.read_parquet(path))
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
        coin_data[sym] = {
            "proba_fb": proba_fb,
            "proba_lstm": get_lstm_proba(lstm_model, lstm_scaler, X_lstm, n),
            "hmm": hmm,
            "close": df["close"].values, "high": df["high"].values, "low": df["low"].values,
            "atr": df["atr_14_h1"].values,
            "h4_sh": df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan),
            "h4_sl": df["h4_swing_low"].values if "h4_swing_low" in df.columns else np.full(n, np.nan),
            "h4_tr": df["h4_trend"].values if "h4_trend" in df.columns else None,
            "yt": np.array([LM.get(str(v), 1) for v in df["label"].values], dtype=np.int32),
            "index": df.index,
        }

    lstm_cfgs = build_lstm_configs()
    print(f"\nLSTM CONTRIBUTION SWEEP | {len(coin_data)} coins | {len(HMM_BASES)} HMM × {len(lstm_cfgs)} LSTM modes")
    print(f"Guardian OFF | Pass: pnl > baseline & trade_drop <= {MAX_TRADE_DROP}%\n")

    all_results = []
    contributors = []

    for hmm_cfg in HMM_BASES:
        baselines = {}
        for sym, d in coin_data.items():
            yp, conf = hmm_entry(
                d["proba_fb"], d["hmm"],
                hmm_cfg["thr_T"], hmm_cfg["thr_R"], hmm_cfg["short_off"],
            )
            baselines[sym] = (yp, conf)

        base_yp = {s: baselines[s][0] for s in baselines}
        base_cf = {s: baselines[s][1] for s in baselines}
        base_sc = eval_config(coin_data, base_yp, base_cf)
        base_trades = max(base_sc["trades"], 1)

        for lc in lstm_cfgs:
            yp_map, conf_map = {}, {}
            for sym, d in coin_data.items():
                yp, conf = baselines[sym]
                params = dict(lc["params"])
                if lc["mode"] == "candidate_unlock":
                    params["proba_fb"] = d["proba_fb"]
                yp, conf = apply_mode(
                    lc["mode"], yp, conf, d["proba_lstm"], d["hmm"],
                    hmm_cfg["thr_T"], hmm_cfg["thr_R"], hmm_cfg["short_off"], params,
                )
                yp_map[sym] = yp
                conf_map[sym] = conf

            sc = eval_config(coin_data, yp_map, conf_map)
            drop = (base_trades - sc["trades"]) / base_trades * 100
            pnl_delta = sc["pnl"] - base_sc["pnl"]
            contributes = pnl_delta > 0 and drop <= MAX_TRADE_DROP

            row = {
                "hmm": hmm_cfg["label"],
                "thr_T": hmm_cfg["thr_T"],
                "thr_R": hmm_cfg["thr_R"],
                "short_off": hmm_cfg["short_off"],
                "lstm_mode": lc["mode"],
                "lstm_tag": lc["tag"],
                "lstm_params": lc["params"],
                "baseline_trades": base_sc["trades"],
                "baseline_pnl": base_sc["pnl"],
                "baseline_ppt": base_sc["pnl_per_trade"],
                "trade_drop_pct": round(drop, 2),
                "pnl_delta": round(pnl_delta, 2),
                "ppt_delta": round(sc["pnl_per_trade"] - base_sc["pnl_per_trade"], 4),
                "lstm_contributes": contributes,
                **sc,
            }
            all_results.append(row)
            if contributes and lc["mode"] != "lgbm_only":
                contributors.append(row)

    df = pd.DataFrame(all_results)
    contrib = pd.DataFrame(contributors).sort_values("pnl", ascending=False) if contributors else pd.DataFrame()

    print(f"{'='*115}")
    print(f"  BASELINES (lgbm_only per HMM)")
    for hmm_cfg in HMM_BASES:
        sub = df[(df["hmm"] == hmm_cfg["label"]) & (df["lstm_mode"] == "lgbm_only")]
        if not sub.empty:
            r = sub.iloc[0]
            print(f"    {r['hmm']}: {int(r['trades'])} tr | WR {r['wr']:.1f}% | PF {r['pf']:.2f} | PnL ${r['pnl']:+.0f}")

    print(f"\n  LSTM CONTRIBUTORS (pnl > baseline, drop <= {MAX_TRADE_DROP}%): {len(contrib)} configs")
    print(f"  {'HMM':<14} {'LSTM mode':<16} {'tag':<28} {'Tr':>5} {'WR':>6} {'PF':>5} {'PnL':>8} {'ΔPnL':>7} {'drop%':>6}")
    print("-" * 115)
    for r in (contrib.head(25).to_dict("records") if len(contrib) else []):
        print(
            f"  {r['hmm']:<14} {r['lstm_mode']:<16} {r['lstm_tag']:<28} "
            f"{int(r['trades']):>5} {r['wr']:>5.1f}% {r['pf']:>5.2f} "
            f"${r['pnl']:>+7.0f} {r['pnl_delta']:>+6.0f} {r['trade_drop_pct']:>5.1f}%"
        )

    if len(contrib) == 0:
        print("  (none — LSTM tidak beat baseline dengan constraint volume)")
        best_any = df[df["lstm_mode"] != "lgbm_only"].sort_values("pnl_delta", ascending=False).head(10)
        print(f"\n  Top 10 by PnL delta (tanpa constraint):")
        for r in best_any.to_dict("records"):
            print(f"    {r['hmm']} {r['lstm_tag']}: ΔPnL ${r['pnl_delta']:+.0f} drop {r['trade_drop_pct']:.1f}% PnL ${r['pnl']:+.0f}")

    out = {
        "evaluated_at": datetime.now().isoformat(),
        "scope": "lgbm_hmm_lstm_entry_only",
        "guardian_enabled": False,
        "pass_criteria": {"pnl_delta_gt": 0, "max_trade_drop_pct": MAX_TRADE_DROP},
        "hmm_bases": HMM_BASES,
        "n_lstm_modes": len(lstm_cfgs),
        "n_contributors": len(contrib),
        "contributors_top25": contrib.head(25).to_dict("records") if len(contrib) else [],
        "all_results": df.to_dict("records"),
    }
    out_path = MODEL_DIR / "runs" / FB_RUN / "lstm_contribution_tune.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved -> {out_path}")
    print(f"{'='*115}\n")


if __name__ == "__main__":
    main()