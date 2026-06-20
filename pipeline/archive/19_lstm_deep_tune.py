"""
pipeline/19_lstm_deep_tune.py
LSTM deep contribution sweep — mode historis + produksi yang belum di sweep 18.

Tambahan vs 18_lstm_contribution_tune.py:
  - soft_pen_hmm (09_tune_hmm_adaptive — mode produksi asli)
  - veto_hmm, soft_gate, lstm_dominant, flat_unlock, soft_vote
  - prod_soft_veto dengan no_veto_thr (inference.py cascade)
  - 2 checkpoint LSTM: directional (lstm_best.pt) + widyawardhana v1
  - HMM grid lebih lebar (termasuk T50_R60 dari hmm_adaptive)

Pass "LSTM contributes": pnl_delta > 0 vs lgbm_only (same HMM) AND trade_drop <= 20%

Usage:
  python pipeline/19_lstm_deep_tune.py
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

logger = setup_logger("19_lstm_deep_tune")

FB_RUN = "tb_lgbm_flatboost_v2"
HOLDOUT_START = "2026-04-01"
PERIOD_MONTHS = 2.5
LM = LABEL_MAP if isinstance(LABEL_MAP, dict) else {"SHORT": 0, "FLAT": 1, "LONG": 2}
SHORT, FLAT, LONG = 0, 1, 2
TRENDING = {0, 3}
MAX_TRADE_DROP = 20.0

LSTM_VARIANTS = {
    "directional": {
        "model": MODEL_DIR / "lstm_best.pt",
        "scaler": MODEL_DIR / "lstm_scaler.pkl",
        "feats": MODEL_DIR / "feature_cols_lstm_temporal.json",
    },
    "widyawardhana": {
        "model": MODEL_DIR / "runs" / "tb_lstm_widyawardhana_v1" / "lstm.pt",
        "scaler": MODEL_DIR / "runs" / "tb_lstm_widyawardhana_v1" / "lstm_scaler.pkl",
        "feats": MODEL_DIR / "runs" / "tb_lstm_widyawardhana_v1" / "tb_lstm_widyawardhana_v1_features.json",
    },
}

HMM_BASES = [
    {"label": "T48_R55_s5", "thr_T": 0.48, "thr_R": 0.55, "short_off": 0.05},
    {"label": "T50_R55_s5", "thr_T": 0.50, "thr_R": 0.55, "short_off": 0.05},
    {"label": "T50_R60_s5", "thr_T": 0.50, "thr_R": 0.60, "short_off": 0.05},
    {"label": "T52_R55_s5", "thr_T": 0.52, "thr_R": 0.55, "short_off": 0.05},
    {"label": "T54_R60_s5", "thr_T": 0.54, "thr_R": 0.60, "short_off": 0.05},
    {"label": "T45_R55_s5", "thr_T": 0.45, "thr_R": 0.55, "short_off": 0.05},
    {"label": "T50_R65_s5", "thr_T": 0.50, "thr_R": 0.65, "short_off": 0.05},
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


def _max_safe_pen(proba, signal_idx):
    other = [i for i in range(3) if i != signal_idx]
    o = float(max(proba[i] for i in other))
    f = float(sum(proba[i] for i in other)) - o
    total_other = o + f
    s = float(proba[signal_idx])
    if total_other > 0 and o < s:
        return max(0.0, float((s - o) * total_other / (2 * o + f)) - 0.01)
    return 0.0


def apply_mode(mode, yp, conf, proba_fb, proba_lstm, hmm, thr_t, thr_r, short_off, params):
    yp_o, conf_o = yp.copy(), conf.copy()
    scope = params.get("scope", "all")

    if mode == "lgbm_only":
        return yp_o, conf_o

    if mode == "prod_soft_veto":
        opp = params["opposite_pen"]
        agr = params.get("agree_boost", 0.05)
        neu = params.get("neutral_pen", 0.05)
        no_veto = params.get("no_veto_thr", 0.50)
        for i in range(len(yp_o)):
            if yp_o[i] == FLAT or not _scope_ok(scope, hmm[i]):
                continue
            sig = int(yp_o[i])
            p = proba_fb[i].copy()
            li = int(np.argmax(proba_lstm[i]))
            if li == sig:
                adj = agr
            elif li == FLAT:
                adj = -neu
            elif p[sig] > no_veto:
                adj = -min(opp, _max_safe_pen(p, sig))
            else:
                adj = -opp
            conf_o[i] = float(np.clip(p[sig] + adj, 0.0, 1.0))
        return regate(yp_o, conf_o, hmm, thr_t, thr_r, short_off)

    if mode == "soft_pen_hmm":
        thr_lstm = params["thr_lstm"]
        lstm_scope = params.get("lstm_scope", "trending")
        for i in range(len(yp_o)):
            if yp_o[i] == FLAT:
                continue
            if lstm_scope == "trending" and hmm[i] not in TRENDING:
                continue
            sig = int(yp_o[i])
            thr = _thr_for_bar(hmm[i], sig, thr_t, thr_r, short_off)
            if sig == LONG:
                lstm_opp = float(proba_lstm[i, 0])
                entry_conf = float(proba_fb[i, 2])
            else:
                lstm_opp = float(proba_lstm[i, 2])
                entry_conf = float(proba_fb[i, 0])
            if lstm_opp >= thr_lstm:
                adjusted = entry_conf - lstm_opp * (entry_conf - thr + 0.05)
                if adjusted < thr:
                    yp_o[i] = FLAT
                    conf_o[i] = 0.0
        return yp_o, conf_o

    if mode == "veto_hmm":
        thr_lstm = params["thr_lstm"]
        lstm_scope = params.get("lstm_scope", "trending")
        for i in range(len(yp_o)):
            if yp_o[i] == FLAT:
                continue
            if lstm_scope == "trending" and hmm[i] not in TRENDING:
                continue
            sig = int(yp_o[i])
            lstm_opp = float(proba_lstm[i, 0] if sig == LONG else proba_lstm[i, 2])
            if lstm_opp >= thr_lstm:
                yp_o[i] = FLAT
                conf_o[i] = 0.0
        return yp_o, conf_o

    if mode == "soft_gate":
        opp_max = params["opp_max"]
        lstm_scope = params.get("lstm_scope", "all")
        for i in range(len(yp_o)):
            if yp_o[i] == FLAT:
                continue
            if lstm_scope == "trending" and hmm[i] not in TRENDING:
                continue
            sig = int(yp_o[i])
            lstm_opp = float(proba_lstm[i, 0] if sig == LONG else proba_lstm[i, 2])
            if lstm_opp >= opp_max:
                yp_o[i] = FLAT
                conf_o[i] = 0.0
        return yp_o, conf_o

    if mode == "lstm_dominant":
        dom_thr = params["dom_thr"]
        for i in range(len(yp_o)):
            if yp_o[i] == FLAT:
                continue
            sig = int(yp_o[i])
            lstm_l, lstm_s = float(proba_lstm[i, 2]), float(proba_lstm[i, 0])
            if lstm_l > lstm_s:
                ld, dom = LONG, lstm_l
            elif lstm_s > lstm_l:
                ld, dom = SHORT, lstm_s
            else:
                yp_o[i] = FLAT
                conf_o[i] = 0.0
                continue
            if ld != sig or dom < dom_thr:
                yp_o[i] = FLAT
                conf_o[i] = 0.0
        return yp_o, conf_o

    if mode == "flat_unlock":
        override_thr = params["override_thr"]
        lgbm_dir_min = params.get("lgbm_dir_min", 0.35)
        n = len(yp_o)
        for i in range(n):
            if yp_o[i] != FLAT:
                continue
            p = proba_fb[i]
            lgbm_dir_score = max(float(p[2]), float(p[0]))
            if lgbm_dir_score < lgbm_dir_min:
                continue
            li = int(np.argmax(proba_lstm[i]))
            lc = float(proba_lstm[i, li])
            if li == FLAT or lc < override_thr:
                continue
            is_t = hmm[i] in TRENDING
            thr_l = thr_t if is_t else thr_r
            thr_s = thr_l + short_off
            if li == LONG and float(p[2]) >= thr_l - 0.03:
                yp_o[i] = LONG
                conf_o[i] = lc
            elif li == SHORT and float(p[0]) >= thr_s - 0.03:
                yp_o[i] = SHORT
                conf_o[i] = lc
        return regate(yp_o, conf_o, hmm, thr_t, thr_r, short_off)

    if mode == "soft_vote":
        lgbm_w = params["lgbm_w"]
        lstm_w = 1.0 - lgbm_w
        for i in range(len(yp_o)):
            if yp_o[i] == FLAT:
                continue
            sig = int(yp_o[i])
            combined = lgbm_w * float(proba_fb[i, sig]) + lstm_w * float(proba_lstm[i, sig])
            conf_o[i] = float(np.clip(combined, 0.0, 1.0))
        return regate(yp_o, conf_o, hmm, thr_t, thr_r, short_off)

    if mode == "boost_only":
        agr = params["agree_boost"]
        for i in range(len(yp_o)):
            if yp_o[i] == FLAT or not _scope_ok(scope, hmm[i]):
                continue
            if int(np.argmax(proba_lstm[i])) == int(yp_o[i]):
                conf_o[i] = float(np.clip(conf_o[i] + agr, 0.0, 1.0))
        return regate(yp_o, conf_o, hmm, thr_t, thr_r, short_off)

    if mode == "opp_ranging_only":
        opp = params["opposite_pen"]
        for i in range(len(yp_o)):
            if yp_o[i] == FLAT or hmm[i] in TRENDING:
                continue
            sig = int(yp_o[i])
            li = int(np.argmax(proba_lstm[i]))
            if li != sig and li != FLAT:
                conf_o[i] = float(np.clip(conf_o[i] - opp, 0.0, 1.0))
        return regate(yp_o, conf_o, hmm, thr_t, thr_r, short_off)

    if mode == "unlock_addon":
        delta = params["thr_delta"]
        lstm_min = params["lstm_min"]
        n = len(yp_o)
        for i in range(n):
            is_t = hmm[i] in TRENDING
            thr_l = (thr_t - delta) if is_t else (thr_r - delta)
            thr_s = thr_l + short_off
            p = proba_fb[i]
            sig = FLAT
            c = 0.0
            if p[2] >= thr_l and p[2] >= p[0]:
                sig, c = LONG, float(p[2])
            elif p[0] >= thr_s and p[0] > p[2]:
                sig, c = SHORT, float(p[0])
            if sig == FLAT:
                continue
            lstm_l, lstm_s = float(proba_lstm[i, 2]), float(proba_lstm[i, 0])
            dom = lstm_l if sig == LONG else lstm_s
            ld = LONG if lstm_l > lstm_s else (SHORT if lstm_s > lstm_l else FLAT)
            if yp_o[i] == FLAT and ld == sig and dom >= lstm_min:
                yp_o[i] = sig
                conf_o[i] = c
        return yp_o, conf_o

    return yp_o, conf_o


def build_configs():
    cfgs = [{"mode": "lgbm_only", "params": {}, "tag": "lgbm_only", "lstm_variant": "directional"}]

    for scope, opp, no_veto in itertools.product(
        ["all", "trending"],
        [0.05, 0.08, 0.10],
        [0.45, 0.50, 0.55],
    ):
        cfgs.append({
            "mode": "prod_soft_veto",
            "params": {"scope": scope, "opposite_pen": opp, "no_veto_thr": no_veto},
            "tag": f"prod_{scope[0]}_o{int(opp*100)}_nv{int(no_veto*100)}",
            "lstm_variant": "directional",
        })

    for thr in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65]:
        for scope in ["trending", "all"]:
            cfgs.append({
                "mode": "soft_pen_hmm",
                "params": {"thr_lstm": thr, "lstm_scope": scope},
                "tag": f"spen_{scope[0]}_t{int(thr*100)}",
                "lstm_variant": "directional",
            })

    for thr in [0.50, 0.55, 0.60, 0.65]:
        cfgs.append({
            "mode": "veto_hmm",
            "params": {"thr_lstm": thr, "lstm_scope": "trending"},
            "tag": f"veto_t_t{int(thr*100)}",
            "lstm_variant": "directional",
        })

    for opp_max in [0.30, 0.35, 0.40, 0.45]:
        for scope in ["all", "trending"]:
            cfgs.append({
                "mode": "soft_gate",
                "params": {"opp_max": opp_max, "lstm_scope": scope},
                "tag": f"sgate_{scope[0]}_m{int(opp_max*100)}",
                "lstm_variant": "directional",
            })

    for dom in [0.28, 0.32, 0.35, 0.38, 0.42]:
        cfgs.append({
            "mode": "lstm_dominant",
            "params": {"dom_thr": dom},
            "tag": f"ldom_{int(dom*100)}",
            "lstm_variant": "directional",
        })

    for ovr in [0.55, 0.60, 0.65]:
        cfgs.append({
            "mode": "flat_unlock",
            "params": {"override_thr": ovr},
            "tag": f"unlock_{int(ovr*100)}",
            "lstm_variant": "directional",
        })

    for w in [0.55, 0.60, 0.65]:
        cfgs.append({
            "mode": "soft_vote",
            "params": {"lgbm_w": w},
            "tag": f"svote_w{int(w*100)}",
            "lstm_variant": "directional",
        })

    for scope, agr in itertools.product(["all", "trending"], [0.05, 0.08, 0.12]):
        cfgs.append({
            "mode": "boost_only",
            "params": {"scope": scope, "agree_boost": agr},
            "tag": f"boost_{scope[0]}_a{int(agr*100)}",
            "lstm_variant": "directional",
        })

    for opp in [0.05, 0.08, 0.12]:
        cfgs.append({
            "mode": "opp_ranging_only",
            "params": {"opposite_pen": opp},
            "tag": f"oppR_p{int(opp*100)}",
            "lstm_variant": "directional",
        })

    for delta, lm in itertools.product([0.02, 0.03, 0.04], [0.30, 0.35, 0.38]):
        cfgs.append({
            "mode": "unlock_addon",
            "params": {"thr_delta": delta, "lstm_min": lm},
            "tag": f"addon_d{int(delta*100)}_m{int(lm*100)}",
            "lstm_variant": "directional",
        })

    # widyawardhana LSTM — mode terbaik dari directional
    for mode, tag, params in [
        ("prod_soft_veto", "wa_prod_a", {"scope": "all", "opposite_pen": 0.08, "no_veto_thr": 0.50}),
        ("prod_soft_veto", "wa_prod_t", {"scope": "trending", "opposite_pen": 0.08, "no_veto_thr": 0.50}),
        ("soft_pen_hmm", "wa_spen_t50", {"thr_lstm": 0.50, "lstm_scope": "trending"}),
        ("soft_pen_hmm", "wa_spen_t45", {"thr_lstm": 0.45, "lstm_scope": "trending"}),
        ("soft_gate", "wa_sgate_35", {"opp_max": 0.35, "lstm_scope": "all"}),
        ("boost_only", "wa_boost_a8", {"scope": "all", "agree_boost": 0.08}),
        ("flat_unlock", "wa_unlock_60", {"override_thr": 0.60}),
    ]:
        cfgs.append({
            "mode": mode, "params": params, "tag": tag, "lstm_variant": "widyawardhana",
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


def load_lstm_variant(coin_data_base, variant_key, fb_feats):
    spec = LSTM_VARIANTS[variant_key]
    if not spec["model"].exists():
        logger.warning(f"LSTM variant {variant_key} not found: {spec['model']}")
        return None
    model = load_lstm(spec["model"])
    scaler = joblib.load(spec["scaler"])
    with open(spec["feats"], encoding="utf-8") as f:
        feats = json.load(f)

    out = {}
    for sym, d in coin_data_base.items():
        df = d["df"]
        n = len(df)
        X_lstm = np.zeros((n, len(feats)), dtype=np.float64)
        for i, c in enumerate(feats):
            if c in df.columns:
                X_lstm[:, i] = df[c].ffill().fillna(0).values.astype(np.float64)
        out[sym] = get_lstm_proba(model, scaler, X_lstm, n)
    return out


def main():
    fb_model = joblib.load(MODEL_DIR / "runs" / FB_RUN / "lgbm.pkl")
    with open(MODEL_DIR / "runs" / FB_RUN / f"{FB_RUN}_features.json", encoding="utf-8") as f:
        fb_feats = json.load(f)

    coin_data_base = {}
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
        coin_data_base[sym] = {
            "df": df,
            "proba_fb": proba_fb,
            "hmm": hmm,
            "close": df["close"].values, "high": df["high"].values, "low": df["low"].values,
            "atr": df["atr_14_h1"].values,
            "h4_sh": df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan),
            "h4_sl": df["h4_swing_low"].values if "h4_swing_low" in df.columns else np.full(n, np.nan),
            "h4_tr": df["h4_trend"].values if "h4_trend" in df.columns else None,
            "yt": np.array([LM.get(str(v), 1) for v in df["label"].values], dtype=np.int32),
            "index": df.index,
        }

    lstm_proba_cache = {}
    for vk in LSTM_VARIANTS:
        proba = load_lstm_variant(coin_data_base, vk, fb_feats)
        if proba:
            lstm_proba_cache[vk] = proba

    coin_data = {}
    for sym, d in coin_data_base.items():
        coin_data[sym] = {k: v for k, v in d.items() if k != "df"}

    configs = build_configs()
    n_eval = len(HMM_BASES) * len(configs)
    print(f"\nLSTM DEEP TUNE | {len(coin_data)} coins | {len(HMM_BASES)} HMM × {len(configs)} configs = {n_eval} evals")
    print(f"LSTM variants: {list(lstm_proba_cache.keys())}")
    print(f"Guardian OFF | Pass: pnl_delta > 0 & trade_drop <= {MAX_TRADE_DROP}%\n")

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

        for lc in configs:
            if lc["mode"] == "lgbm_only":
                sc = base_sc
                drop = 0.0
                pnl_delta = 0.0
            else:
                vk = lc.get("lstm_variant", "directional")
                if vk not in lstm_proba_cache:
                    continue
                yp_map, conf_map = {}, {}
                for sym, d in coin_data.items():
                    yp, conf = baselines[sym]
                    yp, conf = apply_mode(
                        lc["mode"], yp, conf, d["proba_fb"],
                        lstm_proba_cache[vk][sym], d["hmm"],
                        hmm_cfg["thr_T"], hmm_cfg["thr_R"], hmm_cfg["short_off"],
                        lc["params"],
                    )
                    yp_map[sym] = yp
                    conf_map[sym] = conf
                sc = eval_config(coin_data, yp_map, conf_map)
                drop = (base_trades - sc["trades"]) / base_trades * 100
                pnl_delta = sc["pnl"] - base_sc["pnl"]

            contributes = pnl_delta > 0 and drop <= MAX_TRADE_DROP and lc["mode"] != "lgbm_only"
            row = {
                "hmm": hmm_cfg["label"],
                "thr_T": hmm_cfg["thr_T"],
                "thr_R": hmm_cfg["thr_R"],
                "short_off": hmm_cfg["short_off"],
                "lstm_variant": lc.get("lstm_variant", "directional"),
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
            if contributes:
                contributors.append(row)

    df = pd.DataFrame(all_results)
    contrib = pd.DataFrame(contributors).sort_values("pnl", ascending=False) if contributors else pd.DataFrame()

    print(f"{'='*120}")
    print("  BASELINES (lgbm_only per HMM)")
    for hmm_cfg in HMM_BASES:
        sub = df[(df["hmm"] == hmm_cfg["label"]) & (df["lstm_mode"] == "lgbm_only")]
        if not sub.empty:
            r = sub.iloc[0]
            print(f"    {r['hmm']}: {int(r['trades'])} tr | WR {r['wr']:.1f}% | PF {r['pf']:.2f} | PnL ${r['pnl']:+.0f}")

    print(f"\n  LSTM CONTRIBUTORS: {len(contrib)} configs")
    print(f"  {'HMM':<14} {'variant':<14} {'mode':<16} {'tag':<22} {'Tr':>5} {'WR':>6} {'PnL':>8} {'ΔPnL':>7} {'drop%':>6}")
    print("-" * 120)
    for r in (contrib.head(30).to_dict("records") if len(contrib) else []):
        print(
            f"  {r['hmm']:<14} {r['lstm_variant']:<14} {r['lstm_mode']:<16} {r['lstm_tag']:<22} "
            f"{int(r['trades']):>5} {r['wr']:>5.1f}% ${r['pnl']:>+7.0f} {r['pnl_delta']:>+6.0f} {r['trade_drop_pct']:>5.1f}%"
        )

    if len(contrib) == 0:
        print("  (none with strict pass criteria)")
        best = df[df["lstm_mode"] != "lgbm_only"].sort_values("pnl_delta", ascending=False).head(15)
        print("\n  Top 15 by PnL delta (any drop%):")
        for r in best.to_dict("records"):
            print(
                f"    {r['hmm']} {r['lstm_variant']} {r['lstm_tag']}: "
                f"ΔPnL ${r['pnl_delta']:+.0f} drop {r['trade_drop_pct']:.1f}% PnL ${r['pnl']:+.0f} "
                f"ppt {r['pnl_per_trade']:+.3f}"
            )

    out = {
        "evaluated_at": datetime.now().isoformat(),
        "scope": "lstm_deep_tune_historical_modes",
        "guardian_enabled": False,
        "pass_criteria": {"pnl_delta_gt": 0, "max_trade_drop_pct": MAX_TRADE_DROP},
        "lstm_variants": list(lstm_proba_cache.keys()),
        "hmm_bases": HMM_BASES,
        "n_configs": len(configs),
        "n_contributors": len(contrib),
        "contributors_top30": contrib.head(30).to_dict("records") if len(contrib) else [],
        "all_results": df.to_dict("records"),
    }
    out_path = MODEL_DIR / "runs" / FB_RUN / "lstm_deep_tune.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\n  Saved -> {out_path}")
    print(f"{'='*120}\n")


if __name__ == "__main__":
    main()