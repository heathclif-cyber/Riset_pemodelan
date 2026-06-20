"""
Fine-tune 2 mode LSTM yang berkontribusi dari sweep 19:
  1. soft_pen_hmm (filter) — T48_R55
  2. unlock_addon (unlock) — T50_R55 production HMM
"""
import itertools
import json
import sys
import warnings
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
from core.utils import ensure_utc_index
from pipeline.backtest_utils import get_lstm_proba
import importlib.util

_spec = importlib.util.spec_from_file_location("tune19", ROOT / "pipeline" / "19_lstm_deep_tune.py")
_tune19 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_tune19)
hmm_entry = _tune19.hmm_entry
apply_mode = _tune19.apply_mode
eval_config = _tune19.eval_config
FB_RUN = _tune19.FB_RUN
HOLDOUT_START = _tune19.HOLDOUT_START
LM = _tune19.LM

HMM_TARGETS = [
    {"label": "T48_filter", "thr_T": 0.48, "thr_R": 0.55, "short_off": 0.05, "mode": "soft_pen_hmm"},
    {"label": "T50_unlock", "thr_T": 0.50, "thr_R": 0.55, "short_off": 0.05, "mode": "unlock_addon"},
]


def fine_configs(mode):
    if mode == "soft_pen_hmm":
        return [{"thr_lstm": round(t, 2), "lstm_scope": "trending"}
                for t in np.arange(0.46, 0.541, 0.01)]
    return [{"thr_delta": d, "lstm_min": m}
            for d, m in itertools.product(
                [0.015, 0.02, 0.025, 0.03],
                [0.28, 0.30, 0.32, 0.34, 0.36],
            )]


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

    results = []
    for hmm_cfg in HMM_TARGETS:
        baselines = {}
        for sym, d in coin_data.items():
            baselines[sym] = hmm_entry(
                d["proba_fb"], d["hmm"],
                hmm_cfg["thr_T"], hmm_cfg["thr_R"], hmm_cfg["short_off"],
            )
        base_yp = {s: baselines[s][0] for s in baselines}
        base_cf = {s: baselines[s][1] for s in baselines}
        base_sc = eval_config(coin_data, base_yp, base_cf)
        base_tr = max(base_sc["trades"], 1)

        for params in fine_configs(hmm_cfg["mode"]):
            yp_map, conf_map = {}, {}
            for sym, d in coin_data.items():
                yp, conf = baselines[sym]
                yp, conf = apply_mode(
                    hmm_cfg["mode"], yp, conf, d["proba_fb"], d["proba_lstm"], d["hmm"],
                    hmm_cfg["thr_T"], hmm_cfg["thr_R"], hmm_cfg["short_off"], params,
                )
                yp_map[sym] = yp
                conf_map[sym] = conf
            sc = eval_config(coin_data, yp_map, conf_map)
            drop = (base_tr - sc["trades"]) / base_tr * 100
            results.append({
                "hmm": hmm_cfg["label"],
                "mode": hmm_cfg["mode"],
                "params": params,
                "baseline_pnl": base_sc["pnl"],
                "pnl_delta": round(sc["pnl"] - base_sc["pnl"], 2),
                "trade_drop_pct": round(drop, 2),
                **sc,
            })

    df = pd.DataFrame(results)
    print("\n=== FINE TUNE RESULTS ===\n")
    for hmm_label in ["T48_filter", "T50_unlock"]:
        sub = df[df["hmm"] == hmm_label].sort_values("pnl_delta", ascending=False)
        print(f"--- {hmm_label} baseline PnL ${sub.iloc[0]['baseline_pnl']:.0f} ---")
        for r in sub.head(10).to_dict("records"):
            print(
                f"  {r['params']} | tr={r['trades']} dPnL={r['pnl_delta']:+.1f} "
                f"drop={r['trade_drop_pct']:.1f}% ppt={r['pnl_per_trade']:.3f}"
            )
        print()

    out_path = MODEL_DIR / "runs" / FB_RUN / "lstm_fine_tune.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"results": df.to_dict("records")}, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()