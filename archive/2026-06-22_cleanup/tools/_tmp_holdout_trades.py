"""Extract per-trade detail dari holdout ic32_regime_v2 (Apr-Jun 22 2026)."""
import json, sys, warnings, numpy as np, pandas as pd
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import joblib
from core.evaluator import simulate_trades_swing
from core.utils import ensure_utc_index
from config import (
    ALL_COINS,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP,
    SWING_LABEL_MAX_SL, TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    MODEL_DIR, HOLDOUT_DIR,
)

LGBM_RUN     = "ic32_regime_v2"
GUARDIAN_RUN = "ic32_rv2_guard_oof_v3mom"
THR_LONG           = 0.75
THR_SHORT_BASE     = 0.70
THR_SHORT_TREND_UP = 0.80
GUARDIAN_EXIT_THR  = 0.90
GUARDIAN_MIN_HOLD  = 2
OOS_START = pd.Timestamp("2026-04-01", tz="UTC")
OOS_END   = pd.Timestamp("2026-06-22 09:00", tz="UTC")
HMM_TREND_UP = 3
LM = {"SHORT": 0, "FLAT": 1, "LONG": 2}
DYNAMIC_FEATS = {
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct",
    "direction", "entry_price_ratio", "lgbm_entry_conf", "mfe_atr_ratio",
}

lgbm_dir  = MODEL_DIR / "runs" / LGBM_RUN
gdir      = MODEL_DIR / "runs" / GUARDIAN_RUN
lgbm_model = joblib.load(lgbm_dir / "lgbm.pkl")
lgbm_feats = json.load(open(lgbm_dir / "features.json"))
g_model    = joblib.load(gdir / "guardian.pkl")
g_scaler   = joblib.load(gdir / "guardian_scaler.pkl")
g_feats    = json.load(open(gdir / "guardian_features.json"))
static_names = [f for f in g_feats if f not in DYNAMIC_FEATS]

all_trades = []

for sym in ALL_COINS:
    path = HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet"
    if not path.exists():
        continue
    df = pd.read_parquet(path)
    df = ensure_utc_index(df).sort_index()
    df = df[(df.index >= OOS_START) & (df.index < OOS_END)]
    if len(df) < 30:
        continue

    n = len(df)
    X_lgbm = np.zeros((n, len(lgbm_feats)), dtype=np.float64)
    for i, col in enumerate(lgbm_feats):
        if col in df.columns:
            X_lgbm[:, i] = df[col].ffill().fillna(0).values
    proba = lgbm_model.predict_proba(X_lgbm)
    p0, p2 = proba[:, 0], proba[:, 2]

    regime = df["hmm_regime_enc"].fillna(-1).astype(int).values if "hmm_regime_enc" in df.columns \
             else np.zeros(n, dtype=int)

    y_pred = np.full(n, LM["FLAT"], np.int32)
    y_pred[p2 >= THR_LONG] = LM["LONG"]
    for i in range(n):
        if y_pred[i] == LM["LONG"]:
            continue
        thr_s = THR_SHORT_TREND_UP if regime[i] == HMM_TREND_UP else THR_SHORT_BASE
        if p0[i] >= thr_s:
            y_pred[i] = LM["SHORT"]

    lgbm_conf = np.where(y_pred == LM["LONG"], p2, np.where(y_pred == LM["SHORT"], p0, 0.0))

    h4_sh = df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan)
    h4_sl = df["h4_swing_low"].values  if "h4_swing_low"  in df.columns else np.full(n, np.nan)

    X_guard = np.zeros((n, len(static_names)), dtype=np.float64)
    for i, col in enumerate(static_names):
        if col in df.columns:
            X_guard[:, i] = df[col].ffill().fillna(0).values

    result = simulate_trades_swing(
        y_pred=y_pred, close=df["close"].values, high=df["high"].values, low=df["low"].values,
        atr=df["atr_14_h1"].values, h4_swing_highs=h4_sh, h4_swing_lows=h4_sl,
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0], fee_per_side=FEE_PER_SIDE,
        slippage=SLIPPAGE_PER_SIDE, max_hold=MAX_HOLDING_BARS,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP, max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        guardian_enabled=True, guardian_model=g_model, guardian_scaler=g_scaler,
        X_guardian=X_guard, guardian_feat_cols=g_feats, guardian_static_names=static_names,
        lgbm_conf_arr=lgbm_conf, guardian_exit_threshold=GUARDIAN_EXIT_THR,
        guardian_min_hold_bars=GUARDIAN_MIN_HOLD,
    )

    timestamps = df.index
    for t in result.get("trades", []):
        bar_in  = t.get("bar_in",  0)
        bar_out = t.get("bar_out", 0)
        all_trades.append({
            "coin":        sym,
            "entry_time":  str(timestamps[bar_in])  if bar_in  < len(timestamps) else "",
            "exit_time":   str(timestamps[min(bar_out, len(timestamps)-1)]),
            "direction":   t.get("direction", ""),
            "entry_price": round(t.get("entry", 0), 6),
            "exit_price":  round(t.get("exit",  0), 6),
            "tp":          round(t.get("tp", 0), 6),
            "sl":          round(t.get("sl", 0), 6),
            "rr":          round(t.get("rr", 0), 3),
            "bars_held":   bar_out - bar_in,
            "net_pnl":     round(t.get("net_pnl", 0), 4),
            "outcome":     t.get("outcome", ""),
            "lgbm_conf":   round(float(lgbm_conf[bar_in]), 4) if bar_in < n else 0,
        })

df_trades = pd.DataFrame(all_trades)
df_trades = df_trades.sort_values("entry_time").reset_index(drop=True)

out = ROOT / "reports" / "experiments" / "2026-06-22_ic32_regime_v2_trades.csv"
df_trades.to_csv(out, index=False)
print(f"Total trades: {len(df_trades)}")
print(f"Saved -> {out}")
print()
print(df_trades.to_string(max_rows=None))
