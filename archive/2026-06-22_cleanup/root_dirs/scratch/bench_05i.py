import itertools, time, joblib, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAIN_CUTOFF_DATE, LABEL_DIR, MODEL_DIR, MODAL_PER_TRADE, LEVERAGE_SIM,
    FEE_PER_SIDE, SLIPPAGE_PER_SIDE, SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP,
    SWING_LABEL_MAX_SL, TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL, MAX_HOLDING_BARS,
    GUARDIAN_ACTIVATION_ATR, ALL_COINS,
)
from core.evaluator import simulate_trades_swing
from core.utils import ensure_utc_index

LGBM_DIR = MODEL_DIR / "runs" / "tb_lgbm_genuine_v2"
GUARDIAN_DIR = MODEL_DIR / "runs" / "tb_guardian_genuine_v2_hmm_v2"
DYNAMIC = [
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct", "direction", "entry_price_ratio",
]

oof = pd.read_parquet(LGBM_DIR / "oof_predictions.parquet")
g_model = joblib.load(GUARDIAN_DIR / "guardian.pkl")
g_scaler = joblib.load(GUARDIAN_DIR / "guardian_scaler.pkl")
with open(GUARDIAN_DIR / "guardian_features.json") as f:
    static = [x for x in json.load(f) if x not in DYNAMIC]

total_bars = 0
t_coin = []
for sym in ALL_COINS[:3]:
    path = LABEL_DIR / f"{sym}_features_v3.parquet"
    if not path.exists():
        continue
    df = ensure_utc_index(pd.read_parquet(path)).sort_index()
    df = df[df.index < TRAIN_CUTOFF_DATE]
    sym_oof = oof[(oof.coin == sym) & oof.has_oof][["p0", "p2"]]
    proba = sym_oof.reindex(df.index)
    has = proba.p0.notna()
    df_oof = df[has]
    n = len(df_oof)
    total_bars += n
    p0 = proba.p0[has].values
    p2 = proba.p2[has].values
    y = np.ones(n, dtype=np.int32)
    y[p2 >= 0.45] = 2
    y[(p0 >= 0.45) & (y != 2)] = 0
    X = np.zeros((n, len(static)))
    for i, c in enumerate(static):
        if c in df_oof.columns:
            X[:, i] = df_oof[c].ffill().fillna(0).values
    kw = dict(
        close=df_oof.close.values, high=df_oof.high.values, low=df_oof.low.values,
        atr=df_oof["atr_14_h1"].values,
        h4_swing_highs=df_oof["h4_swing_high"].values if "h4_swing_high" in df_oof else np.full(n, np.nan),
        h4_swing_lows=df_oof["h4_swing_low"].values if "h4_swing_low" in df_oof else np.full(n, np.nan),
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0], fee_per_side=FEE_PER_SIDE,
        slippage=SLIPPAGE_PER_SIDE, max_hold=MAX_HOLDING_BARS,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP, max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
    )
    t0 = time.time()
    r = simulate_trades_swing(
        y_pred=y, guardian_enabled=True, guardian_model=g_model, guardian_scaler=g_scaler,
        X_guardian=X, guardian_exit_threshold=0.55, guardian_min_hold_bars=2,
        guardian_activation_atr=GUARDIAN_ACTIVATION_ATR, **kw,
    )
    dt = time.time() - t0
    t_coin.append(dt)
    print(f"{sym}: bars={n:,} sim={dt:.2f}s trades={len(r.get('trades', []))}")

avg = sum(t_coin) / len(t_coin)
configs = 1 + 2 * 2 * 4 * 3 * 3
coins = 21
print(f"\navg coin sim: {avg:.2f}s")
print(f"grid configs: {configs}")
print(f"full backtests: {configs * coins:,}")
print(f"est total (sim only): {configs * coins * avg / 60:.1f} min")
print(f"+ build_signal_panel x144 (pivot) adds significant overhead")