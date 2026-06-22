"""Quick filter ablation sample (5 coins) vs frozen OOF stack."""
import sys, warnings, json
import numpy as np
import pandas as pd
from pathlib import Path
import joblib

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from config import *
from core.evaluator import simulate_trades_swing
from core.utils import ensure_utc_index

HMM_B = {0: (0.55, 0.55), 1: (0.55, 0.55), 2: (0.50, 0.50), 3: (0.45, 0.50), -1: (0.45, 0.45)}
LGBM_DIR = MODEL_DIR / "runs/tb_lgbm_genuine_v2"
GUARDIAN_DIR = MODEL_DIR / "runs/tb_guardian_genuine_v2_hmm_v2"

g_model = joblib.load(GUARDIAN_DIR / "guardian.pkl")
g_scaler = joblib.load(GUARDIAN_DIR / "guardian_scaler.pkl")
with open(GUARDIAN_DIR / "guardian_features.json") as f:
    gfe = json.load(f)
DYN = [
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct", "direction", "entry_price_ratio",
]
g_static = [f for f in gfe if f not in DYN]
oof = pd.read_parquet(LGBM_DIR / "oof_predictions.parquet")


def apply_hmm(p0, p2, hmm):
    n = len(p0)
    tl = np.full(n, 0.45)
    ts = np.full(n, 0.45)
    for s, (a, b) in HMM_B.items():
        if s == -1:
            continue
        m = hmm == s
        tl[m] = a
        ts[m] = b
    y = np.ones(n, dtype=np.int32)
    y[p2 >= tl] = 2
    y[(p0 >= ts) & (y != 2)] = 0
    return y


def summarize(trades):
    if not trades:
        return dict(n=0, wr=0, ppt=0, pf=0)
    n = len(trades)
    w = sum(1 for t in trades if t["net_pnl"] > 0)
    g = sum(t["net_pnl"] for t in trades if t["net_pnl"] > 0)
    l = sum(abs(t["net_pnl"]) for t in trades if t["net_pnl"] < 0)
    pnl = sum(t["net_pnl"] for t in trades)
    return dict(n=n, wr=w / n * 100, ppt=pnl / n, pf=g / l if l else 9)


variants = {
    "baseline_oof": {"structural_filter": True, "vcb_enabled": False, "min_rr": SWING_LABEL_MIN_RR},
    "all_on_prod": {"structural_filter": True, "vcb_enabled": True, "min_rr": SWING_LABEL_MIN_RR},
    "no_struct": {"structural_filter": False, "vcb_enabled": False, "min_rr": SWING_LABEL_MIN_RR},
    "no_rr": {
        "structural_filter": True, "vcb_enabled": False,
        "min_rr": 0.0, "min_tp_atr": 0.0, "max_sl_atr": 999.0,
    },
    "all_off": {
        "structural_filter": False, "vcb_enabled": False,
        "min_rr": 0.0, "min_tp_atr": 0.0, "max_sl_atr": 999.0,
    },
}

results = {k: [] for k in variants}
vcb_total = 0
sample = ALL_COINS[:7]

for sym in sample:
    p = LABEL_DIR / f"{sym}_features_v3.parquet"
    if not p.exists():
        continue
    df = ensure_utc_index(pd.read_parquet(p)).sort_index()
    df = df[df.index < TRAIN_CUTOFF_DATE]
    so = oof[(oof.coin == sym) & (oof.has_oof)][["p0", "p2"]].reindex(df.index)
    m = so.p0.notna()
    df = df[m]
    so = so[m]
    if len(df) < 100:
        continue
    p0 = so.p0.values.astype(np.float32)
    p2 = so.p2.values.astype(np.float32)
    hmm = (
        df.hmm_regime_enc.fillna(-1).values.astype(np.int8)
        if "hmm_regime_enc" in df.columns else np.full(len(df), -1, np.int8)
    )
    y = apply_hmm(p0, p2, hmm)
    X = np.zeros((len(df), len(g_static)))
    for i, c in enumerate(g_static):
        if c in df.columns:
            X[:, i] = df[c].ffill().fillna(0).values
    base = dict(
        y_pred=y,
        close=df.close.values, high=df.high.values, low=df.low.values,
        atr=df.atr_14_h1.values,
        h4_swing_highs=df.h4_swing_high.values if "h4_swing_high" in df else np.full(len(df), np.nan),
        h4_swing_lows=df.h4_swing_low.values if "h4_swing_low" in df else np.full(len(df), np.nan),
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        max_hold=MAX_HOLDING_BARS,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        guardian_enabled=True, guardian_model=g_model, guardian_scaler=g_scaler, X_guardian=X,
        guardian_exit_threshold=0.55, guardian_min_hold_bars=2,
        guardian_activation_atr=GUARDIAN_ACTIVATION_ATR,
        structural_tolerance_pct=0.03,
        vcb_atr_multiplier=3.0, vcb_lookback_bars=24,
    )
    for name, extra in variants.items():
        kw = {**base, **extra}
        if "min_tp_atr" not in extra:
            kw["min_tp_atr"] = SWING_LABEL_MIN_TP
            kw["max_sl_atr"] = SWING_LABEL_MAX_SL
        r = simulate_trades_swing(**kw)
        results[name].extend(r.get("trades", []))
        if name == "all_on_prod":
            vcb_total += r.get("n_vcb_blocked", 0)

print(f"Filter ablation sample ({len(sample)} coins, HMM-B + Guardian):")
for name, tr in results.items():
    s = summarize(tr)
    print(f"  {name:<16} N={s['n']:>5}  WR={s['wr']:>5.1f}%  PPT={s['ppt']:>+.4f}  PF={s['pf']:.2f}")
print(f"  VCB blocks (all_on_prod): {vcb_total}")