"""
scratch/compare_ic32_vs_tb.py
Head-to-head: ic32_regime_v1+Guardian vs TB+LSTM+Guardian
Live-like eval: SL=1.5xATR, max_hold=36, NO TP | Nov 2025 - Apr 2026
"""
import json, sys, warnings
import numpy as np
import pandas as pd
import torch
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import joblib
from core.models import load_lstm
from core.utils import ensure_utc_index
from config import *

PROD = Path("D:/Apps-Dev/swint_tradev2/models")
SL_MULT  = TP_SL_FALLBACK_SL
MAX_HOLD = MAX_HOLDING_BARS
MODAL    = MODAL_PER_TRADE
LEVERAGE = LEVERAGE_SIM[0] if isinstance(LEVERAGE_SIM, list) else LEVERAGE_SIM
COST_RT  = (FEE_PER_SIDE + SLIPPAGE_PER_SIDE) * 2
SEQ_LEN  = 48
LSTM_ALPHA = 0.3
GDN_MIN  = 2
GDN_THR  = 0.65
REGIME_THRESH = {0: 0.45, 1: 0.50, 2: 0.50, 3: 0.45}
DYNAMIC_NAMES = frozenset({
    "bars_held_norm", "current_pnl_pct", "current_pnl_atr",
    "max_favorable_pnl_pct", "drawdown_from_peak_pct",
    "direction", "entry_price_ratio",
})

# ── Load models ─────────────────────────────────────────────────────────────────
ic32_model = joblib.load(PROD / "lgbm_baseline.pkl")
with open(PROD / "feature_cols_v2.json") as f:
    ic32_feats = json.load(f)

ic32g_model  = joblib.load(PROD / "guardian_best.pkl")
ic32g_scaler = joblib.load(PROD / "guardian_scaler.pkl")
with open(PROD / "guardian_feature_cols.json") as f:
    ic32g_all = json.load(f)

tb_model = joblib.load(MODEL_DIR / "runs/tb_lgbm_widyawardhana_v3/lgbm.pkl")
with open(MODEL_DIR / "runs/tb_lgbm_widyawardhana_v3/tb_lgbm_widyawardhana_v3_features.json") as f:
    tb_feats = json.load(f)

tbg_model  = joblib.load(MODEL_DIR / "runs/tb_guardian_widyawardhana_v2/guardian.pkl")
tbg_scaler = joblib.load(MODEL_DIR / "runs/tb_guardian_widyawardhana_v2/guardian_scaler.pkl")
with open(MODEL_DIR / "runs/tb_guardian_widyawardhana_v2/tb_guardian_widyawardhana_v2_feature_cols.json") as f:
    tbg_all = json.load(f)

lstm_model  = load_lstm(MODEL_DIR / "runs/tb_lstm_v1/lstm.pt", device="cpu")
lstm_scaler = joblib.load(MODEL_DIR / "runs/tb_lstm_v1/lstm_scaler.pkl")
with open(MODEL_DIR / "runs/tb_lstm_v1/tb_lstm_v1_features.json") as f:
    lstm_feats = json.load(f)

print("Models loaded OK")


def gdn_setup(all_feats):
    static = [f for f in all_feats if f not in DYNAMIC_NAMES]
    smap   = {n: i for i, n in enumerate(static)}
    order  = [("static", smap[f]) if f not in DYNAMIC_NAMES else ("dyn", f)
               for f in all_feats]
    return static, order


ic32g_static, ic32g_order = gdn_setup(ic32g_all)
tbg_static,   tbg_order   = gdn_setup(tbg_all)


def gdn_row(j, i, close, atr, direction, max_fav, X_static, feat_order):
    bh  = j - i
    pnl = (close[j] - close[i]) / close[i] * direction
    atr_pct = atr[i] / close[i] if close[i] > 0 else 0.01
    new_max = max(max_fav, pnl)
    dyn = {
        "bars_held_norm"        : bh / MAX_HOLD,
        "current_pnl_pct"       : pnl,
        "current_pnl_atr"       : pnl / atr_pct if atr_pct > 0 else 0.0,
        "max_favorable_pnl_pct" : new_max,
        "drawdown_from_peak_pct": (new_max - pnl) / new_max if new_max > 0.001 else 0.0,
        "direction"             : float(direction),
        "entry_price_ratio"     : close[i] / close[j] if close[j] > 0 else 1.0,
    }
    row = np.zeros(len(feat_order), dtype=np.float64)
    for k, (src, key) in enumerate(feat_order):
        row[k] = X_static[j, key] if src == "static" else dyn.get(key, 0.0)
    return row, new_max


def simulate(yp, close, high, low, atr, X_gdn=None, gdn_order=None, gdn_scl=None, gdn_mdl=None):
    n = len(yp); trades = []; i = 0
    while i < n:
        sig = yp[i]
        if sig == 1:
            i += 1; continue
        direction = 1 if sig == 2 else -1
        entry    = close[i]
        sl_price = entry - direction * SL_MULT * atr[i]
        max_fav  = 0.0
        exit_p, exit_b, outcome = close[min(i + MAX_HOLD, n - 1)], min(i + MAX_HOLD, n - 1), "TIME"
        for j in range(i + 1, min(i + MAX_HOLD + 1, n)):
            if direction == 1 and low[j]  <= sl_price: exit_p, exit_b, outcome = sl_price, j, "SL"; break
            if direction ==-1 and high[j] >= sl_price: exit_p, exit_b, outcome = sl_price, j, "SL"; break
            bh = j - i
            if X_gdn is not None and bh >= GDN_MIN:
                row, max_fav = gdn_row(j, i, close, atr, direction, max_fav, X_gdn, gdn_order)
                scaled = gdn_scl.transform(row.reshape(1, -1))
                prob   = gdn_mdl.predict_proba(scaled)[0]
                ep     = prob[2] if len(prob) > 2 else prob[1]
                if ep >= GDN_THR:
                    exit_p, exit_b, outcome = close[j], j, "GDN"; break
            else:
                max_fav = max(max_fav, (close[j] - entry) / entry * direction)
        ret = (exit_p - entry) / entry * direction
        net = ret * MODAL * LEVERAGE - COST_RT * MODAL * LEVERAGE
        trades.append({
            "net_pnl"   : net,
            "outcome"   : outcome,
            "direction" : "LONG" if direction == 1 else "SHORT",
            "bars_held" : exit_b - i,
        })
        i = exit_b + 1
    return trades


available = [s for s in ALL_COINS
             if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()]

print(f"Running on {len(available)} coins...")
all_trades = {"ic32_gdn": [], "tb_gdn_lstm": []}

for sym in available:
    df = pd.read_parquet(HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet")
    df = ensure_utc_index(df).sort_index()
    mask = df["label"].isin({"SHORT": 0, "FLAT": 1, "LONG": 2})
    df   = df[mask].copy()
    n    = len(df)

    close = df["close"].values.astype(np.float64)
    high  = df["high"].values.astype(np.float64)
    low   = df["low"].values.astype(np.float64)
    atr   = df["atr_14_h1"].values.astype(np.float64)

    # ── ic32 predictions ──────────────────────────────────────────────────────
    X_ic32 = np.zeros((n, len(ic32_feats)))
    for idx, c in enumerate(ic32_feats):
        if c in df.columns:
            X_ic32[:, idx] = df[c].ffill().fillna(0).values
    p_ic32   = ic32_model.predict_proba(X_ic32).astype(np.float32)
    yp_ic32  = np.argmax(p_ic32, axis=1).astype(np.int32)
    yp_ic32[(yp_ic32 == 2) & (p_ic32[:, 2] < LGBM_THRESHOLD_LONG)]  = 1
    yp_ic32[(yp_ic32 == 0) & (p_ic32[:, 0] < LGBM_THRESHOLD_SHORT)] = 1

    X_ic32g = np.zeros((n, len(ic32g_static)))
    for idx, c in enumerate(ic32g_static):
        if c in df.columns:
            X_ic32g[:, idx] = df[c].ffill().fillna(0).values
    all_trades["ic32_gdn"].extend(
        simulate(yp_ic32, close, high, low, atr, X_ic32g, ic32g_order, ic32g_scaler, ic32g_model)
    )

    # ── TB LGBM + LSTM + Guardian ─────────────────────────────────────────────
    rp  = HOLDOUT_DIR / "labeled" / f"{sym}_regime_h1.parquet"
    hmm = np.full(n, 1, np.int32)
    if rp.exists():
        reg = pd.read_parquet(rp)
        if "hmm_regime_enc" in reg.columns:
            hmm = reg["hmm_regime_enc"].reindex(df.index, fill_value=1).values.astype(np.int32)

    X_tb = np.zeros((n, len(tb_feats)))
    for idx, c in enumerate(tb_feats):
        if c in df.columns:
            X_tb[:, idx] = df[c].ffill().fillna(0).values
    p_tb    = tb_model.predict_proba(X_tb).astype(np.float32)
    conf_tb = np.max(p_tb, axis=1)
    yp_tb   = np.argmax(p_tb, axis=1).astype(np.int32)
    for r, th in REGIME_THRESH.items():
        yp_tb[(hmm == r) & (yp_tb != 1) & (conf_tb < th)] = 1

    X_lstm_raw = np.zeros((n, len(lstm_feats)))
    for idx, c in enumerate(lstm_feats):
        if c in df.columns:
            X_lstm_raw[:, idx] = df[c].ffill().fillna(0).values
    X_lstm_sc = lstm_scaler.transform(X_lstm_raw)

    seqs = [X_lstm_sc[k - SEQ_LEN:k] for k in range(SEQ_LEN, n)]
    p_lstm = p_tb.copy()
    if seqs:
        with torch.no_grad():
            logits = lstm_model(torch.FloatTensor(np.array(seqs)))
            p_lstm[SEQ_LEN:] = torch.softmax(logits, dim=-1).numpy()

    p_comb   = (1 - LSTM_ALPHA) * p_tb + LSTM_ALPHA * p_lstm
    conf_c   = np.max(p_comb, axis=1)
    yp_comb  = np.argmax(p_comb, axis=1).astype(np.int32)
    for r, th in REGIME_THRESH.items():
        yp_comb[(hmm == r) & (yp_comb != 1) & (conf_c < th)] = 1

    X_tbg = np.zeros((n, len(tbg_static)))
    for idx, c in enumerate(tbg_static):
        if c in df.columns:
            X_tbg[:, idx] = df[c].ffill().fillna(0).values
    all_trades["tb_gdn_lstm"].extend(
        simulate(yp_comb, close, high, low, atr, X_tbg, tbg_order, tbg_scaler, tbg_model)
    )

    print(f"  [{sym}] ic32_gdn={len(all_trades['ic32_gdn'])} tb_gdn_lstm={len(all_trades['tb_gdn_lstm'])}")


def scorecard(trades, name):
    pnls = np.array([t["net_pnl"] for t in trades])
    wins = pnls > 0
    losses = pnls <= 0
    n = len(trades)
    sl_hits    = sum(1 for t in trades if t["outcome"] == "SL")
    gdn_exits  = sum(1 for t in trades if t["outcome"] == "GDN")
    longs      = sum(1 for t in trades if t["direction"] == "LONG")
    long_wins  = sum(1 for t in trades if t["direction"] == "LONG" and t["net_pnl"] > 0)
    short_wins = sum(1 for t in trades if t["direction"] == "SHORT" and t["net_pnl"] > 0)
    shorts = n - longs
    avg_hold = float(np.mean([t["bars_held"] for t in trades]))

    gross_profit = float(pnls[wins].sum())  if wins.any()   else 0.0
    gross_loss   = float(abs(pnls[losses].sum())) if losses.any() else 1e-9
    pf = gross_profit / gross_loss

    equity      = np.cumsum(pnls)
    running_max = np.maximum.accumulate(equity)
    dd          = running_max - equity
    max_dd      = float(dd.max())
    peak_eq     = float(running_max.max())
    max_dd_pct  = max_dd / peak_eq * 100 if peak_eq > 0 else 0.0

    max_cl = cur = 0
    for p in pnls:
        if p <= 0: cur += 1; max_cl = max(max_cl, cur)
        else: cur = 0

    return {
        "name"            : name,
        "trades"          : n,
        "wr"              : float(wins.mean() * 100),
        "long_wr"         : long_wins / max(longs, 1) * 100,
        "short_wr"        : short_wins / max(shorts, 1) * 100,
        "long_pct"        : longs / n * 100,
        "sl_rate"         : sl_hits / n * 100,
        "gdn_rate"        : gdn_exits / n * 100,
        "avg_hold"        : avg_hold,
        "pnl"             : float(pnls.sum()),
        "ppm"             : float(pnls.sum() / 5),
        "ppt"             : float(pnls.mean()),
        "avg_win"         : float(pnls[wins].mean()) if wins.any() else 0.0,
        "avg_loss"        : float(pnls[losses].mean()) if losses.any() else 0.0,
        "profit_factor"   : round(pf, 3),
        "max_dd"          : max_dd,
        "max_dd_pct"      : max_dd_pct,
        "max_consec_loss" : max_cl,
    }


sc = {k: scorecard(all_trades[k], k) for k in all_trades}

W = 22
print(f"\n{'='*74}")
print(f"  HEAD-TO-HEAD: ic32+Guardian  vs  TB+LSTM+Guardian")
print(f"  Live-like eval | SL=1.5xATR | max_hold=36 bar | NO TP | Nov2025-Apr2026")
print(f"  21 koin | $10/trade | 5x leverage | Entry+exit fee included")
print(f"{'='*74}")
print(f"  {'Metrik':<24} {'ic32+Guardian':>{W}} {'TB+LSTM+Guardian':>{W}}")
print(f"  {'-'*68}")

rows = [
    ("Total Trades",    lambda x: f"{x['trades']:,}"),
    ("Trades/bulan",    lambda x: f"{x['trades']//5:,}"),
    ("Win Rate",        lambda x: f"{x['wr']:.1f}%"),
    ("  LONG WR",       lambda x: f"{x['long_wr']:.1f}% ({x['long_pct']:.0f}%)"),
    ("  SHORT WR",      lambda x: f"{x['short_wr']:.1f}%"),
    ("SL hit rate",     lambda x: f"{x['sl_rate']:.1f}%"),
    ("Guardian exits",  lambda x: f"{x['gdn_rate']:.1f}%"),
    ("Avg hold (bar)",  lambda x: f"{x['avg_hold']:.1f}"),
    ("Net PnL (5bln)",  lambda x: f"${x['pnl']:+.0f}"),
    ("PnL/bulan",       lambda x: f"${x['ppm']:+.0f}"),
    ("PnL/trade",       lambda x: f"${x['ppt']:+.3f}"),
    ("Avg Win",         lambda x: f"${x['avg_win']:+.3f}"),
    ("Avg Loss",        lambda x: f"${x['avg_loss']:+.3f}"),
    ("Profit Factor",   lambda x: f"{x['profit_factor']:.2f}"),
    ("Max Drawdown",    lambda x: f"${x['max_dd']:.0f} ({x['max_dd_pct']:.1f}%)"),
    ("Max Consec Loss", lambda x: f"{x['max_consec_loss']}"),
]

for label, fn in rows:
    a = fn(sc["ic32_gdn"])
    b = fn(sc["tb_gdn_lstm"])
    row = f"  {label:<24} {a:>{W}} {b:>{W}}"
    if "PnL (5bln)" in label or "Factor" in label:
        row += "  <--"
    print(row)

a, b = sc["ic32_gdn"], sc["tb_gdn_lstm"]
print(f"\n  {'Delta (TB - ic32)':<24}")
print(f"  {'  PnL':<24} {b['pnl']-a['pnl']:>+{W}.0f}")
print(f"  {'  WR':<24} {b['wr']-a['wr']:>+{W}.1f}pp")
print(f"  {'  PF':<24} {b['profit_factor']-a['profit_factor']:>+{W}.2f}")
print(f"  {'  MaxDD':<24} {b['max_dd']-a['max_dd']:>+{W}.0f}")
print(f"  {'  PnL/trade':<24} {b['ppt']-a['ppt']:>+{W}.3f}")
print(f"{'='*74}")

out_path = MODEL_DIR / "runs/tb_lgbm_widyawardhana_v3/comparison_ic32_vs_tb.json"
with open(out_path, "w") as f:
    json.dump({k: sc[k] for k in sc}, f, indent=2, default=float)
print(f"\n  Saved: {out_path}")
