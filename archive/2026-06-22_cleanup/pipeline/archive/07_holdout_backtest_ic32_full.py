"""
pipeline/07_holdout_backtest_ic32_full.py
Holdout backtest ic32_regime_v1 dengan full production stack:
  LGBM (ic32_regime_v1) + LSTM (lstm_best.pt) hard_consensus + Guardian clean_v2

Model diambil dari D:/Apps-Dev/swint_tradev2/models — READ ONLY.
Tidak ada modifikasi pada model atau parameter produksi.
"""
import json, sys, warnings, numpy as np, pandas as pd, torch
from pathlib import Path
warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import joblib
from core.models import TradingLSTM
from core.evaluator import full_trading_report
from core.utils import setup_logger, ensure_utc_index
from config import *

logger = setup_logger("07_ic32_full")

PROD = Path("D:/Apps-Dev/swint_tradev2/models")

# ── Cascade params (dari inference_config.json) ────────────────────────────────
AGREE_BOOST       = 0.05
NEUTRAL_PEN       = 0.0
OPPOSITE_PEN      = 0.65
NO_VETO_THR       = 0.50
LGBM_THR_LONG     = LGBM_THRESHOLD_LONG    # 0.69
LGBM_THR_SHORT    = LGBM_THRESHOLD_SHORT   # 0.59
DIR_REVIEW_THR    = 0.35
LSTM_OVERRIDE_THR = 0.70
SEQ_LEN           = 32

LM = {"SHORT": 0, "FLAT": 1, "LONG": 2}

# ── Load all models (READ ONLY from production) ────────────────────────────────
logger.info("Loading ic32 LGBM model...")
ic32_model = joblib.load(MODEL_DIR / "runs" / "ic32_regime_v1" / "lgbm.pkl")
ic32_feats = list(ic32_model.feature_name_)

logger.info("Loading Guardian clean_v2...")
gd_model  = joblib.load(PROD / "guardian_best.pkl")
gd_scaler = joblib.load(PROD / "guardian_scaler.pkl")
with open(PROD / "guardian_feature_cols.json") as f:
    gd_all_feats = json.load(f)
DYNAMIC_FEATS = {"bars_held_norm", "current_pnl_pct", "current_pnl_atr",
                 "max_favorable_pnl_pct", "drawdown_from_peak_pct",
                 "direction", "entry_price_ratio"}
static_feats = [f for f in gd_all_feats if f not in DYNAMIC_FEATS]

logger.info("Loading LSTM model...")
lstm_scaler = joblib.load(PROD / "lstm_scaler.pkl")
with open(PROD / "feature_cols_lstm_temporal.json") as f:
    lstm_feats = json.load(f)

state = torch.load(PROD / "lstm_best.pt", map_location="cpu", weights_only=False)
W_ih = state["lstm.cells.0.W_ih"]
hidden  = W_ih.shape[0] // 4
n_inp   = W_ih.shape[1]
n_layers = sum(1 for k in state if k.startswith("lstm.cells") and k.endswith("W_ih"))
lstm_model = TradingLSTM(n_features=n_inp, hidden_size=hidden,
                          num_layers=n_layers, num_classes=3, dropout=0.0)
lstm_model.load_state_dict(state)
lstm_model.eval()
logger.info(f"LSTM loaded: input={n_inp}, hidden={hidden}, layers={n_layers}, seq={SEQ_LEN}")


def apply_hard_consensus(lgbm_proba: np.ndarray,
                          lstm_proba_list: list) -> tuple:
    """
    Apply hard_consensus cascade for each bar.
    Returns (yp array int32, conf array float64).
    lstm_proba_list[i] = None if not enough history (first SEQ_LEN-1 bars).
    """
    n  = len(lgbm_proba)
    yp   = np.ones(n, dtype=np.int32)
    conf = np.zeros(n, dtype=np.float64)

    for i in range(n):
        lp   = lgbm_proba[i].copy().astype(np.float64)
        lstm = lstm_proba_list[i]  # None or ndarray(3,)

        # Stage 1: LGBM threshold gate
        if lp[2] >= LGBM_THR_LONG:
            sig = 2
        elif lp[0] >= LGBM_THR_SHORT:
            sig = 0
        else:
            # Flat review: only if LSTM available and LGBM was close-ish
            best_dir = max(lp[2], lp[0])
            if lstm is not None and best_dir >= DIR_REVIEW_THR:
                li = int(np.argmax(lstm))
                lc = float(lstm[li])
                if li != 1 and lc >= LSTM_OVERRIDE_THR:
                    sig = li
                    lp  = lstm.copy().astype(np.float64)
                else:
                    continue   # FLAT
            else:
                continue       # FLAT

        # Stage 2: LSTM confidence adjustment
        if lstm is None:
            yp[i]   = sig
            conf[i] = float(lp[sig])
            continue

        li = int(np.argmax(lstm))
        proba = lp.copy()

        if li == sig:
            adj = AGREE_BOOST
        elif li == 1:
            adj = -NEUTRAL_PEN
        else:
            # Opposite — veto logic
            if proba[sig] > NO_VETO_THR:
                # Soft penalty: don't flip direction
                other_idxs = [j for j in range(3) if j != sig]
                o  = max(proba[j] for j in other_idxs)
                f_ = sum(proba[j] for j in other_idxs) - o
                tot = o + f_
                if tot > 0 and o < proba[sig]:
                    msp = (proba[sig] - o) * tot / (2 * o + f_)
                    msp = max(0.0, msp - 0.01)
                else:
                    msp = 0.0
                adj = -min(OPPOSITE_PEN, msp)
            else:
                adj = -OPPOSITE_PEN

        new_conf = float(np.clip(proba[sig] + adj, 0.0, 1.0))
        thr = LGBM_THR_LONG if sig == 2 else LGBM_THR_SHORT
        if new_conf < thr:
            continue   # filtered out after LSTM penalty

        yp[i]   = sig
        conf[i] = new_conf

    return yp, conf


available = [s for s in ALL_COINS
             if (HOLDOUT_DIR / "labeled" / f"{s}_features_v3.parquet").exists()]

print(f"\n{'='*72}")
print(f"  HOLDOUT: ic32_regime_v1 — Full Production Stack")
print(f"  LGBM (ic32) + LSTM hard_consensus (seq={SEQ_LEN}) + Guardian clean_v2")
print(f"  Coins: {len(available)} | Period: Nov 2025 – Apr 2026 | $10/trade 5x")
print(f"{'='*72}\n")

# ── Aggregators (4 variants) ──────────────────────────────────────────────────
def new_agg():
    return {"trades": 0, "wins": 0, "pnl": 0.0,
            "longs": 0, "long_wins": 0, "sl_hits": 0, "gd_exits": 0}

agg_bare      = new_agg()   # ic32 LGBM only (no LSTM, no Guardian)
agg_gd_only   = new_agg()   # ic32 LGBM + Guardian (no LSTM)
agg_lstm      = new_agg()   # ic32 LGBM + LSTM (no Guardian)
agg_full      = new_agg()   # ic32 LGBM + LSTM + Guardian (full stack)

for sym in available:
    df = pd.read_parquet(HOLDOUT_DIR / "labeled" / f"{sym}_features_v3.parquet")
    df = ensure_utc_index(df).sort_index()

    # HMM regime (33rd feature of ic32)
    rp = HOLDOUT_DIR / "labeled" / f"{sym}_regime_h1.parquet"
    hmm = np.full(len(df), 1, np.int32)
    if rp.exists():
        reg = pd.read_parquet(rp)
        if "hmm_regime_enc" in reg.columns:
            hmm = reg["hmm_regime_enc"].reindex(df.index, fill_value=1).values.astype(np.int32)

    mask = df["label"].isin(LM)
    df   = df[mask].copy()
    hmm  = hmm[mask.values]
    n    = len(df)

    close = df["close"].values; high = df["high"].values; low = df["low"].values
    atr   = df["atr_14_h1"].values
    h4_sh = df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan)
    h4_sl = df["h4_swing_low"].values  if "h4_swing_low"  in df.columns else np.full(n, np.nan)
    h4_tr = df["h4_trend"].values      if "h4_trend"      in df.columns else None
    yt    = df["label"].map(LM).values.astype(np.int32)

    # ── LGBM predictions ───────────────────────────────────────────────────
    X_ic = np.zeros((n, len(ic32_feats)), dtype=np.float64)
    for idx, c in enumerate(ic32_feats):
        if c in df.columns:
            X_ic[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)
        elif c == "hmm_regime_enc":
            X_ic[:, idx] = hmm.astype(np.float64)

    lgbm_proba = ic32_model.predict_proba(X_ic)   # (n, 3) — [SHORT, FLAT, LONG]

    # Bare ic32 thresholds (no LSTM)
    yp_bare  = np.ones(n, dtype=np.int32)
    yp_bare[lgbm_proba[:, 2] >= LGBM_THR_LONG]  = 2
    short_m = (lgbm_proba[:, 0] >= LGBM_THR_SHORT) & (yp_bare != 2)
    yp_bare[short_m] = 0
    conf_bare = np.max(lgbm_proba, axis=1)

    # ── LSTM rolling window predictions ───────────────────────────────────
    X_lstm = np.zeros((n, len(lstm_feats)), dtype=np.float32)
    for idx, c in enumerate(lstm_feats):
        if c in df.columns:
            X_lstm[:, idx] = df[c].ffill().fillna(0).values.astype(np.float32)

    X_lstm_sc = lstm_scaler.transform(X_lstm)
    lstm_proba_list = [None] * n
    with torch.no_grad():
        for i in range(SEQ_LEN - 1, n):
            seq = torch.tensor(
                X_lstm_sc[i - SEQ_LEN + 1 : i + 1], dtype=torch.float32
            ).unsqueeze(0)
            out  = lstm_model(seq)
            prob = torch.softmax(out, dim=-1).squeeze(0).numpy()
            lstm_proba_list[i] = prob

    # ── Apply hard_consensus ───────────────────────────────────────────────
    yp_lstm, conf_lstm = apply_hard_consensus(lgbm_proba, lstm_proba_list)

    # ── Guardian static feature matrix ────────────────────────────────────
    X_gd = np.zeros((n, len(static_feats)), dtype=np.float64)
    for idx, c in enumerate(static_feats):
        if c in df.columns:
            X_gd[:, idx] = df[c].ffill().fillna(0).values.astype(np.float64)
        elif c == "hmm_regime_enc":
            X_gd[:, idx] = hmm.astype(np.float64)

    # ── Run 4 variants ─────────────────────────────────────────────────────
    base_kwargs = dict(
        y_actual=yt, atr=atr, close=close, high=high, low=low,
        h4_swing_highs=h4_sh, h4_swing_lows=h4_sl,
        index=df.index, modal=MODAL_PER_TRADE, leverages=LEVERAGE_SIM,
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        max_hold=MAX_HOLDING_BARS, symbol=sym,
        trailing_stop_enabled=False, h4_trend=h4_tr,
    )

    rep_bare = full_trading_report(
        y_pred=yp_bare, confidence=conf_bare,
        guardian_enabled=False,
        **base_kwargs,
    )

    rep_gd_only = full_trading_report(
        y_pred=yp_bare, confidence=conf_bare,
        guardian_model=gd_model, guardian_scaler=gd_scaler,
        X_guardian=X_gd, guardian_exit_threshold=0.65,
        guardian_min_hold_bars=2, guardian_enabled=True,
        **base_kwargs,
    )

    rep_lstm = full_trading_report(
        y_pred=yp_lstm, confidence=conf_lstm,
        guardian_enabled=False,
        **base_kwargs,
    )

    rep_full = full_trading_report(
        y_pred=yp_lstm, confidence=conf_lstm,
        guardian_model=gd_model, guardian_scaler=gd_scaler,
        X_guardian=X_gd, guardian_exit_threshold=0.65,
        guardian_min_hold_bars=2, guardian_enabled=True,
        **base_kwargs,
    )

    def accumulate(agg, rep):
        lev = rep.get("lev5x", rep)
        tr  = lev.get("trades", [])
        agg["trades"]    += len(tr)
        agg["wins"]      += sum(1 for t in tr if t.get("net_pnl", 0) > 0)
        agg["pnl"]       += sum(t.get("net_pnl", 0) for t in tr)
        agg["longs"]     += sum(1 for t in tr if t.get("direction") == "LONG")
        agg["long_wins"] += sum(1 for t in tr if t.get("direction") == "LONG"
                                and t.get("net_pnl", 0) > 0)
        agg["sl_hits"]   += sum(1 for t in tr if "SL" in str(t.get("outcome", ""))
                                or t.get("exit_reason") == "sl")
        agg["gd_exits"]  += sum(1 for t in tr if "GUARDIAN" in str(t.get("outcome", "")))

    accumulate(agg_bare,    rep_bare)
    accumulate(agg_gd_only, rep_gd_only)
    accumulate(agg_lstm,    rep_lstm)
    accumulate(agg_full,    rep_full)

    logger.info(f"[{sym}] bare={agg_bare['trades']} gd_only={agg_gd_only['trades']} "
                f"lstm={agg_lstm['trades']} full={agg_full['trades']}")


def scorecard(name, agg):
    t  = agg["trades"]; w = agg["wins"]; p = agg["pnl"]
    lw = agg["long_wins"]; l = agg["longs"]
    sl = agg["sl_hits"]; gd = agg["gd_exits"]
    wr   = w  / max(t, 1) * 100
    lwr  = lw / max(l, 1) * 100
    swr  = (w - lw) / max(t - l, 1) * 100
    return {
        "name": name, "trades": t,
        "wr": round(wr, 2), "long_wr": round(lwr, 2), "short_wr": round(swr, 2),
        "long_pct": round(l / max(t, 1) * 100, 2),
        "pnl": round(p, 2),
        "pnl_per_trade": round(p / max(t, 1), 3),
        "pnl_per_month": round(p / 5, 2),
        "sl_rate": round(sl / max(t, 1) * 100, 2),
        "gd_exit_rate": round(gd / max(t, 1) * 100, 2),
    }

sc_bare    = scorecard("ic32 LGBM only",           agg_bare)
sc_gd_only = scorecard("ic32 + Guardian",           agg_gd_only)
sc_lstm    = scorecard("ic32 + LSTM",               agg_lstm)
sc_full    = scorecard("ic32 + LSTM + Guardian",    agg_full)

# ── Print scorecard ────────────────────────────────────────────────────────────
w1, w2, w3, w4 = 19, 17, 17, 20
print(f"\n{'='*76}")
print(f"  SCORECARD — ic32_regime_v1 Full Production Stack")
print(f"  Holdout: Nov 2025 – Apr 2026 | {len(available)} coins | $10/trade 5x")
print(f"{'='*76}")
hdr = (f"  {'Metrik':<26} {'LGBM only':>{w1}} {'+ Guardian':>{w2}} "
       f"{'+ LSTM':>{w3}} {'LSTM+Guardian':>{w4}}")
print(hdr)
print("-" * 76)

rows = [
    ("Total Trades",
     f"{sc_bare['trades']:,}",
     f"{sc_gd_only['trades']:,}",
     f"{sc_lstm['trades']:,}",
     f"{sc_full['trades']:,}"),
    ("Trades/bulan",
     f"{sc_bare['trades']/5:.0f}",
     f"{sc_gd_only['trades']/5:.0f}",
     f"{sc_lstm['trades']/5:.0f}",
     f"{sc_full['trades']/5:.0f}"),
    ("Win Rate",
     f"{sc_bare['wr']:.1f}%",
     f"{sc_gd_only['wr']:.1f}%",
     f"{sc_lstm['wr']:.1f}%",
     f"{sc_full['wr']:.1f}%"),
    ("  LONG WR",
     f"{sc_bare['long_wr']:.1f}%",
     f"{sc_gd_only['long_wr']:.1f}%",
     f"{sc_lstm['long_wr']:.1f}%",
     f"{sc_full['long_wr']:.1f}%"),
    ("  SHORT WR",
     f"{sc_bare['short_wr']:.1f}%",
     f"{sc_gd_only['short_wr']:.1f}%",
     f"{sc_lstm['short_wr']:.1f}%",
     f"{sc_full['short_wr']:.1f}%"),
    ("  LONG share",
     f"{sc_bare['long_pct']:.1f}%",
     f"{sc_gd_only['long_pct']:.1f}%",
     f"{sc_lstm['long_pct']:.1f}%",
     f"{sc_full['long_pct']:.1f}%"),
    ("Net PnL (5 bln)",
     f"${sc_bare['pnl']:+.0f}",
     f"${sc_gd_only['pnl']:+.0f}",
     f"${sc_lstm['pnl']:+.0f}",
     f"${sc_full['pnl']:+.0f}"),
    ("PnL/bulan",
     f"${sc_bare['pnl_per_month']:+.0f}",
     f"${sc_gd_only['pnl_per_month']:+.0f}",
     f"${sc_lstm['pnl_per_month']:+.0f}",
     f"${sc_full['pnl_per_month']:+.0f}"),
    ("PnL/trade",
     f"${sc_bare['pnl_per_trade']:+.3f}",
     f"${sc_gd_only['pnl_per_trade']:+.3f}",
     f"${sc_lstm['pnl_per_trade']:+.3f}",
     f"${sc_full['pnl_per_trade']:+.3f}"),
    ("SL Hit Rate",
     f"{sc_bare['sl_rate']:.1f}%",
     f"{sc_gd_only['sl_rate']:.1f}%",
     f"{sc_lstm['sl_rate']:.1f}%",
     f"{sc_full['sl_rate']:.1f}%"),
    ("Guardian exits",
     "—",
     f"{sc_gd_only['gd_exit_rate']:.1f}%",
     "—",
     f"{sc_full['gd_exit_rate']:.1f}%"),
]
for r in rows:
    label = r[0]
    vals  = r[1:]
    marker = " <--" if label == "Net PnL (5 bln)" else ""
    print(f"  {label:<26} {vals[0]:>{w1}} {vals[1]:>{w2}} "
          f"{vals[2]:>{w3}} {vals[3]:>{w4}}{marker}")

best_pnl  = max(sc_bare["pnl"], sc_gd_only["pnl"], sc_lstm["pnl"], sc_full["pnl"])
best_name = ["LGBM only", "+ Guardian", "+ LSTM", "LSTM+Guardian"][
    [sc_bare["pnl"], sc_gd_only["pnl"], sc_lstm["pnl"], sc_full["pnl"]].index(best_pnl)]
print(f"\n  Best combination: {best_name}  (PnL ${best_pnl:+.0f})")
print(f"\n  Note: Swing-based TP/SL. Production scorecard (WR 67.5%, $848) uses")
print(f"  different sim environment (live exec, partial exits, continuous stream).")

out = {
    "ic32_lgbm_only":         sc_bare,
    "ic32_plus_guardian":     sc_gd_only,
    "ic32_plus_lstm":         sc_lstm,
    "ic32_plus_lstm_guardian":sc_full,
    "best": best_name,
    "note": "swing-based TP/SL, full_trading_report(), 21 coins Nov2025-Apr2026, READ-ONLY models from swint_tradev2",
}
out_path = MODEL_DIR / "runs" / "ic32_regime_v1" / "holdout_full_stack.json"
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\n  Saved -> {out_path}")
print(f"{'='*76}")
