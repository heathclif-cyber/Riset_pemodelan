"""
Fine-grained sweep: cari sweet threshold LGBM x LSTM (Mode C).
Target: WR >= 63%, PF >= 2.3, trades >= 8k, PnL >= $2,500
Baseline: LGBM 0.75/0.75 alone → 8,523 trades, WR 67.4%, PF 2.807, PnL $2,627
"""
import sys, numpy as np, pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    LABEL_DIR, TRAIN_CUTOFF_DATE,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
)
from core.evaluator import simulate_trades_swing
from core.utils import ensure_utc_index

EXCLUDE = {"1000SHIBUSDT", "1000PEPEUSDT"}

lgbm_oof = pd.read_parquet("models/runs/ic32_regime_v2/oof_predictions.parquet")
lgbm_oof = lgbm_oof[lgbm_oof["has_oof"]].copy()
lgbm_oof.index = pd.to_datetime(lgbm_oof.index, utc=True)

lstm_oof = pd.read_parquet("models/runs/ic32_lstm_candidate_v2/oof_lstm_predictions.parquet")
lstm_oof = lstm_oof[lstm_oof["has_oof"]].copy()
lstm_oof.index = pd.to_datetime(lstm_oof.index, utc=True)
lstm_oof = lstm_oof.rename(columns={"p0": "p0_lstm", "p2": "p2_lstm"})


def _sim(y_pred, df):
    n = len(df)
    h4sh = df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(n, np.nan)
    h4sl = df["h4_swing_low"].values  if "h4_swing_low"  in df.columns else np.full(n, np.nan)
    return simulate_trades_swing(
        y_pred=y_pred, close=df["close"].values, high=df["high"].values,
        low=df["low"].values, atr=df["atr_14_h1"].values,
        h4_swing_highs=h4sh, h4_swing_lows=h4sl,
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        max_hold=MAX_HOLDING_BARS, min_rr=SWING_LABEL_MIN_RR,
        min_tp_atr=SWING_LABEL_MIN_TP, max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        guardian_enabled=False,
    )


def _accum(agg, r):
    t2 = r.get("total_trades", 0)
    if t2 == 0: return
    tlog = r.get("trade_log", [])
    lt2 = sum(1 for x in tlog if x.get("direction") == "LONG")
    st2 = sum(1 for x in tlog if x.get("direction") == "SHORT")
    wl  = r.get("win_by_class", {}).get("LONG", 0)
    ws  = r.get("win_by_class", {}).get("SHORT", 0)
    ppl = r.get("pnl_per_trade", [])
    arr = np.array([float(x) for x in ppl]) if isinstance(ppl, (list, np.ndarray)) else np.array([])
    agg["t"]  += t2;  agg["w"]   += r.get("wins", 0)
    agg["pnl"]+= r.get("net_pnl_total", 0.0)
    agg["sw"] += float(arr[arr>0].sum()) if (arr>0).any() else 0.0
    agg["sl"] += float(abs(arr[arr<0].sum())) if (arr<0).any() else 0.0
    agg["lt"] += lt2; agg["lw"]  += round(wl*lt2)
    agg["st"] += st2; agg["sw2"] += round(ws*st2)


def run(tl, ts, mode, lstm_thr=0.35):
    agg = dict(t=0, w=0, pnl=0.0, sw=0.0, sl=0.0, lt=0, lw=0, st=0, sw2=0)
    for sym in sorted(lgbm_oof["coin"].unique()):
        if sym in EXCLUDE: continue
        fp = LABEL_DIR / f"{sym}_features_v3.parquet"
        if not fp.exists(): continue
        df = pd.read_parquet(fp)
        df = ensure_utc_index(df).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]
        sl  = lgbm_oof[lgbm_oof["coin"]==sym][["p0","p2"]]
        df  = df.join(sl, how="inner").dropna(subset=["p0","p2"])
        if len(df) < 30: continue
        sl2 = lstm_oof[lstm_oof["coin"]==sym][["p0_lstm","p2_lstm"]]
        df  = df.join(sl2, how="left")
        n   = len(df)
        p2v = df["p2"].values; p0v = df["p0"].values
        p2l = df["p2_lstm"].values; p0l = df["p0_lstm"].values
        has = df["p0_lstm"].notna().values
        y   = np.full(n, 1, np.int32)
        for i in range(n):
            il = p2v[i] >= tl
            is_ = (p0v[i] >= ts) and not il
            if not (il or is_): continue
            if mode == "A":
                y[i] = 2 if il else 0
            elif mode == "C":
                if has[i]:
                    agree = (p2l[i] >= lstm_thr) if il else (p0l[i] >= lstm_thr)
                    if agree: y[i] = 2 if il else 0
                # else skip (no OOF → skip)
        if (y!=1).sum()==0: continue
        _accum(agg, _sim(y, df))
    return agg


def metrics(a):
    t=a["t"]; w=a["w"]; lt=a["lt"]; st=a["st"]
    wr  = 100*w/t  if t  else 0
    wrl = 100*a["lw"]/lt if lt else 0
    wrs = 100*a["sw2"]/st if st else 0
    pf  = a["sw"]/a["sl"] if a["sl"]>0 else 0
    ppt = a["pnl"]/t if t else 0
    return t, wr, wrl, wrs, pf, a["pnl"], ppt


# Baseline
BL_T, BL_WR, _, _, BL_PF, BL_PNL, BL_PPT = metrics(run(0.75, 0.75, "A"))

print("="*108)
print("  SWEETSPOT SWEEP — LGBM x LSTM Candidate (Mode C)")
print(f"  Baseline LGBM 0.75/0.75: {BL_T:,} trades | WR {BL_WR:.1f}% | PF {BL_PF:.3f} | PnL ${BL_PNL:,.0f} | PPT ${BL_PPT:.4f}")
print("  Target: trades >= 8k, WR >= 63%, PF >= 2.30, PnL >= $2,500")
print("="*108)

HDR = f"  {'Config':<38} {'T':>7}  {'dT':>7}  {'WR%':>5}  {'WRL%':>5}  {'WRS%':>5}  {'PF':>6}  {'PnL':>8}  {'dPnL':>7}  {'PPT':>7}  {'OK?':>4}"
print(HDR)

# LGBM thresholds fine-grained
LGBM_PAIRS = [
    (0.62, 0.57), (0.63, 0.58), (0.64, 0.59),
    (0.65, 0.60), (0.66, 0.61), (0.67, 0.62),
    (0.68, 0.63), (0.69, 0.64), (0.70, 0.65),
    (0.71, 0.66), (0.72, 0.67), (0.73, 0.68),
    (0.74, 0.69), (0.75, 0.70), (0.75, 0.72),
]

for lstm_thr in [0.33, 0.35, 0.38, 0.40]:
    print(f"\n--- Mode C LSTM >= {lstm_thr} ---")
    for tl, ts in LGBM_PAIRS:
        # Also compute LGBM alone for reference
        t, wr, wrl, wrs, pf, pnl, ppt = metrics(run(tl, ts, "C", lstm_thr))
        dt  = f"{(t-BL_T)/BL_T*100:+.1f}%"
        dp  = f"{(pnl-BL_PNL)/abs(BL_PNL)*100:+.0f}%"
        ok  = "OK" if (t >= 8000 and wr >= 63.0 and pf >= 2.30 and pnl >= 2500) else ""
        star= " <<" if ok else ""
        print(f"  C  {tl}/{ts} LSTM>={lstm_thr}              {t:>7,}  {dt:>7}  {wr:>5.1f}  {wrl:>5.1f}  {wrs:>5.1f}  {pf:>6.3f}  {pnl:>8.0f}  {dp:>7}  {ppt:>7.4f}  {ok}{star}")

# Juga tampilkan LGBM alone untuk referensi
print(f"\n--- LGBM ONLY (referensi per threshold) ---")
for tl, ts in LGBM_PAIRS + [(0.75, 0.75)]:
    t, wr, wrl, wrs, pf, pnl, ppt = metrics(run(tl, ts, "A"))
    dt = f"{(t-BL_T)/BL_T*100:+.1f}%"
    dp = f"{(pnl-BL_PNL)/abs(BL_PNL)*100:+.0f}%"
    print(f"  A  {tl}/{ts}                         {t:>7,}  {dt:>7}  {wr:>5.1f}  {wrl:>5.1f}  {wrs:>5.1f}  {pf:>6.3f}  {pnl:>8.0f}  {dp:>7}  {ppt:>7.4f}")
