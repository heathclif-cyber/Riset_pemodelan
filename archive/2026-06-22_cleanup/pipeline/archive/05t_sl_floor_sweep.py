"""
05t — SL% Floor Minimum sweep (genuine OOF).

Motivasi: trade live TRX kena SL di -2.6% (leveraged) padahal harga gerak -0.51%.
ATR koin low-vol terlalu rendah -> band SL (1.5xATR) hanya ~0.46% -> kena noise.
Hipotesis: floor jarak SL minimum (independen ATR) kurangi stop-out prematur.

Genuine: OOF only, fusion config = stack aktif tb_genuine_v2_dynsize_lstm_cond,
HMM Config B frozen, Guardian tb_guardian_genuine_v2_hmm_v2, holdout TIDAK disentuh.
Keputusan floor terbaik HANYA dari metrik OOF (Aturan 1).

Output: models/runs/tb_lgbm_genuine_v2/sl_floor_sweep.json

Usage:
  python pipeline/05t_sl_floor_sweep.py
"""
import json
import sys
import time
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
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL, MAX_HOLDING_BARS,
    GUARDIAN_ACTIVATION_ATR,
)
from core.cascade_utils import apply_conditional_momentum_fusion_pre
from core.evaluator import simulate_trades_swing
from pipeline.lstm_fusion_shared import (
    LGBM_DIR, GUARDIAN_DIR, DYNAMIC_FEATS, DYNSIZE_CFG, FLAT,
    apply_hmm_thr, build_y_pred, compute_dynamic_modal, genuine_audit_block,
    load_guardian_params, load_hmm_cfg, preload_coins,
)

# Stack aktif: tb_genuine_v2_dynsize_lstm_cond (lihat model_registry.json)
ACTIVE_CFG = {
    "fusion": "lstm", "mode": "conditional_momentum",
    "bull_thr": 0.38, "bear_thr": 0.50,
    "boost": 0.10, "opposite_pen": 0.14, "near_miss_gap": 0.03,
    "vol_thr": 2.0, "proportional": True,
    "enable_boost": True, "enable_penalty": True,
}

SL_FLOORS = [0.000, 0.006, 0.008, 0.010, 0.012, 0.015]
# Ambang slice low-vol: ATR% (atr/close) di bar entry di bawah ini = koin/bar low-vol.
LOWVOL_ATR_PCT = 0.004  # 0.4%


def summarize(trades: list, base_modal: float = MODAL_PER_TRADE) -> dict:
    if not trades:
        return {"n": 0, "wr": 0.0, "pnl": 0.0, "ppt": 0.0, "pf": 0.0,
                "sl_pct": 0.0, "ppt_norm": 0.0}
    n = len(trades)
    wins = sum(1 for t in trades if t["net_pnl"] > 0)
    sl_hit = sum(1 for t in trades if t["outcome"] == "LOSS")
    gpnl = sum(t["net_pnl"] for t in trades if t["net_pnl"] > 0)
    lloss = sum(abs(t["net_pnl"]) for t in trades if t["net_pnl"] < 0)
    tpnl = sum(t["net_pnl"] for t in trades)
    pf = gpnl / lloss if lloss > 0 else float("inf")
    modals = [t.get("modal_used", base_modal) for t in trades]
    avg_modal = float(np.mean(modals)) if modals else base_modal
    ppt_norm = (tpnl / n) * (base_modal / avg_modal) if avg_modal > 0 else 0.0
    return {
        "n": n, "wr": round(wins / n * 100, 2), "pnl": round(tpnl, 2),
        "ppt": round(tpnl / n, 4), "pf": round(pf, 3),
        "sl_pct": round(sl_hit / n * 100, 2), "ppt_norm": round(ppt_norm, 4),
    }


def run_floor(coins, hmm_cfg, g_model, g_scaler, g_params, min_sl_pct: float) -> dict:
    common = dict(
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        max_hold=MAX_HOLDING_BARS,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL,
        tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
        min_sl_pct=min_sl_pct,
    )
    all_trades, lowvol_trades = [], []
    for c in coins:
        cfg = ACTIVE_CFG
        y = build_y_pred(c, cfg, hmm_cfg)
        p0, p2 = c["p0"].copy(), c["p2"].copy()
        _, _, tl, ts = apply_hmm_thr(p0, p2, c["hmm"], hmm_cfg)
        p0, p2 = apply_conditional_momentum_fusion_pre(
            p0, p2, c["lstm_p"], tl, ts, c["vol_spike"],
            vol_thr=cfg["vol_thr"], bull_thr=cfg["bull_thr"], bear_thr=cfg["bear_thr"],
            near_miss_gap=cfg["near_miss_gap"], boost=cfg["boost"],
            opposite_pen=cfg["opposite_pen"], enable_boost=cfg["enable_boost"],
            enable_penalty=cfg["enable_penalty"], lstm_valid=c["lstm_valid"],
            proportional=cfg["proportional"],
        )
        _, conf, tl, ts = apply_hmm_thr(p0, p2, c["hmm"], hmm_cfg)
        modal_arr = compute_dynamic_modal(p0, p2, c["hmm"], y, MODAL_PER_TRADE, DYNSIZE_CFG, tl, ts)
        rep = simulate_trades_swing(
            y_pred=y, guardian_enabled=True,
            guardian_model=g_model, guardian_scaler=g_scaler, X_guardian=c["X_grd"],
            guardian_exit_threshold=g_params["exit_threshold"],
            guardian_min_hold_bars=g_params["min_hold_bars"],
            guardian_activation_atr=GUARDIAN_ACTIVATION_ATR,
            modal_arr=modal_arr,
            close=c["close"], high=c["high"], low=c["low"], atr=c["atr"],
            h4_swing_highs=c["h4_sh"], h4_swing_lows=c["h4_sl"],
            **common,
        )
        close, atr = c["close"], c["atr"]
        for t in rep.get("trades", []):
            bi = t.get("bar_in", 0)
            atr_pct = (atr[bi] / close[bi]) if (bi < len(close) and close[bi] > 0) else np.nan
            all_trades.append(t)
            if not np.isnan(atr_pct) and atr_pct < LOWVOL_ATR_PCT:
                lowvol_trades.append(t)
    return {
        "min_sl_pct": min_sl_pct,
        "portfolio": summarize(all_trades),
        "lowvol": summarize(lowvol_trades),
    }


def main():
    SEP = "=" * 84
    t0 = time.time()
    hmm_cfg = load_hmm_cfg()
    g_params = load_guardian_params()

    print(f"\n{SEP}")
    print("  05t: SL% Floor Minimum sweep (genuine OOF)")
    print(f"  Stack: tb_genuine_v2_dynsize_lstm_cond | floors={SL_FLOORS}")
    print(f"  Low-vol slice: ATR% < {LOWVOL_ATR_PCT*100:.1f}% di bar entry")
    print(SEP)

    lgbm_oof = pd.read_parquet(LGBM_DIR / "oof_predictions.parquet")
    lstm_oof = pd.read_parquet(ROOT / "models" / "runs" / "tb_lstm_genuine_v2" / "oof_lstm_predictions.parquet")

    g_model = joblib.load(GUARDIAN_DIR / "guardian.pkl")
    g_scaler = joblib.load(GUARDIAN_DIR / "guardian_scaler.pkl")
    with open(GUARDIAN_DIR / "guardian_features.json", encoding="utf-8") as f:
        g_static = [feat for feat in json.load(f) if feat not in DYNAMIC_FEATS]
    coins = preload_coins(lgbm_oof, lstm_oof, g_static)
    print(f"  Coins: {len(coins)}")

    results = []
    baseline = None
    print(f"\n  {'floor':>7} | {'N':>6} {'WR%':>6} {'PF':>6} {'PnL':>9} {'pptN':>8} {'SL%':>6} "
          f"|| LOWVOL {'N':>5} {'WR%':>6} {'PF':>6} {'SL%':>6}")
    for floor in SL_FLOORS:
        r = run_floor(coins, hmm_cfg, g_model, g_scaler, g_params, floor)
        if floor == 0.0:
            baseline = r
        p, lv = r["portfolio"], r["lowvol"]
        if baseline is not None:
            r["delta_pnl"] = round(p["pnl"] - baseline["portfolio"]["pnl"], 2)
            r["delta_pptnorm"] = round(p["ppt_norm"] - baseline["portfolio"]["ppt_norm"], 4)
            r["delta_wr"] = round(p["wr"] - baseline["portfolio"]["wr"], 2)
            r["delta_pf"] = round(p["pf"] - baseline["portfolio"]["pf"], 3)
        results.append(r)
        print(f"  {floor:>7.3f} | {p['n']:>6,} {p['wr']:>6.1f} {p['pf']:>6.2f} {p['pnl']:>9.2f} "
              f"{p['ppt_norm']:>8.4f} {p['sl_pct']:>6.1f} || {'':6}{lv['n']:>5,} "
              f"{lv['wr']:>6.1f} {lv['pf']:>6.2f} {lv['sl_pct']:>6.1f}")

    # Decision: floor terbaik = WR>=base AND PF>=base AND ppt_norm>=base, pilih PnL tertinggi
    b = baseline["portfolio"]
    qualified = [
        r for r in results if r["min_sl_pct"] > 0.0
        and r["portfolio"]["wr"] >= b["wr"]
        and r["portfolio"]["pf"] >= b["pf"]
        and r["portfolio"]["ppt_norm"] >= b["ppt_norm"]
    ]
    best = max(qualified, key=lambda r: r["portfolio"]["pnl"]) if qualified else None

    print(f"\n{SEP}")
    if best:
        print(f"  WINNER: min_sl_pct={best['min_sl_pct']:.3f} | "
              f"PnL {best['portfolio']['pnl']:+.2f} (d{best['delta_pnl']:+.2f}) "
              f"WR {best['portfolio']['wr']:.1f}% PF {best['portfolio']['pf']:.2f}")
        decision = "PROMOTE_CANDIDATE"
    else:
        print("  NO floor lulus kriteria (WR & PF & ppt_norm >= baseline). Tetap baseline.")
        decision = "NO_PROMOTE"
    print(SEP)

    elapsed = time.time() - t0
    audit = genuine_audit_block()
    out = {
        **audit,
        "eval": "sl_floor_sweep",
        "created": datetime.now().isoformat(),
        "elapsed_sec": round(elapsed, 1),
        "active_fusion_cfg": ACTIVE_CFG,
        "sl_floors": SL_FLOORS,
        "lowvol_atr_pct_threshold": LOWVOL_ATR_PCT,
        "baseline": baseline,
        "results": results,
        "decision": decision,
        "best": best,
    }
    out_path = LGBM_DIR / "sl_floor_sweep.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"  Decision: {decision} | Elapsed: {elapsed/60:.1f} min")
    print(f"  Saved: {out_path}\n")


if __name__ == "__main__":
    main()
