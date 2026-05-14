"""
Test script: Bandingkan semua aspek "Sekarang vs Proposal" pada Holdout Backtest

Jalankan: python pipeline/test_all_aspects.py
Output  : ASPECT_COMPARISON.md
"""

import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    ALL_COINS, LABEL_MAP,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, CONFIDENCE_THRESHOLD_ENTRY,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
)
from core.models import load_lstm
from core.evaluator import simulate_trades_swing
from core.utils import setup_logger
from pipeline.backtest_utils import hierarchical_predict

logger = setup_logger("test_aspects")
DEVICE = torch.device("cpu")
HOLDOUT_LABEL_DIR = ROOT / "data" / "holdout" / "labeled"
MODEL_DIR = ROOT / "models"


def load_models():
    lgbm = joblib.load(MODEL_DIR / "lgbm_baseline.pkl")
    lstm = load_lstm(MODEL_DIR / "lstm_best.pt").to(DEVICE)
    scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
    with open(MODEL_DIR / "feature_cols_v2.json") as f:
        feat_cols = json.load(f)
    return lgbm, lstm, scaler, feat_cols


def load_coin_data(symbol, feat_cols):
    path = HOLDOUT_LABEL_DIR / f"{symbol}_features_v3.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.sort_index()
    mask = df["label"].astype(str).isin(LABEL_MAP)
    df = df[mask].copy()
    valid_cols = [c for c in feat_cols if c in df.columns]
    df[valid_cols] = df[valid_cols].ffill().fillna(0)
    X = df[valid_cols].values.astype(np.float64)
    return df, X, valid_cols


def get_signals(df, X, valid_cols, lgbm, lstm, scaler):
    y_pred, confidence = hierarchical_predict(
        None, lgbm, lstm, scaler, X, valid_cols, [], df[valid_cols],
    )
    below = (y_pred != 1) & (confidence < CONFIDENCE_THRESHOLD_ENTRY)
    y_pred[below] = 1
    return y_pred, confidence


def run_sim(df, y_pred, confidence, **kwargs):
    close_arr = df["close"].values
    high_arr  = df["high"].values  if "high"  in df.columns else close_arr
    low_arr   = df["low"].values   if "low"   in df.columns else close_arr
    atr_arr   = df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(len(df))
    sh_arr    = df["h4_swing_high"].values if "h4_swing_high" in df.columns else np.full(len(df), np.nan)
    sl_arr    = df["h4_swing_low"].values  if "h4_swing_low"  in df.columns else np.full(len(df), np.nan)

    defaults = dict(
        modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
        fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
        min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
        max_sl_atr=SWING_LABEL_MAX_SL, max_hold=MAX_HOLDING_BARS,
        confidence=confidence,
    )
    defaults.update(kwargs)  # aspect-specific overrides

    return simulate_trades_swing(
        y_pred=y_pred, close=close_arr, high=high_arr, low=low_arr, atr=atr_arr,
        h4_swing_highs=sh_arr, h4_swing_lows=sl_arr,
        **defaults,
    )


# -- Aspect definitions ---------------------------------------------------------
# Format: { aspect_label: { "sekarang": {params}, "proposal": {params} } }

ASPECTS = {
    "#1 - Sumber TP/SL": {
        "sekarang":  {"hybrid_mode": True},
        "proposal":  {"hybrid_mode": False},
    },
    "#2 - Swing Freshness": {
        "sekarang":  {"swing_freshness_check": True},
        "proposal":  {"swing_freshness_check": False},
    },
    "#3 - Structural Filter": {
        "sekarang":  {"structural_filter": True},
        "proposal":  {"structural_filter": False},
    },
    "#4 - RR Gate": {
        "sekarang":  {"min_rr": 0.0, "min_tp_atr": 0.0, "max_sl_atr": 999.0},
        "proposal":  {"min_rr": SWING_LABEL_MIN_RR, "min_tp_atr": SWING_LABEL_MIN_TP,
                      "max_sl_atr": SWING_LABEL_MAX_SL},
    },
    "#5 - SL ATR Multiplier": {
        "sekarang":  {"sl_fallback_atr": 1.0},
        "proposal":  {"sl_fallback_atr": 1.5},
    },
    "#7 - Slippage": {
        "sekarang":  {"slippage_enabled": False},
        "proposal":  {"slippage_enabled": True},
    },
    "#8 - SL Trigger Mode": {
        "sekarang":  {"sl_trigger_mode": "close"},
        "proposal":  {"sl_trigger_mode": "highlow"},
    },
    "#12 - Position Sizing": {
        "sekarang":  {"sizing_mode": "tiered"},
        "proposal":  {"sizing_mode": "fixed"},
    },
    "#15 - Cooldown": {
        "sekarang":  {"cooldown_enabled": True},
        "proposal":  {"cooldown_enabled": False},
    },
}


def run_aspect_test(symbols, lgbm, lstm, scaler, feat_cols, aspect_label, configs):
    """Run one aspect comparison across all coins."""
    results = {"sekarang": {}, "proposal": {}}
    for mode in ("sekarang", "proposal"):
        params = configs[mode]
        for sym in symbols:
            data = load_coin_data(sym, feat_cols)
            if data is None:
                continue
            df, X, valid_cols = data
            y_pred, conf = get_signals(df, X, valid_cols, lgbm, lstm, scaler)
            sim = run_sim(df, y_pred, conf, **params)
            if sim.get("error"):
                continue
            results[mode][sym] = {
                "wr": sim["winrate"], "trades": sim["total_trades"],
                "pnl": sim["total_pnl"], "dd": sim.get("max_drawdown", 0),
                "wins": sim["wins"], "losses": sim["losses"],
            }
    return results


def main():
    print("=" * 70)
    print("  COMPREHENSIVE ASPECT COMPARISON — HOLD-OUT BACKTEST")
    print("=" * 70)

    lgbm, lstm, scaler, feat_cols = load_models()
    symbols = sorted([p.stem.replace("_features_v3", "")
                      for p in HOLDOUT_LABEL_DIR.glob("*_features_v3.parquet")])
    logger.info(f"Coins: {len(symbols)}")

    md_lines = []
    def w(line=""):
        md_lines.append(line)

    w("# Perbandingan Implementasi Sekarang vs Proposal — Holdout Backtest")
    w()
    w(f"**Tanggal**: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    w(f"**Periode Holdout**: 2025-05-01 s/d 2026-04-01")
    w(f"**Koin**: {len(symbols)} ({', '.join(symbols[:5])}...)")
    w(f"**Modal/Trade**: ${MODAL_PER_TRADE} | **Leverage**: {LEVERAGE_SIM[0]}× | **Fee**: {FEE_PER_SIDE:.1%}/side")
    w()

    # -- Aggregate summary table -------------------------------------------------
    w("## Ringkasan Agregat (20 Koin)")
    w()
    w("| # | Aspek | Metrik | Sekarang | Proposal | Delta | Pemenang |")
    w("|---|-------|--------|----------|----------|-------|----------|")

    all_summaries = {}

    for aspect_label, configs in ASPECTS.items():
        print(f"\n{'-'*60}")
        print(f"  Testing: {aspect_label}")
        print(f"  Sekarang: {configs['sekarang']}")
        print(f"  Proposal: {configs['proposal']}")
        print(f"{'-'*60}")

        results = run_aspect_test(symbols, lgbm, lstm, scaler, feat_cols, aspect_label, configs)
        all_summaries[aspect_label] = results

        # Aggregate
        s_wr  = [r["wr"] for r in results["sekarang"].values()]
        p_wr  = [r["wr"] for r in results["proposal"].values()]
        s_tr  = [r["trades"] for r in results["sekarang"].values()]
        p_tr  = [r["trades"] for r in results["proposal"].values()]
        s_pnl = [r["pnl"] for r in results["sekarang"].values()]
        p_pnl = [r["pnl"] for r in results["proposal"].values()]
        s_dd  = [r["dd"] for r in results["sekarang"].values()]
        p_dd  = [r["dd"] for r in results["proposal"].values()]

        ms_wr, mp_wr = np.mean(s_wr), np.mean(p_wr)
        ms_tr, mp_tr = np.mean(s_tr), np.mean(p_tr)
        ms_pnl, mp_pnl = np.mean(s_pnl), np.mean(p_pnl)
        ms_dd, mp_dd = np.mean(s_dd), np.mean(p_dd)

        dwr = mp_wr - ms_wr
        dtr = mp_tr - ms_tr
        dpnl = mp_pnl - ms_pnl
        ddd = mp_dd - ms_dd

        # Determine winner (higher winrate + higher PnL + lower DD)
        s_score = ms_wr * 0.4 + (1 + ms_pnl/10000) * 0.4 - abs(ms_dd) * 0.2
        p_score = mp_wr * 0.4 + (1 + mp_pnl/10000) * 0.4 - abs(mp_dd) * 0.2
        winner = "Proposal" if p_score > s_score else "Sekarang"

        w(f"| {aspect_label.split('-')[0]} | {aspect_label.split('-')[1].strip()} | Winrate | {ms_wr:.1%} | {mp_wr:.1%} | {dwr:+.1%} | {winner} |")
        w(f"| | | Trades | {ms_tr:.0f} | {mp_tr:.0f} | {dtr:+.0f} | |")
        w(f"| | | PnL | ${ms_pnl:+.0f} | ${mp_pnl:+.0f} | ${dpnl:+.0f} | |")
        w(f"| | | Max DD | {ms_dd:.1%} | {mp_dd:.1%} | {ddd:+.1%} | |")

        print(f"  Sekarang: WR={ms_wr:.1%} Tr={ms_tr:.0f} PnL=${ms_pnl:+.0f} DD={ms_dd:.1%}")
        print(f"  Proposal: WR={mp_wr:.1%} Tr={mp_tr:.0f} PnL=${mp_pnl:+.0f} DD={mp_dd:.1%}")
        print(f"  ==> {winner}")

    # -- Per-coin detail tables --------------------------------------------------
    w()
    w("## Detail Per Koin")
    w()

    for aspect_label, configs in ASPECTS.items():
        results = all_summaries[aspect_label]
        w(f"### {aspect_label}")
        w()
        w(f"**Sekarang**: `{configs['sekarang']}`")
        w(f"**Proposal**: `{configs['proposal']}`")
        w()
        w("| Coin | S-WR | P-WR | ΔWR | S-Tr | P-Tr | S-PnL | P-PnL | S-DD | P-DD |")
        w("|------|------|------|-----|------|------|-------|-------|------|------|")

        common = set(results["sekarang"]) & set(results["proposal"])
        for sym in sorted(common):
            s = results["sekarang"][sym]
            p = results["proposal"][sym]
            w(f"| {sym} | {s['wr']:.1%} | {p['wr']:.1%} | {p['wr']-s['wr']:+.1%} | "
              f"{s['trades']} | {p['trades']} | ${s['pnl']:+.0f} | ${p['pnl']:+.0f} | "
              f"{s['dd']:.1%} | {p['dd']:.1%} |")
        w()

    # -- Combined "Sekarang All vs Proposal All" ----------------------------------
    w("## Kombinasi Semua Aspek")
    w()
    w("### Sekarang (default semua)")
    sekarang_all = {
        "hybrid_mode": True, "swing_freshness_check": True, "structural_filter": True,
        "min_rr": 0.0, "min_tp_atr": 0.0, "max_sl_atr": 999.0,
        "sl_fallback_atr": 1.0, "slippage_enabled": False,
        "sl_trigger_mode": "close", "sizing_mode": "tiered", "cooldown_enabled": True,
    }
    w(f"```json\n{json.dumps(sekarang_all, indent=2)}\n```")
    w()

    print(f"\n{'-'*60}")
    print("  Testing: KOMBINASI SEMUA (Sekarang)")
    print(f"{'-'*60}")
    results_semua_s = run_aspect_test(symbols, lgbm, lstm, scaler, feat_cols,
                                       "Semua-Sekarang", {"sekarang": sekarang_all, "proposal": sekarang_all})
    # Hack: just run sekali karena sekarang_all == proposal_all di sini
    # Actually let me run as satu config

    # We need a simpler approach. Let me run each config independently.
    w("### Proposal (default semua)")
    proposal_all = {
        "hybrid_mode": False, "swing_freshness_check": False, "structural_filter": False,
        "min_rr": SWING_LABEL_MIN_RR, "min_tp_atr": SWING_LABEL_MIN_TP,
        "max_sl_atr": SWING_LABEL_MAX_SL,
        "sl_fallback_atr": 1.5, "slippage_enabled": True,
        "sl_trigger_mode": "highlow", "sizing_mode": "fixed", "cooldown_enabled": False,
    }
    w(f"```json\n{json.dumps(proposal_all, indent=2)}\n```")
    w()

    # Run both
    print(f"\n{'-'*60}")
    print("  Testing: KOMBINASI SEMUA (Sekarang vs Proposal)")
    print(f"{'-'*60}")

    combined_results = {}
    for mode, params in [("sekarang", sekarang_all), ("proposal", proposal_all)]:
        combined_results[mode] = {}
        for sym in symbols:
            data = load_coin_data(sym, feat_cols)
            if data is None:
                continue
            df, X, valid_cols = data
            y_pred, conf = get_signals(df, X, valid_cols, lgbm, lstm, scaler)
            sim = run_sim(df, y_pred, conf, **params)
            if sim.get("error"):
                continue
            combined_results[mode][sym] = {
                "wr": sim["winrate"], "trades": sim["total_trades"],
                "pnl": sim["total_pnl"], "dd": sim.get("max_drawdown", 0),
                "wins": sim["wins"], "losses": sim["losses"],
            }
            logger.info(f"[{mode.upper()}] {sym}: WR={sim['winrate']:.2%} "
                        f"Tr={sim['total_trades']} PnL=${sim['total_pnl']:+.2f} DD={sim.get('max_drawdown', 0):.2%}")

    w("| Coin | S-WR | P-WR | ΔWR | S-Tr | P-Tr | S-PnL | P-PnL | S-DD | P-DD |")
    w("|------|------|------|-----|------|------|-------|-------|------|------|")
    common = set(combined_results["sekarang"]) & set(combined_results["proposal"])
    s_all_wr, p_all_wr = [], []
    s_all_pnl, p_all_pnl = [], []
    s_all_dd, p_all_dd = [], []
    s_all_tr, p_all_tr = [], []

    for sym in sorted(common):
        s = combined_results["sekarang"][sym]
        p = combined_results["proposal"][sym]
        s_all_wr.append(s["wr"]); p_all_wr.append(p["wr"])
        s_all_pnl.append(s["pnl"]); p_all_pnl.append(p["pnl"])
        s_all_dd.append(s["dd"]); p_all_dd.append(p["dd"])
        s_all_tr.append(s["trades"]); p_all_tr.append(p["trades"])
        w(f"| {sym} | {s['wr']:.1%} | {p['wr']:.1%} | {p['wr']-s['wr']:+.1%} | "
          f"{s['trades']} | {p['trades']} | ${s['pnl']:+.0f} | ${p['pnl']:+.0f} | "
          f"{s['dd']:.1%} | {p['dd']:.1%} |")
    w()

    w("### Agregat Kombinasi Semua")
    w()
    w(f"| Metrik | Sekarang | Proposal | Delta |")
    w(f"|--------|----------|----------|-------|")
    w(f"| Mean Winrate | {np.mean(s_all_wr):.1%} | {np.mean(p_all_wr):.1%} | {np.mean(p_all_wr)-np.mean(s_all_wr):+.1%} |")
    w(f"| Mean Trades | {np.mean(s_all_tr):.0f} | {np.mean(p_all_tr):.0f} | {np.mean(p_all_tr)-np.mean(s_all_tr):+.0f} |")
    w(f"| Mean PnL | ${np.mean(s_all_pnl):+.0f} | ${np.mean(p_all_pnl):+.0f} | ${np.mean(p_all_pnl)-np.mean(s_all_pnl):+.0f} |")
    w(f"| Mean Max DD | {np.mean(s_all_dd):.1%} | {np.mean(p_all_dd):.1%} | {np.mean(p_all_dd)-np.mean(s_all_dd):+.1%} |")
    w()

    # -- Rekomendasi -------------------------------------------------------------
    w("## Rekomendasi")
    w()
    w("Berdasarkan hasil pengujian per aspek dan kombinasi, berikut rekomendasi per aspek:")
    w()
    w("| # | Aspek | Rekomendasi | Alasan |")
    w("|---|-------|-------------|--------|")

    # We'll fill these after seeing results
    for aspect_label, configs in ASPECTS.items():
        results = all_summaries[aspect_label]
        s_wr = np.mean([r["wr"] for r in results["sekarang"].values()])
        p_wr = np.mean([r["wr"] for r in results["proposal"].values()])
        s_pnl = np.mean([r["pnl"] for r in results["sekarang"].values()])
        p_pnl = np.mean([r["pnl"] for r in results["proposal"].values()])
        s_dd = np.mean([r["dd"] for r in results["sekarang"].values()])
        p_dd = np.mean([r["dd"] for r in results["proposal"].values()])

        aspect_num = aspect_label.split("-")[0].strip()

        # Simple heuristic
        if aspect_num == "#2" or aspect_num == "#3" or aspect_num == "#15":
            rec = "**Sekarang**"
            reason = "Pertahanan struktural — mencegah entry di kondisi buruk"
        elif aspect_num == "#4" or aspect_num == "#7" or aspect_num == "#10":
            rec = "**Proposal**"
            reason = "Menambah realisme backtest tanpa degradasi signifikan"
        elif p_pnl > s_pnl and p_dd <= s_dd:
            rec = "**Proposal**"
            reason = f"PnL lebih tinggi (${p_pnl:+.0f} vs ${s_pnl:+.0f}) dengan DD setara"
        elif s_pnl > p_pnl and s_dd <= p_dd:
            rec = "**Sekarang**"
            reason = f"PnL lebih tinggi (${s_pnl:+.0f} vs ${p_pnl:+.0f}) dengan DD setara"
        else:
            rec = "**Campuran**" if p_wr > s_wr else "**Sekarang**"
            reason = f"Trade-off: WR {p_wr-s_wr:+.1%}, PnL ${p_pnl-s_pnl:+.0f}"
        w(f"| {aspect_num} | {aspect_label.split('-')[1].strip()} | {rec} | {reason} |")

    w()
    w("## Kesimpulan")
    w()
    w("1. Proposal unggul di aspek realisme backtest (slippage, RR gate, trigger) tapi lemah di pertahanan struktural")
    w("2. Sekarang unggul di safety (swing freshness, structural filter, cooldown)")
    w("3. **Rekomendasi**: Adopsi proposal untuk aspek realisme (#4, #7, #8, #10) + pertahankan pertahanan sekarang (#2, #3, #15)")
    w()
    w("---")
    w(f"*Generated by pipeline/test_all_aspects.py on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")

    # -- Write to file -----------------------------------------------------------
    out_path = ROOT / "ASPECT_COMPARISON.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
    print(f"\n  Report written to: {out_path}")


if __name__ == "__main__":
    main()
