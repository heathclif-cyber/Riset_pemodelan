"""
pipeline/10_visualize.py — Fase 10: Visualisasi Backtest & Swing Verification

Jalankan:
python pipeline/10_visualize.py                         # ETHUSDT, data training
python pipeline/10_visualize.py --symbol SOLUSDT       # koin lain
python pipeline/10_visualize.py --holdout              # data hold-out
python pipeline/10_visualize.py --all                  # semua koin
python pipeline/10_visualize.py --verify-swing         # verifikasi swing levels

Output:
reports/{SYMBOL}_backtest_visual.png
reports/{SYMBOL}_swing_verify.png
"""

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    ALL_COINS,
    LABEL_MAP,
    LABEL_MAP_INV,
    NUM_CLASSES,
    LABEL_DIR,
    MODEL_DIR,
    REPORT_DIR,
    SWING_LABEL_MIN_RR,
    SWING_LABEL_MIN_TP,
    SWING_LABEL_MAX_SL,
    MAX_HOLDING_BARS,
    CONFIDENCE_THRESHOLD_ENTRY,
    MODAL_PER_TRADE,
    LEVERAGE_SIM,
    FEE_PER_SIDE,
)
from core.models import load_lstm, ProbabilityCalibrator
from core.evaluator import simulate_trades_swing
from core.utils import setup_logger, ensure_utc_index
from pipeline.p05_utils import SequenceDataset

logger = setup_logger("10_visualize")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NON_FEATURE_COLS = {"label", "h4_swing_high", "h4_swing_low"}
HOLDOUT_LABEL_DIR = ROOT / "data" / "holdout" / "labeled"


# ═══════════════════════════════════════════════════════════════════════════════
# LOAD MODELS
# ═══════════════════════════════════════════════════════════════════════════════
def load_models():
    lgbm_model   = joblib.load(MODEL_DIR / "lgbm_baseline.pkl")
    lstm_model   = load_lstm(MODEL_DIR / "lstm_best.pt", device=str(DEVICE)).to(DEVICE)
    lstm_scaler  = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
    meta_learner = joblib.load(MODEL_DIR / "ensemble_meta.pkl")
    calibrator   = ProbabilityCalibrator.load(MODEL_DIR / "calibrator.pkl")

    with open(MODEL_DIR / "feature_cols_v2.json") as f:
        feat_cols = json.load(f)
    with open(MODEL_DIR / "inference_config.json") as f:
        cfg = json.load(f)

    logger.info(f"Models loaded | Device: {DEVICE} | Features: {len(feat_cols)}")
    return lgbm_model, lstm_model, lstm_scaler, meta_learner, calibrator, feat_cols, cfg


# ═══════════════════════════════════════════════════════════════════════════════
# INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════
def run_inference(
    df: pd.DataFrame,
    feat_cols: list,
    lgbm_model,
    lstm_model,
    lstm_scaler,
    meta_learner,
    calibrator,
    confidence_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return y_pred_raw, y_pred_filtered, confidence."""
    mask = df["label"].astype(str).isin(LABEL_MAP)
    df   = df[mask].copy()

    valid_feat = [c for c in feat_cols if c in df.columns]
    X_df = df[valid_feat].ffill().fillna(0)
    X    = X_df.values.astype(np.float64)

    # LGBM
    lgbm_proba = lgbm_model.predict_proba(X_df)

    # LSTM
    X_sc   = lstm_scaler.transform(X)
    dummy  = np.zeros(len(X_sc), dtype=np.int64)
    ds     = SequenceDataset(X_sc, dummy)
    loader = DataLoader(ds, batch_size=1024, shuffle=False)

    lstm_list = []
    lstm_model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            logits = lstm_model(xb.to(DEVICE))
            lstm_list.append(torch.softmax(logits, dim=1).cpu().numpy())
    lstm_proba = np.vstack(lstm_list)

    if len(lstm_proba) < len(lgbm_proba):
        pad = np.ones((len(lgbm_proba) - len(lstm_proba), NUM_CLASSES)) / NUM_CLASSES
        lstm_proba = np.vstack([pad, lstm_proba])

    # Ensemble
    meta_input = np.hstack([lgbm_proba, lstm_proba])
    cal_proba  = calibrator.transform(meta_learner.predict_proba(meta_input))
    y_pred     = cal_proba.argmax(axis=1)
    confidence = cal_proba.max(axis=1)

    # Confidence filter
    y_filtered = y_pred.copy()
    y_filtered[(y_filtered != 1) & (confidence < confidence_threshold)] = 1

    n_long  = int((y_filtered == 2).sum())
    n_short = int((y_filtered == 0).sum())
    n_flat  = int((y_filtered == 1).sum())
    logger.info(f"Signal: LONG={n_long}, SHORT={n_short}, FLAT={n_flat}")

    return df, y_pred, y_filtered, confidence


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD TRADES DataFrame
# ═══════════════════════════════════════════════════════════════════════════════
def build_trades_df(
    df: pd.DataFrame,
    y_filtered: np.ndarray,
    feat_cols: list,
) -> tuple[dict, pd.DataFrame]:
    mask = df["label"].astype(str).isin(LABEL_MAP)
    df   = df[mask].copy()
    
    y    = df["label"].map(LABEL_MAP).values.astype(np.int64)
    
    atr_arr   = df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(len(df))
    close_arr = df["close"].values
    high_arr  = df["high"].values
    low_arr   = df["low"].values
    
    h4_sh_arr = df["h4_swing_high"].values if "h4_swing_high" in df.columns else None
    h4_sl_arr = df["h4_swing_low"].values  if "h4_swing_low"  in df.columns else None

    # Jalankan simulate_trades_swing langsung untuk dapat trades + equity_curve
    sim = simulate_trades_swing(
        y_pred         = y_filtered,
        close          = close_arr,
        high           = high_arr,
        low            = low_arr,
        atr            = atr_arr,
        h4_swing_highs = h4_sh_arr,
        h4_swing_lows  = h4_sl_arr,
        modal          = MODAL_PER_TRADE,
        leverage       = LEVERAGE_SIM[0],
        fee_per_side   = FEE_PER_SIDE,
        min_rr         = SWING_LABEL_MIN_RR,
        min_tp_atr     = SWING_LABEL_MIN_TP,
        max_sl_atr     = SWING_LABEL_MAX_SL,
        max_hold       = MAX_HOLDING_BARS,
    )
    
    trades_raw = sim.get("trades", [])
    
    # Map bar index ke timestamp dan build trades_df
    if trades_raw:
        idx = df.index
        records = []
        for t in trades_raw:
            bar_in  = t.get("bar_in", 0)
            bar_out = t.get("bar_out", bar_in)
            
            records.append({
                "ts_in":     idx[bar_in]  if bar_in  < len(idx) else idx[-1],
                "ts_out":    idx[bar_out] if bar_out < len(idx) else idx[-1],
                "entry":     t["entry"],
                "exit":      t["exit"],
                "tp":        t["tp"],
                "sl":        t["sl"],
                "direction": t["direction"],
                "outcome":   "WIN" if t["outcome"] == "WIN" else "LOSS",
                "pnl":       t["net_pnl"],
                "rr":        t["rr"],
            })
        trades_df = pd.DataFrame(records)
    else:
        trades_df = pd.DataFrame(
            columns=["ts_in", "ts_out", "entry", "exit", "tp", "sl", "direction", "outcome", "pnl", "rr"]
        )
        
    # Build result dict yang kompatibel dengan plot_backtest
    wins   = [t for t in trades_raw if t["outcome"] == "WIN"]
    losses = [t for t in trades_raw if t["outcome"] == "LOSS"]
    total  = len(trades_raw)
    
    result = {
        "winrate":            sim.get("winrate", 0),
        "total_trades":       total,
        "wins":               len(wins),
        "losses":             len(losses),
        "profit_factor":      sim.get("profit_factor", 0),
        "avg_rr":             sim.get("avg_rr", 0),
        "equity_curve":       sim.get("equity_curve", [0]),
        "pnl_lev3x":          sim.get("total_pnl", 0),
        "max_drawdown_lev3x": abs(sim.get("max_drawdown", 0)),
    }
    
    return result, trades_df


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT BACKTEST VISUAL
# ═══════════════════════════════════════════════════════════════════════════════
def plot_backtest(
    symbol: str,
    df: pd.DataFrame,
    result: dict,
    trades_df: pd.DataFrame,
    n_bars: int = 1500,
    out_dir: Path = REPORT_DIR,
) -> Path:
    """Buat visualisasi 3-panel: price+trades, equity curve, rolling winrate."""
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_df = df.tail(n_bars).copy()

    plot_t  = trades_df[trades_df["ts_in"] >= plot_df.index[0]].copy() \
              if not trades_df.empty else pd.DataFrame()

    fig, axes = plt.subplots(
        3, 1,
        figsize=(22, 14),
        gridspec_kw={"height_ratios": [4, 1, 1]},
        facecolor="#0d1117"
    )

    # ── Panel 1: Price + Swing + Trades ──────────────────────────────────────
    ax1 = axes[0]
    ax1.set_facecolor("#161b22")

    up   = plot_df[plot_df["close"] >= plot_df["open"]]
    down = plot_df[plot_df["close"] <  plot_df["open"]]
    W    = pd.Timedelta(minutes=45)

    ax1.bar(up.index,   up["close"]   - up["open"],   bottom=up["open"],   color="#26a69a", width=W, alpha=0.85)
    ax1.bar(down.index, down["close"] - down["open"], bottom=down["open"], color="#ef5350", width=W, alpha=0.85)
    ax1.vlines(plot_df.index, plot_df["low"], plot_df["high"], color="#777777", linewidth=0.5)

    if "h4_swing_high" in plot_df.columns:
        ax1.step(plot_df.index, plot_df["h4_swing_high"], color="#f5a623", lw=1.0, alpha=0.7, where="post", label="H4 Swing High — TP LONG")
    if "h4_swing_low" in plot_df.columns:
        ax1.step(plot_df.index, plot_df["h4_swing_low"], color="#7ed321", lw=1.0, alpha=0.7, where="post", label="H4 Swing Low  — TP SHORT")

    if not plot_t.empty:
        for _, t in plot_t.iterrows():
            is_long = t.get("direction") == "LONG"
            is_win  = t.get("outcome")   == "WIN"
            ec  = "#26a69a" if is_long else "#ef5350"
            rc  = "#00e676" if is_win  else "#ff1744"
            mk  = "^"       if is_long else "v"

            ax1.scatter(t["ts_in"],  t["entry"], marker=mk, color=ec, s=70, zorder=6, edgecolors="white", linewidths=0.3)
            ax1.scatter(t["ts_out"], t["exit"],  marker="D", color=rc, s=40, zorder=6, alpha=0.9)
            ax1.plot([t["ts_in"], t["ts_out"]], [t["entry"], t["exit"]], color=rc, lw=0.6, alpha=0.35)

            if "tp" in t:
                ax1.hlines(t["tp"], t["ts_in"], t["ts_out"], colors="#26a69a", lw=0.7, linestyles="--", alpha=0.45)
            if "sl" in t:
                ax1.hlines(t["sl"], t["ts_in"], t["ts_out"], colors="#ef5350", lw=0.7, linestyles="--", alpha=0.45)

    patches = [
        mpatches.Patch(color="#26a69a", label="LONG entry (▲)"),
        mpatches.Patch(color="#ef5350", label="SHORT entry (▼)"),
        mpatches.Patch(color="#00e676", label="WIN exit (◆)"),
        mpatches.Patch(color="#ff1744", label="LOSS exit (◆)"),
        mpatches.Patch(color="#f5a623", label="H4 Swing High — TP LONG"),
        mpatches.Patch(color="#7ed321", label="H4 Swing Low  — TP SHORT"),
    ]
    ax1.legend(handles=patches, loc="upper left", facecolor="#1a1d2e", labelcolor="white", fontsize=8.5, framealpha=0.8)

    wr  = result.get("winrate", 0)
    pf  = result.get("profit_factor", 0)
    rr  = result.get("avg_rr", 0)
    ax1.set_title(
        f"{symbol}  |  Winrate {wr:.1%}  |  Trades {len(trades_df)}  |  "
        f"PF {pf:.2f}  |  Avg R:R {rr:.2f}  |  {n_bars} bar terakhir",
        color="#e8eaf6", fontsize=12, pad=8
    )
    ax1.tick_params(colors="#b0b8d0")
    for sp in ax1.spines.values():
        sp.set_color("#2a2d3e")
    ax1.set_ylabel("Price", color="#b0b8d0")

    # ── Panel 2: Equity Curve ─────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#161b22")
    eq = result.get("equity_curve", [0])

    ax2.plot(eq, color="#4f8ef7", lw=1.3)
    ax2.fill_between(range(len(eq)), eq, alpha=0.15, color="#4f8ef7")
    ax2.axhline(0, color="#888", lw=0.7, ls="--")
    ax2.set_title("Equity Curve — Cumulative PnL ($)", color="#e8eaf6", fontsize=10)
    ax2.tick_params(colors="#b0b8d0")
    for sp in ax2.spines.values():
        sp.set_color("#2a2d3e")
    ax2.set_ylabel("PnL ($)", color="#b0b8d0")

    # ── Panel 3: Rolling Winrate ──────────────────────────────────────────────
    ax3 = axes[2]
    ax3.set_facecolor("#161b22")

    if not trades_df.empty and "outcome" in trades_df.columns:
        win_bin = (trades_df["outcome"] == "WIN").astype(int)
        rolling = win_bin.rolling(50, min_periods=10).mean() * 100

        ax3.plot(rolling.values, color="#50e3c2", lw=1.2)
        ax3.fill_between(range(len(rolling)), rolling, alpha=0.12, color="#50e3c2")
        ax3.axhline(win_bin.mean() * 100, color="#f5a623", lw=0.9, ls="--", label=f"Overall {win_bin.mean()*100:.1f}%")
        ax3.axhline(60, color="#888", lw=0.5, ls=":", label="60% baseline")
        ax3.set_ylim(0, 100)
        ax3.legend(facecolor="#1a1d2e", labelcolor="white", fontsize=8.5)

    ax3.set_title("Rolling Winrate (50-trade window)", color="#e8eaf6", fontsize=10)
    ax3.tick_params(colors="#b0b8d0")
    for sp in ax3.spines.values():
        sp.set_color("#2a2d3e")
    ax3.set_ylabel("Winrate %", color="#b0b8d0")

    plt.tight_layout(h_pad=1.5)
    out_path = out_dir / f"{symbol}_backtest_visual.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    logger.info(f"[{symbol}] Chart disimpan → {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# PLOT SWING VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════
def plot_swing_verify(
    symbol: str,
    df: pd.DataFrame,
    n_bars: int = 200,
    out_dir: Path = REPORT_DIR,
) -> Path:
    """Verifikasi swing levels — titik konfirmasi harus SETELAH puncak/lembah."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = df[["close", "h4_swing_high", "h4_swing_low"]].tail(n_bars)

    fig, ax = plt.subplots(figsize=(18, 6), facecolor="#0d1117")
    ax.set_facecolor("#161b22")

    ax.plot(sample.index, sample["close"], color="white", lw=0.8, label="Close")

    if "h4_swing_high" in sample.columns:
        ax.step(sample.index, sample["h4_swing_high"], color="#f5a623", lw=1.2, where="post", label="H4 Swing High (ffill)")
        sh_change = sample["h4_swing_high"] != sample["h4_swing_high"].shift(1)
        ax.scatter(sample.index[sh_change], sample["h4_swing_high"][sh_change], color="#f5a623", s=80, zorder=5, marker="o", label="Swing High baru terkonfirmasi")

    if "h4_swing_low" in sample.columns:
        ax.step(sample.index, sample["h4_swing_low"], color="#7ed321", lw=1.2, where="post", label="H4 Swing Low (ffill)")
        sl_change = sample["h4_swing_low"] != sample["h4_swing_low"].shift(1)
        ax.scatter(sample.index[sl_change], sample["h4_swing_low"][sl_change], color="#7ed321", s=80, zorder=5, marker="o", label="Swing Low baru terkonfirmasi")

    ax.legend(facecolor="#1a1d2e", labelcolor="white", fontsize=9)
    ax.set_title(
        f"{symbol} — Verifikasi Swing Levels (titik harus muncul SETELAH puncak/lembah)",
        color="#e8eaf6", fontsize=12
    )
    ax.tick_params(colors="#b0b8d0")
    for sp in ax.spines.values():
        sp.set_color("#2a2d3e")

    plt.tight_layout()
    out_path = out_dir / f"{symbol}_swing_verify.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()

    # Print tabel perubahan swing
    if "h4_swing_high" in sample.columns:
        sh_change = sample["h4_swing_high"] != sample["h4_swing_high"].shift(1)
        changes   = sample[sh_change][["close", "h4_swing_high"]].head(10)
        print(f"\n[{symbol}] Kapan swing high berubah (10 pertama):")
        print(changes.to_string())

    logger.info(f"[{symbol}] Swing verify chart → {out_path}")
    return out_path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def parse_args():
    parser = argparse.ArgumentParser(description="Visualisasi Backtest Pipeline")
    parser.add_argument("--symbol",       default="ETHUSDT", help="Koin target")
    parser.add_argument("--all",          action="store_true", help="Semua koin")
    parser.add_argument("--holdout",      action="store_true", help="Gunakan data hold-out")
    parser.add_argument("--verify-swing", action="store_true", help="Verifikasi swing levels")
    parser.add_argument("--n-bars",       type=int, default=1500, help="Jumlah bar untuk chart")
    return parser.parse_args()


def process_symbol(
    symbol: str,
    holdout: bool,
    verify_swing: bool,
    n_bars: int,
    lgbm_model,
    lstm_model,
    lstm_scaler,
    meta_learner,
    calibrator,
    feat_cols: list,
    cfg: dict,
) -> None:
    label_dir = HOLDOUT_LABEL_DIR if holdout else LABEL_DIR
    suffix    = "holdout" if holdout else "training"
    out_dir   = REPORT_DIR / suffix

    parquet_path = label_dir / f"{symbol}_features_v3.parquet"
    if not parquet_path.exists():
        logger.warning(f"[{symbol}] File tidak ditemukan: {parquet_path}")
        return

    df = pd.read_parquet(parquet_path)
    df = ensure_utc_index(df)
    df = df.sort_index()

    logger.info(f"[{symbol}] Loaded: {len(df):,} rows ({suffix})")
    confidence_threshold = cfg["inference"]["confidence_threshold_entry"]

    # Inference
    df_masked, y_pred, y_filtered, confidence = run_inference(
        df, feat_cols, lgbm_model, lstm_model, lstm_scaler, meta_learner, calibrator, confidence_threshold,
    )

    # Build trades
    result, trades_df = build_trades_df(df_masked, y_filtered, feat_cols)

    # Plot backtest visual
    plot_backtest(symbol, df_masked, result, trades_df, n_bars=n_bars, out_dir=out_dir)

    # Plot swing verify (optional)
    if verify_swing:
        plot_swing_verify(symbol, df_masked, n_bars=200, out_dir=out_dir)

    # Print ringkasan
    print(f"\n{'='*50}")
    print(f"  {symbol} ({suffix})")
    print(f"{'='*50}")
    print(f"  Winrate        : {result.get('winrate', 0):.2%}")
    print(f"  Total trades   : {result.get('total_trades', 0)}")
    print(f"  Profit Factor  : {result.get('profit_factor', 0):.2f}")
    print(f"  Avg R:R        : {result.get('avg_rr', 0):.2f}")
    print(f"  Max DD Lev3x   : {result.get('max_drawdown_lev3x', 0):.2%}")
    print(f"  PnL Lev3x      : ${result.get('pnl_lev3x', 0):+,.2f}")
    print(f"{'='*50}\n")


def main():
    args = parse_args()
    lgbm_model, lstm_model, lstm_scaler, meta_learner, calibrator, feat_cols, cfg = load_models()
    coins = ALL_COINS if args.all else [args.symbol.upper()]

    for symbol in coins:
        try:
            process_symbol(
                symbol       = symbol,
                holdout      = args.holdout,
                verify_swing = args.verify_swing,
                n_bars       = args.n_bars,
                lgbm_model   = lgbm_model,
                lstm_model   = lstm_model,
                lstm_scaler  = lstm_scaler,
                meta_learner = meta_learner,
                calibrator   = calibrator,
                feat_cols    = feat_cols,
                cfg          = cfg,
            )
        except Exception as e:
            logger.error(f"[{symbol}] Error: {e}")
            import traceback
            logger.error(traceback.format_exc())

if __name__ == "__main__":
    main()