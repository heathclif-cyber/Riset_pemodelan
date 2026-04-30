"""
pipeline/08_backtest.py — Fase 8: Walk-Forward Backtest

Walk-forward backtest menggunakan ensemble model (LGBM + LSTM + Calibrator).
Berbeda dengan 07_evaluate.py yang hanya pakai LGBM untuk SHAP,
08_backtest.py mensimulasikan trade nyata menggunakan full ensemble pipeline.

Jalankan:
  python pipeline/08_backtest.py                 # training coins
  python pipeline/08_backtest.py --all           # semua 20 koin
  python pipeline/08_backtest.py --coins SOLUSDT ETHUSDT
  python pipeline/08_backtest.py --run-id my_run

Output:
  models/runs/{run_id}/backtest_results.json
  models/inference_config.json   ★ BARU: semua parameter untuk deployment
"""

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, ALL_COINS, LABEL_DIR, MODEL_DIR,
    TRAIN_START, TRAIN_END,
    N_FOLDS, PURGE_GAP_BARS,
    LABEL_MAP, LABEL_MAP_INV, NUM_CLASSES,
    LSTM_SEQ_LEN, LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE,
    TP_ATR_MULT, SL_ATR_MULT, MAX_HOLDING_BARS,
    CONFIDENCE_THRESHOLD_ENTRY, CONFIDENCE_FULL, CONFIDENCE_HALF,
    MIN_HOLD_BARS,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    VP_WINDOW, VP_BINS, FVG_MIN_GAP_ATR, SWING_LOOKBACK,
    SIGNAL_FLIP_CONF_MIN, FLIP_CONFIRM_BARS, FLIP_COOLDOWN_SECS, SAME_DIR_COOLDOWN_HOURS,
    VCB_ENABLED, VCB_ATR_MULTIPLIER, VCB_LOOKBACK_BARS,
    MONITOR_POLL_INTERVAL_SECS,
)
from core.models import load_lstm, ProbabilityCalibrator
from core.evaluator import full_trading_report
from core.utils import setup_logger, update_model_metrics
from pipeline.p05_utils import SequenceDataset

logger = setup_logger("08_backtest")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NON_FEATURE_COLS = {"label", "h4_swing_high", "h4_swing_low"}


# ─── Data Loading ─────────────────────────────────────────────────────────────

def load_symbol(symbol: str, feat_cols: list[str]) -> tuple[pd.DataFrame, np.ndarray] | None:
    path = LABEL_DIR / f"{symbol}_features_v3.parquet"
    if not path.exists():
        logger.warning(f"[{symbol}] File tidak ditemukan: {path}")
        return None

    df = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.sort_index()

    mask = df["label"].astype(str).isin(LABEL_MAP)
    df   = df[mask].copy()
    y    = df["label"].map(LABEL_MAP).values.astype(np.int64)

    valid_cols = [c for c in feat_cols if c in df.columns]
    df[valid_cols] = df[valid_cols].ffill().fillna(0)

    logger.info(f"[{symbol}] Loaded: {len(df):,} rows")
    return df, y


# ─── Ensemble Inference ───────────────────────────────────────────────────────

def get_ensemble_proba(
    lgbm_model,
    lstm_model,
    lstm_scaler,
    meta_learner,
    calibrator: ProbabilityCalibrator,
    X: np.ndarray,
    feat_cols: list[str],
    df_slice: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    lgbm_proba = lgbm_model.predict_proba(df_slice[feat_cols])

    X_sc  = lstm_scaler.transform(X)
    dummy = np.zeros(len(X_sc), dtype=np.int64)
    ds    = SequenceDataset(X_sc, dummy)
    loader = DataLoader(ds, batch_size=1024, shuffle=False)

    lstm_list = []
    lstm_model.eval()
    with torch.no_grad():
        for xb, _ in loader:
            logits = lstm_model(xb.to(DEVICE))
            proba  = torch.softmax(logits, dim=1).cpu().numpy()
            lstm_list.append(proba)
    lstm_proba = np.vstack(lstm_list)

    if len(lstm_proba) < len(lgbm_proba):
        pad = np.ones((len(lgbm_proba) - len(lstm_proba), NUM_CLASSES)) / NUM_CLASSES
        lstm_proba = np.vstack([pad, lstm_proba])

    meta_input = np.hstack([lgbm_proba, lstm_proba])
    meta_proba = meta_learner.predict_proba(meta_input)
    cal_proba  = calibrator.transform(meta_proba)

    return cal_proba.argmax(axis=1), cal_proba.max(axis=1)


# ─── Walk-Forward Folds ───────────────────────────────────────────────────────

def build_purged_folds(n: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splits = np.array_split(np.arange(n), N_FOLDS + 1)
    folds  = []
    for k in range(1, N_FOLDS + 1):
        train_raw = np.concatenate(splits[:k])
        test_raw  = splits[k]
        train_idx = train_raw[:-PURGE_GAP_BARS] if len(train_raw) > PURGE_GAP_BARS else train_raw
        test_idx  = test_raw[PURGE_GAP_BARS:]   if len(test_raw)  > PURGE_GAP_BARS  else test_raw
        folds.append((train_idx, test_idx))
    return folds


# ─── Trade Chart ─────────────────────────────────────────────────────────────

def build_trades_from_predictions(
    df_valid: pd.DataFrame,
    y_pred: np.ndarray,
    conf: np.ndarray,
    tp_mult: float,
    sl_mult: float,
    max_hold: int,
    min_hold: int,
) -> pd.DataFrame:
    """
    Simulate trade-by-trade dari OOF predictions.
    Return DataFrame satu baris per trade dengan entry/exit/TP/SL info.
    """
    close_arr = df_valid["close"].values
    atr_arr   = df_valid["atr_14_m15"].values if "atr_14_m15" in df_valid.columns else np.ones(len(df_valid))
    index_arr = df_valid.index

    trades    = []
    trade_num = 0
    pos       = None  # current open position

    for i in range(len(df_valid)):
        pred = y_pred[i]
        c    = conf[i]

        # ── Close existing position ────────────────────────────────────────────
        if pos is not None:
            price   = close_arr[i]
            held    = i - pos["entry_bar"]
            outcome = None

            if pos["direction"] == "LONG":
                if price >= pos["tp"]:
                    outcome, exit_price = "TP", pos["tp"]
                elif price <= pos["sl"]:
                    outcome, exit_price = "SL", pos["sl"]
                elif held >= max_hold:
                    outcome, exit_price = "EXPIRED", price
            else:  # SHORT
                if price <= pos["tp"]:
                    outcome, exit_price = "TP", pos["tp"]
                elif price >= pos["sl"]:
                    outcome, exit_price = "SL", pos["sl"]
                elif held >= max_hold:
                    outcome, exit_price = "EXPIRED", price

            if outcome:
                pnl = (exit_price - pos["entry_price"]) if pos["direction"] == "LONG" \
                      else (pos["entry_price"] - exit_price)
                trades.append({
                    "trade_num":   pos["trade_num"],
                    "direction":   pos["direction"],
                    "entry_bar":   pos["entry_bar"],
                    "exit_bar":    i,
                    "entry_time":  pos["entry_time"],
                    "exit_time":   index_arr[i],
                    "entry_price": pos["entry_price"],
                    "exit_price":  exit_price,
                    "tp":          pos["tp"],
                    "sl":          pos["sl"],
                    "outcome":     outcome,
                    "pnl":         pnl,
                    "pnl_atr":     pnl / pos["atr"],
                    "conf":        pos["conf"],
                    "bars_held":   held,
                })
                pos = None

        # ── Open new position ──────────────────────────────────────────────────
        if pos is None and pred in [0, 2]:  # 0=SHORT, 2=LONG
            direction   = "LONG" if pred == 2 else "SHORT"
            entry_price = close_arr[i]
            atr_i       = atr_arr[i]
            trade_num  += 1

            if direction == "LONG":
                tp = entry_price + tp_mult * atr_i
                sl = entry_price - sl_mult * atr_i
            else:
                tp = entry_price - tp_mult * atr_i
                sl = entry_price + sl_mult * atr_i

            pos = {
                "trade_num":   trade_num,
                "direction":   direction,
                "entry_bar":   i,
                "entry_time":  index_arr[i],
                "entry_price": entry_price,
                "tp":          tp,
                "sl":          sl,
                "atr":         atr_i,
                "conf":        c,
            }

    return pd.DataFrame(trades)


def plot_trade_chart(
    df_valid: pd.DataFrame,
    trades_df: pd.DataFrame,
    symbol: str,
    run_dir: Path,
    max_trades_plot: int = 80,
) -> None:
    """
    Plot price chart dengan entry/exit markers, TP/SL lines, dan nomor trade.

    Layout:
    ├─ Panel 1 (besar): Price + EMA + trade markers
    └─ Panel 2 (kecil): Cumulative PnL (equity curve)
    """
    if trades_df.empty:
        logger.warning(f"[{symbol}] Tidak ada trades untuk diplot.")
        return

    close = df_valid["close"]
    idx   = df_valid.index

    # Ambil subset trades supaya grafik tidak terlalu penuh
    plot_trades = trades_df.head(max_trades_plot)

    # ── Tentukan window chart berdasarkan trades yang diplot ──────────────────
    first_bar = int(plot_trades["entry_bar"].min())
    last_bar  = int(plot_trades["exit_bar"].max())
    pad       = max(20, (last_bar - first_bar) // 20)
    start_bar = max(0, first_bar - pad)
    end_bar   = min(len(df_valid) - 1, last_bar + pad)

    close_slice = close.iloc[start_bar:end_bar + 1]
    idx_slice   = idx[start_bar:end_bar + 1]
    x_range     = range(len(close_slice))

    # ── Colors ────────────────────────────────────────────────────────────────
    CLR = {
        "bg":       "#0d1117",
        "panel":    "#161b22",
        "grid":     "#21262d",
        "text":     "#e6edf3",
        "price":    "#58a6ff",
        "long_in":  "#3fb950",
        "short_in": "#f85149",
        "tp":       "#3fb950",
        "sl":       "#f85149",
        "expired":  "#d29922",
        "equity":   "#a5d6ff",
        "flat":     "#8b949e",
    }

    fig = plt.figure(figsize=(22, 12), facecolor=CLR["bg"])
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax1 = fig.add_subplot(gs[0])  # price chart
    ax2 = fig.add_subplot(gs[1], sharex=ax1)  # equity curve

    for ax in [ax1, ax2]:
        ax.set_facecolor(CLR["panel"])
        ax.tick_params(colors=CLR["text"], labelsize=8)
        ax.spines[:].set_color(CLR["grid"])
        ax.grid(True, color=CLR["grid"], linewidth=0.5, alpha=0.7)

    # ── Panel 1: Price line ────────────────────────────────────────────────────
    ax1.plot(x_range, close_slice.values, color=CLR["price"],
             linewidth=1.2, zorder=2, label="Close")

    # Optional: EMA H4 overlay jika tersedia
    if "ema_21_h4" in df_valid.columns:
        ema = df_valid["ema_21_h4"].iloc[start_bar:end_bar + 1]
        ax1.plot(x_range, ema.values, color="#d29922",
                 linewidth=0.8, alpha=0.7, linestyle="--", label="EMA 21 H4")

    # ── Plot setiap trade ─────────────────────────────────────────────────────
    for _, t in plot_trades.iterrows():
        eb = int(t["entry_bar"]) - start_bar
        xb = int(t["exit_bar"])  - start_bar

        if eb < 0 or xb < 0 or eb >= len(close_slice):
            continue
        xb = min(xb, len(close_slice) - 1)

        num     = int(t["trade_num"])
        is_long = t["direction"] == "LONG"
        outcome = t["outcome"]

        entry_clr  = CLR["long_in"] if is_long else CLR["short_in"]
        exit_clr   = CLR["tp"] if outcome == "TP" else (
                     CLR["sl"] if outcome == "SL" else CLR["expired"])
        marker_in  = "^" if is_long else "v"
        marker_out = "D" if outcome == "TP" else ("x" if outcome == "SL" else "o")

        ep  = t["entry_price"]
        tp  = t["tp"]
        sl  = t["sl"]
        xp  = t["exit_price"]
        x_range_trade = range(eb, xb + 1)

        # TP line (dashed)
        ax1.hlines(tp, eb, xb, colors=CLR["tp"],
                   linewidth=0.8, linestyle=":", alpha=0.6)
        # SL line (dashed)
        ax1.hlines(sl, eb, xb, colors=CLR["sl"],
                   linewidth=0.8, linestyle=":", alpha=0.6)
        # Shaded region antara TP dan SL
        ax1.fill_between(x_range_trade,
                         [sl] * len(x_range_trade),
                         [tp] * len(x_range_trade),
                         color=entry_clr, alpha=0.04)

        # Entry marker
        ax1.scatter(eb, ep, marker=marker_in, color=entry_clr,
                    s=90, zorder=5, linewidths=1.5)
        # Exit marker
        if xb < len(close_slice):
            ax1.scatter(xb, xp, marker=marker_out, color=exit_clr,
                        s=90, zorder=5, linewidths=1.5)

        # Nomor trade (label di entry)
        ax1.annotate(
            f"{num}",
            xy=(eb, ep),
            xytext=(eb, ep * (1.003 if is_long else 0.997)),
            fontsize=6.5, color=entry_clr, ha="center", va="bottom",
            fontweight="bold", zorder=6,
        )

        # Nomor trade di exit (TP/SL label)
        if xb < len(close_slice):
            outcome_label = f"{num} {'TP' if outcome=='TP' else ('SL' if outcome=='SL' else 'EXP')}"
            ax1.annotate(
                outcome_label,
                xy=(xb, xp),
                xytext=(xb, xp * (1.003 if outcome == "TP" else 0.997)),
                fontsize=6, color=exit_clr, ha="center",
                va="bottom" if outcome == "TP" else "top",
                zorder=6,
            )

    # ── Axis labels ───────────────────────────────────────────────────────────
    # X axis: tampilkan tanggal setiap N bar
    n_ticks = min(12, len(close_slice) // 10 + 1)
    tick_pos = np.linspace(0, len(close_slice) - 1, n_ticks, dtype=int)
    ax1.set_xticks(tick_pos)
    ax1.set_xticklabels(
        [str(idx_slice[p])[:10] for p in tick_pos],
        rotation=30, ha="right", fontsize=7, color=CLR["text"],
    )
    ax1.set_ylabel("Price (USDT)", color=CLR["text"], fontsize=9)
    ax1.set_title(
        f"{symbol} — Trade Chart  |  "
        f"{len(plot_trades)} trades shown  |  "
        f"TP rate: {(trades_df['outcome']=='TP').mean():.1%}  "
        f"SL rate: {(trades_df['outcome']=='SL').mean():.1%}",
        color=CLR["text"], fontsize=11, pad=10,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=CLR["long_in"],  label="LONG entry (▲)"),
        mpatches.Patch(color=CLR["short_in"], label="SHORT entry (▼)"),
        mpatches.Patch(color=CLR["tp"],       label="TP hit (◆)"),
        mpatches.Patch(color=CLR["sl"],       label="SL hit (✕)"),
        mpatches.Patch(color=CLR["expired"],  label="EXPIRED (●)"),
    ]
    ax1.legend(handles=legend_items, loc="upper left",
               facecolor=CLR["panel"], edgecolor=CLR["grid"],
               labelcolor=CLR["text"], fontsize=8)

    # ── Panel 2: Equity curve ─────────────────────────────────────────────────
    # Gunakan semua trades (bukan hanya yang diplot)
    cumulative_pnl = trades_df["pnl_atr"].cumsum().values
    trade_x        = trades_df["exit_bar"].values - start_bar

    # Filter yang masuk window
    mask_in_window = (trade_x >= 0) & (trade_x < len(close_slice))
    if mask_in_window.any():
        tx_plot  = trade_x[mask_in_window]
        pnl_plot = cumulative_pnl[: mask_in_window.sum()]
        colors_eq = [CLR["tp"] if p >= 0 else CLR["sl"] for p in np.diff(
            np.concatenate([[0], pnl_plot]))]

        ax2.plot(tx_plot, pnl_plot, color=CLR["equity"],
                 linewidth=1.2, zorder=3)
        ax2.fill_between(tx_plot, 0, pnl_plot,
                         where=(pnl_plot >= 0), color=CLR["tp"],
                         alpha=0.15, interpolate=True)
        ax2.fill_between(tx_plot, 0, pnl_plot,
                         where=(pnl_plot < 0), color=CLR["sl"],
                         alpha=0.15, interpolate=True)
        ax2.axhline(0, color=CLR["flat"], linewidth=0.8, linestyle="--")

    ax2.set_ylabel("Cum. PnL (ATR)", color=CLR["text"], fontsize=9)
    ax2.set_xlabel("Bar index", color=CLR["text"], fontsize=9)
    plt.setp(ax1.get_xticklabels(), visible=False)

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = run_dir / f"{symbol}_trade_chart.svg"
    fig.savefig(out_path, dpi=150, bbox_inches="tight",
                facecolor=CLR["bg"])
    plt.close(fig)
    logger.info(f"[{symbol}] Trade chart → {out_path}")


def plot_summary_chart(results: dict, trades_all: dict, run_dir: Path) -> None:
    """
    Summary chart untuk semua koin:
    ├─ Win rate per koin
    ├─ TP vs SL vs EXPIRED breakdown
    └─ Overall equity curve gabungan
    """
    CLR = {
        "bg": "#0d1117", "panel": "#161b22",
        "grid": "#21262d", "text": "#e6edf3",
        "tp": "#3fb950", "sl": "#f85149",
        "expired": "#d29922", "bar": "#58a6ff",
    }

    symbols = list(results.keys())
    if not symbols:
        return

    fig, axes = plt.subplots(1, 3, figsize=(20, 6), facecolor=CLR["bg"])

    # ── Panel 1: Win Rate per koin ────────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(CLR["panel"])
    ax.spines[:].set_color(CLR["grid"])
    ax.grid(True, color=CLR["grid"], linewidth=0.5, alpha=0.7)

    wrs    = [results[s].get("winrate", 0) for s in symbols]
    colors = [CLR["tp"] if w >= 0.60 else CLR["sl"] for w in wrs]
    bars   = ax.barh(symbols, wrs, color=colors, alpha=0.8, height=0.6)
    ax.axvline(0.60, color=CLR["expired"], linewidth=1.5,
               linestyle="--", label="60% threshold")
    for bar, wr in zip(bars, wrs):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                f"{wr:.1%}", va="center", fontsize=8, color=CLR["text"])
    ax.set_xlabel("Win Rate", color=CLR["text"])
    ax.set_title("Win Rate per Koin", color=CLR["text"])
    ax.tick_params(colors=CLR["text"])
    ax.legend(facecolor=CLR["panel"], labelcolor=CLR["text"], fontsize=8)

    # ── Panel 2: TP / SL / EXPIRED breakdown ─────────────────────────────────
    ax = axes[1]
    ax.set_facecolor(CLR["panel"])
    ax.spines[:].set_color(CLR["grid"])
    ax.grid(True, color=CLR["grid"], linewidth=0.5, alpha=0.7)

    all_trades_concat = pd.concat(list(trades_all.values())) if trades_all else pd.DataFrame()
    if not all_trades_concat.empty:
        outcome_counts = all_trades_concat["outcome"].value_counts()
        labels  = list(outcome_counts.index)
        sizes   = list(outcome_counts.values)
        clr_map = {"TP": CLR["tp"], "SL": CLR["sl"], "EXPIRED": CLR["expired"]}
        pie_clr = [clr_map.get(l, CLR["bar"]) for l in labels]
        wedges, texts, autotexts = ax.pie(
            sizes, labels=labels, colors=pie_clr,
            autopct="%1.1f%%", startangle=90,
            textprops={"color": CLR["text"], "fontsize": 9},
        )
        for at in autotexts:
            at.set_color(CLR["bg"])
            at.set_fontweight("bold")
    ax.set_title("TP / SL / EXPIRED Distribution", color=CLR["text"])

    # ── Panel 3: Equity curve gabungan ────────────────────────────────────────
    ax = axes[2]
    ax.set_facecolor(CLR["panel"])
    ax.spines[:].set_color(CLR["grid"])
    ax.grid(True, color=CLR["grid"], linewidth=0.5, alpha=0.7)
    ax.axhline(0, color=CLR["expired"], linewidth=0.8, linestyle="--")

    for sym, tdf in trades_all.items():
        if tdf.empty:
            continue
        cum = tdf["pnl_atr"].cumsum().values
        ax.plot(range(len(cum)), cum, linewidth=1.0, alpha=0.7, label=sym)

    ax.set_xlabel("Trade #", color=CLR["text"])
    ax.set_ylabel("Cumulative PnL (ATR)", color=CLR["text"])
    ax.set_title("Equity Curve per Koin", color=CLR["text"])
    ax.tick_params(colors=CLR["text"])
    ax.legend(facecolor=CLR["panel"], labelcolor=CLR["text"],
              fontsize=7, ncol=2)

    plt.suptitle("Backtest Summary — All Coins", color=CLR["text"],
                 fontsize=13, y=1.01)

    out_path = run_dir / "backtest_summary_chart.svg"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=CLR["bg"])
    plt.close(fig)
    logger.info(f"Summary chart → {out_path}")


# ─── Backtest Per Symbol ──────────────────────────────────────────────────────

def backtest_symbol(
    symbol: str,
    feat_cols: list[str],
    lgbm_model,
    lstm_model,
    lstm_scaler,
    meta_learner,
    calibrator: ProbabilityCalibrator,
) -> dict | None:
    result = load_symbol(symbol, feat_cols)
    if result is None:
        return None

    df, y = result
    X     = df[feat_cols].values.astype(np.float64)
    folds = build_purged_folds(len(df))

    oof_pred  = np.full(len(y), -1, dtype=np.int64)
    oof_conf  = np.zeros(len(y), dtype=np.float64)
    oof_valid = np.zeros(len(y), dtype=bool)

    for fold_num, (_, te_pos) in enumerate(folds, 1):
        if len(te_pos) == 0:
            continue
        df_te = df.iloc[te_pos]
        X_te  = X[te_pos]
        y_pred, confidence = get_ensemble_proba(
            lgbm_model, lstm_model, lstm_scaler,
            meta_learner, calibrator,
            X_te, feat_cols, df_te,
        )
        oof_pred[te_pos]  = y_pred
        oof_conf[te_pos]  = confidence
        oof_valid[te_pos] = True
        logger.info(f"[{symbol}] Fold {fold_num} done — test={len(te_pos):,} bars")

    valid_idx = np.where(oof_valid)[0]
    if len(valid_idx) == 0:
        logger.warning(f"[{symbol}] Tidak ada valid OOF predictions.")
        return None

    y_pred_filtered = oof_pred[valid_idx].copy()
    conf_filtered   = oof_conf[valid_idx]
    below_threshold = (y_pred_filtered != 1) & (conf_filtered < CONFIDENCE_THRESHOLD_ENTRY)
    y_pred_filtered[below_threshold] = 1
    n_filtered = int(below_threshold.sum())
    logger.info(f"[{symbol}] Confidence filter: {n_filtered} sinyal di-skip (< {CONFIDENCE_THRESHOLD_ENTRY:.0%})")

    df_valid  = df.iloc[valid_idx]
    y_valid   = y[valid_idx]
    atr_arr   = df_valid["atr_14_h1"].values if "atr_14_h1" in df_valid.columns else np.ones(len(df_valid))
    close_arr = df_valid["close"].values       if "close"      in df_valid.columns else np.ones(len(df_valid))
    h4_sh_arr = df_valid["h4_swing_high"].values if "h4_swing_high" in df_valid.columns else None
    h4_sl_arr = df_valid["h4_swing_low"].values  if "h4_swing_low"  in df_valid.columns else None
    high_arr  = df_valid["high"].values if "high" in df_valid.columns else close_arr
    low_arr   = df_valid["low"].values if "low" in df_valid.columns else close_arr

    report = full_trading_report(
        y_pred          = y_pred_filtered,
        y_actual        = y_valid,
        atr             = atr_arr,
        close           = close_arr,
        high            = high_arr,
        low             = low_arr,
        h4_swing_highs  = h4_sh_arr,
        h4_swing_lows   = h4_sl_arr,
        index           = df_valid.index,
        modal           = MODAL_PER_TRADE,
        leverages       = LEVERAGE_SIM,
        fee_per_side    = FEE_PER_SIDE,
        min_rr          = SWING_LABEL_MIN_RR,
        min_tp_atr      = SWING_LABEL_MIN_TP,
        max_sl_atr      = SWING_LABEL_MAX_SL,
        max_hold        = MAX_HOLDING_BARS,
        symbol          = symbol,
    )

    report["n_filtered_by_confidence"] = n_filtered
    report["confidence_threshold"]     = CONFIDENCE_THRESHOLD_ENTRY

    # ── Build trade-by-trade DataFrame (untuk chart) ──────────────────────────
    trades_df = build_trades_from_predictions(
        df_valid   = df_valid,
        y_pred     = y_pred_filtered,
        conf       = conf_filtered,
        tp_mult    = TP_ATR_MULT,
        sl_mult    = SL_ATR_MULT,
        max_hold   = MAX_HOLDING_BARS,
        min_hold   = MIN_HOLD_BARS,
    )
    report["trades_df"] = trades_df  # simpan untuk chart di main

    return report


# ─── Generate Inference Config ────────────────────────────────────────────────

def generate_inference_config(
    aggregate: dict,
    results: dict,
    feat_cols: list[str],
) -> dict:
    """
    Generate inference_config.json — semua parameter yang dibutuhkan
    aplikasi untuk plug-and-play deployment model.

    Aplikasi cukup baca file ini untuk tahu:
      - Parameter inference (threshold, hold bars, dll)
      - Parameter labeling (TP/SL mult)
      - Parameter risk (leverage, modal, fee)
      - Koin mana yang validated dan direkomendasikan
      - Hasil backtest per koin
      - Arsitektur model (n_features, hidden size, dll)
    """
    # Tentukan kategori koin berdasarkan winrate dan drawdown
    recommended, acceptable, caution = [], [], []
    for sym, r in results.items():
        wr  = r.get("winrate", 0)
        dd3 = r.get("max_drawdown_lev3x", 1)
        if wr >= 0.63 and dd3 <= 0.20:
            recommended.append(sym)
        elif wr >= 0.60 and dd3 <= 0.30:
            acceptable.append(sym)
        else:
            caution.append(sym)

    return {
        "model_version": "ensemble_v2",
        "created_at":    datetime.now(timezone.utc).isoformat(),
        "training_period": {
            "start": str(TRAIN_START.date()),
            "end":   str(TRAIN_END.date()),
        },

        # ── Parameter Inference ───────────────────────────────────────────────
        # Semua parameter yang dibutuhkan saat generate sinyal live
        "inference": {
            "confidence_threshold_entry": CONFIDENCE_THRESHOLD_ENTRY,
            "confidence_full_size":       CONFIDENCE_FULL,
            "confidence_half_size":       CONFIDENCE_HALF,
            "min_hold_bars":              MIN_HOLD_BARS,
            "max_hold_bars":              MAX_HOLDING_BARS,
            "timeframe":                  "1h",
            "seq_len":                    LSTM_SEQ_LEN,
            "label_map":                  LABEL_MAP,
            "label_map_inv":              {str(v): k for k, v in LABEL_MAP.items()},
        },

        # ── Parameter Feature Engineering ─────────────────────────────────────
        "feature_engineering": {
            "vp_window":       VP_WINDOW,
            "vp_bins":         VP_BINS,
            "fvg_min_gap_atr": FVG_MIN_GAP_ATR,
            "swing_lookback":  SWING_LOOKBACK,
        },

        # ── Parameter Signal Stability ────────────────────────────────────────
        "signal_stability": {
            "flip_conf_min":           SIGNAL_FLIP_CONF_MIN,
            "flip_confirm_bars":       FLIP_CONFIRM_BARS,
            "flip_cooldown_secs":      FLIP_COOLDOWN_SECS,
            "same_dir_cooldown_hours": SAME_DIR_COOLDOWN_HOURS,
        },

        # ── Parameter Circuit Breaker ─────────────────────────────────────────
        "volatility_circuit_breaker": {
            "enabled":        VCB_ENABLED,
            "atr_multiplier": VCB_ATR_MULTIPLIER,
            "lookback_bars":  VCB_LOOKBACK_BARS,
        },

        # ── Parameter Monitoring ──────────────────────────────────────────────
        "monitor": {
            "pairs":              ALL_COINS,
            "poll_interval_secs": MONITOR_POLL_INTERVAL_SECS,
        },

        # ── Parameter Labeling ────────────────────────────────────────────────
        # Dipakai untuk hitung TP/SL di aplikasi
        "labeling": {
            "tp_atr_mult":      TP_ATR_MULT,
            "sl_atr_mult":      SL_ATR_MULT,
            "max_holding_bars": MAX_HOLDING_BARS,
        },

        # ── Parameter Risk ────────────────────────────────────────────────────
        "risk": {
            "modal_per_trade":      MODAL_PER_TRADE,
            "leverage_recommended": 3.0,
            "leverage_max":         5.0,
            "fee_per_side":         FEE_PER_SIDE,
        },

        # ── Koin yang Sudah Divalidasi ────────────────────────────────────────
        # recommended : winrate ≥ 63% dan DD lev3x ≤ 20%
        # acceptable  : winrate ≥ 60% dan DD lev3x ≤ 30%
        # caution     : di bawah threshold — pakai dengan hati-hati
        "coins_validated": {
            "recommended": recommended,
            "acceptable":  acceptable,
            "caution":     caution,
        },

        # ── Hasil Backtest ────────────────────────────────────────────────────
        "backtest_summary": {
            "mean_winrate":         aggregate["mean_winrate"],
            "mean_trade_per_month": aggregate["mean_trade_per_month"],
            "mean_drawdown_lev3x":  aggregate["mean_drawdown_lev3x"],
            "max_consecutive_loss": aggregate["max_consecutive_loss"],
        },
        "backtest_per_coin": {
            sym: {
                "winrate":         r.get("winrate", 0),
                "trade_per_month": r.get("trade_per_month", 0),
                "dd_lev3x":        r.get("max_drawdown_lev3x", 0),
                "dd_lev5x":        r.get("max_drawdown_lev5x", 0),
                "pnl_lev3x":       r.get("pnl_lev3x", 0),
                "pnl_lev5x":       r.get("pnl_lev5x", 0),
                "max_consec_loss": r.get("max_consecutive_loss", 0),
                "total_trades":    r.get("total_trades", 0),
            }
            for sym, r in results.items()
        },

        # ── File Model ────────────────────────────────────────────────────────
        # Path relatif dari folder models/
        "model_files": {
            "lgbm":       "lgbm_baseline.pkl",
            "lstm":       "lstm_best.pt",
            "scaler":     "lstm_scaler.pkl",
            "meta":       "ensemble_meta.pkl",
            "calibrator": "calibrator.pkl",
            "features":   "feature_cols_v2.json",
        },

        # ── Arsitektur Model ──────────────────────────────────────────────────
        # Dipakai untuk load LSTM dengan parameter yang benar
        "model_architecture": {
            "n_features":   len(feat_cols),
            "lstm_hidden":  LSTM_HIDDEN,
            "lstm_layers":  LSTM_LAYERS,
            "lstm_dropout": LSTM_DROPOUT,
            "num_classes":  NUM_CLASSES,
        },
    }


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Walk-Forward Backtest v2")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--all",   action="store_true", help="Semua koin")
    group.add_argument("--coins", nargs="+", metavar="SYMBOL")
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main():
    args    = parse_args()
    run_id  = args.run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = MODEL_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if args.all:
        coins = ALL_COINS
    elif args.coins:
        coins = [c.upper() for c in args.coins]
    else:
        coins = TRAINING_COINS

    # ── Load models ───────────────────────────────────────────────────────────
    for path, name in [
        (MODEL_DIR / "lgbm_baseline.pkl",    "LightGBM"),
        (MODEL_DIR / "lstm_best.pt",         "LSTM"),
        (MODEL_DIR / "lstm_scaler.pkl",      "LSTM Scaler"),
        (MODEL_DIR / "ensemble_meta.pkl",    "Meta-learner"),
        (MODEL_DIR / "calibrator.pkl",       "Calibrator"),
        (MODEL_DIR / "feature_cols_v2.json", "Feature cols"),
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"{name} tidak ditemukan: {path}\n"
                f"Jalankan pipeline 04-06 terlebih dahulu."
            )

    lgbm_model   = joblib.load(MODEL_DIR / "lgbm_baseline.pkl")
    lstm_model   = load_lstm(MODEL_DIR / "lstm_best.pt", device=str(DEVICE)).to(DEVICE)
    lstm_scaler  = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
    meta_learner = joblib.load(MODEL_DIR / "ensemble_meta.pkl")
    calibrator   = ProbabilityCalibrator.load(MODEL_DIR / "calibrator.pkl")

    with open(MODEL_DIR / "feature_cols_v2.json") as f:
        feat_cols = json.load(f)

    logger.info(f"Models loaded | Device: {DEVICE} | Features: {len(feat_cols)} | Coins: {coins}")

    # ── Backtest per symbol ───────────────────────────────────────────────────
    results         = {}
    success, failed = [], []

    for symbol in coins:
        try:
            report = backtest_symbol(
                symbol, feat_cols,
                lgbm_model, lstm_model, lstm_scaler,
                meta_learner, calibrator,
            )
            if report:
                results[symbol] = report
                success.append(symbol)
            else:
                failed.append(symbol)
        except Exception as e:
            logger.exception(f"[{symbol}] Backtest error: {e}")
            failed.append(symbol)

    if not results:
        logger.error("Tidak ada hasil backtest — semua koin gagal.")
        return

    # ── Generate trade charts per symbol ──────────────────────────────────────
    trades_all = {}
    for symbol, report in results.items():
        trades_df = report.pop("trades_df", pd.DataFrame())
        trades_all[symbol] = trades_df

        if not trades_df.empty:
            # Load df_valid untuk chart (ambil dari labeled)
            res = load_symbol(symbol, feat_cols)
            if res is not None:
                df_sym, _ = res
                valid_idx = np.where(
                    np.ones(len(df_sym), dtype=bool)
                )[0]
                # Ambil valid subset (sama logika seperti backtest_symbol)
                folds     = build_purged_folds(len(df_sym))
                oof_valid = np.zeros(len(df_sym), dtype=bool)
                for _, te_pos in folds:
                    if len(te_pos) > PURGE_GAP_BARS:
                        oof_valid[te_pos[PURGE_GAP_BARS:]] = True
                    else:
                        oof_valid[te_pos] = True
                df_valid = df_sym.iloc[np.where(oof_valid)[0]]

                plot_trade_chart(
                    df_valid        = df_valid,
                    trades_df       = trades_df,
                    symbol          = symbol,
                    run_dir         = run_dir,
                    max_trades_plot = 80,
                )

    # ── Summary chart semua koin ──────────────────────────────────────────────
    plot_summary_chart(results, trades_all, run_dir)

    # ── Aggregate metrics ─────────────────────────────────────────────────────
    all_wr   = [r["winrate"]             for r in results.values()]
    all_tpm  = [r["trade_per_month"]     for r in results.values()]
    all_mcl  = [r["max_consecutive_loss"] for r in results.values()]
    all_pnl3 = [r.get("pnl_lev3x", 0)   for r in results.values()]
    all_pnl5 = [r.get("pnl_lev5x", 0)   for r in results.values()]
    all_dd3  = [r.get("max_drawdown_lev3x", 0) for r in results.values()]

    aggregate = {
        "run_id":               run_id,
        "coins":                coins,
        "success":              success,
        "failed":               failed,
        "n_coins":              len(success),
        "confidence_threshold": CONFIDENCE_THRESHOLD_ENTRY,
        "modal_per_trade":      MODAL_PER_TRADE,
        "leverage_sim":         LEVERAGE_SIM,
        "mean_winrate":         round(float(np.mean(all_wr)),   4),
        "std_winrate":          round(float(np.std(all_wr)),    4),
        "mean_trade_per_month": round(float(np.mean(all_tpm)), 2),
        "mean_pnl_lev3x":       round(float(np.mean(all_pnl3)), 2),
        "mean_pnl_lev5x":       round(float(np.mean(all_pnl5)), 2),
        "mean_drawdown_lev3x":  round(float(np.mean(all_dd3)), 4),
        "max_consecutive_loss": int(max(all_mcl)),
        "per_symbol":           results,
    }

    # ── Simpan backtest results ───────────────────────────────────────────────
    out_path = run_dir / "backtest_results.json"
    with open(out_path, "w") as f:
        json.dump(aggregate, f, indent=2, default=str)
    logger.info(f"Backtest results → {out_path}")

    # ── Update model registry ─────────────────────────────────────────────────
    update_model_metrics(
        "ensemble_v2",
        winrate              = aggregate["mean_winrate"],
        trade_per_month      = aggregate["mean_trade_per_month"],
        pnl_lev3x            = aggregate["mean_pnl_lev3x"],
        pnl_lev5x            = aggregate["mean_pnl_lev5x"],
        max_drawdown         = aggregate["mean_drawdown_lev3x"],
        max_consecutive_loss = aggregate["max_consecutive_loss"],
        status               = "active",
    )
    logger.info("Model registry updated.")

    # ── Generate inference_config.json ★ BARU ─────────────────────────────────
    inference_cfg      = generate_inference_config(aggregate, results, feat_cols)
    inference_cfg_path = MODEL_DIR / "inference_config.json"
    with open(inference_cfg_path, "w") as f:
        json.dump(inference_cfg, f, indent=2, default=str)
    logger.info(f"Inference config → {inference_cfg_path}")

    # ── Print summary ─────────────────────────────────────────────────────────
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  BACKTEST SELESAI — {run_id}")
    print(f"{sep}")
    print(f"  Koin berhasil  : {len(success)} — {success}")
    print(f"  Koin gagal     : {len(failed)}  — {failed}")
    print(f"{sep}")
    print(f"  {'Metric':<28}  {'Value':>10}")
    print(f"  {'-'*28}  {'-'*10}")
    print(f"  {'Mean Winrate':<28}  {aggregate['mean_winrate']:>10.2%}")
    print(f"  {'Mean Trade/Bulan':<28}  {aggregate['mean_trade_per_month']:>10.1f}")
    print(f"  {'Mean PnL Lev3x (USD)':<28}  {aggregate['mean_pnl_lev3x']:>+10.2f}")
    print(f"  {'Mean PnL Lev5x (USD)':<28}  {aggregate['mean_pnl_lev5x']:>+10.2f}")
    print(f"  {'Mean Max Drawdown Lev3x':<28}  {aggregate['mean_drawdown_lev3x']:>10.2%}")
    print(f"  {'Max Consecutive Loss':<28}  {aggregate['max_consecutive_loss']:>10}")
    print(f"{sep}")
    print(f"\n  Per-symbol winrate:")
    for sym, r in results.items():
        bar = "█" * int(r["winrate"] * 20)
        print(f"  {sym:<14} {r['winrate']:.2%}  {bar}")
    print(f"\n  Backtest : {out_path}")
    print(f"  Inference config : {inference_cfg_path}")
    print(f"  Trade charts     : {run_dir}/<SYMBOL>_trade_chart.svg")
    print(f"  Summary chart    : {run_dir}/backtest_summary_chart.svg")

    # Print coin categories
    cv = inference_cfg["coins_validated"]
    print(f"\n  Koin recommended : {cv['recommended']}")
    print(f"  Koin acceptable  : {cv['acceptable']}")
    print(f"  Koin caution     : {cv['caution']}")
    print(f"{sep}\n")


if __name__ == "__main__":
    main()