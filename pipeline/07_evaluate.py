"""
pipeline/07_evaluate.py — Fase 7: Multi-Model Evaluation + SHAP Analysis

Mengevaluasi semua komponen dalam hierarchical cascade:
  1. H4 LGBM  — SHAP + regime classification metrics (accuracy per direction)
  2. H1 LGBM  — SHAP + entry precision/recall untuk LONG dan SHORT
  3. Cascade  — end-to-end trading simulation (winrate, Sharpe, drawdown)

Perbaikan dari versi lama:
  ❌ Sebelumnya: hanya evaluasi H1 LGBM standalone (blind spot — path production berbeda)
  ✅ Sekarang  : evaluasi seluruh hierarki sesuai arsitektur actual inference

Jalankan:
  python pipeline/07_evaluate.py                 # evaluasi lengkap
  python pipeline/07_evaluate.py --run-id my_run
  python pipeline/07_evaluate.py --skip-h4       # skip H4 jika belum dilatih
  python pipeline/07_evaluate.py --skip-cascade  # hanya SHAP, tanpa trading sim

Output: models/runs/{run_id}/
  shap_h1_ranking.json     ← H1 LGBM SHAP importance
  shap_h4_ranking.json     ← H4 LGBM SHAP importance (jika tersedia)
  cascade_metrics.json     ← end-to-end cascade trading metrics
  shap_h1_importance.png
  shap_h4_importance.png
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, LABEL_DIR, MODEL_DIR, REPORT_DIR,
    LABEL_MAP, NUM_CLASSES, MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE,
    SLIPPAGE_PER_SIDE,
    TP_ATR_MULT, SL_ATR_MULT, CONFIDENCE_THRESHOLD_ENTRY,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL, MAX_HOLDING_BARS,
    H4_FEATURE_COLS, LSTM_SEQ_LEN, LSTM_HIDDEN, LSTM_LAYERS, LSTM_DROPOUT,
)
from core.utils import setup_logger, update_model_metrics
from core.evaluator import full_trading_report
from core.models import load_lstm
from pipeline.backtest_utils import hierarchical_predict

logger = setup_logger("07_evaluate")

SAMPLE_SYMBOL = "SOLUSDT"
SAMPLE_N      = 10_000
SAMPLE_SEED   = 42
TOP_N         = 20
DEVICE        = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_data(symbol: str) -> pd.DataFrame:
    path = LABEL_DIR / f"{symbol}_features_v3.parquet"
    df   = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    df = df.sort_index()
    if "OB_price" in df.columns:
        df.drop(columns=["OB_price"], inplace=True)
    return df


def set_style():
    plt.style.use("dark_background")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d", "grid.color": "#21262d",
        "text.color": "#e6edf3", "axes.labelcolor": "#e6edf3",
    })


def _shap_bar_plot(imp_df: pd.DataFrame, title: str, out_path: Path):
    """Render SHAP bar plot dan simpan ke out_path."""
    set_style()
    top20   = imp_df.head(TOP_N)
    colors  = plt.cm.plasma(np.linspace(0.2, 0.85, len(top20)))[::-1]
    fig, ax = plt.subplots(figsize=(12, 8))
    bars = ax.barh(top20["feature"][::-1], top20["mean_abs_shap"][::-1],
                   color=colors, edgecolor="none", height=0.7)
    for bar, val in zip(bars, top20["mean_abs_shap"][::-1]):
        ax.text(bar.get_width() + top20["mean_abs_shap"].max() * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.5f}", va="center", ha="left", fontsize=9, color="#8b949e")
    ax.set_xlabel("Mean |SHAP Value|")
    ax.set_title(title)
    ax.grid(axis="x", linewidth=0.5, alpha=0.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    logger.info(f"Plot → {out_path}")


def _compute_shap(model, X_sample: pd.DataFrame, feat_cols: list[str]) -> pd.DataFrame:
    """Hitung mean |SHAP| per fitur, return DataFrame sorted descending."""
    explainer   = shap.TreeExplainer(model.booster_)
    shap_values = explainer.shap_values(X_sample)
    if isinstance(shap_values, list):
        shap_arr = np.stack(shap_values, axis=2)
    else:
        shap_arr = shap_values
    mean_abs = np.mean(np.abs(shap_arr), axis=(0, 2)) if shap_arr.ndim == 3 \
               else np.mean(np.abs(shap_arr), axis=0)
    imp_df = pd.DataFrame({
        "feature": feat_cols, "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp_df["rank"] = imp_df.index + 1
    return imp_df


# ─── STEP 1: H1 LGBM SHAP ────────────────────────────────────────────────────

def evaluate_h1_lgbm(run_dir: Path) -> dict:
    logger.info("=== EVALUASI H1 LGBM ===")

    model_path = MODEL_DIR / "lgbm_baseline.pkl"
    if not model_path.exists():
        logger.warning("lgbm_baseline.pkl tidak ditemukan — skip H1 LGBM eval")
        return {}

    model = joblib.load(model_path)
    feat_cols_path = MODEL_DIR / "feature_cols_v2.json"
    with open(feat_cols_path) as f:
        feat_cols = json.load(f)

    df = load_data(SAMPLE_SYMBOL)
    mask = df["label"].astype(str).isin(LABEL_MAP)
    df   = df[mask].copy()
    feat_cols = [c for c in feat_cols if c in df.columns]
    X    = df[feat_cols].ffill().fillna(0)

    rng        = np.random.default_rng(SAMPLE_SEED)
    sample_idx = np.sort(rng.choice(len(X), size=min(SAMPLE_N, len(X)), replace=False))
    X_sample   = X.iloc[sample_idx]

    logger.info(f"H1 SHAP: {len(X_sample):,} samples × {len(feat_cols)} features")
    imp_df = _compute_shap(model, X_sample, feat_cols)

    # Entry precision/recall per direction
    y_actual = df["label"].map(LABEL_MAP).values
    y_pred   = model.predict(X)
    from sklearn.metrics import precision_score, recall_score, f1_score
    prec = precision_score(y_actual, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
    rec  = recall_score(y_actual, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
    f1p  = f1_score(y_actual, y_pred, average=None, zero_division=0, labels=[0, 1, 2])

    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  H1 LGBM — Entry Signal Evaluation ({SAMPLE_SYMBOL})")
    print(f"{sep}")
    print(f"  {'Class':<8}  {'Precision':>9}  {'Recall':>9}  {'F1':>9}")
    print(f"  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*9}")
    for i, name in enumerate(["SHORT", "FLAT", "LONG"]):
        print(f"  {name:<8}  {prec[i]:>9.4f}  {rec[i]:>9.4f}  {f1p[i]:>9.4f}")
    print(f"\n  TOP-{TOP_N} FEATURES (H1 LGBM):")
    for _, row in imp_df.head(TOP_N).iterrows():
        print(f"  {int(row['rank']):>4}  {row['feature']:<28}  {row['mean_abs_shap']:>11.6f}")
    print(f"{sep}\n")

    ranking_list = [{"rank": int(r["rank"]), "feature": r["feature"],
                     "mean_abs_shap": float(r["mean_abs_shap"])} for _, r in imp_df.iterrows()]
    out = {
        "model": "h1_lgbm", "symbol": SAMPLE_SYMBOL, "n_samples": len(X_sample),
        "n_features": len(feat_cols),
        "entry_metrics": {
            "SHORT": {"precision": round(float(prec[0]), 4), "recall": round(float(rec[0]), 4), "f1": round(float(f1p[0]), 4)},
            "FLAT":  {"precision": round(float(prec[1]), 4), "recall": round(float(rec[1]), 4), "f1": round(float(f1p[1]), 4)},
            "LONG":  {"precision": round(float(prec[2]), 4), "recall": round(float(rec[2]), 4), "f1": round(float(f1p[2]), 4)},
        },
        "shap_ranking": ranking_list,
    }

    rank_path = run_dir / "shap_h1_ranking.json"
    with open(rank_path, "w") as f:
        json.dump(out, f, indent=2)
    with open(MODEL_DIR / "shap_ranking.json", "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"H1 SHAP ranking → {rank_path}")

    _shap_bar_plot(
        imp_df,
        title    = f"H1 LGBM — Top-{TOP_N} Features ({SAMPLE_SYMBOL}, n={len(X_sample):,})",
        out_path = run_dir / "shap_h1_importance.png",
    )
    return out


# ─── STEP 2: H4 LGBM SHAP ────────────────────────────────────────────────────

def evaluate_h4_lgbm(run_dir: Path) -> dict:
    logger.info("=== EVALUASI H4 LGBM ===")

    model_path = MODEL_DIR / "lgbm_h4.pkl"
    feat_path  = MODEL_DIR / "h4_feature_cols.json"
    if not model_path.exists():
        logger.warning("lgbm_h4.pkl tidak ditemukan — skip. Jalankan 04_train_lgbm_h4.py dulu.")
        return {}

    model = joblib.load(model_path)
    with open(feat_path) as f:
        h4_feat_cols = json.load(f)

    # Import H4 resampler dari trainer H4
    from pipeline.p04_train_lgbm_h4 import load_and_resample_to_h4
    df_h4 = load_and_resample_to_h4(SAMPLE_SYMBOL)
    if df_h4 is None or len(df_h4) < 50:
        logger.warning(f"H4 data untuk {SAMPLE_SYMBOL} terlalu sedikit — skip")
        return {}

    mask = df_h4["label"].astype(str).isin(LABEL_MAP)
    df_h4 = df_h4[mask].copy()
    feat_cols = [c for c in h4_feat_cols if c in df_h4.columns]
    X = df_h4[feat_cols].ffill().fillna(0)

    rng        = np.random.default_rng(SAMPLE_SEED)
    sample_idx = np.sort(rng.choice(len(X), size=min(min(SAMPLE_N, len(X)), 5000), replace=False))
    X_sample   = X.iloc[sample_idx]

    logger.info(f"H4 SHAP: {len(X_sample):,} samples × {len(feat_cols)} features")
    imp_df = _compute_shap(model, X_sample, feat_cols)

    # Regime classification accuracy
    y_actual = df_h4["label"].map(LABEL_MAP).values
    y_pred   = model.predict(X)
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    regime_mask = y_actual != 1  # exclude FLAT
    regime_acc  = float(accuracy_score(y_actual[regime_mask], y_pred[regime_mask])) \
                  if regime_mask.sum() > 0 else 0.0
    prec = precision_score(y_actual, y_pred, average=None, zero_division=0, labels=[0, 1, 2])
    rec  = recall_score(y_actual, y_pred, average=None, zero_division=0, labels=[0, 1, 2])

    label_dist = pd.Series(df_h4["label"]).value_counts(normalize=True).to_dict()

    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  H4 LGBM — Regime Filter Evaluation ({SAMPLE_SYMBOL})")
    print(f"{sep}")
    print(f"  Label distribution: {label_dist}")
    print(f"  Regime accuracy (non-FLAT): {regime_acc:.4f}")
    print(f"  {'Class':<8}  {'Precision':>9}  {'Recall':>9}")
    for i, name in enumerate(["SHORT", "FLAT", "LONG"]):
        print(f"  {name:<8}  {prec[i]:>9.4f}  {rec[i]:>9.4f}")
    print(f"\n  TOP-{TOP_N} FEATURES (H4 LGBM):")
    for _, row in imp_df.head(TOP_N).iterrows():
        print(f"  {int(row['rank']):>4}  {row['feature']:<28}  {row['mean_abs_shap']:>11.6f}")
    print(f"{sep}\n")

    ranking_list = [{"rank": int(r["rank"]), "feature": r["feature"],
                     "mean_abs_shap": float(r["mean_abs_shap"])} for _, r in imp_df.iterrows()]
    out = {
        "model": "h4_lgbm", "symbol": SAMPLE_SYMBOL, "n_samples": len(X_sample),
        "n_features": len(feat_cols),
        "label_distribution": {k: round(float(v), 4) for k, v in label_dist.items()},
        "regime_accuracy_non_flat": round(regime_acc, 4),
        "class_metrics": {
            "SHORT": {"precision": round(float(prec[0]), 4), "recall": round(float(rec[0]), 4)},
            "FLAT":  {"precision": round(float(prec[1]), 4), "recall": round(float(rec[1]), 4)},
            "LONG":  {"precision": round(float(prec[2]), 4), "recall": round(float(rec[2]), 4)},
        },
        "shap_ranking": ranking_list,
    }

    rank_path = run_dir / "shap_h4_ranking.json"
    with open(rank_path, "w") as f:
        json.dump(out, f, indent=2)
    logger.info(f"H4 SHAP ranking → {rank_path}")

    _shap_bar_plot(
        imp_df,
        title    = f"H4 LGBM — Top-{TOP_N} Features ({SAMPLE_SYMBOL}, n={len(X_sample):,})",
        out_path = run_dir / "shap_h4_importance.png",
    )
    return out


# ─── STEP 3: Cascade End-to-End Trading Metrics ───────────────────────────────

def evaluate_cascade(run_dir: Path) -> dict:
    logger.info("=== EVALUASI CASCADE END-TO-END ===")

    required = [
        (MODEL_DIR / "lgbm_baseline.pkl", "H1 LGBM"),
        (MODEL_DIR / "lstm_best.pt",      "LSTM"),
        (MODEL_DIR / "lstm_scaler.pkl",   "LSTM Scaler"),
        (MODEL_DIR / "feature_cols_v2.json", "H1 feature cols"),
    ]
    for path, name in required:
        if not path.exists():
            logger.warning(f"{name} tidak ditemukan — skip cascade eval")
            return {}

    h1_model    = joblib.load(MODEL_DIR / "lgbm_baseline.pkl")
    lstm_model  = load_lstm(MODEL_DIR / "lstm_best.pt", device=str(DEVICE)).to(DEVICE)
    lstm_scaler = joblib.load(MODEL_DIR / "lstm_scaler.pkl")
    with open(MODEL_DIR / "feature_cols_v2.json") as f:
        feat_cols = json.load(f)

    h4_model_path = MODEL_DIR / "lgbm_h4.pkl"
    h4_feat_path  = MODEL_DIR / "h4_feature_cols.json"
    if h4_model_path.exists() and h4_feat_path.exists():
        h4_model = joblib.load(h4_model_path)
        with open(h4_feat_path) as f:
            h4_feat_cols = json.load(f)
        logger.info(f"H4 model loaded: {len(h4_feat_cols)} features")
    else:
        h4_model, h4_feat_cols = None, []
        logger.warning("lgbm_h4.pkl tidak ada — H4 bias = FLAT (partial cascade)")

    df = load_data(SAMPLE_SYMBOL)
    mask = df["label"].astype(str).isin(LABEL_MAP)
    df   = df[mask].copy()
    y    = df["label"].map(LABEL_MAP).values.astype(np.int64)
    valid_cols = [c for c in feat_cols if c in df.columns]
    df[valid_cols] = df[valid_cols].ffill().fillna(0)
    X = df[valid_cols].values.astype(np.float64)

    logger.info(f"Cascade eval: {len(df):,} bars ({SAMPLE_SYMBOL})")

    y_pred, confidence = hierarchical_predict(
        h4_model, h1_model, lstm_model, lstm_scaler,
        X, valid_cols, h4_feat_cols, df[valid_cols],
    )

    # Statistik sinyal
    n_long  = int((y_pred == 2).sum())
    n_short = int((y_pred == 0).sum())
    n_flat  = int((y_pred == 1).sum())
    signal_rate = (n_long + n_short) / len(y_pred)

    logger.info(f"Signal distribution: LONG={n_long} SHORT={n_short} FLAT={n_flat} "
                f"signal_rate={signal_rate:.1%}")

    atr_arr   = df["atr_14_h1"].values if "atr_14_h1" in df.columns else np.ones(len(df))
    close_arr = df["close"].values
    high_arr  = df["high"].values  if "high" in df.columns else close_arr
    low_arr   = df["low"].values   if "low"  in df.columns else close_arr
    h4_sh_arr = df["h4_swing_high"].values if "h4_swing_high" in df.columns else None
    h4_sl_arr = df["h4_swing_low"].values  if "h4_swing_low"  in df.columns else None

    trading_report = full_trading_report(
        y_pred         = y_pred,
        y_actual       = y,
        atr            = atr_arr,
        close          = close_arr,
        high           = high_arr,
        low            = low_arr,
        h4_swing_highs = h4_sh_arr,
        h4_swing_lows  = h4_sl_arr,
        index          = df.index,
        modal          = MODAL_PER_TRADE,
        leverages      = LEVERAGE_SIM,
        fee_per_side   = FEE_PER_SIDE,
        slippage       = SLIPPAGE_PER_SIDE,
        min_rr         = SWING_LABEL_MIN_RR,
        min_tp_atr     = SWING_LABEL_MIN_TP,
        max_sl_atr     = SWING_LABEL_MAX_SL,
        max_hold       = MAX_HOLDING_BARS,
        symbol         = SAMPLE_SYMBOL,
    )

    trading_report["signal_stats"] = {
        "n_long": n_long, "n_short": n_short, "n_flat": n_flat,
        "signal_rate": round(signal_rate, 4),
        "h4_model_active": h4_model is not None,
    }

    cascade_path = run_dir / "cascade_metrics.json"
    with open(cascade_path, "w") as f:
        json.dump(trading_report, f, indent=2, default=str)
    logger.info(f"Cascade metrics → {cascade_path}")

    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  HIERARCHICAL CASCADE — End-to-End ({SAMPLE_SYMBOL})")
    print(f"{sep}")
    print(f"  H4 bias aktif : {'Ya' if h4_model else 'Tidak (fallback FLAT)'}")
    print(f"  Signal rate   : {signal_rate:.1%}  (LONG={n_long} SHORT={n_short} FLAT={n_flat})")
    print(f"  {'Metric':<28}  {'Value':>10}")
    print(f"  {'-'*28}  {'-'*10}")
    print(f"  {'Winrate':<28}  {trading_report.get('winrate', 0):>10.2%}")
    print(f"  {'Trade/Bulan':<28}  {trading_report.get('trade_per_month', 0):>10.1f}")
    print(f"  {'Max DD Lev5x':<28}  {trading_report.get('max_drawdown_lev5x', 0):>10.2%}")
    print(f"  {'Sharpe Ratio':<28}  {trading_report.get('sharpe_ratio', 0):>10.2f}")
    print(f"  {'Profit Factor':<28}  {trading_report.get('profit_factor', 0):>10.2f}")
    print(f"  {'Max Consec Loss':<28}  {trading_report.get('max_consecutive_loss', 0):>10}")
    print(f"{sep}\n")

    update_model_metrics(
        "hierarchical_v1",
        winrate              = trading_report.get("winrate", 0),
        trade_per_month      = trading_report.get("trade_per_month", 0),
        pnl_lev5x            = trading_report.get("pnl_lev5x"),
        max_drawdown         = trading_report.get("max_drawdown_lev5x"),
        sharpe_ratio         = trading_report.get("sharpe_ratio"),
        profit_factor        = trading_report.get("profit_factor"),
        max_consecutive_loss = trading_report.get("max_consecutive_loss", 0),
        trained_date         = datetime.now().strftime("%Y-%m-%d"),
        status               = "active",
    )
    logger.info("Model registry updated → hierarchical_v1")
    return trading_report


# ─── Main ─────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Multi-Model Evaluation Pipeline")
    parser.add_argument("--run-id",       default=None)
    parser.add_argument("--skip-h4",      action="store_true", help="Skip H4 LGBM eval")
    parser.add_argument("--skip-cascade", action="store_true", help="Skip cascade trading sim")
    return parser.parse_args()


def main():
    args   = parse_args()
    run_id = args.run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir = MODEL_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"Run ID: {run_id} | Symbol: {SAMPLE_SYMBOL}")

    # Evaluasi H1 LGBM (selalu)
    h1_result = evaluate_h1_lgbm(run_dir)

    # Evaluasi H4 LGBM (jika ada)
    h4_result = {} if args.skip_h4 else evaluate_h4_lgbm(run_dir)

    # Cascade end-to-end
    cascade_result = {} if args.skip_cascade else evaluate_cascade(run_dir)

    print(f"\n  Output dir: {run_dir}")
    if h1_result:
        print(f"  H1 SHAP  : {run_dir}/shap_h1_ranking.json")
    if h4_result:
        print(f"  H4 SHAP  : {run_dir}/shap_h4_ranking.json")
    if cascade_result:
        print(f"  Cascade  : {run_dir}/cascade_metrics.json")


if __name__ == "__main__":
    main()
