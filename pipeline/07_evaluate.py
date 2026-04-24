"""
pipeline/07_evaluate.py — Fase 7: Evaluasi Model + SHAP Analysis

Jalankan:
  python pipeline/07_evaluate.py                 # SHAP analysis LGBM
  python pipeline/07_evaluate.py --run-id my_run

Output: models/runs/{run_id}/shap_importance.png, shap_summary.png, shap_ranking.json
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

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, LABEL_DIR, MODEL_DIR, REPORT_DIR,
    LABEL_MAP, NUM_CLASSES, MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE,
    TP_ATR_MULT, SL_ATR_MULT, CONFIDENCE_THRESHOLD_ENTRY,
    SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL, MAX_HOLDING_BARS,
)
from core.utils import setup_logger, update_model_metrics
from core.evaluator import full_trading_report

logger = setup_logger("07_evaluate")

SAMPLE_SYMBOL = "SOLUSDT"
SAMPLE_N      = 10_000
SAMPLE_SEED   = 42
TOP_N         = 20


def load_data(symbol: str) -> pd.DataFrame:
    path = LABEL_DIR / f"{symbol}_features_v3.parquet"
    df   = pd.read_parquet(path)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df.sort_index()


def set_style():
    plt.style.use("dark_background")
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11,
        "figure.facecolor": "#0d1117", "axes.facecolor": "#161b22",
        "axes.edgecolor": "#30363d", "grid.color": "#21262d",
        "text.color": "#e6edf3", "axes.labelcolor": "#e6edf3",
    })


def run_shap(run_id: str):
    run_dir = MODEL_DIR / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODEL_DIR / "lgbm_baseline.pkl"
    cv_path    = MODEL_DIR / "cv_results.json"
    if not model_path.exists():
        raise FileNotFoundError(f"Model tidak ditemukan: {model_path}")

    model = joblib.load(model_path)
    logger.info(f"Model loaded: {model_path}")

    # ★ v2: prioritas baca dari feature_cols_v2.json
    feat_cols_v2_path = MODEL_DIR / "feature_cols_v2.json"
    feat_cols = None
    if feat_cols_v2_path.exists():
        with open(feat_cols_v2_path) as f:
            feat_cols = json.load(f)
        logger.info(f"Feature cols v2 loaded: {len(feat_cols)} fitur")
    elif cv_path.exists():
        with open(cv_path) as f:
            feat_cols = json.load(f).get("feature_cols")

    df = load_data(SAMPLE_SYMBOL)
    if "OB_price" in df.columns:
        df.drop(columns=["OB_price"], inplace=True)

    mask = df["label"].astype(str).isin(LABEL_MAP)
    df   = df[mask].copy()

    if feat_cols is None:
        feat_cols = [c for c in df.columns if c != "label"]
    feat_cols = [c for c in feat_cols if c in df.columns]

    X = df[feat_cols].ffill().fillna(0)
    rng        = np.random.default_rng(SAMPLE_SEED)
    sample_idx = np.sort(rng.choice(len(X), size=min(SAMPLE_N, len(X)), replace=False))
    X_sample   = X.iloc[sample_idx]

    logger.info(f"SHAP calculation: {len(X_sample):,} samples × {len(feat_cols)} features")
    explainer   = shap.TreeExplainer(model.booster_)
    shap_values = explainer.shap_values(X_sample)

    if isinstance(shap_values, list):
        shap_arr = np.stack(shap_values, axis=2)
    else:
        shap_arr = shap_values

    mean_abs = np.mean(np.abs(shap_arr), axis=(0, 2))
    imp_df = pd.DataFrame({
        "feature": feat_cols, "mean_abs_shap": mean_abs
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp_df["rank"] = imp_df.index + 1

    # Print top-N
    sep = "=" * 55
    print(f"\n{sep}")
    print(f"  TOP-{TOP_N} FEATURES (Mean |SHAP|) — {SAMPLE_SYMBOL}")
    print(f"{sep}")
    print(f"  {'Rank':>4}  {'Feature':<28}  {'Mean |SHAP|':>11}")
    print(f"  {'-'*4}  {'-'*28}  {'-'*11}")
    for _, row in imp_df.head(TOP_N).iterrows():
        print(f"  {int(row['rank']):>4}  {row['feature']:<28}  {row['mean_abs_shap']:>11.6f}")
    print(f"{sep}\n")

    # Simpan JSON
    ranking_path = run_dir / "shap_ranking.json"
    ranking_list = [{"rank": int(r["rank"]), "feature": r["feature"],
                     "mean_abs_shap": float(r["mean_abs_shap"])}
                    for _, r in imp_df.iterrows()]
    with open(ranking_path, "w") as f:
        json.dump({"symbol": SAMPLE_SYMBOL, "n": len(X_sample),
                   "n_features": len(feat_cols), "ranking": ranking_list}, f, indent=2)
    # Copy ke root
    with open(MODEL_DIR / "shap_ranking.json", "w") as f:
        json.dump({"symbol": SAMPLE_SYMBOL, "n": len(X_sample),
                   "n_features": len(feat_cols), "ranking": ranking_list}, f, indent=2)
    logger.info(f"SHAP ranking → {ranking_path}")

    # ★ v2: Trading metrics + PnL simulation
    logger.info("Menghitung trading metrics v2...")
    y_actual = df["label"].map(LABEL_MAP).values
    lgbm_model = joblib.load(MODEL_DIR / "lgbm_baseline.pkl")
    y_pred  = lgbm_model.predict(X)
    atr_arr = df["atr_14_h1"].ffill().fillna(0).values if "atr_14_h1" in df.columns \
          else np.ones(len(df))
    close_arr = df["close"].ffill().fillna(1).values    if "close"      in df.columns else np.ones(len(df))
    h4_sh_arr = df["h4_swing_high"].values if "h4_swing_high" in df.columns else None
    h4_sl_arr = df["h4_swing_low"].values  if "h4_swing_low"  in df.columns else None
    high_arr  = df["high"].values if "high" in df.columns else close_arr
    low_arr   = df["low"].values if "low" in df.columns else close_arr

    trading_report = full_trading_report(
        y_pred          = y_pred,
        y_actual        = y_actual,
        atr             = atr_arr,
        close           = close_arr,
        high            = high_arr,
        low             = low_arr,
        h4_swing_highs  = h4_sh_arr,
        h4_swing_lows   = h4_sl_arr,
        index           = df.index,
        modal           = MODAL_PER_TRADE,
        leverages       = LEVERAGE_SIM,
        fee_per_side    = FEE_PER_SIDE,
        min_rr          = SWING_LABEL_MIN_RR,
        min_tp_atr      = SWING_LABEL_MIN_TP,
        max_sl_atr      = SWING_LABEL_MAX_SL,
        max_hold        = MAX_HOLDING_BARS,
        symbol          = SAMPLE_SYMBOL,
    )

    # Simpan trading report ke run dir
    trading_path = run_dir / "trading_metrics.json"
    with open(trading_path, "w") as f:
        json.dump(trading_report, f, indent=2, default=str)
    logger.info(f"Trading metrics → {trading_path}")

    # Update model registry
    update_model_metrics(
        "ensemble_v2",
        f1_macro    = float(imp_df["mean_abs_shap"].mean()),  # placeholder sampai 08_backtest
        winrate     = trading_report["winrate"],
        trade_per_month = trading_report["trade_per_month"],
        pnl_lev3x   = trading_report.get("pnl_lev3x"),
        pnl_lev5x   = trading_report.get("pnl_lev5x"),
        max_drawdown = trading_report.get("max_drawdown_lev3x"),
        max_consecutive_loss = trading_report["max_consecutive_loss"],
        trained_date = datetime.now().strftime("%Y-%m-%d"),
        status       = "active",
    )
    logger.info("Model registry updated.")

    # Bar plot
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
    ax.set_title(f"Top-{TOP_N} Feature Importance — {SAMPLE_SYMBOL} (n={len(X_sample):,})")
    ax.grid(axis="x", linewidth=0.5, alpha=0.5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    plt.tight_layout()
    bar_path = run_dir / "shap_importance.png"
    fig.savefig(bar_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    # Copy ke reports
    fig.savefig(REPORT_DIR / "eda" / "shap_importance.png", dpi=150,
                bbox_inches="tight") if (REPORT_DIR / "eda").exists() else None
    logger.info(f"Bar plot → {bar_path}")

    print(f"\nOutput:")
    print(f"  JSON   : {ranking_path}")
    print(f"  Plot   : {bar_path}")
    print(f"  Top-1  : {imp_df.iloc[0]['feature']} ({imp_df.iloc[0]['mean_abs_shap']:.6f})")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default=None)
    return parser.parse_args()


def main():
    args   = parse_args()
    run_id = args.run_id or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_shap(run_id)


if __name__ == "__main__":
    main()
