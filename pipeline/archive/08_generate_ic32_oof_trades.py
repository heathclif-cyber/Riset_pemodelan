"""
pipeline/08_generate_ic32_oof_trades.py
Generate ic32 OOF trade outcomes untuk binary meta-labeling.

Spec: pipeline/meta_label_spec.json

Walk-forward OOF (no leakage):
  1. Train LGBM per fold (ic32_regime_v1 config)
  2. Predict proba hanya pada val fold
  3. Entry kandidat: max(p_long, p_short) >= candidate_thr (default 0.45)
  4. Simulate outcome: SL=1.5xATR, max_hold=36, no TP (label sim — holdout pakai full stack)
  5. Label: WIN=1 jika net_pnl > 0

Output: data/meta_labels/ic32_oof_trades.parquet

Usage:
  python pipeline/08_generate_ic32_oof_trades.py
  python pipeline/08_generate_ic32_oof_trades.py --candidate-thr 0.45
  python pipeline/08_generate_ic32_oof_trades.py --production-thr
"""
import argparse
import json
import sys
import warnings
from pathlib import Path
from datetime import datetime

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, LABEL_DIR, MODEL_DIR,
    LGBM_PARAMS, LGBM_EARLY_STOPPING, LGBM_CLASS_WEIGHTS,
    N_FOLDS, PURGE_GAP_BARS, LABEL_MAP,
    LGBM_THRESHOLD_LONG, LGBM_THRESHOLD_SHORT,
    LEVERAGE_SIM, MODAL_PER_TRADE, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    TRAIN_CUTOFF_DATE, TP_SL_FALLBACK_SL, MAX_HOLDING_BARS,
)
from pipeline.shared import build_purged_folds
from core.meta_labeling import load_spec, directional_candidate, build_meta_row
from core.utils import setup_logger

logger = setup_logger("08_oof_trades")
SPEC = load_spec()

# ── Config ───────────────────────────────────────────────────────────────────
LEVERAGE = LEVERAGE_SIM[0] if isinstance(LEVERAGE_SIM, list) else LEVERAGE_SIM
SL_MULT  = TP_SL_FALLBACK_SL    # 1.5
MAX_HOLD = MAX_HOLDING_BARS      # 36
COST_RT  = (FEE_PER_SIDE + SLIPPAGE_PER_SIDE) * 2

OUT_DIR  = ROOT / "data" / "meta_labels"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "ic32_oof_trades.parquet"

# Exact 33-feature set dari ic32_regime_v1
with open(MODEL_DIR / "runs" / "ic32_regime_v1" / "lgbm_cv_results.json") as f:
    _cv_meta = json.load(f)
IC32_FEATS = _cv_meta["feature_cols"]

META_CONTEXT = [f for f in SPEC["meta"]["features"]
                if f not in ("p_short", "p_flat", "p_long", "confidence", "direction")]

logger.info(f"ic32 features: {len(IC32_FEATS)}")
logger.info(f"Meta context features: {META_CONTEXT}")
logger.info(f"Simulation: SL={SL_MULT}xATR | max_hold={MAX_HOLD} bar | leverage={LEVERAGE}x")


# ── Data Loading ──────────────────────────────────────────────────────────────

def load_all_coins() -> pd.DataFrame:
    """Load all training coins into a combined sorted DataFrame."""
    frames = []
    for coin in TRAINING_COINS:
        path = LABEL_DIR / f"{coin}_features_v3.parquet"
        if not path.exists():
            logger.warning(f"{coin}: not found, skip")
            continue

        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[df.index < TRAIN_CUTOFF_DATE]

        # Merge HMM regime (same as 04_train_lgbm.py)
        regime_path = LABEL_DIR / f"{coin}_regime_h1.parquet"
        if regime_path.exists():
            try:
                reg = pd.read_parquet(regime_path)
                if "hmm_regime_enc" in df.columns:
                    df = df.drop(columns=["hmm_regime_enc"])
                df = df.join(reg[["hmm_regime_enc"]], how="left")
                df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
            except Exception as e:
                logger.warning(f"{coin}: regime merge failed ({e})")

        df["_coin"] = coin
        frames.append(df)
        logger.info(f"  {coin}: {len(df):,} rows")

    combined = pd.concat(frames).sort_index()
    logger.info(f"Combined: {len(combined):,} rows, {combined['_coin'].nunique()} coins")
    return combined


# ── Trade Simulation ──────────────────────────────────────────────────────────

def simulate_coin(
    df_coin: pd.DataFrame,
    proba: np.ndarray,
    candidate_thr: float,
    use_production_thr: bool,
) -> list[dict]:
    """
    Simulate OOF trades for one coin in a val fold.

    df_coin: subset with OHLCV + features
    proba  : (n, 3) — SHORT=0, FLAT=1, LONG=2
    """
    close = df_coin["close"].values.astype(np.float64)
    high  = df_coin["high"].values.astype(np.float64)
    low   = df_coin["low"].values.astype(np.float64)
    atr   = df_coin["atr_14_h1"].values.astype(np.float64)
    n     = len(df_coin)

    for k in range(1, n):
        if np.isnan(atr[k]):
            atr[k] = atr[k - 1]

    trades = []
    i = 0
    while i < n:
        p = proba[i]
        p_short, p_flat, p_long = float(p[0]), float(p[1]), float(p[2])

        if use_production_thr:
            if p_long >= LGBM_THRESHOLD_LONG and p_long > p_short:
                direction, sig = 1, 2
            elif p_short >= LGBM_THRESHOLD_SHORT and p_short > p_long:
                direction, sig = -1, 0
            else:
                i += 1
                continue
            conf = p_long if sig == 2 else p_short
        else:
            sig, conf = directional_candidate(p, candidate_thr)
            if sig == -1:
                i += 1
                continue
            direction = 1 if sig == 2 else -1

        if np.isnan(atr[i]) or atr[i] <= 0:
            i += 1
            continue

        entry      = close[i]
        sl_price   = entry - direction * SL_MULT * atr[i]
        exit_i     = min(i + MAX_HOLD, n - 1)
        exit_price = close[exit_i]

        for j in range(i + 1, min(i + MAX_HOLD + 1, n)):
            if direction == 1 and low[j] <= sl_price:
                exit_price = sl_price
                exit_i = j
                break
            if direction == -1 and high[j] >= sl_price:
                exit_price = sl_price
                exit_i = j
                break

        ret     = (exit_price - entry) / entry * direction
        net_pnl = ret * MODAL_PER_TRADE * LEVERAGE - COST_RT * MODAL_PER_TRADE * LEVERAGE
        win     = 1 if net_pnl > 0 else 0

        row_series = df_coin.iloc[i]
        meta_feats = build_meta_row(p, sig, row_series, META_CONTEXT)

        record = {
            "timestamp": df_coin.index[i],
            "direction": direction,
            "win": win,
            "net_pnl": round(float(net_pnl), 6),
            "hold_bars": exit_i - i,
            **meta_feats,
        }
        for feat in IC32_FEATS:
            if feat in df_coin.columns and feat not in record:
                record[feat] = float(row_series[feat])

        trades.append(record)
        i = exit_i + 1

    return trades


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate ic32 OOF meta-label dataset")
    parser.add_argument(
        "--candidate-thr", type=float,
        default=float(SPEC["primary"]["candidate_mode"]["candidate_thr"]),
        help="Min max(p_long,p_short) for meta training candidates",
    )
    parser.add_argument(
        "--production-thr", action="store_true",
        help="Use production LGBM thresholds instead of candidate_thr",
    )
    args = parser.parse_args()

    t_start = datetime.now()
    logger.info("=" * 60)
    logger.info("  IC32 OOF Trade Generation (meta-labeling)")
    logger.info(f"  Coins: {len(TRAINING_COINS)} | Folds: {N_FOLDS} | Purge: {PURGE_GAP_BARS}")
    if args.production_thr:
        logger.info(f"  Entry mode: production LONG>={LGBM_THRESHOLD_LONG} SHORT>={LGBM_THRESHOLD_SHORT}")
    else:
        logger.info(f"  Entry mode: candidate_thr={args.candidate_thr}")
    logger.info("=" * 60)

    # Load all coins
    combined = load_all_coins()

    # Encode labels
    mask = combined["label"].isin(LABEL_MAP)
    combined = combined[mask].copy()
    y_all = combined["label"].map(LABEL_MAP).astype(np.int32)

    # Feature matrix (only IC32 features that exist)
    feat_cols_present = [c for c in IC32_FEATS if c in combined.columns]
    missing = set(IC32_FEATS) - set(feat_cols_present)
    if missing:
        logger.warning(f"  Missing features: {sorted(missing)}")
    X_all = combined[feat_cols_present].ffill().fillna(0.0)

    logger.info(f"  X shape: {X_all.shape} | labels: {y_all.value_counts().to_dict()}")

    # Build folds (same structure as 04_train_lgbm.py)
    folds = build_purged_folds(X_all.index, N_FOLDS, PURGE_GAP_BARS)

    all_trades = []
    fold_stats = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        fold_num = fold_idx + 1
        X_tr = X_all.iloc[train_idx]
        y_tr = y_all.iloc[train_idx]
        X_val = X_all.iloc[val_idx]
        y_val = y_all.iloc[val_idx]
        val_combined = combined.iloc[val_idx]  # includes _coin, OHLCV, ATR

        logger.info(f"\nFold {fold_num}/{N_FOLDS} | train={len(X_tr):,} val={len(X_val):,}")

        # Train LGBM from scratch (same as ic32_regime_v1 config)
        sample_w = np.array([LGBM_CLASS_WEIGHTS[int(l)] for l in y_tr], dtype=np.float32)

        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(
            X_tr, y_tr,
            sample_weight=sample_w,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(LGBM_EARLY_STOPPING, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        val_f1 = f1_score(y_val, model.predict(X_val), average="macro", zero_division=0)
        logger.info(f"  LGBM trained | best_iter={model.best_iteration_} | val_f1={val_f1:.4f}")

        # Predict proba on val — shape (n_val, 3)
        proba_val = model.predict_proba(X_val).astype(np.float32)

        # Simulate per coin
        fold_trades = []
        for coin in TRAINING_COINS:
            coin_mask = val_combined["_coin"] == coin
            if coin_mask.sum() == 0:
                continue

            df_coin = val_combined[coin_mask]
            proba_coin = proba_val[coin_mask.values]

            coin_trades = simulate_coin(
                df_coin, proba_coin,
                candidate_thr=args.candidate_thr,
                use_production_thr=args.production_thr,
            )
            for t in coin_trades:
                t["coin"] = coin
                t["fold"] = fold_num
            fold_trades.extend(coin_trades)

        n_win  = sum(t["win"] == 1 for t in fold_trades)
        n_tot  = len(fold_trades)
        wr     = n_win / n_tot * 100 if n_tot > 0 else 0.0
        logger.info(f"  Fold {fold_num} trades: {n_tot} | WR: {wr:.1f}%")
        fold_stats.append({"fold": fold_num, "trades": n_tot, "wr": round(wr, 2)})
        all_trades.extend(fold_trades)

    # ── Save ─────────────────────────────────────────────────────────────────
    df_trades = pd.DataFrame(all_trades)
    df_trades.to_parquet(OUT_PATH, index=False)

    total = len(df_trades)
    total_wr = df_trades["win"].mean() * 100
    elapsed  = (datetime.now() - t_start).seconds // 60

    print("\n" + "=" * 60)
    print("  OOF TRADE GENERATION COMPLETE")
    print("=" * 60)
    print(f"  Total OOF trades : {total:,}")
    print(f"  Overall WR       : {total_wr:.1f}%")
    print(f"  LONG trades      : {(df_trades['direction']==1).sum():,}")
    print(f"  SHORT trades     : {(df_trades['direction']==-1).sum():,}")
    print(f"  Elapsed          : {elapsed} min")
    print(f"  Saved to         : {OUT_PATH}")
    print("-" * 60)
    print("  Per-fold summary:")
    for s in fold_stats:
        print(f"    Fold {s['fold']}: {s['trades']:>5} trades | WR {s['wr']:.1f}%")
    print("=" * 60)

    # Metadata
    meta = {
        "spec_version": SPEC["version"],
        "generated_at": datetime.now().isoformat(),
        "n_trades": total,
        "overall_wr": round(total_wr, 2),
        "n_folds": N_FOLDS,
        "purge_gap": PURGE_GAP_BARS,
        "ic32_features": feat_cols_present,
        "meta_features": SPEC["meta"]["features"],
        "entry_mode": "production" if args.production_thr else "candidate",
        "candidate_thr": None if args.production_thr else args.candidate_thr,
        "simulation": {
            "sl_mult": SL_MULT,
            "max_hold": MAX_HOLD,
            "thr_long": LGBM_THRESHOLD_LONG,
            "thr_short": LGBM_THRESHOLD_SHORT,
            "leverage": LEVERAGE,
            "modal": MODAL_PER_TRADE,
            "guardian": False,
            "tp": "none",
        },
        "fold_stats": fold_stats,
        "leak_check": "OOF — model retrained per fold, labels only from val predictions",
        "note": "Domain-matched binary meta labels for ic32_meta_v1. "
                "Holdout ablation uses full_trading_report (swing+Guardian).",
    }
    with open(OUT_DIR / "ic32_oof_trades_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info("Meta saved.")


if __name__ == "__main__":
    main()
