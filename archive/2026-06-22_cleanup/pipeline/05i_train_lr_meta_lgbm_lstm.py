"""
pipeline/05i_train_lr_meta_lgbm_lstm.py
Logistic Regression meta-learner: stacking LGBM + LSTM OOF predictions.

Input features untuk LR:
  lgbm_p0, lgbm_p1, lgbm_p2   - LGBM class probabilities
  lstm_p0, lstm_p1, lstm_p2   - LSTM class probabilities (0.333 jika bukan candidate bar)
  lstm_in_domain              - 1 jika bar masuk LSTM candidate domain, 0 otherwise

Label: swing label asli (SHORT=0, FLAT=1, LONG=2)

Metodologi: rolling walk-forward OOF (8 fold) dengan purge gap = 36 bar.
Scaler fit di dalam fold — tidak ada leakage.

Output:
  models/runs/ic32_rv2_lr_meta_c55/oof_lr_predictions.parquet
  models/runs/ic32_rv2_lr_meta_c55/meta_lr_meta.json
"""
import json, sys, warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, ALL_COINS, LABEL_DIR, TRAIN_CUTOFF_DATE, MODEL_DIR, LABEL_MAP,
    N_FOLDS,
)
from core.evaluator import simulate_trades_swing
from core.utils import setup_logger, ensure_utc_index
from config import (
    MAX_HOLDING_BARS, MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE,
    SLIPPAGE_PER_SIDE, SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP,
    SWING_LABEL_MAX_SL, TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
)

logger = setup_logger("05i_train_lr_meta_lgbm_lstm")

RUN_NAME  = "ic32_rv2_lr_meta_c55"
LGBM_RUN  = "ic32_regime_v2"
LSTM_RUN  = "ic32_lstm_candidate_v3_seq36"
PURGE_GAP = 36
LR_C      = 1.0        # regularization strength (inverse)
LR_SOLVER = "lbfgs"

OUT_DIR   = MODEL_DIR / "runs" / RUN_NAME
LM        = {"SHORT": 0, "FLAT": 1, "LONG": 2}
LM_REV    = {v: k for k, v in LM.items()}

META_FEATURES = [
    "lgbm_p0", "lgbm_p1", "lgbm_p2",
    "lstm_p0", "lstm_p1", "lstm_p2",
    "lstm_in_domain",
]


def load_and_merge_oof(coins) -> pd.DataFrame:
    """Merge LGBM OOF + LSTM OOF + true labels ke satu DataFrame."""
    lgbm_oof = pd.read_parquet(MODEL_DIR / "runs" / LGBM_RUN / "oof_predictions.parquet")
    lstm_oof = pd.read_parquet(MODEL_DIR / "runs" / LSTM_RUN / "oof_lstm_predictions.parquet")

    lgbm_oof.index = pd.to_datetime(lgbm_oof.index, utc=True)
    lstm_oof.index = pd.to_datetime(lstm_oof.index, utc=True)

    # LGBM: hanya bar dengan OOF valid
    lgbm_oof = lgbm_oof[lgbm_oof["has_oof"]].copy()
    lgbm_oof = lgbm_oof.rename(columns={"p0": "lgbm_p0", "p1": "lgbm_p1", "p2": "lgbm_p2"})

    # LSTM: tandai domain
    lstm_oof["lstm_in_domain"] = lstm_oof["has_oof"].astype(float)
    lstm_oof = lstm_oof.rename(columns={"p0": "lstm_p0", "p1": "lstm_p1", "p2": "lstm_p2"})

    frames = []
    for coin in coins:
        fp = LABEL_DIR / f"{coin}_features_v3.parquet"
        if not fp.exists():
            continue
        df = pd.read_parquet(fp).sort_index()
        df = ensure_utc_index(df).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]
        df = df[df["label"].astype(str).isin(LABEL_MAP)].dropna(subset=["label"])
        if df.empty:
            continue

        df["coin"] = coin
        df["y_true"] = df["label"].map(LABEL_MAP).astype(int)
        df = df[["coin", "y_true", "close", "high", "low", "atr_14_h1",
                 "h4_swing_high", "h4_swing_low"]]

        # Join LGBM OOF
        lg = lgbm_oof[lgbm_oof["coin"] == coin][["lgbm_p0", "lgbm_p1", "lgbm_p2"]]
        df = df.join(lg, how="inner")   # inner: hanya bar dengan LGBM OOF

        # Join LSTM OOF
        lt = lstm_oof[lstm_oof["coin"] == coin][["lstm_p0", "lstm_p1", "lstm_p2", "lstm_in_domain"]]
        df = df.join(lt, how="left")

        # Bar tanpa LSTM domain → uniform prior + in_domain=0
        df["lstm_in_domain"] = df["lstm_in_domain"].fillna(0.0)
        df["lstm_p0"] = df["lstm_p0"].fillna(1/3)
        df["lstm_p1"] = df["lstm_p1"].fillna(1/3)
        df["lstm_p2"] = df["lstm_p2"].fillna(1/3)

        frames.append(df)

    if not frames:
        raise RuntimeError("Tidak ada data setelah merge OOF!")

    merged = pd.concat(frames).sort_index()
    logger.info(f"  Merged OOF: {len(merged):,} bars, {merged['coin'].nunique()} coins")
    return merged


def build_purged_rolling_folds(unique_ts, n_folds=N_FOLDS, purge=PURGE_GAP):
    """Rolling expanding-window folds pada level timestamp, dengan purge gap."""
    splits_ts = np.array_split(unique_ts, n_folds + 1)
    folds = []
    for k in range(1, n_folds + 1):
        train_ts = set(np.concatenate(splits_ts[:k]))
        test_ts  = set(splits_ts[k])
        # purge: buang purge bar terakhir train dan purge bar pertama test
        train_ts_sorted = np.sort(list(train_ts))
        test_ts_sorted  = np.sort(list(test_ts))
        train_ts_purged = train_ts_sorted[:-purge] if len(train_ts_sorted) > purge else train_ts_sorted
        test_ts_purged  = test_ts_sorted[purge:]   if len(test_ts_sorted)  > purge else test_ts_sorted
        folds.append((set(train_ts_purged), set(test_ts_purged)))
    return folds


def train_oof(merged: pd.DataFrame):
    unique_ts = np.sort(merged.index.unique())
    folds = build_purged_rolling_folds(unique_ts)

    oof_records = []
    fold_f1s = []

    for fold_idx, (train_ts_set, test_ts_set) in enumerate(folds, 1):
        train_mask = merged.index.isin(train_ts_set)
        test_mask  = merged.index.isin(test_ts_set)

        X_train = merged.loc[train_mask, META_FEATURES].values.astype(np.float32)
        y_train = merged.loc[train_mask, "y_true"].values.astype(int)
        X_test  = merged.loc[test_mask,  META_FEATURES].values.astype(np.float32)
        y_test  = merged.loc[test_mask,  "y_true"].values.astype(int)

        if len(X_train) < 100 or len(X_test) < 10:
            logger.warning(f"  Fold {fold_idx}: skip (train={len(X_train)}, test={len(X_test)})")
            continue

        # Scaler fit hanya pada train
        sc = RobustScaler()
        X_train_sc = sc.fit_transform(X_train)
        X_test_sc  = sc.transform(X_test)

        lr = LogisticRegression(
            C=LR_C, solver=LR_SOLVER, max_iter=500,
            class_weight="balanced", random_state=42,
        )
        lr.fit(X_train_sc, y_train)

        proba = lr.predict_proba(X_test_sc)   # shape (N, 3)
        y_pred = np.argmax(proba, axis=1)

        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        fold_f1s.append(f1)
        logger.info(f"  Fold {fold_idx}: train={len(X_train):,} test={len(X_test):,} F1={f1:.4f}")

        # Simpan OOF predictions
        test_df = merged[test_mask].copy()
        test_df["lr_p0"]       = proba[:, 0]
        test_df["lr_p1"]       = proba[:, 1]
        test_df["lr_p2"]       = proba[:, 2]
        test_df["lr_pred"]     = y_pred
        test_df["fold"]        = fold_idx
        oof_records.append(test_df)

    if not oof_records:
        raise RuntimeError("Tidak ada OOF records!")

    oof_df = pd.concat(oof_records).sort_index()
    mean_f1 = float(np.mean(fold_f1s))
    logger.info(f"  CV F1 macro: {mean_f1:.4f} (folds: {[round(f,4) for f in fold_f1s]})")
    return oof_df, mean_f1, fold_f1s


def evaluate_oof(oof_df: pd.DataFrame):
    """Evaluate OOF dengan argmax LR signal — tanpa threshold sweep."""
    LM_LONG = LM["LONG"]; LM_SHORT = LM["SHORT"]; LM_FLAT = LM["FLAT"]
    agg = dict(trades=0, wins=0, losses=0, pnl=0.0, gw=0.0, gl=0.0)

    for coin in oof_df["coin"].unique():
        cdf = oof_df[oof_df["coin"] == coin].sort_index()
        if len(cdf) < 30:
            continue

        y_pred = cdf["lr_pred"].values.astype(np.int32)
        if (y_pred != LM_FLAT).sum() == 0:
            continue

        h4_sh = cdf["h4_swing_high"].values if "h4_swing_high" in cdf.columns else np.full(len(cdf), np.nan)
        h4_sl = cdf["h4_swing_low"].values  if "h4_swing_low"  in cdf.columns else np.full(len(cdf), np.nan)

        res = simulate_trades_swing(
            y_pred=y_pred, close=cdf["close"].values, high=cdf["high"].values,
            low=cdf["low"].values, atr=cdf["atr_14_h1"].values,
            h4_swing_highs=h4_sh, h4_swing_lows=h4_sl,
            modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            max_hold=MAX_HOLDING_BARS, min_rr=SWING_LABEL_MIN_RR,
            min_tp_atr=SWING_LABEL_MIN_TP, max_sl_atr=SWING_LABEL_MAX_SL,
            tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
            guardian_enabled=False,
        )
        t = res.get("total_trades", 0)
        w = res.get("wins", 0)
        pnl = res.get("total_pnl", 0.0)
        pf  = res.get("profit_factor", 0.0)
        if t > 0 and pf > 0 and abs(pf - 1) > 1e-6:
            gl = abs(pnl / (pf - 1.0)); gw = pf * gl
            agg["gw"] += gw; agg["gl"] += gl
        agg["trades"] += t; agg["wins"] += w; agg["losses"] += t - w; agg["pnl"] += pnl

    if agg["trades"] == 0:
        return {}
    wr = agg["wins"] / agg["trades"]
    pf = agg["gw"] / agg["gl"] if agg["gl"] > 1e-9 else 0.0
    return {
        "trades": agg["trades"], "wr": round(wr, 4),
        "pf": round(pf, 3), "pnl": round(agg["pnl"], 2),
        "pnl_per_trade": round(agg["pnl"] / agg["trades"], 4),
    }


def main():
    logger.info("="*70)
    logger.info("  LR Meta-Learner: LGBM + LSTM Stacking")
    logger.info(f"  LGBM run  : {LGBM_RUN}")
    logger.info(f"  LSTM run  : {LSTM_RUN}")
    logger.info(f"  LR C      : {LR_C}  |  purge gap: {PURGE_GAP} bars")
    logger.info("="*70)

    logger.info("Loading & merging OOF predictions...")
    merged = load_and_merge_oof(TRAINING_COINS)

    logger.info(f"\nTraining LR meta-learner ({N_FOLDS}-fold rolling OOF)...")
    oof_df, mean_f1, fold_f1s = train_oof(merged)

    logger.info("\nEvaluating OOF signal (argmax LR)...")
    metrics = evaluate_oof(oof_df)
    logger.info(f"  Trades={metrics['trades']:,} WR={metrics['wr']:.4f} "
                f"PF={metrics['pf']:.3f} PnL/T={metrics['pnl_per_trade']:.4f} PnL={metrics['pnl']:.2f}")

    # Baseline LGBM standalone
    lgbm_best_path = MODEL_DIR / "runs" / LGBM_RUN / "best_thresholds.json"
    with open(lgbm_best_path) as f:
        lgbm_best = json.load(f)
    logger.info(f"\n  BASELINE LGBM(0.75/0.70): Trades=12,992 WR=0.6420 PF=2.439 PnL=3,554.73")
    logger.info(f"  LSTM filter (thr=0.38)  : Trades=7,206  WR=0.6599 PF=2.637 PnL=2,019.27")
    logger.info(f"  LR meta (argmax)         : Trades={metrics['trades']:,} "
                f"WR={metrics['wr']:.4f} PF={metrics['pf']:.3f} PnL={metrics['pnl']:.2f}")

    # Save
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    oof_save = oof_df[["coin", "lr_p0", "lr_p1", "lr_p2", "lr_pred", "fold"]].copy()
    oof_save["has_oof"] = True
    oof_save.to_parquet(OUT_DIR / "oof_lr_predictions.parquet")
    logger.info(f"\n  Saved OOF: {OUT_DIR / 'oof_lr_predictions.parquet'}")

    meta = {
        "run_name": RUN_NAME, "lgbm_run": LGBM_RUN, "lstm_run": LSTM_RUN,
        "lr_C": LR_C, "purge_gap": PURGE_GAP, "n_folds": N_FOLDS,
        "meta_features": META_FEATURES,
        "cv_f1_macro_mean": round(mean_f1, 4),
        "cv_f1_macro_folds": [round(f, 4) for f in fold_f1s],
        "oof_metrics_argmax": metrics,
        "created": datetime.now().isoformat(),
    }
    with open(OUT_DIR / "lr_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"  Saved meta: {OUT_DIR / 'lr_meta.json'}")
    logger.info("\nDONE.")


if __name__ == "__main__":
    main()
