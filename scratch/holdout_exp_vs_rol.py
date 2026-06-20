"""
Holdout comparison: Expanding vs Rolling CV (LGBM only, no Guardian, no HMM).
Both methods train the SAME final model. Difference: threshold from OOF sweep.
Tests on holdout period Apr 1 - Jun 13, 2026.
"""
import json, sys, warnings, itertools
import numpy as np, pandas as pd
from pathlib import Path
import lightgbm as lgb

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from core.evaluator import simulate_trades_swing
from core.utils import ensure_utc_index
from config import (
    ALL_COINS, LABEL_DIR, MODEL_DIR, TRAIN_CUTOFF_DATE, HOLDOUT_DIR,
    LGBM_PARAMS, LGBM_EARLY_STOPPING, LABEL_MAP, PURGE_GAP_BARS,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP,
    SWING_LABEL_MAX_SL, TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
)

HOLDOUT_END = pd.Timestamp("2026-06-14", tz="UTC")
FEAT_SOURCE = MODEL_DIR / "feature_cols_v2.json"
COMPARE_DIR = MODEL_DIR / "runs" / "tb_lgbm_cv_comparison"
LM = {"SHORT": 0, "FLAT": 1, "LONG": 2}

def load_data():
    """Load training + holdout data."""
    with open(FEAT_SOURCE) as f:
        features = json.load(f)

    # Training data dari data/training/labeled/
    train_frames = []
    for sym in ALL_COINS:
        path = LABEL_DIR / f"{sym}_features_v3.parquet"
        if not path.exists(): continue
        df = pd.read_parquet(path)
        df = ensure_utc_index(df).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]
        # Merge HMM regime
        regime_path = LABEL_DIR / f"{sym}_regime_h1.parquet"
        if regime_path.exists():
            reg = pd.read_parquet(regime_path)
            if "hmm_regime_enc" in df.columns:
                df = df.drop(columns=["hmm_regime_enc"])
            df = df.join(reg[["hmm_regime_enc"]], how="left")
            df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
        mask = df["label"].astype(str).isin(LABEL_MAP)
        df = df[mask].copy()
        df["coin"] = sym
        if len(df) >= 100: train_frames.append(df)

    # Holdout data dari data/holdout-test/labeled/
    holdout_frames = []
    holdout_label_dir = HOLDOUT_DIR / "labeled"
    for sym in ALL_COINS:
        path = holdout_label_dir / f"{sym}_features_v3.parquet"
        if not path.exists(): continue
        df = pd.read_parquet(path)
        df = ensure_utc_index(df).sort_index()
        df = df[(df.index >= TRAIN_CUTOFF_DATE) & (df.index < HOLDOUT_END)]
        # Merge HMM regime holdout
        regime_path = holdout_label_dir / f"{sym}_regime_h1.parquet"
        if regime_path.exists():
            reg = pd.read_parquet(regime_path)
            if "hmm_regime_enc" in df.columns:
                df = df.drop(columns=["hmm_regime_enc"])
            df = df.join(reg[["hmm_regime_enc"]], how="left")
            df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
        mask = df["label"].astype(str).isin(LABEL_MAP)
        df = df[mask].copy()
        df["coin"] = sym
        if len(df) >= 50: holdout_frames.append(df)

    if not train_frames:
        raise RuntimeError("No training data")
    if not holdout_frames:
        raise RuntimeError(f"No holdout data in {holdout_label_dir}")
    return pd.concat(train_frames).sort_index(), pd.concat(holdout_frames).sort_index(), features


def simulate_on_holdout(df_hold, model, avail, thresholds, label):
    """Predict on holdout with given thresholds, simulate PER-COIN then aggregate.
    WARNING: simulate_trades_swing expects single-coin contiguous data.
    Multi-coin sorted by timestamp will have price jumps between coins -> garbage PnL."""
    agg_trades = agg_wins = 0
    agg_pnl = 0.0
    all_losses = []
    sl_count = 0
    sl_losses = []

    for sym in df_hold["coin"].unique():
        sdf = df_hold[df_hold["coin"] == sym].sort_index()
        if len(sdf) < 50: continue

        X = sdf[avail].ffill().fillna(0).values.astype(np.float32)
        probas = model.predict_proba(X)
        n = len(sdf)
        y_pred = np.full(n, LM["FLAT"], np.int32)
        y_pred[probas[:, 2] >= thresholds["thr_long"]] = LM["LONG"]
        short_m = (probas[:, 0] >= thresholds["thr_short"]) & (y_pred != LM["LONG"])
        y_pred[short_m] = LM["SHORT"]
        if (y_pred != LM["FLAT"]).sum() < 5: continue

        result = simulate_trades_swing(
            y_pred=y_pred,
            close=sdf["close"].values, high=sdf["high"].values, low=sdf["low"].values,
            atr=sdf["atr_14_h1"].values,
            h4_swing_highs=sdf["h4_swing_high"].values if "h4_swing_high" in sdf.columns else np.full(n, np.nan),
            h4_swing_lows=sdf["h4_swing_low"].values if "h4_swing_low" in sdf.columns else np.full(n, np.nan),
            modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            max_hold=MAX_HOLDING_BARS, min_rr=SWING_LABEL_MIN_RR,
            min_tp_atr=SWING_LABEL_MIN_TP, max_sl_atr=SWING_LABEL_MAX_SL,
            tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
            guardian_enabled=False,
        )
        agg_trades += result.get("total_trades", 0)
        agg_wins   += result.get("wins", 0)
        agg_pnl    += result.get("total_pnl", 0.0)
        for t in result.get("trades", []):
            if t["net_pnl"] < 0: all_losses.append(t["net_pnl"])
            if t["outcome"] == "LOSS":
                sl_count += 1
                sl_losses.append(t["net_pnl"])

    wr  = agg_wins / agg_trades if agg_trades > 0 else 0.0
    ppt = agg_pnl / agg_trades if agg_trades > 0 else 0.0
    gross_loss = abs(sum(all_losses)) if all_losses else 1e-9
    gross_win  = agg_pnl + gross_loss  # total_pnl = sum(wins) + sum(losses)
    pf = gross_win / gross_loss
    sl_avg_loss = np.mean(sl_losses) if sl_losses else 0.0

    print(f"\n  [{label}] thr={thresholds['thr_long']}/{thresholds['thr_short']}")
    print(f"    Trades: {agg_trades:>5}  WR: {wr*100:>5.1f}%  PF: {pf:>5.2f}  PnL: ${agg_pnl:>8.2f}  PPT: ${ppt:>6.4f}")
    print(f"    SL hits: {sl_count:>4}  avg SL loss: ${sl_avg_loss:>6.4f}")

    return {
        "label": label, "thresholds": thresholds,
        "trades": agg_trades, "wr": round(wr, 4), "pf": round(pf, 2),
        "pnl": round(agg_pnl, 2), "ppt": round(ppt, 4),
        "sl_hits": sl_count, "sl_avg_loss": round(sl_avg_loss, 4),
    }


def main():
    print("Loading data...")
    df_train, df_hold, features = load_data()
    print(f"  Train: {len(df_train):,} bars, {df_train['coin'].nunique()} coins")
    print(f"  Holdout: {len(df_hold):,} bars, {df_hold['coin'].nunique()} coins "
          f"({df_hold.index.min().date()} -> {df_hold.index.max().date()})")

    # Train final model on all training data
    # Use consistent feature set across train + holdout
    avail_train = [c for c in features if c in df_train.columns]
    avail_hold  = [c for c in features if c in df_hold.columns]
    avail = sorted(set(avail_train) & set(avail_hold))
    missing_train = set(features) - set(avail)
    if missing_train:
        print(f"  Features missing in both: {sorted(missing_train)[:5]}...")
    print(f"  Consistent features: {len(avail)}")

    y_train = df_train["label"].map(LABEL_MAP).values.astype(np.int32)
    X_train = df_train[avail].ffill().fillna(0).values.astype(np.float32)

    print(f"\nTraining final LGBM on {len(X_train):,} bars, {len(avail)} features...")
    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(X_train, y_train)
    print(f"  Trained: {model.n_estimators_} iters")

    # Load OOF-derived thresholds from comparison
    exp_thr_path = COMPARE_DIR / "oof_predictions_expanding.parquet"
    rol_thr_path = COMPARE_DIR / "oof_predictions_rolling.parquet"

    # Threshold dari OOF sweep terbaik (dari hasil sebelumnya)
    # Expanding best: 0.45/0.45, Rolling best: 0.45/0.45 (hampir sama)
    # Kita test beberapa threshold pair untuk fair comparison
    threshold_pairs = [
        {"thr_long": 0.45, "thr_short": 0.45},
        {"thr_long": 0.50, "thr_short": 0.50},
        {"thr_long": 0.50, "thr_short": 0.55},
        {"thr_long": 0.55, "thr_short": 0.55},
        {"thr_long": 0.55, "thr_short": 0.60},
    ]

    sep = "=" * 70
    print(f"\n{sep}")
    print("  HOLDOUT TEST — Expanding vs Rolling OOF Thresholds")
    print(f"  Period: {TRAIN_CUTOFF_DATE.date()} -> {HOLDOUT_END.date()}")
    print(f"  Model: SAME LGBM (trained on all data up to {TRAIN_CUTOFF_DATE.date()})")
    print(f"  Difference: which threshold pair each CV method selects")
    print(f"{sep}")

    all_results = []
    for thr in threshold_pairs:
        r = simulate_on_holdout(df_hold, model, avail, thr,
                                f"thr={thr['thr_long']}/{thr['thr_short']}")
        all_results.append(r)

    # Head-to-head summary
    print(f"\n{'='*70}")
    print("  SUMMARY — Holdout LGBM only (no Guardian, no HMM)")
    print(f"  {'Thr (L/S)':<14} {'Trades':>6} {'WR%':>7} {'PF':>6} {'PnL':>9} {'PPT':>7} {'SL':>5} {'SL avg loss':>12}")
    print(f"  {'-'*14} {'-'*6} {'-'*7} {'-'*6} {'-'*9} {'-'*7} {'-'*5} {'-'*12}")
    for r in all_results:
        t = r["thresholds"]
        print(f"  {t['thr_long']:.2f}/{t['thr_short']:.2f}        "
              f"{r['trades']:>6} {r['wr']*100:>6.1f}% {r['pf']:>5.2f} "
              f"${r['pnl']:>8.2f} ${r['ppt']:>6.4f} {r['sl_hits']:>5} ${r['sl_avg_loss']:>11.4f}")

    print(f"\n  Expanding CV OOF best thr: 0.45/0.45")
    print(f"  Rolling CV OOF best thr:   0.45/0.45")
    print(f"  -> Both CV methods pick same threshold -> HOLD OUT RESULTS IDENTICAL")
    print(f"  -> Holdout comparison shows threshold sensitivity, not CV method difference")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
