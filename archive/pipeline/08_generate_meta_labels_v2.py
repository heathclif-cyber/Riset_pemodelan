"""
pipeline/08_generate_meta_labels_v2.py — Meta-Label Generation (Walk-Forward OOF)

Phase 2 LSTM Binary Meta-Labeling - Step 1.

Generate binary meta-labels: is_good_trade = 1 jika net_pnl > median
MENGGUNAKAN WALK-FORWARD OOF (bukan training simulation).

Kenapa walk-forward: hindari in-sample bias yang membuat meta-labeling v1 gagal.
Model fold k memprediksi fold k+1 → trade outcomes adalah GENUINE OOS.

Output per coin:
  {coin}_meta_labels_v2.parquet:
    timestamp (entry bar), net_pnl, is_good_trade, fold
  {coin}_meta_sequences_v2.npy:
    sequence features (40 bars before entry × 25 features) per trade

Usage:
  python pipeline/08_generate_meta_labels_v2.py --all
  python pipeline/08_generate_meta_labels_v2.py --coins BTCUSDT ETHUSDT
"""

import argparse, json, sys, warnings, numpy as np, pandas as pd
from pathlib import Path
import lightgbm as lgb

ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

from config import (
    TRAINING_COINS, LABEL_DIR, MODEL_DIR, TRAIN_CUTOFF_DATE,
    LABEL_MAP, N_FOLDS, PURGE_GAP_BARS, LGBM_PARAMS, LGBM_EARLY_STOPPING,
    MODAL_PER_TRADE, LEVERAGE_SIM, FEE_PER_SIDE, SLIPPAGE_PER_SIDE,
    MAX_HOLDING_BARS, SWING_LABEL_MIN_RR, SWING_LABEL_MIN_TP, SWING_LABEL_MAX_SL,
    TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
)
from core.utils import setup_logger
from core.evaluator import simulate_trades_swing
from pipeline.shared import build_purged_folds
from pipeline.backtest_utils import hierarchical_predict

logger = setup_logger("08_meta_labels_v2")

# ── Feature Configuration ────────────────────────────────────────────────────
# 25 features untuk LSTM meta-model (sequence sebelum entry)
# 15 multi-TF dari LSTM + 4 HMM probs + h4_trend + 5 structural
LSTM_FEATS = json.load(open(MODEL_DIR / "feature_cols_lstm_temporal.json"))
META_EXTRA_FEATS = [
    "hmm_prob_0", "hmm_prob_1", "hmm_prob_2", "hmm_prob_3",
    "h4_trend", "trend_strength",
    "atr_14_h1", "atr_percentile_h1", "vol_ratio_20",
]
META_FEATS = [f for f in LSTM_FEATS + META_EXTRA_FEATS if f != "btc_h1_return"]

SEQ_LEN = 40   # bars sebelum entry
LABEL_PCTILE = 50  # top 50% profit = good trade

# ── LGBM features (dari feature_cols_v2.json) ─────────────────────────────────
LGBM_FEATS = json.load(open(MODEL_DIR / "feature_cols_v2.json"))


def load_coin_data(coin):
    """Load features + HMM probs + regime for one coin."""
    feat_path = LABEL_DIR / f"{coin}_features_v3.parquet"
    prob_path = LABEL_DIR / f"{coin}_hmm_probs.parquet"
    reg_path = LABEL_DIR / f"{coin}_regime_h1.parquet"

    if not feat_path.exists():
        return None

    df = pd.read_parquet(feat_path).sort_index()
    df = df[df.index < TRAIN_CUTOFF_DATE]

    if prob_path.exists():
        probs = pd.read_parquet(prob_path).sort_index()
        for i in range(4):
            c = f"hmm_prob_{i}"
            if c in probs.columns:
                df[c] = probs[c]

    if reg_path.exists():
        reg = pd.read_parquet(reg_path)
        if "hmm_regime_enc" in df.columns:
            df = df.drop(columns=["hmm_regime_enc"])
        df = df.join(reg[["hmm_regime_enc"]], how="left")
        df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")

    # Ensure all META_FEATS exist
    for c in META_FEATS:
        if c not in df.columns:
            df[c] = 0.0

    mask = df["label"].astype(str).isin(LABEL_MAP)
    df = df[mask].copy()
    if len(df) < SEQ_LEN + 50:
        return None
    return df


def get_sequence(df, bar_idx):
    """Extract SEQ_LEN bars of META_FEATS ending at bar_idx."""
    start = bar_idx - SEQ_LEN + 1
    if start < 0:
        return None
    seq = np.zeros((SEQ_LEN, len(META_FEATS)), dtype=np.float32)
    for j, col in enumerate(META_FEATS):
        seq[:, j] = df[col].iloc[start:bar_idx+1].ffill().fillna(0).values
    return seq


def generate_for_coin(coin, fold_df_mapping):
    """
    Walk-forward per fold:
    - Train LGBM on training folds
    - Predict + simulate trades on OOF fold
    - Label each trade: is_good_trade = 1 if PnL > median(positive PnL)
    - Extract sequence before entry bar
    """
    results = []
    df = load_coin_data(coin)
    if df is None:
        return results

    # Build timestamp-based folds
    ts_index = pd.DatetimeIndex(df.index)
    folds = build_purged_folds(ts_index, N_FOLDS, PURGE_GAP_BARS)

    for fold_idx, (tr_idx, te_idx) in enumerate(folds):
        if len(te_idx) < 100:
            continue

        # Train LGBM on this fold's training data
        df_tr = df.iloc[tr_idx]
        df_te = df.iloc[te_idx]

        feat_cols = [c for c in LGBM_FEATS if c in df_tr.columns]
        X_tr = df_tr[feat_cols].ffill().fillna(0)
        y_tr = df_tr["label"].map(LABEL_MAP).values.astype(np.int64)

        if len(np.unique(y_tr)) < 2:
            continue

        # GENUINE OOF: retrain LGBM from scratch on this fold's training data only.
        # Tidak pakai pre-trained model — itu menyebabkan in-sample bias
        # karena model sudah lihat 2020-2025 (termasuk fold k+1).
        if len(np.unique(y_tr)) < 3:
            continue

        # Train LGBM on training folds only
        params = {**LGBM_PARAMS}
        fold_model = lgb.LGBMClassifier(**params)
        fold_model.fit(
            X_tr, y_tr,
            eval_set=[(X_tr, y_tr)],
            eval_metric="multi_logloss",
            callbacks=[lgb.early_stopping(LGBM_EARLY_STOPPING), lgb.log_evaluation(0)],
        )

        # Predict on OOF fold
        n_te = len(df_te)
        X_te = np.zeros((n_te, len(feat_cols)), dtype=np.float64)
        for i, col in enumerate(feat_cols):
            if col in df_te.columns:
                X_te[:, i] = df_te[col].ffill().fillna(0).values

        y_pred, confidence = hierarchical_predict(
            None, fold_model, None, None, X_te, feat_cols, [], df_te,
            trend_alignment_enabled=False,
        )
        below = (y_pred != 1) & (confidence < 0.59)
        y_pred[below] = 1

        atr = df_te["atr_14_h1"].values if "atr_14_h1" in df_te.columns else np.ones(n_te)
        close = df_te["close"].values
        high = df_te["high"].values if "high" in df_te.columns else close
        low = df_te["low"].values if "low" in df_te.columns else close

        result = simulate_trades_swing(
            y_pred=y_pred, close=close, high=high, low=low, atr=atr,
            h4_swing_highs=np.full(n_te, np.nan),
            h4_swing_lows=np.full(n_te, np.nan),
            modal=MODAL_PER_TRADE, leverage=LEVERAGE_SIM[0],
            fee_per_side=FEE_PER_SIDE, slippage=SLIPPAGE_PER_SIDE,
            max_hold=MAX_HOLDING_BARS,
            min_rr=SWING_LABEL_MIN_RR, min_tp_atr=SWING_LABEL_MIN_TP,
            max_sl_atr=SWING_LABEL_MAX_SL,
            tp_fallback_atr=TP_SL_FALLBACK_TP, sl_fallback_atr=TP_SL_FALLBACK_SL,
            confidence=confidence,
            guardian_enabled=False,
        )

        trades = result.get("trades", [])
        if len(trades) < 5:
            continue

        # Compute median profit for labeling
        profits = [t["net_pnl"] for t in trades]
        median_profit = np.median(profits)
        # If median is negative, use median of positive profits
        if median_profit <= 0:
            pos = [p for p in profits if p > 0]
            median_profit = np.median(pos) if pos else 0.01

        for t in trades:
            bar_in = t["bar_in"]
            if bar_in < SEQ_LEN:
                continue  # Need enough history for sequence

            seq = get_sequence(df_te, bar_in)
            if seq is None:
                continue

            is_good = 1 if t["net_pnl"] > median_profit else 0
            results.append({
                "coin": coin,
                "fold": fold_idx,
                "bar_in": bar_in,
                "timestamp": df_te.index[bar_in],
                "direction": t["direction"],
                "entry_price": t["entry"],
                "net_pnl": round(float(t["net_pnl"]), 4),
                "is_good_trade": is_good,
                "median_profit": round(float(median_profit), 4),
                "sequence": seq,
            })

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--coins", nargs="+", default=None)
    args = parser.parse_args()

    coins = args.coins or (TRAINING_COINS if args.all else TRAINING_COINS[:5])

    print(f"\n{'='*65}")
    print(f"  META-LABEL GENERATION V2 | Walk-Forward OOF")
    print(f"  Coins: {len(coins)} | Seq len: {SEQ_LEN} | Features: {len(META_FEATS)}")
    print(f"  Label: is_good_trade = 1 if PnL > median | Folds: {N_FOLDS}")
    print(f"{'='*65}\n")

    all_results = []
    for coin in coins:
        res = generate_for_coin(coin, None)
        if res:
            all_results.extend(res)
            n_good = sum(1 for r in res if r["is_good_trade"] == 1)
            logger.info(f"{coin}: {len(res)} trades, {n_good} good ({n_good/len(res)*100:.0f}%)")

    if not all_results:
        print("ERROR: No trades generated")
        return

    # Save labels
    labels_df = pd.DataFrame([{
        "coin": r["coin"], "fold": r["fold"], "bar_in": r["bar_in"],
        "timestamp": r["timestamp"], "direction": r["direction"],
        "entry_price": r["entry_price"], "net_pnl": r["net_pnl"],
        "is_good_trade": r["is_good_trade"], "median_profit": r["median_profit"],
    } for r in all_results])

    out_dir = Path("data/meta_labels_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    labels_path = out_dir / "meta_labels_training_v2.parquet"
    labels_df.to_parquet(labels_path)

    # Save sequences as .npy
    sequences = np.stack([r["sequence"] for r in all_results])
    seq_path = out_dir / "meta_sequences_training_v2.npy"
    np.save(seq_path, sequences)

    # Save metadata
    meta = {
        "n_trades": len(all_results),
        "n_good": int(sum(1 for r in all_results if r["is_good_trade"] == 1)),
        "good_pct": round(sum(1 for r in all_results if r["is_good_trade"] == 1) / len(all_results) * 100, 1),
        "seq_len": SEQ_LEN,
        "n_features": len(META_FEATS),
        "features": META_FEATS,
        "label_pctile": LABEL_PCTILE,
        "n_folds": N_FOLDS,
    }
    with open(out_dir / "meta_labels_v2_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*65}")
    print(f"  META-LABELS SAVED")
    print(f"  Trades : {len(all_results)}")
    print(f"  Good   : {meta['n_good']} ({meta['good_pct']}%)")
    print(f"  Seq    : {sequences.shape} ({SEQ_LEN} bars × {len(META_FEATS)} features)")
    print(f"  Labels : {labels_path}")
    print(f"  Seqs   : {seq_path}")
    print(f"{'='*65}")
    print(f"\n  Next: python pipeline/09_train_lstm_meta.py")


if __name__ == "__main__":
    main()
