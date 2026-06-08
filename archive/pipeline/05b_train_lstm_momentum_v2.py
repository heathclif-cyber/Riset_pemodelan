"""
pipeline/05b_train_lstm_momentum_v2.py — Train LSTM dengan Momentum Flow Labels V2

PERBEDAAN FUNDAMENTAL dari 05_train_lstm_v2_style.py:
  1. Label berbeda: BEARISH/NEUTRAL/BULLISH (flow-based)
                   bukan SHORT/FLAT/LONG (structure-based swing labels)
  2. Fitur berbeda: fokus pada flow (OFI, CVD, volume) bukan semua LGBM features
  3. Per-coin z-score untuk volume_delta sebelum menggabung sequences (fix scale mismatch)

Tujuan: LSTM mengukur "apakah momentum flow mendukung arah entry?"
        Bukan menduplikasi prediksi LGBM yang sudah berbasis structure.

Marginal IC test:
  IC(LSTM_momentum | LGBM_structure sudah ada) harus > 0
  Ini yang memvalidasi LSTM worth it di ensemble.

Compatibility:
  Model output: 3-class (0=BEARISH, 1=NEUTRAL, 2=BULLISH)
  Numerik sama dengan LABEL_MAP -> compatible dengan inference pipeline
  hard_consensus mode: LGBM=LONG + LSTM=BULLISH(2) -> konfirmasi
                        LGBM=LONG + LSTM=BEARISH(0) -> penalize

Usage:
  # Probe 5 coin dulu
  python pipeline/05b_train_lstm_momentum_v2.py --run-id ic32_lstm_momentum_v2

  # Full training
  python pipeline/05b_train_lstm_momentum_v2.py --run-id ic32_lstm_momentum_v2 --all

Prerequisite:
  python pipeline/05a_momentum_labels_v2.py --all
"""

import argparse
import gc
import json
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message="The operator 'aten::lerp")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, accuracy_score
from torch.utils.data import DataLoader, Dataset
import joblib

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, LABEL_DIR, TRAIN_CUTOFF_DATE, MODEL_DIR,
    N_FOLDS, PURGE_GAP_BARS,
    LSTM_SEQ_LEN, LSTM_BATCH_SIZE, LSTM_EPOCHS, LSTM_PATIENCE,
    LSTM_V2_HIDDEN, LSTM_V2_LAYERS, LSTM_V2_DROPOUT,
    LSTM_V2_WEIGHT_DECAY, LSTM_V2_LR,
)
from core.models import TradingLSTM, save_lstm
from core.utils import setup_logger, get_lstm_device
from pipeline.shared import build_purged_folds

logger = setup_logger("05b_train_lstm_momentum_v2")
DEVICE = get_lstm_device()

# ─── Fitur LSTM Momentum V2 ─────────────────────────────────────────────────
# Flow-focused: z-scored atau bounded → cross-coin comparable tanpa rescaling.
# volume_delta & cvd_momentum_adv: di-z-score per-coin di dalam load_data_for_run()
#   sebelum digabung, bukan after, untuk fix scale mismatch BTC vs DOGE.
LSTM_MOMENTUM_V2_FEATURES = [
    # Order flow — sudah z-scored atau acceleration-based
    "ofi_z_score",          # OFI z-score (normalized by design)
    "ofi_acceleration",     # OFI acceleration (per-coin z-score in builder)
    "cvd_momentum_adv",     # CVD momentum derivative (per-coin z-score in builder)
    "absorption_z",         # absorption at swing (z-scored)

    # Volume imbalance — per-coin z-score in builder
    "volume_delta",         # buy - sell (AKAN di-z-score per-coin)
    "vol_ratio_20",         # volume / rolling 20-bar avg (bounded ~0-5)

    # Price trajectory — log returns (naturally comparable across coins)
    "log_ret_1",
    "log_ret_5",
    "log_ret_20",

    # Oscillator — bounded
    "rsi_6",

    # Cross-market context
    "btc_h1_return",
]

# Fitur yang butuh per-coin z-score sebelum digabung
_PERCOIN_ZSCORE_FEATS = {"volume_delta", "cvd_momentum_adv", "ofi_acceleration"}
_ZSCORE_WINDOW = 500  # rolling window ~3 minggu H1


# ─── Label map untuk momentum V2 ─────────────────────────────────────────────
MOMENTUM_LABEL_MAP     = {"BEARISH": 0, "NEUTRAL": 1, "BULLISH": 2}
MOMENTUM_LABEL_MAP_INV = {0: "BEARISH", 1: "NEUTRAL", 2: "BULLISH"}


class PrebuiltSeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self): return len(self.X)

    def __getitem__(self, idx): return self.X[idx], self.y[idx]

    def get_labels(self): return self.y.numpy()


def _percoin_z(series: np.ndarray, window: int = _ZSCORE_WINDOW) -> np.ndarray:
    """Rolling z-score untuk normalize raw volume features per coin."""
    s = pd.Series(series)
    mean = s.rolling(window=window, min_periods=50).mean()
    std  = s.rolling(window=window, min_periods=50).std().clip(lower=1e-8)
    return ((s - mean) / std).clip(-4, 4).fillna(0).values.astype(np.float32)


def fit_scaler(X: np.ndarray) -> RobustScaler:
    n, s, f = X.shape
    scaler = RobustScaler()
    scaler.fit(X.reshape(-1, f))
    return scaler


def scale_X(X: np.ndarray, scaler: RobustScaler) -> np.ndarray:
    n, s, f = X.shape
    return scaler.transform(X.reshape(-1, f)).reshape(n, s, f).astype(np.float32)


def load_data_for_run(run_id: str, coins: list[str]):
    """
    Load features_v3 + momentum_v2_labels per coin, bangun sequences.

    Key difference vs 05_train_lstm_v2_style.py:
    - Label dari {coin}_momentum_v2_labels.parquet (flow-based)
    - Per-coin z-score untuk volume_delta, cvd_momentum_adv, ofi_acceleration
      SEBELUM sequences digabung — ini fix utama untuk scale mismatch.
    """
    available_feat = LSTM_MOMENTUM_V2_FEATURES

    X_seqs, y_seqs, ts_seqs = [], [], []
    skipped = []

    for coin in coins:
        feat_path  = LABEL_DIR / f"{coin}_features_v3.parquet"
        label_path = LABEL_DIR / f"{coin}_momentum_v2_labels.parquet"

        if not feat_path.exists():
            logger.warning(f"{coin}: features_v3 tidak ditemukan, skip")
            skipped.append(coin)
            continue

        if not label_path.exists():
            logger.warning(
                f"{coin}: momentum_v2_labels tidak ditemukan. "
                f"Jalankan: python pipeline/05a_momentum_labels_v2.py --all"
            )
            skipped.append(coin)
            continue

        df = pd.read_parquet(feat_path).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]

        # Merge label
        lbl_df = pd.read_parquet(label_path)
        df = df.join(lbl_df["momentum_v2_label"], how="inner")
        df = df.dropna(subset=["momentum_v2_label"])

        if len(df) < LSTM_SEQ_LEN + 10:
            logger.warning(f"{coin}: terlalu sedikit data ({len(df)}), skip")
            skipped.append(coin)
            continue

        # Kolom fitur yang tersedia di parquet
        cols = [c for c in available_feat if c in df.columns]
        missing = set(available_feat) - set(cols)
        if missing:
            logger.info(f"{coin}: fitur tidak ada di parquet (diisi 0): {sorted(missing)}")

        # Build feature matrix dengan per-coin z-score untuk kolom sensitif
        feat_vals = {}
        for c in available_feat:
            if c in df.columns:
                vals = df[c].ffill().fillna(0).values.astype(np.float32)
                # Per-coin z-score untuk fitur yang raw (scale mismatch)
                if c in _PERCOIN_ZSCORE_FEATS:
                    vals = _percoin_z(vals.astype(np.float64)).astype(np.float32)
                feat_vals[c] = vals
            else:
                feat_vals[c] = np.zeros(len(df), dtype=np.float32)

        X_c  = np.column_stack([feat_vals[c] for c in available_feat])
        y_c  = df["momentum_v2_label"].values.astype(np.int64)
        ts_c = df.index.astype(np.int64).values

        # Sequence per coin (no cross-coin contamination)
        n_coin_seqs = 0
        for i in range(LSTM_SEQ_LEN - 1, len(X_c)):
            X_seqs.append(X_c[i - LSTM_SEQ_LEN + 1 : i + 1])
            y_seqs.append(y_c[i])
            ts_seqs.append(ts_c[i])
            n_coin_seqs += 1

        # Log distribusi label per koin
        n_bull = int((y_c == 2).sum())
        n_neu  = int((y_c == 1).sum())
        n_bear = int((y_c == 0).sum())
        logger.info(
            f"{coin}: {len(df):,} bars | "
            f"BULL={n_bull/len(y_c)*100:.0f}% NEU={n_neu/len(y_c)*100:.0f}% "
            f"BEAR={n_bear/len(y_c)*100:.0f}% | seqs={n_coin_seqs:,}"
        )

    if skipped:
        logger.warning(f"Skipped {len(skipped)} coins: {skipped}")

    if not X_seqs:
        raise ValueError(
            "Tidak ada sequences terbangun. "
            "Cek apakah momentum_v2_labels sudah digenerate."
        )

    X  = np.stack(X_seqs)
    y  = np.array(y_seqs, dtype=np.int64)
    ts = np.array(ts_seqs, dtype=np.int64)

    # Sort by real timestamp
    order = np.argsort(ts)
    return X[order], y[order], ts[order], available_feat


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    classes, counts = np.unique(y, return_counts=True)
    total   = len(y)
    weights = {c: total / (len(classes) * count) for c, count in zip(classes, counts)}
    return torch.tensor([weights.get(i, 1.0) for i in range(3)], dtype=torch.float32).to(DEVICE)


def train_one_fold(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray, y_te: np.ndarray,
    fold_num: int,
):
    n_features = X_tr.shape[2]
    n_train    = len(X_tr)
    n_val      = len(X_te)

    fold_scaler = fit_scaler(X_tr)
    X_tr_sc = scale_X(X_tr, fold_scaler); del X_tr; gc.collect()
    X_te_sc = scale_X(X_te, fold_scaler); del X_te; gc.collect()

    train_ds = PrebuiltSeqDataset(X_tr_sc, y_tr)
    test_ds  = PrebuiltSeqDataset(X_te_sc, y_te)

    train_loader = DataLoader(train_ds, batch_size=LSTM_BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=LSTM_BATCH_SIZE, shuffle=False, num_workers=0)

    model     = TradingLSTM(n_features, LSTM_V2_HIDDEN, LSTM_V2_LAYERS, LSTM_V2_DROPOUT).to(DEVICE)
    cw        = compute_class_weights(y_tr)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LSTM_V2_LR,
        weight_decay=LSTM_V2_WEIGHT_DECAY, foreach=False,
    )

    best_f1, best_state, patience_count, best_epoch = -1.0, None, 0, 1

    for epoch in range(1, LSTM_EPOCHS + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        preds_v, labels_v = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                preds_v.extend(model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy())
                labels_v.extend(yb.numpy())

        f1 = float(f1_score(labels_v, preds_v, average="macro", zero_division=0))

        if f1 > best_f1:
            best_f1, best_state, patience_count, best_epoch = f1, {
                k: v.cpu() for k, v in model.state_dict().items()
            }, 0, epoch
        else:
            patience_count += 1
            if patience_count >= LSTM_PATIENCE:
                logger.info(f"[Fold {fold_num}] Early stop epoch {epoch} | best F1={best_f1:.4f}")
                break

        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"[Fold {fold_num}] Epoch {epoch:>3} | F1={f1:.4f} | Best={best_f1:.4f}")

    model.load_state_dict(best_state)
    model.eval()

    # Final val metrics
    pv, lv = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            pv.extend(model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy())
            lv.extend(yb.numpy())

    val_f1m  = float(f1_score(lv, pv, average="macro", zero_division=0))
    val_f1_p = f1_score(lv, pv, average=None, zero_division=0, labels=[0, 1, 2])
    val_acc  = float(accuracy_score(lv, pv))

    # Train metrics (overfitting check)
    eval_loader = DataLoader(train_ds, batch_size=LSTM_BATCH_SIZE, shuffle=False, num_workers=0)
    pt, lt = [], []
    with torch.no_grad():
        for xb, yb in eval_loader:
            pt.extend(model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy())
            lt.extend(yb.numpy())

    train_f1m  = float(f1_score(lt, pt, average="macro", zero_division=0))
    train_f1_p = f1_score(lt, pt, average=None, zero_division=0, labels=[0, 1, 2])
    train_acc  = float(accuracy_score(lt, pt))

    metrics = {
        "fold": fold_num, "n_train": n_train, "n_val": n_val, "best_epoch": best_epoch,
        "train_f1_macro": round(train_f1m, 4),
        "train_f1_BEARISH": round(float(train_f1_p[0]), 4),
        "train_f1_NEUTRAL": round(float(train_f1_p[1]), 4),
        "train_f1_BULLISH": round(float(train_f1_p[2]), 4),
        "train_accuracy":   round(train_acc, 4),
        "f1_macro":         round(val_f1m, 4),
        "f1_BEARISH":       round(float(val_f1_p[0]), 4),
        "f1_NEUTRAL":       round(float(val_f1_p[1]), 4),
        "f1_BULLISH":       round(float(val_f1_p[2]), 4),
        "accuracy":         round(val_acc, 4),
    }

    gap = train_f1m - val_f1m
    logger.info(
        f"[Fold {fold_num}] Train={train_f1m:.4f} | Val={val_f1m:.4f} | "
        f"Gap={gap:+.4f} | Epoch={best_epoch}"
    )
    return model, fold_scaler, metrics


def retrain_final(X_all: np.ndarray, y_all: np.ndarray, n_epochs: int):
    n_features   = X_all.shape[2]
    final_scaler = fit_scaler(X_all)
    X_sc         = scale_X(X_all, final_scaler); del X_all; gc.collect()

    ds     = PrebuiltSeqDataset(X_sc, y_all)
    loader = DataLoader(ds, batch_size=LSTM_BATCH_SIZE, shuffle=True, num_workers=0)

    model     = TradingLSTM(n_features, LSTM_V2_HIDDEN, LSTM_V2_LAYERS, LSTM_V2_DROPOUT).to(DEVICE)
    cw        = compute_class_weights(y_all)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LSTM_V2_LR,
        weight_decay=LSTM_V2_WEIGHT_DECAY, foreach=False,
    )

    model.train()
    for epoch in range(1, n_epochs + 1):
        total_loss = sum(
            (lambda: (
                optimizer.zero_grad(),
                criterion(model((xb := xb.to(DEVICE)), yb.to(DEVICE))),
            )[-1].backward() or
            nn.utils.clip_grad_norm_(model.parameters(), 1.0) or
            optimizer.step() or 0
            )()
            for xb, yb in loader
        )
        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"[Final] Epoch {epoch:>3}/{n_epochs} | batches={len(loader)}")

    model.eval()
    return model, final_scaler


def retrain_final(X_all: np.ndarray, y_all: np.ndarray, n_epochs: int):
    """Clean final retrain tanpa lambda complexity."""
    n_features   = X_all.shape[2]
    final_scaler = fit_scaler(X_all)
    X_sc         = scale_X(X_all, final_scaler); del X_all; gc.collect()

    ds     = PrebuiltSeqDataset(X_sc, y_all)
    loader = DataLoader(ds, batch_size=LSTM_BATCH_SIZE, shuffle=True, num_workers=0)

    model     = TradingLSTM(n_features, LSTM_V2_HIDDEN, LSTM_V2_LAYERS, LSTM_V2_DROPOUT).to(DEVICE)
    cw        = compute_class_weights(y_all)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LSTM_V2_LR,
        weight_decay=LSTM_V2_WEIGHT_DECAY, foreach=False,
    )

    model.train()
    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"[Final] Epoch {epoch:>3}/{n_epochs} | loss={total_loss/len(loader):.4f}")

    model.eval()
    return model, final_scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True, help="Nama run (contoh: ic32_lstm_momentum_v2)")
    parser.add_argument("--all", action="store_true", help="Gunakan semua 21 training coins")
    args = parser.parse_args()

    coins = TRAINING_COINS if args.all else TRAINING_COINS[:5]

    print(f"\n{'='*65}")
    print(f"  LSTM MOMENTUM V2 TRAINING | run_id={args.run_id}")
    print(f"  Coins: {len(coins)} | Labels: FLOW-BASED (bukan swing)")
    print(f"  Fitur: {len(LSTM_MOMENTUM_V2_FEATURES)} (OFI + CVD + volume + returns)")
    print(f"  Per-coin z-score: {sorted(_PERCOIN_ZSCORE_FEATS)}")
    print(f"{'='*65}\n")

    torch.manual_seed(42)
    np.random.seed(42)

    X, y, ts, feat_cols = load_data_for_run(args.run_id, coins)
    n_features = X.shape[2]

    logger.info(f"Dataset: X={X.shape} | y={y.shape} | features={n_features}")
    for lbl_int, lbl_str in MOMENTUM_LABEL_MAP_INV.items():
        n = int((y == lbl_int).sum())
        logger.info(f"  {lbl_str:8}: {n:>8,} ({n/len(y)*100:.1f}%)")

    run_dir = MODEL_DIR / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "lstm_momentum_v2_feature_cols.json", "w") as f:
        json.dump(feat_cols, f, indent=2)

    ts_index = pd.to_datetime(ts, unit="ns", utc=True)
    folds    = build_purged_folds(ts_index, n_folds=N_FOLDS, purge=PURGE_GAP_BARS)

    all_metrics = []
    for fold_idx, (tr_idx, te_idx) in enumerate(folds):
        _, _, metrics = train_one_fold(
            X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], fold_idx + 1,
        )
        all_metrics.append(metrics)

    avg_best_epoch = int(np.median([m.get("best_epoch", 30) for m in all_metrics]))
    final_epochs   = max(20, min(avg_best_epoch + 5, LSTM_EPOCHS))

    logger.info(f"Retraining final LSTM on 100% data for {final_epochs} epochs...")
    final_model, final_scaler = retrain_final(X, y, final_epochs)

    save_lstm(final_model, run_dir / "lstm_momentum_v2.pt")
    joblib.dump(final_scaler, run_dir / "lstm_momentum_v2_scaler.pkl")

    meta = {
        "run_id":             args.run_id,
        "model_type":         "lstm_momentum_v2",
        "label_type":         "momentum_flow_v2",
        "label_map":          MOMENTUM_LABEL_MAP,
        "n_features":         n_features,
        "features":           feat_cols,
        "percoin_z_features": sorted(_PERCOIN_ZSCORE_FEATS),
        "seq_len":            LSTM_SEQ_LEN,
        "hidden":             LSTM_V2_HIDDEN,
        "layers":             LSTM_V2_LAYERS,
        "dropout":            LSTM_V2_DROPOUT,
        "weight_decay":       LSTM_V2_WEIGHT_DECAY,
        "lr":                 LSTM_V2_LR,
        "scaler_type":        "RobustScaler",
        "final_epochs":       final_epochs,
        "cv_folds":           len(all_metrics),
        "note": (
            "Label: BEARISH(0)/NEUTRAL(1)/BULLISH(2) based on OFI+CVD+volume flow vote. "
            "Independent dari swing labels LGBM. Marginal IC LSTM vs LGBM harus ditest "
            "sebelum deploy ke ensemble."
        ),
    }
    with open(run_dir / "lstm_momentum_v2_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(run_dir / "lstm_momentum_v2_cv_results.json", "w") as f:
        json.dump({
            "run_id": args.run_id, "n_folds": N_FOLDS,
            "gap_bars": PURGE_GAP_BARS, "metrics": all_metrics,
        }, f, indent=2)

    # Summary
    val_f1s   = [m["f1_macro"]       for m in all_metrics]
    train_f1s = [m["train_f1_macro"] for m in all_metrics]
    gaps      = [t - v for t, v in zip(train_f1s, val_f1s)]

    random_baseline = 1 / 3

    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  LSTM MOMENTUM V2 SELESAI — {args.run_id}")
    print(f"{sep}")
    print(f"  Mean Val F1   : {np.mean(val_f1s):.4f} +/- {np.std(val_f1s):.4f}")
    print(f"  Random baseline: {random_baseline:.4f}")
    print(f"  Gain vs random: {np.mean(val_f1s) - random_baseline:+.4f}")
    print(f"  Mean Train F1  : {np.mean(train_f1s):.4f}")
    print(f"  Mean Gap (overfit): {np.mean(gaps):+.4f}  (target: < +0.08)")
    print(f"  Final epochs   : {final_epochs}")
    print(f"{sep}")
    print(f"\n  {'Fold':>4}  {'TrainF1':>8}  {'ValF1':>7}  {'Gap':>7}  {'Epoch':>6}")
    print(f"  {'-'*42}")
    for m in all_metrics:
        g = m["train_f1_macro"] - m["f1_macro"]
        print(f"  {m['fold']:>4}  {m['train_f1_macro']:>8.4f}  {m['f1_macro']:>7.4f}"
              f"  {g:>+7.4f}  {m['best_epoch']:>6}")
    print()
    print(f"  Model: {run_dir / 'lstm_momentum_v2.pt'}")
    print()
    print(f"  Next step — validasi marginal IC:")
    print(f"    1. Backtest: bandingkan LGBM-only vs LGBM+LSTM(momentum_v2)")
    print(f"    2. Jika WR atau PnL naik -> LSTM menambah informasi unik")
    print(f"    3. Jika tidak -> LSTM masih duplikasi LGBM signal")
    print()


if __name__ == "__main__":
    main()
