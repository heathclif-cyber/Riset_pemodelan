"""
pipeline/05_train_lstm_v2_style.py — Train LSTM "v2-style" (same features as LGBM)

Tujuan khusus untuk cascade_v2.5_hybrid:
- LSTM menggunakan **fitur yang persis sama** dengan LGBM entry model untuk versi tersebut.
- Label yang sama dengan LGBM (kolom 'label' dari swing labeling).
- Pendekatan sederhana seperti di era cascade_v2 yang perform lebih baik di live.

Scaling:
- Menggunakan **RobustScaler** (bukan StandardScaler).
- Alasan: Data crypto memiliki outlier ekstrem dan skala antar coin sangat berbeda (volume, CVD, OFI, dll).
- LSTM sangat sensitif terhadap skala → RobustScaler jauh lebih stabil.

Regularisasi (Round 4 - Light Round 2, 2026-05-31):
- Hidden=96, Dropout=0.45, WeightDecay=2e-4, LR=0.0007
- Tujuan: memperbaiki Round 2 (gap +0.36~0.40) tanpa membuat model underfit seperti Round 3.

Ini berbeda dengan jalur advanced momentum detector (yang sudah dipindah ke archive).

Cara pakai:
    # Probe 5 coin dulu (sangat disarankan untuk cek gap)
    python pipeline/05_train_lstm_v2_style.py --run-id cascade_v2.5_hybrid_pruned

    # Training penuh 21 coin
    python pipeline/05_train_lstm_v2_style.py --run-id cascade_v2.5_hybrid_pruned --all

Fix (2026-05-31):
  1. Sequences dibangun per-koin — tidak ada cross-coin contamination di SequenceDataset
  2. WeightedRandomSampler dihapus — double-weighting dengan class weights di loss dihilangkan
  3. Real timestamps digunakan untuk build_purged_folds — bukan dummy index
"""

import argparse
import gc
import json
import sys
import warnings
from pathlib import Path

# Suppress DirectML CPU fallback warnings — lerp_ tidak didukung DirectML,
# fallback ke CPU per-tensor. Dampak performa minimal untuk batch kecil.
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
    TRAINING_COINS, PROC_DIR, LABEL_DIR, TRAIN_CUTOFF_DATE,
    N_FOLDS, PURGE_GAP_BARS, MODEL_DIR,
    LSTM_SEQ_LEN, LSTM_BATCH_SIZE, LSTM_EPOCHS, LSTM_PATIENCE,
    LABEL_MAP,
    LSTM_V2_HIDDEN, LSTM_V2_LAYERS, LSTM_V2_DROPOUT,
    LSTM_V2_WEIGHT_DECAY, LSTM_V2_LR,
)
from core.models import TradingLSTM, save_lstm
from core.utils import setup_logger, get_lstm_device
from pipeline.shared import build_purged_folds

logger = setup_logger("05_train_lstm_v2_style")
DEVICE = get_lstm_device()


class PrebuiltSeqDataset(Dataset):
    """Simple Dataset untuk sequences yang sudah dibangun per-koin (shape: n, seq_len, feat)."""
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    def get_labels(self) -> np.ndarray:
        return self.y.numpy()


def fit_scaler(X: np.ndarray) -> RobustScaler:
    """Fit RobustScaler pada data 3D (n, seq_len, feat) — reshape ke 2D untuk fit."""
    n, s, f = X.shape
    scaler = RobustScaler()
    scaler.fit(X.reshape(-1, f))
    return scaler


def scale_X(X: np.ndarray, scaler: RobustScaler) -> np.ndarray:
    n, s, f = X.shape
    return scaler.transform(X.reshape(-1, f)).reshape(n, s, f).astype(np.float32)


def load_data_for_run(run_id: str, coins: list[str], feature_cols_path: str = None):
    """
    Load data dan bangun sequences per-koin sebelum digabung.

    FIX: Sebelumnya data di-concat dulu lalu di-pass ke SequenceDataset, menyebabkan
    sliding window memotong lintas batas koin (cross-coin contamination). Sekarang
    sequence dibangun per-koin lalu digabung — tidak ada leakage antar koin.

    Returns: (X, y, ts, feat_cols)
      X  — shape (n_samples, LSTM_SEQ_LEN, n_features)
      y  — shape (n_samples,)
      ts — shape (n_samples,) int64 nanoseconds UTC
    """
    if feature_cols_path:
        feat_cols_path = Path(feature_cols_path)
    else:
        run_dir = MODEL_DIR / "runs" / run_id
        feat_cols_path = run_dir / "feature_cols_v2.json"
        if not feat_cols_path.exists():
            feat_cols_path = MODEL_DIR / "feature_cols_v2.json"

    with open(feat_cols_path) as f:
        feat_cols = json.load(f)

    X_seqs, y_seqs, ts_seqs = [], [], []

    for coin in coins:
        path_v3 = LABEL_DIR / f"{coin}_features_v3.parquet"
        path_old = PROC_DIR / f"{coin}_engineered.parquet"

        if path_v3.exists():
            path = path_v3
        elif path_old.exists():
            path = path_old
        else:
            logger.warning(f"{coin}: data file not found, skip")
            continue

        df = pd.read_parquet(path)
        df = df[df.index < TRAIN_CUTOFF_DATE]
        if "label" not in df.columns:
            logger.warning(f"{coin}: no label column, skip")
            continue

        # Merge HMM regime labels jika tersedia dan dibutuhkan
        if "hmm_regime_enc" in feat_cols:
            regime_path = LABEL_DIR / f"{coin}_regime_h1.parquet"
            if regime_path.exists():
                try:
                    reg = pd.read_parquet(regime_path)
                    if "hmm_regime_enc" in df.columns:
                        df = df.drop(columns=["hmm_regime_enc"])
                    df = df.join(reg[["hmm_regime_enc"]], how="left")
                    df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(1).astype("int32")
                except Exception as e:
                    logger.warning(f"{coin}: regime merge gagal ({e}), fill 1")
                    df["hmm_regime_enc"] = 1

        available = [c for c in feat_cols if c in df.columns]
        df = df[available + ["label"]].dropna()

        mask = df["label"].isin(LABEL_MAP)
        if (~mask).sum():
            df = df[mask].copy()

        if len(df) < LSTM_SEQ_LEN:
            logger.warning(f"{coin}: only {len(df)} rows after filter, skip")
            continue

        X_c  = df[available].values.astype(np.float32)
        y_c  = df["label"].map(LABEL_MAP).astype(np.int64).values
        ts_c = df.index.astype(np.int64).values

        # Sequence dibangun dalam batas koin ini saja — tidak ada kontaminasi lintas koin
        for i in range(LSTM_SEQ_LEN - 1, len(X_c)):
            X_seqs.append(X_c[i - LSTM_SEQ_LEN + 1 : i + 1])
            y_seqs.append(y_c[i])
            ts_seqs.append(ts_c[i])

    if not X_seqs:
        raise ValueError("No sequences built — check data paths and LSTM_SEQ_LEN")

    X  = np.stack(X_seqs)                        # (n_samples, seq_len, n_features) float32
    y  = np.array(y_seqs, dtype=np.int64)
    ts = np.array(ts_seqs, dtype=np.int64)

    # Sort by real timestamp untuk fold ordering yang konsisten
    order = np.argsort(ts)
    return X[order], y[order], ts[order], feat_cols


def compute_class_weights(y: np.ndarray) -> torch.Tensor:
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = {c: total / (len(classes) * count) for c, count in zip(classes, counts)}
    return torch.tensor([weights.get(i, 1.0) for i in range(3)], dtype=torch.float32).to(DEVICE)


def train_one_fold(X_tr: np.ndarray, y_tr: np.ndarray,
                   X_te: np.ndarray, y_te: np.ndarray,
                   fold_num: int):
    """
    FIX: WeightedRandomSampler dihapus — double-weighting dengan class weights di loss
    menyebabkan model over-fit ke minority class di training. Sekarang class balancing
    hanya lewat CrossEntropyLoss(weight=...), konsisten dengan fix di v4 series.
    """
    # Fit scaler hanya dari training fold — cegah val leakage
    n_features = X_tr.shape[2]   # simpan SEBELUM del X_tr
    n_train    = len(X_tr)        # simpan SEBELUM del X_tr
    n_val      = len(X_te)        # simpan SEBELUM del X_te
    fold_scaler = fit_scaler(X_tr)
    X_tr_sc = scale_X(X_tr, fold_scaler)
    del X_tr   # bebaskan SEGERA — cegah X_tr + X_tr_sc exist bersamaan (fold 6+ = 5-7 GB!)
    gc.collect()
    X_te_sc = scale_X(X_te, fold_scaler)
    del X_te   # bebaskan original setelah scaled version dibuat
    gc.collect()

    train_ds = PrebuiltSeqDataset(X_tr_sc, y_tr)
    test_ds  = PrebuiltSeqDataset(X_te_sc, y_te)

    # Shuffle saja, tanpa WeightedRandomSampler
    train_loader = DataLoader(train_ds, batch_size=LSTM_BATCH_SIZE, shuffle=True,  num_workers=0)
    test_loader  = DataLoader(test_ds,  batch_size=LSTM_BATCH_SIZE, shuffle=False, num_workers=0)

    # n_features sudah di-set di atas (sebelum del X_tr)
    model     = TradingLSTM(n_features, LSTM_V2_HIDDEN, LSTM_V2_LAYERS, LSTM_V2_DROPOUT).to(DEVICE)
    cw        = compute_class_weights(y_tr)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_V2_LR, weight_decay=LSTM_V2_WEIGHT_DECAY, foreach=False)

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
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                preds = model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(yb.numpy())

        f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
        if f1 > best_f1:
            best_f1       = f1
            best_state    = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_count = 0
            best_epoch    = epoch
        else:
            patience_count += 1
            if patience_count >= LSTM_PATIENCE:
                logger.info(f"[Fold {fold_num}] Early stop at epoch {epoch} (best F1={best_f1:.4f})")
                break

        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"[Fold {fold_num}] Epoch {epoch:>2} | F1-macro={f1:.4f} | Best={best_f1:.4f}")

    model.load_state_dict(best_state)
    model.eval()

    # === Validation metrics ===
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            preds = model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())

    val_f1_macro = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    val_f1_per   = f1_score(all_labels, all_preds, average=None, zero_division=0, labels=[0, 1, 2])
    val_acc      = float(accuracy_score(all_labels, all_preds))

    # === Training metrics (untuk deteksi overfitting) ===
    # Gunakan loader terpisah (shuffle=False) agar coverage 100% dan deterministik
    eval_train_loader = DataLoader(train_ds, batch_size=LSTM_BATCH_SIZE, shuffle=False, num_workers=0)
    train_preds, train_labels_list = [], []
    with torch.no_grad():
        for xb, yb in eval_train_loader:
            preds = model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy()
            train_preds.extend(preds)
            train_labels_list.extend(yb.numpy())

    train_f1_macro = float(f1_score(train_labels_list, train_preds, average="macro", zero_division=0))
    train_f1_per   = f1_score(train_labels_list, train_preds, average=None, zero_division=0, labels=[0, 1, 2])
    train_acc      = float(accuracy_score(train_labels_list, train_preds))

    metrics = {
        "fold": fold_num,
        "n_train": n_train,
        "n_val": n_val,
        "best_epoch": best_epoch,
        "train_accuracy": round(train_acc, 4),
        "train_f1_macro": round(train_f1_macro, 4),
        "train_f1_SHORT": round(float(train_f1_per[0]), 4),
        "train_f1_FLAT":  round(float(train_f1_per[1]), 4),
        "train_f1_LONG":  round(float(train_f1_per[2]), 4),
        "accuracy": round(val_acc, 4),
        "f1_macro": round(val_f1_macro, 4),
        "f1_SHORT": round(float(val_f1_per[0]), 4),
        "f1_FLAT":  round(float(val_f1_per[1]), 4),
        "f1_LONG":  round(float(val_f1_per[2]), 4),
    }

    gap = train_f1_macro - val_f1_macro
    logger.info(
        f"[Fold {fold_num}] Train F1={train_f1_macro:.4f} | Val F1={val_f1_macro:.4f} "
        f"| Gap={gap:+.4f} | Best Epoch={best_epoch}"
    )
    return model, fold_scaler, metrics


def retrain_final(X_all: np.ndarray, y_all: np.ndarray, n_epochs: int) -> tuple[TradingLSTM, RobustScaler]:
    """
    Train final LSTM on 100% data (no validation split).
    FIX: WeightedRandomSampler dihapus — konsisten dengan train_one_fold.
    """
    n_features = X_all.shape[2]          # simpan SEBELUM del X_all
    final_scaler = fit_scaler(X_all)
    X_sc = scale_X(X_all, final_scaler)
    del X_all   # KRITIS: X_all (2.05 GB) + X_sc (2.05 GB) = 4.1 GB — bebaskan segera
    gc.collect()

    ds     = PrebuiltSeqDataset(X_sc, y_all)
    loader = DataLoader(ds, batch_size=LSTM_BATCH_SIZE, shuffle=True, num_workers=0)

    model     = TradingLSTM(n_features, LSTM_V2_HIDDEN, LSTM_V2_LAYERS, LSTM_V2_DROPOUT).to(DEVICE)
    cw        = compute_class_weights(y_all)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_V2_LR, weight_decay=LSTM_V2_WEIGHT_DECAY, foreach=False)

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
            logger.info(f"[Final] Epoch {epoch:>3}/{n_epochs} | loss={total_loss / len(loader):.4f}")

    model.eval()
    return model, final_scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--feature-cols", default=None, help="Path ke custom feature columns JSON")
    args = parser.parse_args()

    coins = TRAINING_COINS if args.all else TRAINING_COINS[:5]
    logger.info(f"Training LSTM v2-style for run: {args.run_id}")
    logger.info(
        f"[Reg] Hidden={LSTM_V2_HIDDEN}, Layers={LSTM_V2_LAYERS}, "
        f"Dropout={LSTM_V2_DROPOUT}, WeightDecay={LSTM_V2_WEIGHT_DECAY}, LR={LSTM_V2_LR}"
    )
    logger.info(f"[CV] N_FOLDS={N_FOLDS}, PURGE_GAP_BARS={PURGE_GAP_BARS}")
    logger.info("[Fix] per-coin sequences | no WeightedRandomSampler | real timestamps for purging")

    # Fix random seed untuk reproducibility — cegah bad initialization
    import torch as _torch
    _torch.manual_seed(42)
    np.random.seed(42)

    # FIX: load_data_for_run sekarang return (X_3d, y, ts_real, feat_cols)
    X, y, ts, feat_cols = load_data_for_run(args.run_id, coins, args.feature_cols)
    n_features = X.shape[2]

    logger.info(f"Dataset: X={X.shape} | y={y.shape} | features={n_features}")
    for lbl_str, lbl_int in LABEL_MAP.items():
        n = (y == lbl_int).sum()
        logger.info(f"  {lbl_str:5}: {n:>8,} ({n / len(y) * 100:.1f}%)")

    run_dir = MODEL_DIR / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    with open(run_dir / "lstm_v2_style_feature_cols.json", "w") as f:
        json.dump(feat_cols, f, indent=2)

    # FIX: gunakan real timestamps (bukan dummy) untuk purging yang akurat
    ts_index = pd.to_datetime(ts, unit="ns", utc=True)
    folds = build_purged_folds(ts_index, n_folds=N_FOLDS, purge=PURGE_GAP_BARS)

    all_metrics = []
    for fold_idx, (tr_idx, te_idx) in enumerate(folds):
        model, scaler, metrics = train_one_fold(
            X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], fold_idx + 1
        )
        all_metrics.append(metrics)

    if all_metrics:
        avg_best_epoch = int(np.median([m.get("best_epoch", 30) for m in all_metrics]))
        final_epochs   = max(20, min(avg_best_epoch + 5, LSTM_EPOCHS))
    else:
        final_epochs = max(30, LSTM_EPOCHS // 2)

    logger.info(f"Retraining final LSTM on 100% data for ~{final_epochs} epochs (median best from CV)...")
    final_model, final_scaler = retrain_final(X, y, final_epochs)

    model_path  = run_dir / "lstm_v2_style.pt"
    scaler_path = run_dir / "lstm_v2_style_scaler.pkl"
    save_lstm(final_model, model_path)
    joblib.dump(final_scaler, scaler_path)

    logger.info(f"Model saved: {model_path}")

    meta = {
        "run_id": args.run_id,
        "model_type": "lstm_v2_style",
        "n_features": n_features,
        "seq_len": LSTM_SEQ_LEN,
        "hidden": LSTM_V2_HIDDEN,
        "layers": LSTM_V2_LAYERS,
        "dropout": LSTM_V2_DROPOUT,
        "weight_decay": LSTM_V2_WEIGHT_DECAY,
        "lr": LSTM_V2_LR,
        "scaler_type": "RobustScaler",
        "final_epochs_trained": final_epochs,
        "cv_folds": len(all_metrics),
        "feature_file": "lstm_v2_style_feature_cols.json",
        "fixes_applied": [
            "per_coin_sequences",
            "no_weighted_sampler",
            "real_timestamps_purging",
            "fold_scaler",
        ],
    }
    with open(run_dir / "lstm_v2_style_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    with open(run_dir / "lstm_v2_style_cv_results.json", "w") as f:
        json.dump({
            "run_id": args.run_id,
            "n_folds": N_FOLDS,
            "gap_bars": PURGE_GAP_BARS,
            "metrics": all_metrics,
        }, f, indent=2)

    # Summary
    val_f1s   = [m["f1_macro"]       for m in all_metrics]
    train_f1s = [m["train_f1_macro"] for m in all_metrics]
    gaps      = [m["train_f1_macro"] - m["f1_macro"] for m in all_metrics]

    sep = "=" * 65
    print(f"\n{sep}")
    print(f"  LSTM V2-STYLE TRAINING SELESAI — {args.run_id}")
    print(f"{sep}")
    print(f"  Mean Val F1   : {np.mean(val_f1s):.4f} +/- {np.std(val_f1s):.4f}")
    print(f"  Mean Train F1 : {np.mean(train_f1s):.4f}")
    print(f"  Mean Gap      : {np.mean(gaps):+.4f}  (target: < +0.10)")
    print(f"  Final epochs  : {final_epochs}")
    print(f"{sep}")
    print(f"\n  {'Fold':>4}  {'TrainF1':>8}  {'ValF1':>7}  {'Gap':>7}  {'Epoch':>6}")
    print(f"  {'-' * 42}")
    for m in all_metrics:
        g = m["train_f1_macro"] - m["f1_macro"]
        print(f"  {m['fold']:>4}  {m['train_f1_macro']:>8.4f}  {m['f1_macro']:>7.4f}  {g:>+7.4f}  {m['best_epoch']:>6}")
    print()


if __name__ == "__main__":
    main()
