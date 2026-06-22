"""
pipeline/05_train_lstm_ic32_rv2_momentum.py
LSTM Momentum Veto untuk ic32_regime_v2 — identik pendekatan V1, seq_len=72.

Label  : momentum_v2_label (BEARISH=0/NEUTRAL=1/BULLISH=2, flow voting)
Gate   : semua bar (bukan complement filter)
Loss   : WeightedCrossEntropy
Peran  : hard_consensus VETO — LGBM LONG + LSTM BEARISH>0.5 → cancel

Optimasi memori: raw features disimpan per-koin (n×11), sequences
dibuat on-the-fly di DataLoader — tidak pre-build 250k × 72 × 11 array.

Usage:
  python pipeline/05_train_lstm_ic32_rv2_momentum.py         # 5 koin probe
  python pipeline/05_train_lstm_ic32_rv2_momentum.py --all   # 21 koin full
"""
import argparse, gc, json, sys, warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, LABEL_DIR, TRAIN_CUTOFF_DATE, MODEL_DIR,
    N_FOLDS, LSTM_SEQ_LEN, LSTM_BATCH_SIZE,
    LSTM_EPOCHS, LSTM_PATIENCE,
    LSTM_V2_HIDDEN, LSTM_V2_LAYERS, LSTM_V2_DROPOUT,
    LSTM_V2_WEIGHT_DECAY, LSTM_V2_LR,
)

PURGE_GAP_BARS = LSTM_SEQ_LEN  # 72 bar — window tidak overlap antara train/val
from core.models import TradingLSTM, save_lstm
from core.utils import setup_logger, get_lstm_device, ensure_utc_index
from pipeline.shared import build_purged_folds

logger = setup_logger("05_train_lstm_ic32_rv2_momentum")
DEVICE = get_lstm_device()

RUN_NAME   = "ic32_rv2_lstm_momentum"
LABEL_COL  = "momentum_v2_label"
LABEL_MAP_FLOW = {0: "BEARISH", 1: "NEUTRAL", 2: "BULLISH"}

FEAT_COLS = [
    "ofi_z_score", "ofi_acceleration", "cvd_momentum_adv", "absorption_z",
    "volume_delta", "vol_ratio_20",
    "log_ret_1", "log_ret_5", "log_ret_20",
    "rsi_6", "btc_h1_return",
]


# ─── On-the-fly sequence dataset ─────────────────────────────────────────────

class CoinSeqDataset(Dataset):
    """
    Dataset multi-koin dengan on-the-fly slicing — hemat memori.
    Menyimpan raw arrays (n, n_feat) per koin dan boundaries,
    sequences dibentuk di __getitem__ agar tidak pre-build n_seq × seq_len array.
    Cross-koin contamination dicegah via valid_indices (hanya index ≥ seq_len dari awal koin).
    """
    def __init__(self, X_list: list[np.ndarray], y_list: list[np.ndarray],
                 seq_len: int, scaler: RobustScaler | None = None):
        self.seq_len = seq_len
        # Gabung dengan offset per koin agar boundary jelas
        offsets, valid_idx, labels = [], [], []
        offset = 0
        X_parts, y_parts = [], []
        for Xc, yc in zip(X_list, y_list):
            n = len(Xc)
            X_parts.append(Xc)
            y_parts.append(yc)
            for i in range(seq_len - 1, n):
                valid_idx.append(offset + i)
                labels.append(yc[i])
            offsets.append(offset)
            offset += n

        self.X = np.concatenate(X_parts, axis=0).astype(np.float32)
        # Scale setelah concat
        if scaler is not None:
            n_total, n_feat = self.X.shape
            self.X = scaler.transform(self.X)
        self.X = torch.from_numpy(self.X.astype(np.float32))

        self.valid_idx = np.array(valid_idx, dtype=np.int64)
        self.labels    = np.array(labels, dtype=np.int64)
        # Boundary set: index awal setiap koin — tidak boleh span koin
        self.boundaries = set(offsets)

    def __len__(self): return len(self.valid_idx)

    def __getitem__(self, idx):
        end = int(self.valid_idx[idx])
        seq = self.X[end - self.seq_len + 1: end + 1]  # (seq_len, n_feat)
        return seq, self.labels[idx]

    def get_labels(self): return self.labels


# ─── Data loading ────────────────────────────────────────────────────────────

def load_data(coins: list[str]):
    """
    Return: X_list (list of ndarray n×11 per koin),
            y_list (list of ndarray n per koin),
            ts (global timestamp array untuk CV splitting)
    """
    X_list, y_list, ts_all = [], [], []
    skipped = []

    btc_ret = None
    btc_fp = LABEL_DIR / "BTCUSDT_features_v3.parquet"
    if btc_fp.exists():
        btc_df = pd.read_parquet(btc_fp, columns=["log_ret_1"])
        btc_df = ensure_utc_index(btc_df).sort_index()
        btc_df = btc_df[btc_df.index < TRAIN_CUTOFF_DATE]
        btc_ret = btc_df["log_ret_1"].rename("btc_h1_return")
        logger.info(f"BTC loaded: {len(btc_ret):,} bars")

    for coin in coins:
        fp = LABEL_DIR / f"{coin}_features_v3.parquet"
        lp = LABEL_DIR / f"{coin}_momentum_v2_labels.parquet"
        if not fp.exists() or not lp.exists():
            skipped.append(coin); continue

        df = pd.read_parquet(fp).sort_index()
        df = ensure_utc_index(df)
        df = df[df.index < TRAIN_CUTOFF_DATE]
        ldf = pd.read_parquet(lp).sort_index()
        ldf = ensure_utc_index(ldf)
        df  = df.join(ldf[[LABEL_COL]], how="inner").dropna(subset=[LABEL_COL])
        if len(df) < LSTM_SEQ_LEN + 50:
            skipped.append(coin); continue

        if btc_ret is not None and "btc_h1_return" not in df.columns:
            df = df.join(btc_ret, how="left")
        if "btc_h1_return" not in df.columns:
            df["btc_h1_return"] = 0.0

        avail = [c for c in FEAT_COLS if c in df.columns]
        if len(avail) < len(FEAT_COLS):
            logger.warning(f"[{coin}] missing: {[c for c in FEAT_COLS if c not in avail]}")
            skipped.append(coin); continue

        feat_vals = []
        for c in avail:
            v = df[c].ffill().fillna(0).values.astype(np.float32)
            feat_vals.append(v)

        Xc = np.column_stack(feat_vals).astype(np.float32)   # (n, 11)
        yc = df[LABEL_COL].values.astype(np.int64)
        tc = df.index

        X_list.append(Xc)
        y_list.append(yc)
        ts_all.append(tc)

        dist = yc[LSTM_SEQ_LEN - 1:]
        logger.info(
            f"[{coin}] {len(Xc):,} bars | "
            f"BEARISH={(dist==0).mean()*100:.1f}% "
            f"NEUTRAL={(dist==1).mean()*100:.1f}% "
            f"BULLISH={(dist==2).mean()*100:.1f}%"
        )

    if skipped:
        logger.warning(f"Skipped: {skipped}")
    if not X_list:
        raise ValueError("No data loaded.")

    # ts global: semua timestamp (untuk CV splitting)
    ts_global = np.concatenate([t.values for t in ts_all])
    total_seq = sum(len(Xc) - LSTM_SEQ_LEN + 1 for Xc in X_list)
    logger.info(f"Total: {total_seq:,} valid sequences, {len(X_list)} coins")
    return X_list, y_list, ts_global


# ─── CV fold helpers ──────────────────────────────────────────────────────────

def subset_lists(X_list, y_list, ts_global, indices):
    """Kembalikan X_list/y_list subset yang sesuai dengan index set dari ts_global."""
    # indices adalah row-level indices dari ts_global (per bar, bukan per sequence)
    idx_set = set(indices.tolist())
    result_X, result_y = [], []
    offset = 0
    for Xc, yc in zip(X_list, y_list):
        n = len(Xc)
        local = sorted(i - offset for i in range(offset, offset + n) if i in idx_set)
        if len(local) >= LSTM_SEQ_LEN:
            # Ambil slice: dari min ke max+1 agar sequence window tetap valid dalam koin
            lo, hi = local[0], local[-1] + 1
            # Pastikan lo cukup besar untuk window
            lo = max(0, lo)
            result_X.append(Xc[lo:hi])
            result_y.append(yc[lo:hi])
        offset += n
    return result_X, result_y


# ─── Train one fold ───────────────────────────────────────────────────────────

def train_one_fold(X_tr_list, y_tr_list, X_te_list, y_te_list, fold_num):
    # Fit scaler pada training data (Aturan 3: scaler di dalam loop fold)
    X_tr_raw = np.concatenate(X_tr_list, axis=0)
    fold_scaler = RobustScaler()
    fold_scaler.fit(X_tr_raw)
    del X_tr_raw; gc.collect()

    tr_ds = CoinSeqDataset(X_tr_list, y_tr_list, LSTM_SEQ_LEN, fold_scaler)
    te_ds = CoinSeqDataset(X_te_list, y_te_list, LSTM_SEQ_LEN, fold_scaler)
    tr_ld = DataLoader(tr_ds, batch_size=LSTM_BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=False)
    te_ld = DataLoader(te_ds, batch_size=LSTM_BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=False)

    y_tr_flat = tr_ds.get_labels()
    classes, counts = np.unique(y_tr_flat, return_counts=True)
    w = torch.tensor(
        [len(y_tr_flat) / (3 * counts[list(classes).index(i)]) for i in range(3)],
        dtype=torch.float32,
    ).to(DEVICE)

    model = TradingLSTM(len(FEAT_COLS), LSTM_V2_HIDDEN, LSTM_V2_LAYERS, LSTM_V2_DROPOUT).to(DEVICE)
    criterion = nn.CrossEntropyLoss(weight=w)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LSTM_V2_LR,
        weight_decay=LSTM_V2_WEIGHT_DECAY, foreach=False,
    )

    best_f1, best_state, patience_count, best_epoch = -1.0, None, 0, 0
    for epoch in range(1, LSTM_EPOCHS + 1):
        model.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            criterion(model(xb), yb).backward()
            optimizer.step()

        model.eval()
        pv, lv = [], []
        with torch.no_grad():
            for xb, yb in te_ld:
                pv.extend(model(xb.to(DEVICE)).argmax(1).cpu().numpy())
                lv.extend(yb.numpy())
        f1 = float(f1_score(lv, pv, average="macro", zero_division=0))

        if f1 > best_f1:
            best_f1, best_state, patience_count, best_epoch = (
                f1, {k: v.cpu() for k, v in model.state_dict().items()}, 0, epoch
            )
        else:
            patience_count += 1
            if patience_count >= LSTM_PATIENCE:
                break

        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"  Fold {fold_num} epoch {epoch:3d}: val_f1={f1:.4f} (best={best_f1:.4f})")

    logger.info(f"Fold {fold_num} done: F1={best_f1:.4f} @ epoch {best_epoch}")
    model.load_state_dict(best_state)
    return model, fold_scaler, best_f1, best_epoch


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    coins = TRAINING_COINS if args.all else TRAINING_COINS[:5]
    logger.info(f"=== {RUN_NAME} | {len(coins)} coins | seq={LSTM_SEQ_LEN} purge={PURGE_GAP_BARS} device={DEVICE} ===")

    run_dir = MODEL_DIR / "runs" / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    X_list, y_list, ts_global = load_data(coins)
    n_features = X_list[0].shape[1]
    total_seq = sum(len(Xc) - LSTM_SEQ_LEN + 1 for Xc in X_list)
    logger.info(f"Dataset: {total_seq:,} sequences x {LSTM_SEQ_LEN} x {n_features}")

    # Build CV folds dari ts_global (per-bar timestamps)
    ts_idx = pd.DatetimeIndex(ts_global)
    folds = build_purged_folds(ts_idx, n_folds=N_FOLDS, purge=PURGE_GAP_BARS)
    logger.info(f"CV folds: {N_FOLDS}, purge={PURGE_GAP_BARS}")

    fold_results, oof_proba_rows = [], []

    for fold_idx, (tr_idx, te_idx) in enumerate(folds, 1):
        logger.info(f"--- Fold {fold_idx}/{N_FOLDS}: train_bars={len(tr_idx):,} val_bars={len(te_idx):,} ---")

        X_tr_list, y_tr_list = subset_lists(X_list, y_list, ts_global, tr_idx)
        X_te_list, y_te_list = subset_lists(X_list, y_list, ts_global, te_idx)

        n_tr = sum(max(0, len(Xc) - LSTM_SEQ_LEN + 1) for Xc in X_tr_list)
        n_te = sum(max(0, len(Xc) - LSTM_SEQ_LEN + 1) for Xc in X_te_list)
        logger.info(f"  sequences: train={n_tr:,} val={n_te:,}")

        model, fold_scaler, best_f1, best_epoch = train_one_fold(
            X_tr_list, y_tr_list, X_te_list, y_te_list, fold_idx,
        )

        # Collect OOF predictions
        te_ds = CoinSeqDataset(X_te_list, y_te_list, LSTM_SEQ_LEN, fold_scaler)
        te_ld = DataLoader(te_ds, batch_size=LSTM_BATCH_SIZE, shuffle=False, num_workers=0)
        model.eval()
        with torch.no_grad():
            for xb, yb in te_ld:
                proba = torch.softmax(model(xb.to(DEVICE)), dim=1).cpu().numpy()
                for p, l in zip(proba, yb.numpy()):
                    oof_proba_rows.append({"p0": p[0], "p1": p[1], "p2": p[2], "label": int(l)})

        fold_results.append({
            "fold": fold_idx, "n_train_seq": n_tr, "n_val_seq": n_te,
            "best_f1_macro": round(best_f1, 4), "best_epoch": best_epoch,
        })
        gc.collect()

    # OOF global F1
    oof_df = pd.DataFrame(oof_proba_rows)
    oof_pred = oof_df[["p0", "p1", "p2"]].values.argmax(1)
    oof_true = oof_df["label"].values
    global_f1 = float(f1_score(oof_true, oof_pred, average="macro", zero_division=0))
    logger.info(f"=== OOF global F1 macro: {global_f1:.4f} ===")
    for cls, name in LABEL_MAP_FLOW.items():
        mask = oof_true == cls
        if mask.any():
            f1_c = float(f1_score(oof_true, oof_pred, labels=[cls], average="macro", zero_division=0))
            logger.info(f"  {name}: F1={f1_c:.4f}  n={mask.sum():,}")

    # Train final model pada semua data
    logger.info("Training final model on all data...")
    X_all_raw = np.concatenate(X_list, axis=0)
    final_scaler = RobustScaler().fit(X_all_raw); del X_all_raw; gc.collect()
    all_ds = CoinSeqDataset(X_list, y_list, LSTM_SEQ_LEN, final_scaler)
    all_ld = DataLoader(all_ds, batch_size=LSTM_BATCH_SIZE, shuffle=True, num_workers=0)

    y_all = all_ds.get_labels()
    classes, counts = np.unique(y_all, return_counts=True)
    w = torch.tensor(
        [len(y_all) / (3 * counts[list(classes).index(i)]) for i in range(3)],
        dtype=torch.float32,
    ).to(DEVICE)
    final_model = TradingLSTM(n_features, LSTM_V2_HIDDEN, LSTM_V2_LAYERS, LSTM_V2_DROPOUT).to(DEVICE)
    criterion  = nn.CrossEntropyLoss(weight=w)
    optimizer  = torch.optim.Adam(final_model.parameters(), lr=LSTM_V2_LR, weight_decay=LSTM_V2_WEIGHT_DECAY, foreach=False)
    median_ep  = int(np.median([r["best_epoch"] for r in fold_results]))
    logger.info(f"Final: {median_ep} epochs (median best epoch dari fold)")
    for epoch in range(1, median_ep + 1):
        final_model.train()
        for xb, yb in all_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
            criterion(final_model(xb), yb).backward()
            optimizer.step()
        if epoch % 5 == 0:
            logger.info(f"  Final epoch {epoch}/{median_ep}")

    save_lstm(final_model, run_dir / "lstm_momentum.pt")
    joblib.dump(final_scaler, run_dir / "lstm_momentum_scaler.pkl")
    with open(run_dir / "lstm_v4_selected_features.json", "w") as f:
        json.dump(FEAT_COLS, f, indent=2)

    meta = {
        "run_name": RUN_NAME,
        "model_type": "lstm_momentum_v2_style",
        "role": "hard_consensus_veto",
        "label_type": "momentum_flow_v2",
        "label_source": f"{LABEL_COL} from momentum_v2_labels.parquet",
        "label_map": {"BEARISH": 0, "NEUTRAL": 1, "BULLISH": 2},
        "sample_filter": "all_bars",
        "n_features": len(FEAT_COLS),
        "features": FEAT_COLS,
        "seq_len": LSTM_SEQ_LEN,
        "purge_gap": PURGE_GAP_BARS,
        "hidden": LSTM_V2_HIDDEN,
        "layers": LSTM_V2_LAYERS,
        "dropout": LSTM_V2_DROPOUT,
        "n_folds": N_FOLDS,
        "n_samples_approx": int(total_seq),
        "oof_f1_macro": round(global_f1, 4),
        "fold_results": fold_results,
        "final_epochs": median_ep,
        "created": datetime.now().isoformat(),
        "train_cutoff": str(TRAIN_CUTOFF_DATE.date()),
        "methodology": "purged CV OOF, scaler per fold (Aturan 3), all bars, momentum labels, on-the-fly seq",
        "inference_note": "seq_len=72 matches production _lstm_momentum_seq_len default=72",
    }
    with open(run_dir / f"{RUN_NAME}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    logger.info(f"Saved to {run_dir}")
    print(f"\n=== DONE: OOF F1 macro={global_f1:.4f} ===")
    for r in fold_results:
        print(f"  Fold {r['fold']}: F1={r['best_f1_macro']}  ep={r['best_epoch']}")


if __name__ == "__main__":
    main()
