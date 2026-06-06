"""
pipeline/09_train_lstm_meta.py — Train LSTM Meta-Model (Binary)

Phase 2 LSTM Binary Meta-Labeling - Step 2.

Binary classifier: prediksi apakah trade akan profit (is_good_trade=1)
menggunakan 40-bar sequence sebelum entry.

Arsitektur: LSTM + Attention + Dropout
Output: P(good_trade) — sigmoid

Usage:
  python pipeline/09_train_lstm_meta.py
  python pipeline/09_train_lstm_meta.py --run-id lstm_meta_v1
"""

import argparse, gc, json, sys, warnings, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.preprocessing import RobustScaler
import joblib

from config import MODEL_DIR, N_FOLDS, PURGE_GAP_BARS
from core.utils import setup_logger
from pipeline.shared import build_purged_folds

logger = setup_logger("09_lstm_meta")

# ─── Config ──────────────────────────────────────────────────────────────────
META_LSTM_HIDDEN = 96
META_LSTM_LAYERS = 2
META_LSTM_DROPOUT = 0.40
META_LSTM_LR = 0.0007
META_LSTM_WD = 2e-4
META_LSTM_EPOCHS = 80
META_LSTM_PATIENCE = 12
META_BATCH_SIZE = 256

DEVICE = torch.device("cpu")


# ─── Attention ────────────────────────────────────────────────────────────────
class AttentionPooling(nn.Module):
    """Soft attention over time steps → weighted average."""
    def __init__(self, hidden_dim):
        super().__init__()
        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, lstm_out):
        # lstm_out: (batch, seq, hidden)
        weights = torch.softmax(self.attn(lstm_out), dim=1)  # (batch, seq, 1)
        pooled = (lstm_out * weights).sum(dim=1)  # (batch, hidden)
        return pooled


# ─── Model ────────────────────────────────────────────────────────────────────
class MetaLSTM(nn.Module):
    def __init__(self, n_features, hidden, num_layers, dropout):
        super().__init__()
        self.lstm = nn.LSTM(
            n_features, hidden, num_layers,
            batch_first=True, dropout=dropout if num_layers > 1 else 0,
        )
        self.attention = AttentionPooling(hidden)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, 1)

    def forward(self, x):
        # x: (batch, seq, features)
        out, _ = self.lstm(x)       # (batch, seq, hidden)
        pooled = self.attention(out)  # (batch, hidden)
        pooled = self.dropout(pooled)
        return torch.sigmoid(self.classifier(pooled)).squeeze(-1)


# ─── Dataset ──────────────────────────────────────────────────────────────────
class MetaSeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.float32))

    def __len__(self): return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ─── Training ─────────────────────────────────────────────────────────────────
def fit_scaler(X):
    n, s, f = X.shape
    scl = RobustScaler()
    scl.fit(X.reshape(-1, f))
    return scl


def scale_X(X, scl):
    n, s, f = X.shape
    return scl.transform(X.reshape(-1, f)).reshape(n, s, f).astype(np.float32)


def train_fold(X_tr, y_tr, X_te, y_te, fold_num):
    scaler = fit_scaler(X_tr)
    X_tr_s = scale_X(X_tr, scaler); X_te_s = scale_X(X_te, scaler)

    tr_ds = MetaSeqDataset(X_tr_s, y_tr)
    te_ds = MetaSeqDataset(X_te_s, y_te)
    tr_ld = DataLoader(tr_ds, batch_size=META_BATCH_SIZE, shuffle=True)
    te_ld = DataLoader(te_ds, batch_size=META_BATCH_SIZE, shuffle=False)

    model = MetaLSTM(
        X_tr.shape[2], META_LSTM_HIDDEN, META_LSTM_LAYERS, META_LSTM_DROPOUT
    ).to(DEVICE)

    # Class weight: balanced
    n_pos = (y_tr == 1).sum()
    n_neg = (y_tr == 0).sum()
    pos_weight = torch.tensor([n_neg / max(n_pos, 1)]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=META_LSTM_LR, weight_decay=META_LSTM_WD,
    )

    best_auc, best_state, patience = 0.0, None, 0
    for epoch in range(1, META_LSTM_EPOCHS + 1):
        model.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for xb, yb in te_ld:
                p = model(xb.to(DEVICE)).cpu().numpy()
                preds.extend(p); labels.extend(yb.numpy())

        auc = roc_auc_score(labels, preds)
        if auc > best_auc:
            best_auc = auc
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= META_LSTM_PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    # Final metrics
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in te_ld:
            preds.extend(model(xb.to(DEVICE)).cpu().numpy())
            labels.extend(yb.numpy())
    preds_bin = (np.array(preds) >= 0.5).astype(int)
    f1 = f1_score(labels, preds_bin)
    acc = accuracy_score(labels, preds_bin)

    return model, scaler, {"fold": fold_num, "auc": round(best_auc, 4),
                           "f1": round(f1, 4), "acc": round(acc, 4)}


# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="lstm_meta_v1")
    args = parser.parse_args()

    data_dir = Path("data/meta_labels_v2")
    X = np.load(data_dir / "meta_sequences_training_v2.npy")
    labels_df = pd.read_parquet(data_dir / "meta_labels_training_v2.parquet")
    y = labels_df["is_good_trade"].values.astype(np.float32)

    print(f"\n{'='*60}")
    print(f"  LSTM META-MODEL TRAINING | {args.run_id}")
    print(f"  Samples: {len(X)} | Seq: {X.shape[1]}×{X.shape[2]}")
    print(f"  Good trades: {y.sum():.0f} ({y.mean()*100:.1f}%)")
    print(f"{'='*60}\n")

    # Build purged folds by timestamp
    ts = pd.DatetimeIndex(labels_df["timestamp"])
    folds = build_purged_folds(ts, N_FOLDS, PURGE_GAP_BARS)

    all_metrics = []
    for fi, (tr_idx, te_idx) in enumerate(folds):
        if len(te_idx) < 5: continue
        _, _, m = train_fold(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], fi+1)
        all_metrics.append(m)
        logger.info(f"Fold {fi+1}: AUC={m['auc']:.4f} F1={m['f1']:.4f} Acc={m['acc']:.4f}")

    # Summary
    aucs = [m["auc"] for m in all_metrics]
    f1s = [m["f1"] for m in all_metrics]
    print(f"\n{'='*60}")
    print(f"  META-MODEL RESULTS — {args.run_id}")
    print(f"  Mean AUC: {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")
    print(f"  Mean F1 : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print(f"  Baseline: 0.500 (random)")
    print(f"  Gain    : {np.mean(aucs) - 0.5:+.4f}")
    print(f"{'='*60}")

    if np.mean(aucs) > 0.55:
        print(f"\n  [OK] AUC > 0.55 - LSTM meta-model shows predictive power!")
        print(f"  Next: integrate with LGBM cascade -> backtest ensemble")
    else:
        print(f"\n  [X] AUC <= 0.55 - LSTM cannot predict trade quality")
        print(f"  Revisit: features, seq_len, or label definition")

    # Save model
    run_dir = MODEL_DIR / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    # Train final on all data
    scaler = fit_scaler(X)
    X_s = scale_X(X, scaler)
    final_model = MetaLSTM(
        X.shape[2], META_LSTM_HIDDEN, META_LSTM_LAYERS, META_LSTM_DROPOUT
    )
    torch.save(final_model.state_dict(), run_dir / "lstm_meta.pt")
    joblib.dump(scaler, run_dir / "lstm_meta_scaler.pkl")
    with open(run_dir / "lstm_meta_config.json", "w") as f:
        json.dump({
            "run_id": args.run_id, "n_features": X.shape[2], "seq_len": X.shape[1],
            "hidden": META_LSTM_HIDDEN, "layers": META_LSTM_LAYERS,
            "folds": all_metrics, "mean_auc": round(np.mean(aucs), 4),
        }, f, indent=2)
    print(f"\n  Model saved: {run_dir}")


if __name__ == "__main__":
    main()
