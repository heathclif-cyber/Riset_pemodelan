"""
pipeline/05_train_lstm_binary_meta_v2.py
Binary Meta LSTM v2 — domain-matched labels dari ic32 OOF trades.

Perbedaan dari v1:
  v1: labels = TB-WIN/LOSS (TP hit first pada TB simulation)
  v2: labels = ic32-WIN/LOSS (trade ic32 profitable setelah SL+time exit)
       → domain-matched dengan deployment target

Arsitektur sama (seq=32, hidden=32, layers=1) — arsitektur bukan masalah.
Evaluasi: AUC + lift WR conditional (Challenge 4 dari reviewer).

Input  : data/meta_labels/ic32_oof_trades.parquet
Output : models/runs/ic32_binary_meta_v2/

Usage:
  python pipeline/05_train_lstm_binary_meta_v2.py
"""
import gc, json, sys, warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
import joblib

warnings.filterwarnings("ignore", message="The operator 'aten::lerp")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, LABEL_DIR, MODEL_DIR, TRAIN_CUTOFF_DATE,
    N_FOLDS, PURGE_GAP_BARS,
)
from core.models import TradingLSTM, save_lstm
from core.utils import setup_logger
from pipeline.shared import build_purged_folds

logger = setup_logger("05_lstm_meta_v2")
try:
    import torch_directml
    DEVICE = torch_directml.device()
    logger.info("Device: DirectML (AMD GPU)")
except Exception:
    DEVICE = torch.device("cpu")
    logger.info("Device: CPU")

RUN_NAME = "ic32_binary_meta_v2"
OUT_DIR  = MODEL_DIR / "runs" / RUN_NAME
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Same architecture as v1 (architecture wasn't the issue)
SEQ_LEN  = 32
HIDDEN   = 32
N_LAYERS = 1
DROPOUT  = 0.5
BATCH    = 128
EPOCHS   = 100
PATIENCE = 20
LR       = 0.0005
WD       = 1e-4

# Same 15 features as v1 — IC-selected for binary prediction task
FEAT_COLS = [
    "atr_zscore_20d", "atr_percent_h4", "atr_percentile_h1",
    "funding_rate", "ofi_h4_delta", "ofi_raw",
    "cvd_slope_h4", "cvd_momentum_adv",
    "rsi_6", "rsi_h4",
    "log_ret_1", "log_ret_5",
    "swing_momentum", "trend_strength", "ema_50_slope_h4",
]

OOF_PATH = ROOT / "data" / "meta_labels" / "ic32_oof_trades.parquet"


# ── Sequence Extraction ────────────────────────────────────────────────────────

def load_coin_features() -> dict:
    """Load feature parquets per coin — only FEAT_COLS."""
    coin_feats = {}
    for coin in TRAINING_COINS:
        path = LABEL_DIR / f"{coin}_features_v3.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index, utc=True)
        df = df[df.index < TRAIN_CUTOFF_DATE]
        avail = [c for c in FEAT_COLS if c in df.columns]
        coin_feats[coin] = df[avail].ffill().fillna(0.0)
    return coin_feats


def build_sequences(oof_trades: pd.DataFrame, coin_feats: dict) -> tuple:
    """
    Extract seq_len=32 feature sequences for each OOF trade.
    Returns (X, y, timestamps, folds, feat_cols_used).
    """
    feat_cols_used = None
    X_list, y_list, ts_list, fold_list = [], [], [], []
    skipped = 0

    for _, row in oof_trades.iterrows():
        coin = row["coin"]
        ts   = pd.Timestamp(row["timestamp"])
        win  = int(row["win"])
        fold = int(row["fold"])

        if coin not in coin_feats:
            skipped += 1
            continue

        df_c = coin_feats[coin]
        if feat_cols_used is None:
            feat_cols_used = list(df_c.columns)

        # Find entry bar position
        try:
            pos = df_c.index.get_loc(ts)
        except KeyError:
            # Try nearest bar (tolerance ±1h for timezone edge cases)
            idx_arr = np.searchsorted(df_c.index.values, ts.value)
            if idx_arr >= len(df_c):
                skipped += 1
                continue
            pos = idx_arr

        if pos < SEQ_LEN - 1:
            skipped += 1
            continue

        seq = df_c.iloc[pos - SEQ_LEN + 1 : pos + 1].values  # (SEQ_LEN, n_feat)
        if seq.shape[0] != SEQ_LEN:
            skipped += 1
            continue

        X_list.append(seq)
        y_list.append(win)
        ts_list.append(ts)
        fold_list.append(fold)

    if skipped > 0:
        logger.info(f"  Skipped {skipped} trades (missing data or insufficient history)")

    X  = np.stack(X_list).astype(np.float32)          # (N, SEQ_LEN, n_feat)
    y  = np.array(y_list, dtype=np.float32)            # (N,)
    ts = pd.DatetimeIndex(ts_list)
    folds_arr = np.array(fold_list, dtype=np.int32)

    return X, y, ts, folds_arr, feat_cols_used


# ── Dataset ────────────────────────────────────────────────────────────────────

class BinarySeqDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self): return len(self.X)

    def __getitem__(self, i): return self.X[i], self.y[i]


# ── Training ───────────────────────────────────────────────────────────────────

def scale(X_tr, X_te):
    n, s, f = X_tr.shape
    sc = RobustScaler()
    sc.fit(X_tr.reshape(-1, f))
    X_tr_s = sc.transform(X_tr.reshape(-1, f)).reshape(n, s, f).astype(np.float32)
    X_te_s = sc.transform(X_te.reshape(-1, f)).reshape(len(X_te), s, f).astype(np.float32)
    return X_tr_s, X_te_s, sc


def train_fold(X_tr, y_tr, X_te, y_te, n_feat, fold_num):
    """Train binary LSTM for one fold. Returns (auc, oof_proba, best_epoch)."""
    # pos_weight for class imbalance (base WR ~39%)
    n_pos = y_tr.sum()
    n_neg = len(y_tr) - n_pos
    pos_w = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32).to(DEVICE)

    model = TradingLSTM(n_feat, HIDDEN, N_LAYERS, DROPOUT,
                        use_native=False, binary=True).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD, foreach=False)

    tr_ds = BinarySeqDataset(X_tr, y_tr)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH, shuffle=True, num_workers=0)
    te_ds = BinarySeqDataset(X_te, y_te)
    te_ld = DataLoader(te_ds, batch_size=BATCH, shuffle=False, num_workers=0)

    best_auc, best_st, patience_ct, best_ep = 0.0, None, 0, 1

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb).squeeze(-1), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        proba, labels = [], []
        with torch.no_grad():
            for xb, yb in te_ld:
                xb = xb.to(DEVICE)
                p = torch.sigmoid(model(xb).squeeze(-1)).cpu().numpy()
                proba.extend(p)
                labels.extend(yb.numpy())

        try:
            auc = roc_auc_score(labels, proba)
        except Exception:
            auc = 0.5

        if auc > best_auc:
            best_auc = auc
            best_st  = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_ct = 0
            best_ep = epoch
        else:
            patience_ct += 1
            if patience_ct >= PATIENCE:
                break

        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"  [F{fold_num}] E{epoch:>3} | AUC={auc:.4f} BestAUC={best_auc:.4f}")

    model.load_state_dict(best_st)
    model.eval()
    proba, labels = [], []
    with torch.no_grad():
        for xb, yb in te_ld:
            p = torch.sigmoid(model(xb.to(DEVICE)).squeeze(-1)).cpu().numpy()
            proba.extend(p)
            labels.extend(yb.numpy())

    return best_auc, np.array(proba), np.array(labels), best_ep, model


# ── Lift WR evaluation ────────────────────────────────────────────────────────

def lift_wr_report(oof_proba: np.ndarray, oof_labels: np.ndarray) -> dict:
    """
    Compute conditional WR by quantile of p_win (Challenge 4).
    Returns dict with WR per quintile and spread.
    """
    quantiles = [0.2, 0.4, 0.6, 0.8, 1.0]
    thresholds = np.quantile(oof_proba, quantiles)
    report = []
    prev_t = -np.inf
    for q, t in zip(quantiles, thresholds):
        mask = (oof_proba > prev_t) & (oof_proba <= t)
        n = mask.sum()
        wr = oof_labels[mask].mean() if n > 0 else 0.0
        report.append({"quantile": q, "threshold": round(float(t), 4),
                        "n": int(n), "wr": round(float(wr * 100), 2)})
        prev_t = t

    # Top vs bottom half spread
    top_mask = oof_proba >= np.median(oof_proba)
    bot_mask = oof_proba < np.median(oof_proba)
    top_wr = oof_labels[top_mask].mean() * 100
    bot_wr = oof_labels[bot_mask].mean() * 100
    spread = top_wr - bot_wr

    return {
        "quintile_wr": report,
        "top_half_wr":  round(float(top_wr), 2),
        "bottom_half_wr": round(float(bot_wr), 2),
        "spread_pp":    round(float(spread), 2),
        "deploy_recommendation": "YES" if spread >= 15.0 else "NO",
    }


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not OOF_PATH.exists():
        logger.error(f"OOF trades not found: {OOF_PATH}")
        logger.error("Run: python pipeline/08_generate_ic32_oof_trades.py first")
        return

    logger.info("=" * 60)
    logger.info(f"  Binary Meta LSTM v2 — ic32 domain-matched labels")
    logger.info(f"  Arch: TradingLSTM(n_feat={len(FEAT_COLS)}, hidden={HIDDEN}, "
                f"layers={N_LAYERS}, seq={SEQ_LEN})")
    logger.info("=" * 60)

    # Load OOF trades
    oof_trades = pd.read_parquet(OOF_PATH)
    logger.info(f"OOF trades: {len(oof_trades):,} | base WR: {oof_trades['win'].mean()*100:.1f}%")

    # Load feature parquets
    logger.info("Loading coin features...")
    coin_feats = load_coin_features()
    logger.info(f"  Loaded {len(coin_feats)} coins")

    # Extract sequences
    logger.info(f"Building sequences (seq_len={SEQ_LEN})...")
    X, y, ts_idx, lgbm_folds, feat_cols_used = build_sequences(oof_trades, coin_feats)
    n_feat = X.shape[2]
    logger.info(f"  Sequences: {len(X):,} | n_feat: {n_feat} | "
                f"WIN rate: {y.mean()*100:.1f}%")

    # Purged CV using fold assignments from LGBM OOF
    # Use LGBM fold number as fold boundary (fold k = val for LSTM fold k)
    folds = build_purged_folds(ts_idx, N_FOLDS, PURGE_GAP_BARS)

    cv_aucs  = []
    all_oof_proba  = np.zeros(len(X))
    all_oof_labels = np.zeros(len(X))
    covered = np.zeros(len(X), dtype=bool)
    all_best_epochs = []

    for fold_idx, (train_idx, val_idx) in enumerate(folds):
        if len(val_idx) < 20:
            continue
        fold_num = fold_idx + 1

        X_tr_s, X_te_s, scaler = scale(X[train_idx], X[val_idx])
        y_tr = y[train_idx]
        y_te = y[val_idx]

        logger.info(f"\nFold {fold_num}/{N_FOLDS} | train={len(X_tr_s):,} val={len(X_te_s):,} "
                    f"| val WR={y_te.mean()*100:.1f}%")

        auc, oof_p, oof_l, best_ep, model = train_fold(
            X_tr_s, y_tr, X_te_s, y_te, n_feat, fold_num)

        cv_aucs.append(auc)
        all_best_epochs.append(best_ep)
        all_oof_proba[val_idx] = oof_p
        all_oof_labels[val_idx] = oof_l
        covered[val_idx] = True

        logger.info(f"  Fold {fold_num} FINAL: AUC={auc:.4f} BestEpoch={best_ep}")

    # Final model — retrain on all data
    logger.info("\nRetraining final model on all data...")
    X_all_s, _, sc_all = scale(X, X)  # fit on all
    n_final_epochs = int(np.mean(all_best_epochs))
    _, _, _, _, final_model = train_fold(
        X_all_s, y, X_all_s[:1], y[:1], n_feat, 0)  # dummy val for API compat

    # OOF metrics
    oof_mask = covered
    mean_auc = float(np.mean(cv_aucs))
    std_auc  = float(np.std(cv_aucs))

    # Marginal IC (Spearman) — same gate as v1
    from scipy import stats
    ic, _ = stats.spearmanr(all_oof_proba[oof_mask], all_oof_labels[oof_mask])
    n_oof  = oof_mask.sum()
    t_stat = ic * np.sqrt((n_oof - 2) / (1 - ic ** 2 + 1e-9))
    gate_pass = abs(ic) >= 0.03 and abs(t_stat) >= 2.0

    # Lift WR conditional
    lift = lift_wr_report(all_oof_proba[oof_mask], all_oof_labels[oof_mask])

    # Print summary
    print(f"\n{'='*60}")
    print(f"  BINARY META LSTM v2 — ic32 domain-matched")
    print(f"{'='*60}")
    print(f"  CV AUC   : {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  v1 AUC   : 0.5566 +/- 0.0218  (TB labels, for comparison)")
    print(f"  Marginal IC : {ic:.4f} | t={t_stat:.2f} | gate={'PASS' if gate_pass else 'FAIL'}")
    print(f"\n  Lift WR Conditional (Challenge 4 metric):")
    print(f"  {'Quintile':>10} {'Thresh':>8} {'N':>6} {'WR%':>8}")
    for q in lift["quintile_wr"]:
        print(f"  {q['quantile']:>10.0%} {q['threshold']:>8.4f} "
              f"{q['n']:>6} {q['wr']:>8.1f}%")
    print(f"\n  Top-half WR    : {lift['top_half_wr']:.1f}%")
    print(f"  Bottom-half WR : {lift['bottom_half_wr']:.1f}%")
    print(f"  Spread         : {lift['spread_pp']:+.1f} pp")
    print(f"  Deploy?        : {lift['deploy_recommendation']} "
          f"(threshold: spread >= 15pp)")
    print(f"{'='*60}")

    # Save
    save_lstm(final_model, OUT_DIR / "lstm.pt")
    joblib.dump(sc_all, OUT_DIR / "lstm_scaler.pkl")
    with open(OUT_DIR / f"{RUN_NAME}_features.json", "w") as f:
        json.dump(feat_cols_used, f, indent=2)

    meta = {
        "run_name": RUN_NAME,
        "label_source": "ic32 OOF trades — domain-matched (not TB)",
        "oof_trades_path": str(OOF_PATH),
        "base_win_rate": round(float(y.mean()), 4),
        "n_sequences": int(len(X)),
        "n_features": n_feat,
        "feat_cols": feat_cols_used,
        "architecture": f"TradingLSTM(n_feat={n_feat}, hidden={HIDDEN}, "
                        f"layers={N_LAYERS}, dropout={DROPOUT}, seq_len={SEQ_LEN})",
        "seq_len": SEQ_LEN,
        "cv_mean_auc": round(mean_auc, 4),
        "cv_std_auc":  round(std_auc, 4),
        "cv_aucs": [round(a, 4) for a in cv_aucs],
        "avg_best_epoch": int(np.mean(all_best_epochs)),
        "marginal_ic": {
            "ic":     round(float(ic), 4),
            "t_stat": round(float(t_stat), 2),
            "n":      int(n_oof),
            "gate_pass": gate_pass,
        },
        "lift_wr": lift,
        "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "vs_v1": {
            "v1_auc": 0.5566, "v1_ic": 0.0682, "v1_gate": True,
            "v1_note": "TB-WIN labels — domain mismatch with ic32",
        },
    }
    with open(OUT_DIR / f"{RUN_NAME}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Saved: {OUT_DIR}")
    logger.info(f"Deploy recommendation: {lift['deploy_recommendation']} "
                f"(spread={lift['spread_pp']:+.1f}pp)")


if __name__ == "__main__":
    main()
