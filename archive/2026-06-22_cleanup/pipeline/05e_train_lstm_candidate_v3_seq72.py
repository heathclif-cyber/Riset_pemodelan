"""
pipeline/05e_train_lstm_candidate_v3_seq72.py
Sama persis dengan 05d (14 fitur tinggi-IC) tapi SEQ_LEN=72 untuk perbandingan.

Perbandingan: 05d (SEQ_LEN=24) vs 05e (SEQ_LEN=72), fitur identik.
"""
import argparse, gc, json, sys, warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler
from torch.utils.data import DataLoader, Dataset

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, LABEL_DIR, TRAIN_CUTOFF_DATE, MODEL_DIR, LABEL_MAP,
    N_FOLDS, LSTM_BATCH_SIZE, LSTM_EPOCHS, LSTM_PATIENCE,
    LSTM_V2_HIDDEN, LSTM_V2_LAYERS, LSTM_V2_DROPOUT,
    LSTM_V2_WEIGHT_DECAY, LSTM_V2_LR,
)
from core.models import TradingLSTM, save_lstm
from core.utils import setup_logger, get_lstm_device, ensure_utc_index
from pipeline.shared import build_purged_folds

logger = setup_logger("05e_train_lstm_candidate_v3_seq72")
DEVICE = get_lstm_device()

RUN_NAME  = "ic32_lstm_candidate_v3_seq72"
LGBM_RUN  = "ic32_regime_v2"
SEQ_LEN   = 72
PURGE_GAP = 72
CAND_THR  = 0.55

LSTM_FEATURES = [
    "rsi_6", "log_ret_5", "absorption_z", "cvd_momentum_adv", "log_ret_1", "vol_ratio_20",
    "dist_from_8h_high", "ema_7_h1", "rsi_h4", "swing_momentum",
    "stochrsi_k", "trend_accel_4h", "ultra_high_vol", "dist_liq_50x_short",
]

_PERCOIN_ZSCORE_FEATS = {"cvd_momentum_adv"}
_ZSCORE_WINDOW = 500

FOCAL_CFG = {"focal_gamma": 2.0, "alpha_bear_boost": 1.6, "alpha_bull_boost": 1.3, "alpha_neu_scale": 0.9}


class PrebuiltSeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        with torch.no_grad():
            pt = torch.exp(-ce.detach().cpu()).clamp(1e-7, 1.0 - 1e-7)
            focal_w = (1.0 - pt).pow(self.gamma)
        fw = focal_w.to(logits.device)
        if self.alpha is not None:
            return (self.alpha.to(logits.device)[targets] * fw * ce).mean()
        return (fw * ce).mean()


def focal_alpha(y_train):
    classes, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    base = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    boosts = {0: FOCAL_CFG["alpha_bear_boost"], 1: FOCAL_CFG["alpha_neu_scale"], 2: FOCAL_CFG["alpha_bull_boost"]}
    return torch.tensor([base.get(i, 1.0) * boosts.get(i, 1.0) for i in range(3)], dtype=torch.float32).to(DEVICE)


def _percoin_z(series, window=_ZSCORE_WINDOW):
    s = pd.Series(series)
    mean = s.rolling(window=window, min_periods=50).mean()
    std = s.rolling(window=window, min_periods=50).std().clip(lower=1e-8)
    return ((s - mean) / std).clip(-4, 4).fillna(0).values.astype(np.float32)


def fit_scaler(X):
    n, s, f = X.shape
    sc = RobustScaler()
    sc.fit(X.reshape(-1, f))
    return sc


def scale_X(X, sc):
    n, s, f = X.shape
    return sc.transform(X.reshape(-1, f)).reshape(n, s, f).astype(np.float32)


def load_lgbm_oof():
    path = MODEL_DIR / "runs" / LGBM_RUN / "oof_predictions.parquet"
    df = pd.read_parquet(path)
    df = df.loc[df["has_oof"]].copy()
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def load_data(coins, lgbm_oof):
    X_seqs, y_seqs, ts_seqs, coin_seqs = [], [], [], []
    skipped = []

    for coin in coins:
        fp = LABEL_DIR / f"{coin}_features_v3.parquet"
        if not fp.exists(): skipped.append(coin); continue

        df = pd.read_parquet(fp).sort_index()
        df = ensure_utc_index(df).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]
        df = df[df["label"].astype(str).isin(LABEL_MAP)].dropna(subset=["label"])
        if len(df) < SEQ_LEN + 10: skipped.append(coin); continue

        sym_lgbm = lgbm_oof[lgbm_oof["coin"] == coin][["p0", "p2"]].rename(
            columns={"p0": "p0_lgbm", "p2": "p2_lgbm"})
        df = df.join(sym_lgbm, how="left")
        if df["p0_lgbm"].notna().sum() < 100: skipped.append(coin); continue

        missing = [c for c in LSTM_FEATURES if c not in df.columns]
        if missing: logger.warning(f"  [{coin}] missing: {missing}"); skipped.append(coin); continue

        feat_vals = {}
        for c in LSTM_FEATURES:
            vals = df[c].ffill().fillna(0).values.astype(np.float32)
            if c in _PERCOIN_ZSCORE_FEATS:
                vals = _percoin_z(vals.astype(np.float64)).astype(np.float32)
            feat_vals[c] = vals

        X_c = np.column_stack([feat_vals[c] for c in LSTM_FEATURES])
        y_c = df["label"].map(LABEL_MAP).values.astype(np.int64)
        p0_lgbm = df["p0_lgbm"].ffill().values.astype(np.float32)
        p2_lgbm = df["p2_lgbm"].ffill().values.astype(np.float32)
        ts_c = df.index

        n_incl = 0
        for i in range(SEQ_LEN - 1, len(X_c)):
            if np.isnan(p0_lgbm[i]) or np.isnan(p2_lgbm[i]): continue
            if p2_lgbm[i] < CAND_THR and p0_lgbm[i] < CAND_THR: continue
            X_seqs.append(X_c[i - SEQ_LEN + 1:i + 1])
            y_seqs.append(y_c[i])
            ts_seqs.append(ts_c[i])
            coin_seqs.append(coin)
            n_incl += 1

        if n_incl > 0:
            sub = y_c[(p2_lgbm >= CAND_THR) | (p0_lgbm >= CAND_THR)]
            logger.info(f"  [{coin}] incl={n_incl:,} | LONG={(sub==2).mean()*100:.0f}% FLAT={(sub==1).mean()*100:.0f}% SHORT={(sub==0).mean()*100:.0f}%")

    if skipped: logger.warning(f"Skipped: {skipped}")
    if not X_seqs: raise ValueError("No sequences. Check LGBM OOF + features_v3.")

    X = np.stack(X_seqs); y = np.array(y_seqs, dtype=np.int64); ts = np.array(ts_seqs)
    order = np.argsort(ts)
    return X[order], y[order], ts[order], np.array(coin_seqs)[order]


def train_one_fold(X_tr, y_tr, X_te, y_te, fold_num):
    fold_scaler = fit_scaler(X_tr)
    X_tr_s = scale_X(X_tr, fold_scaler); del X_tr; gc.collect()
    X_te_s = scale_X(X_te, fold_scaler); del X_te; gc.collect()

    tr_ds = PrebuiltSeqDataset(X_tr_s, y_tr)
    te_ds = PrebuiltSeqDataset(X_te_s, y_te)
    tr_ld = DataLoader(tr_ds, batch_size=LSTM_BATCH_SIZE, shuffle=True, num_workers=0)
    te_ld = DataLoader(te_ds, batch_size=LSTM_BATCH_SIZE, shuffle=False, num_workers=0)

    model = TradingLSTM(X_tr_s.shape[2], LSTM_V2_HIDDEN, LSTM_V2_LAYERS, LSTM_V2_DROPOUT).to(DEVICE)
    criterion = FocalLoss(alpha=focal_alpha(y_tr), gamma=FOCAL_CFG["focal_gamma"])
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_V2_LR, weight_decay=LSTM_V2_WEIGHT_DECAY, foreach=False)

    best_f1, best_state, patience_count, best_epoch = -1.0, None, 0, 0
    for epoch in range(1, LSTM_EPOCHS + 1):
        model.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        pv, lv = [], []
        with torch.no_grad():
            for xb, yb in te_ld:
                pv.extend(model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy())
                lv.extend(yb.numpy())
        f1 = float(f1_score(lv, pv, average="macro", zero_division=0))

        if f1 > best_f1:
            best_f1 = f1; best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_count = 0; best_epoch = epoch
        else:
            patience_count += 1
            if patience_count >= LSTM_PATIENCE: break

        if epoch % 5 == 0 or epoch == 1:
            model.eval()
            pt, lt = [], []
            with torch.no_grad():
                for xb, yb in tr_ld:
                    pt.extend(model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy())
                    lt.extend(yb.numpy())
            f1_tr = float(f1_score(lt, pt, average="macro", zero_division=0))
            logger.info(f"  Fold {fold_num} Ep {epoch:>3} | val={f1:.4f} tr={f1_tr:.4f} gap={f1_tr-f1:+.3f} | Best={best_f1:.4f}")

    model.load_state_dict(best_state); model.eval()
    oof_proba = []
    with torch.no_grad():
        for xb, _ in te_ld:
            oof_proba.append(torch.softmax(model(xb.to(DEVICE)), dim=1).cpu().numpy())
    oof_proba = np.vstack(oof_proba)

    pv = oof_proba.argmax(axis=1)
    val_f1 = float(f1_score(y_te, pv, average="macro", zero_division=0))
    f1p = f1_score(y_te, pv, average=None, zero_division=0, labels=[0, 1, 2])
    metrics = {"fold": fold_num, "val_f1": round(val_f1, 4),
               "f1_SHORT": round(float(f1p[0]), 4), "f1_FLAT": round(float(f1p[1]), 4),
               "f1_LONG": round(float(f1p[2]), 4), "best_epoch": best_epoch}
    logger.info(f"  Fold {fold_num} DONE: F1={val_f1:.4f} S={f1p[0]:.3f} F={f1p[1]:.3f} L={f1p[2]:.3f}")
    return model, fold_scaler, metrics, oof_proba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    coins = TRAINING_COINS if args.all else TRAINING_COINS[:5]
    run_dir = MODEL_DIR / "runs" / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    lgbm_oof = load_lgbm_oof()

    print(f"\n{'='*70}")
    print(f"  LSTM IC32 CANDIDATE v3 SEQ72 -- {RUN_NAME}")
    print(f"  Features: {len(LSTM_FEATURES)} | SEQ_LEN: {SEQ_LEN} | PURGE: {PURGE_GAP}")
    print(f"  Compare: 05d (seq=24) vs 05e (seq=72), fitur identik")
    print(f"  Device  : {DEVICE}")
    print(f"{'='*70}\n")

    torch.manual_seed(42)
    np.random.seed(42)

    X, y, ts, coins_arr = load_data(coins, lgbm_oof)
    logger.info(f"Sequences: {X.shape[0]:,} | seq={SEQ_LEN} | feat={X.shape[2]}")
    for lbl_int, lbl_str in {0: "SHORT", 1: "FLAT", 2: "LONG"}.items():
        cnt = (y == lbl_int).sum()
        logger.info(f"  {lbl_str}: {cnt:,} ({cnt/len(y)*100:.1f}%)")

    ts_index = pd.to_datetime(ts, utc=True)
    folds = build_purged_folds(ts_index, n_folds=N_FOLDS, purge=PURGE_GAP)

    all_metrics = []
    oof_proba_all = np.full((len(y), 3), np.nan, dtype=np.float64)
    oof_has = np.zeros(len(y), dtype=bool)

    for fi, (tr_idx, te_idx) in enumerate(folds):
        _, _, m, oof_proba = train_one_fold(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], fi + 1)
        all_metrics.append(m)
        oof_proba_all[te_idx] = oof_proba
        oof_has[te_idx] = True

    val_f1s = [m["val_f1"] for m in all_metrics]
    mean_f1 = float(np.mean(val_f1s))
    std_f1  = float(np.std(val_f1s))

    oof_df = pd.DataFrame({
        "coin": coins_arr, "p0": oof_proba_all[:, 0], "p1": oof_proba_all[:, 1],
        "p2": oof_proba_all[:, 2], "has_oof": oof_has, "swing_label": y.astype(np.int8),
    }, index=pd.to_datetime(ts, utc=True))
    oof_df.to_parquet(run_dir / "oof_lstm_predictions.parquet")

    with open(run_dir / f"{RUN_NAME}_features.json", "w") as f:
        json.dump(LSTM_FEATURES, f, indent=2)

    meta = {
        "run_name": RUN_NAME, "lgbm_partner": LGBM_RUN,
        "seq_len": SEQ_LEN, "purge_gap": PURGE_GAP, "cand_thr": CAND_THR,
        "n_features": len(LSTM_FEATURES), "features": LSTM_FEATURES,
        "n_folds": N_FOLDS, "n_samples": int(X.shape[0]),
        "mean_f1_macro": round(mean_f1, 4), "std_f1_macro": round(std_f1, 4),
        "folds": all_metrics, "focal_cfg": FOCAL_CFG,
        "created": datetime.now().isoformat(),
        "train_cutoff": str(TRAIN_CUTOFF_DATE.date()),
        "benchmark_v2_f1": 0.3748,
        "methodology": "purged CV OOF, scaler per fold, swing labels, holdout not used",
    }
    with open(run_dir / f"{RUN_NAME}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  COMPLETE: {RUN_NAME}")
    print(f"  CV F1 macro: {mean_f1:.4f} +/- {std_f1:.4f}  (benchmark v2: 0.3748)")
    for m in all_metrics:
        print(f"  Fold {m['fold']}: F1={m['val_f1']:.4f} S={m['f1_SHORT']:.3f} F={m['f1_FLAT']:.3f} L={m['f1_LONG']:.3f}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
