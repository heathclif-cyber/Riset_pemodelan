"""
pipeline/10_train_lstm_binary_meta_tb.py
Binary LSTM Meta-Model — Simon Phase 2

Pertanyaan : "Apakah pola temporal 32 bar sebelum entry menunjukkan setup WIN?"
Target     : binary WIN=1 / LOSS=0 (dari LGBM OOF trade outcomes)
Berbeda    : LGBM prediksi ARAH, LSTM prediksi KUALITAS — dua pertanyaan berbeda

Simon Gate  : Marginal IC test wajib setelah CV
  IC(lstm_oof_score, win | lgbm_confidence) > 0  → LSTM add value
  Gagal → LSTM tidak layak masuk ensemble

Architecture: LSTM(hidden=32, layers=1) + sigmoid
Kecil sengaja: 22K samples, Simons rule 10x data per param → max ~2K params
"""
import json, sys, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler
from scipy import stats

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import joblib
from core.utils import setup_logger
from config import *

logger = setup_logger("10_lstm_binary_meta")

RUN_NAME  = "tb_lstm_binary_meta_v1"
SEQ_LEN   = 32
HIDDEN    = 32
N_LAYERS  = 1
DROPOUT   = 0.50
LR        = 1e-3
EPOCHS    = 80
PATIENCE  = 12
BATCH     = 128
N_FOLDS   = 8
PURGE_DAYS = 3

# Features untuk LSTM sequence
# Prinsip: PASS dari IC test (atr, funding) + temporal pattern candidates
# (OFI/CVD buildup, RSI divergence) + direction sebagai constant feature
LSTM_FEATS = [
    # IC PASS (confirmed vs WIN/LOSS)
    "atr_zscore_20d",
    "atr_percent_h4",
    "atr_percentile_h1",
    "funding_rate",
    # OFI / CVD — temporal accumulation
    "ofi_h4_delta",
    "ofi_raw",
    "cvd_slope_h4",
    "cvd_momentum_adv",
    # Momentum — divergence pattern
    "rsi_6",
    "rsi_h4",
    "log_ret_1",
    "log_ret_5",
    # Trend context
    "swing_momentum",
    "trend_strength",
    "ema_50_slope_h4",
]
# direction (LONG=1, SHORT=-1) ditambahkan sebagai feature ke-16 (constant)
N_FEAT_TOTAL = len(LSTM_FEATS) + 1  # +1 direction

OOF_PATH = MODEL_DIR / "runs" / "tb_meta_v1" / "oof_meta_dataset.parquet"
RUN_DIR  = MODEL_DIR / "runs" / RUN_NAME


# ── Model ──────────────────────────────────────────────────────────────────────
class BinaryLSTMMeta(nn.Module):
    def __init__(self, n_feat, hidden=32, n_layers=1, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, n_layers, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc   = nn.Linear(hidden, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return torch.sigmoid(self.fc(self.drop(h[-1])))  # (B, 1)


# ── Build sequence dataset ─────────────────────────────────────────────────────
def build_sequences(df_by_coin, oof_df, feat_cols, seq_len):
    X_list, y_list, ts_list, coin_list = [], [], [], []
    lgbm_conf_list = []

    for sym, grp in oof_df.groupby("coin"):
        if sym not in df_by_coin:
            logger.warning(f"[{sym}] feature parquet not found — skip")
            continue
        df = df_by_coin[sym]

        for ts, row in grp.iterrows():
            ts_utc = pd.Timestamp(ts, tz="UTC") if getattr(ts, "tzinfo", None) is None else ts
            if ts_utc not in df.index:
                continue
            pos = df.index.get_loc(ts_utc)
            if pos < seq_len - 1:
                continue

            seq_raw = df.iloc[pos - seq_len + 1: pos + 1][feat_cols]
            if len(seq_raw) != seq_len:
                continue

            seq = seq_raw.ffill().fillna(0).values.astype(np.float32)
            # Append direction as constant feature
            direction = np.full((seq_len, 1), float(row["direction"]), dtype=np.float32)
            seq = np.concatenate([seq, direction], axis=1)  # (seq_len, n_feat+1)

            X_list.append(seq)
            y_list.append(float(row["win"]))
            ts_list.append(ts_utc)
            coin_list.append(sym)
            lgbm_conf_list.append(float(row["confidence"]))

    X  = np.array(X_list, dtype=np.float32)    # (N, seq_len, n_feat)
    y  = np.array(y_list, dtype=np.float32)     # (N,)
    ts = np.array(ts_list)
    return X, y, ts, np.array(coin_list), np.array(lgbm_conf_list)


# ── Temporal CV folds (trade-level) ───────────────────────────────────────────
def build_folds(timestamps, n_folds=8, purge_days=3):
    ts   = pd.DatetimeIndex(timestamps)
    order = np.argsort(ts)
    ts_sorted = ts[order]
    fold_size = len(ts_sorted) // (n_folds + 1)
    folds = []
    for k in range(1, n_folds + 1):
        val_start = ts_sorted[k * fold_size]
        val_end   = ts_sorted[min((k + 1) * fold_size, len(ts_sorted) - 1)]
        purge_cut = val_start - pd.Timedelta(days=purge_days)

        tr_mask  = ts < purge_cut
        val_mask = (ts >= val_start) & (ts < val_end)
        tr_idx   = np.where(tr_mask)[0]
        val_idx  = np.where(val_mask)[0]

        if len(tr_idx) >= 200 and len(val_idx) >= 50:
            folds.append((tr_idx, val_idx))
    return folds


# ── Train one fold ─────────────────────────────────────────────────────────────
def train_fold(X_tr, y_tr, X_val, y_val, scaler, pos_weight_val):
    # Scale using per-feature stats over (N * seq_len) rows
    n_tr, sl, nf = X_tr.shape
    X_tr_2d  = X_tr.reshape(-1, nf)
    X_val_2d = X_val.reshape(-1, nf)
    scaler.fit(X_tr_2d)
    X_tr_sc  = scaler.transform(X_tr_2d).reshape(n_tr, sl, nf).astype(np.float32)
    X_val_sc = scaler.transform(X_val_2d).reshape(len(X_val), sl, nf).astype(np.float32)

    device = "cpu"
    model  = BinaryLSTMMeta(nf, HIDDEN, N_LAYERS, DROPOUT).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    pw     = torch.tensor([pos_weight_val], dtype=torch.float32)
    crit   = nn.BCELoss(weight=None)  # manual weighting below

    tr_ds  = torch.utils.data.TensorDataset(torch.FloatTensor(X_tr_sc), torch.FloatTensor(y_tr))
    loader = torch.utils.data.DataLoader(tr_ds, batch_size=BATCH, shuffle=True)

    best_auc, best_epoch, best_state = 0.0, 0, None
    patience_cnt = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb).squeeze(1)
            # Manual pos_weight for imbalance
            weights = torch.where(yb == 1, pw, torch.ones_like(yb))
            loss = (weights * nn.functional.binary_cross_entropy(pred, yb, reduction="none")).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(torch.FloatTensor(X_val_sc)).squeeze(1).numpy()
        try:
            auc = roc_auc_score(y_val, val_pred)
        except Exception:
            auc = 0.5

        if auc > best_auc:
            best_auc   = auc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_auc, best_epoch, scaler, val_pred


# ── Marginal IC test ───────────────────────────────────────────────────────────
def marginal_ic_test(lstm_scores, lgbm_conf, win_labels):
    """
    IC(lstm_score, win | lgbm_confidence_already_known)
    Hitung residu setelah regress out lgbm_confidence dari keduanya.
    """
    mask = np.isfinite(lstm_scores) & np.isfinite(lgbm_conf) & np.isfinite(win_labels)
    ls, lc, lw = lstm_scores[mask], lgbm_conf[mask], win_labels[mask]

    def residuals(y, x):
        b, a = np.polyfit(x, y, 1)
        return y - (a + b * x)

    res_lstm = residuals(ls, lc)
    res_win  = residuals(lw, lc)

    ic, pval = stats.spearmanr(res_lstm, res_win)
    n     = mask.sum()
    t_val = ic * np.sqrt(n - 2) / np.sqrt(1 - ic ** 2 + 1e-9)
    return {"ic": round(float(ic), 4), "t_stat": round(float(t_val), 2),
            "n": int(n), "pval": round(float(pval), 4)}


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*66}")
    print(f"  TB LSTM Binary Meta — {RUN_NAME}")
    print(f"  Target: WIN=1 / LOSS=0 (LGBM OOF trade outcomes)")
    print(f"  Seq={SEQ_LEN} | Hidden={HIDDEN} | Features={N_FEAT_TOTAL} (+direction)")
    print(f"  CV: {N_FOLDS} temporal folds | Purge={PURGE_DAYS}d")
    print(f"{'='*66}\n")

    # ── Load OOF dataset ──────────────────────────────────────────────────────
    logger.info("Loading OOF dataset...")
    oof = pd.read_parquet(OOF_PATH)
    oof.index.name = "ts"
    logger.info(f"  OOF trades: {len(oof):,} | WIN={oof['win'].mean()*100:.1f}%")

    # ── Load feature parquets ─────────────────────────────────────────────────
    logger.info("Loading feature parquets...")
    df_by_coin = {}
    avail_coins = oof["coin"].unique()
    for sym in avail_coins:
        path = LABEL_DIR / f"{sym}_features_v3.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if df.index.tz is None:
            df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()
        # Keep only training period
        df = df[df.index < TRAIN_CUTOFF_DATE]
        # Only keep needed feat cols
        avail = [c for c in LSTM_FEATS if c in df.columns]
        df_by_coin[sym] = df[avail]
    logger.info(f"  Loaded {len(df_by_coin)} coins")

    # Detect available features (intersection across all coins)
    feat_avail = set(LSTM_FEATS)
    for sym, df in df_by_coin.items():
        feat_avail &= set(df.columns)
    feat_cols = [f for f in LSTM_FEATS if f in feat_avail]
    logger.info(f"  Features available: {len(feat_cols)}/{len(LSTM_FEATS)}: {feat_cols}")

    # Re-slice to available features
    for sym in df_by_coin:
        df_by_coin[sym] = df_by_coin[sym][feat_cols]

    # ── Build sequences ───────────────────────────────────────────────────────
    logger.info("Building sequences...")
    X, y, ts, coins, lgbm_conf = build_sequences(df_by_coin, oof, feat_cols, SEQ_LEN)
    n_total = len(X)
    n_feat  = X.shape[2]
    logger.info(f"  Sequences: {n_total:,} | Shape: {X.shape}")
    logger.info(f"  WIN={y.mean()*100:.1f}% | LOSS={(1-y.mean())*100:.1f}%")

    pos_weight = float((1 - y.mean()) / (y.mean() + 1e-9))  # LOSS/WIN ratio
    logger.info(f"  pos_weight (WIN upweight): {pos_weight:.2f}")

    # ── CV ────────────────────────────────────────────────────────────────────
    logger.info(f"\nRunning {N_FOLDS}-fold temporal CV...")
    folds = build_folds(ts, N_FOLDS, PURGE_DAYS)
    logger.info(f"  Valid folds: {len(folds)}")

    oof_scores  = np.full(n_total, np.nan)
    fold_aucs   = []
    best_epochs = []

    for fold_idx, (tr_idx, val_idx) in enumerate(folds, 1):
        X_tr, y_tr = X[tr_idx], y[tr_idx]
        X_val, y_val = X[val_idx], y[val_idx]

        scaler = RobustScaler()
        model, auc, ep, scaler, val_pred = train_fold(
            X_tr, y_tr, X_val, y_val, scaler, pos_weight
        )

        # WR at threshold 0.55
        thr_mask = val_pred >= 0.55
        wr_sel = y_val[thr_mask].mean() if thr_mask.sum() > 0 else 0.0
        n_sel  = thr_mask.sum()

        oof_scores[val_idx] = val_pred
        fold_aucs.append(auc)
        best_epochs.append(ep)

        logger.info(
            f"  Fold {fold_idx}/{len(folds)}: "
            f"n_tr={len(tr_idx):,} n_val={len(val_idx):,} | "
            f"AUC={auc:.4f} epoch={ep} | "
            f"WR_sel={wr_sel*100:.1f}% (n={n_sel})"
        )

    mean_auc = float(np.nanmean(fold_aucs))
    std_auc  = float(np.nanstd(fold_aucs))
    avg_ep   = int(np.mean(best_epochs))
    logger.info(f"\n  CV Mean AUC: {mean_auc:.4f} +/- {std_auc:.4f}")
    logger.info(f"  Avg best epoch: {avg_ep}")

    # ── Marginal IC test (Simon Gate) ─────────────────────────────────────────
    valid_mask = np.isfinite(oof_scores)
    mic = marginal_ic_test(oof_scores[valid_mask], lgbm_conf[valid_mask], y[valid_mask])
    logger.info(f"\n  Marginal IC test (Simon Gate):")
    logger.info(f"  IC(lstm | lgbm_confidence) = {mic['ic']:+.4f}  t={mic['t_stat']:+.2f}  p={mic['pval']:.3f}")

    gate_pass = abs(mic["ic"]) >= 0.02 and abs(mic["t_stat"]) >= 2.0
    logger.info(f"  Gate: {'PASS — LSTM adds marginal value' if gate_pass else 'FAIL — LSTM tidak perlu masuk ensemble'}")

    # WR analysis at different thresholds
    print(f"\n  Threshold sweep (OOF scores, n={valid_mask.sum():,}):")
    print(f"  {'Threshold':>10} {'Selected':>10} {'Cover%':>8} {'WR_sel':>8} {'WR_base':>8}")
    base_wr = y[valid_mask].mean()
    for thr in [0.45, 0.50, 0.55, 0.60, 0.65]:
        sel = oof_scores[valid_mask] >= thr
        if sel.sum() == 0:
            continue
        wr = y[valid_mask][sel].mean()
        cover = sel.mean() * 100
        print(f"  {thr:>10.2f} {sel.sum():>10,} {cover:>7.1f}% {wr*100:>7.1f}% {base_wr*100:>7.1f}%")

    # ── Final retrain on ALL data ─────────────────────────────────────────────
    logger.info(f"\nFinal retrain on all data (epochs={avg_ep})...")
    n_all, sl, nf = X.shape
    final_scaler = RobustScaler()
    X_2d = X.reshape(-1, nf)
    final_scaler.fit(X_2d)
    X_sc = final_scaler.transform(X_2d).reshape(n_all, sl, nf).astype(np.float32)

    final_model = BinaryLSTMMeta(nf, HIDDEN, N_LAYERS, DROPOUT)
    opt   = torch.optim.Adam(final_model.parameters(), lr=LR, weight_decay=1e-4)
    pw    = torch.tensor([pos_weight], dtype=torch.float32)
    ds    = torch.utils.data.TensorDataset(torch.FloatTensor(X_sc), torch.FloatTensor(y))
    loader= torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True)

    for epoch in range(1, avg_ep + 1):
        final_model.train()
        for xb, yb in loader:
            opt.zero_grad()
            pred = final_model(xb).squeeze(1)
            weights = torch.where(yb == 1, pw, torch.ones_like(yb))
            loss = (weights * nn.functional.binary_cross_entropy(pred, yb, reduction="none")).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
            opt.step()

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save(final_model.state_dict(), RUN_DIR / "lstm_binary_meta.pt")
    joblib.dump(final_scaler, RUN_DIR / "lstm_binary_meta_scaler.pkl")
    with open(RUN_DIR / f"{RUN_NAME}_features.json", "w") as f:
        json.dump(feat_cols, f, indent=2)

    meta = {
        "run_name"       : RUN_NAME,
        "target"         : "binary WIN=1/LOSS=0",
        "n_samples"      : n_total,
        "base_win_rate"  : round(float(y.mean()), 4),
        "n_features"     : len(feat_cols),
        "n_feat_total"   : n_feat,  # incl direction
        "feat_cols"      : feat_cols,
        "seq_len"        : SEQ_LEN,
        "hidden"         : HIDDEN,
        "n_layers"       : N_LAYERS,
        "dropout"        : DROPOUT,
        "cv_mean_auc"    : round(mean_auc, 4),
        "cv_std_auc"     : round(std_auc, 4),
        "avg_best_epoch" : avg_ep,
        "marginal_ic"    : mic,
        "gate_pass"      : gate_pass,
        "fold_aucs"      : [round(a, 4) for a in fold_aucs],
        "pos_weight"     : round(pos_weight, 3),
    }
    with open(RUN_DIR / f"{RUN_NAME}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*66}")
    print(f"  {RUN_NAME} COMPLETE")
    print(f"  CV Mean AUC : {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Simon Gate  : {'PASS' if gate_pass else 'FAIL'}")
    print(f"  Marginal IC : {mic['ic']:+.4f}  t={mic['t_stat']:+.2f}  p={mic['pval']:.3f}")
    print(f"  Model  : {RUN_DIR}/lstm_binary_meta.pt")
    print(f"  Scaler : {RUN_DIR}/lstm_binary_meta_scaler.pkl")
    print(f"{'='*66}\n")


if __name__ == "__main__":
    main()
