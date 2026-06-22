"""
pipeline/05o_lstm_meta_short_win_v1.py
LSTM Binary Meta — SHORT trades only (Simon meta-labeling).

Pertanyaan: "Kalau LGBM mau entry SHORT sekarang, apakah trade ini menang?"
Label     : win dari OOF trade simulation (tb_lstm_genuine_v2 oof_trade_dataset)
Fitur     : komplementer saja — exclude overlap dengan LGBM 36f (feature_cols_v2.json)
Gate      : Simon marginal IC pada OOF SHORT trades (bukan F1 macro)

Usage:
  python pipeline/05o_lstm_meta_short_win_v1.py
"""
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import RobustScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS,
    LABEL_DIR,
    MODEL_DIR,
    TRAIN_CUTOFF_DATE,
    N_FOLDS,
    TB_PURGE_GAP_BARS,
)
from core.utils import setup_logger
from pipeline.shared import build_purged_folds

logger = setup_logger("05o_lstm_meta_short")
# Small binary meta-LSTM: CPU only (DirectML nn.LSTM fused cell unsupported)
DEVICE = torch.device("cpu")

RUN_NAME = "tb_lstm_meta_short_win_v1"
LGBM_RUN = "tb_lgbm_genuine_v2"
OOF_PATH = MODEL_DIR / "runs" / "tb_lstm_genuine_v2" / "oof_trade_dataset.parquet"
LGBM_FEAT_PATH = MODEL_DIR / "feature_cols_v2.json"
RUN_DIR = MODEL_DIR / "runs" / RUN_NAME

SEQ_LEN = 32
HIDDEN = 32
N_LAYERS = 1
DROPOUT = 0.50
LR = 1e-3
EPOCHS = 80
PATIENCE = 12
BATCH = 128
MIN_MARGINAL_IC = 0.015
MIN_T_STAT = 2.0

# Complement candidates — exclude any name present in LGBM feature_cols_v2.json
META_FEAT_CANDIDATES = [
    "ofi_raw",
    "ofi_z_score",
    "buy_volume",
    "sell_volume",
    "cvd",
    "vol_spike_zscore",
    "vol_accel_3h",
    "absorption_z",
    "ultra_high_vol",
    "range_expansion_h4",
    "vol_ratio_20",
    "no_supply",
    "no_demand",
    "effort_vs_result",
]


class BinaryLSTMMeta(nn.Module):
    def __init__(self, n_feat, hidden=32, n_layers=1, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, n_layers, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden, 1)

    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return torch.sigmoid(self.fc(self.drop(h[-1])))


def load_lgbm_features() -> set[str]:
    with open(LGBM_FEAT_PATH, encoding="utf-8") as f:
        return set(json.load(f))


def resolve_meta_features(lgbm_feats: set[str]) -> list[str]:
    return [c for c in META_FEAT_CANDIDATES if c not in lgbm_feats]


def load_coin_features(feat_cols: list[str]) -> dict[str, pd.DataFrame]:
    out = {}
    for coin in TRAINING_COINS:
        path = LABEL_DIR / f"{coin}_features_v3.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if df.index.tz is None:
            df.index = pd.to_datetime(df.index, utc=True)
        df = df.sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]
        avail = [c for c in feat_cols if c in df.columns]
        if not avail:
            continue
        out[coin] = df[avail].ffill().fillna(0.0)
    return out


def build_sequences(
    oof_short: pd.DataFrame,
    coin_feats: dict[str, pd.DataFrame],
    feat_cols: list[str],
) -> tuple:
    X_list, y_list, ts_list, conf_list, coin_list = [], [], [], [], []
    skipped = 0

    for ts, row in oof_short.iterrows():
        coin = row["coin"]
        if coin not in coin_feats:
            skipped += 1
            continue
        df_c = coin_feats[coin]
        ts_utc = pd.Timestamp(ts, tz="UTC") if getattr(ts, "tzinfo", None) is None else ts
        try:
            pos = df_c.index.get_loc(ts_utc)
        except KeyError:
            idx_arr = np.searchsorted(df_c.index.values, ts_utc.value)
            if idx_arr >= len(df_c):
                skipped += 1
                continue
            pos = idx_arr
        if pos < SEQ_LEN - 1:
            skipped += 1
            continue
        seq = df_c.iloc[pos - SEQ_LEN + 1: pos + 1][feat_cols].values.astype(np.float32)
        if seq.shape[0] != SEQ_LEN:
            skipped += 1
            continue
        X_list.append(seq)
        y_list.append(float(row["win"]))
        ts_list.append(ts_utc)
        conf_list.append(float(row["confidence"]))
        coin_list.append(coin)

    if skipped:
        logger.info(f"  Skipped {skipped:,} SHORT trades (missing history/features)")
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.float32)
    return X, y, np.array(ts_list), np.array(conf_list), np.array(coin_list)


def train_fold(X_tr, y_tr, X_val, y_val, pos_weight_val):
    n_tr, sl, nf = X_tr.shape
    scaler = RobustScaler()
    scaler.fit(X_tr.reshape(-1, nf))
    X_tr_sc = scaler.transform(X_tr.reshape(-1, nf)).reshape(n_tr, sl, nf).astype(np.float32)
    X_val_sc = scaler.transform(X_val.reshape(-1, nf)).reshape(len(X_val), sl, nf).astype(np.float32)

    model = BinaryLSTMMeta(nf, HIDDEN, N_LAYERS, DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    pw = torch.tensor([pos_weight_val], dtype=torch.float32, device=DEVICE)

    tr_ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_tr_sc).to(DEVICE),
        torch.FloatTensor(y_tr).to(DEVICE),
    )
    loader = torch.utils.data.DataLoader(tr_ds, batch_size=BATCH, shuffle=True)

    best_auc, best_epoch, best_state = 0.0, 0, None
    patience_cnt = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            pred = model(xb).squeeze(1)
            weights = torch.where(yb == 1, pw, torch.ones_like(yb))
            loss = (
                weights * nn.functional.binary_cross_entropy(pred, yb, reduction="none")
            ).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(torch.FloatTensor(X_val_sc).to(DEVICE)).squeeze(1).cpu().numpy()
        try:
            auc = roc_auc_score(y_val, val_pred)
        except Exception:
            auc = 0.5

        if auc > best_auc:
            best_auc = auc
            best_epoch = epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_cnt = 0
        else:
            patience_cnt += 1
            if patience_cnt >= PATIENCE:
                break

    model.load_state_dict(best_state)
    return model, best_auc, best_epoch, scaler, val_pred


def marginal_ic_test(lstm_scores, lgbm_conf, win_labels):
    mask = np.isfinite(lstm_scores) & np.isfinite(lgbm_conf) & np.isfinite(win_labels)
    ls, lc, lw = lstm_scores[mask], lgbm_conf[mask], win_labels[mask]

    def residuals(y, x):
        b, a = np.polyfit(x, y, 1)
        return y - (a + b * x)

    res_lstm = residuals(ls, lc)
    res_win = residuals(lw, lc)
    ic, pval = stats.spearmanr(res_lstm, res_win)
    n = mask.sum()
    t_val = ic * np.sqrt(n - 2) / np.sqrt(1 - ic ** 2 + 1e-9)
    return {
        "ic": round(float(ic), 4),
        "t_stat": round(float(t_val), 2),
        "n": int(n),
        "pval": round(float(pval), 4),
    }


def main():
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*66}")
    print(f"  LSTM Meta SHORT Win — {RUN_NAME}")
    print(f"  OOF source: {OOF_PATH.name}")
    print(f"  Seq={SEQ_LEN} | Hidden={HIDDEN} | Gate IC>={MIN_MARGINAL_IC}")
    print(f"{'='*66}\n")

    if not OOF_PATH.exists():
        raise FileNotFoundError(f"Missing OOF trades: {OOF_PATH}")

    lgbm_feats = load_lgbm_features()
    meta_candidates = resolve_meta_features(lgbm_feats)
    logger.info(f"LGBM features: {len(lgbm_feats)} | meta candidates: {len(meta_candidates)}")

    oof = pd.read_parquet(OOF_PATH)
    oof_short = oof[oof["direction"] == -1].copy()
    logger.info(
        f"OOF SHORT trades: {len(oof_short):,} | WR={oof_short['win'].mean()*100:.1f}%"
    )

    coin_feats = load_coin_features(meta_candidates)
    feat_cols = meta_candidates
    for sym, df in coin_feats.items():
        feat_cols = [c for c in feat_cols if c in df.columns]
    feat_cols = list(dict.fromkeys(feat_cols))
    if len(feat_cols) < 3:
        raise RuntimeError(f"Too few complement features available: {feat_cols}")
    logger.info(f"Using {len(feat_cols)} complement features: {feat_cols}")

    for sym in list(coin_feats.keys()):
        coin_feats[sym] = coin_feats[sym][feat_cols]

    X, y, ts, lgbm_conf, coins = build_sequences(oof_short, coin_feats, feat_cols)
    n_total = len(X)
    logger.info(f"Sequences: {n_total:,} | shape={X.shape} | WR={y.mean()*100:.1f}%")

    pos_weight = float((1 - y.mean()) / (y.mean() + 1e-9))
    ts_index = pd.to_datetime(ts, utc=True)
    folds = build_purged_folds(ts_index, n_folds=N_FOLDS, purge=TB_PURGE_GAP_BARS)
    logger.info(f"Purged CV folds: {len(folds)} (purge={TB_PURGE_GAP_BARS} bars)")

    oof_scores = np.full(n_total, np.nan)
    fold_aucs, best_epochs = [], []

    for fi, (tr_idx, val_idx) in enumerate(folds, 1):
        _, auc, ep, _, val_pred = train_fold(
            X[tr_idx], y[tr_idx], X[val_idx], y[val_idx], pos_weight,
        )
        oof_scores[val_idx] = val_pred
        fold_aucs.append(auc)
        best_epochs.append(ep)
        logger.info(
            f"  Fold {fi}/{len(folds)}: n_tr={len(tr_idx):,} n_val={len(val_idx):,} "
            f"AUC={auc:.4f} epoch={ep}"
        )

    mean_auc = float(np.nanmean(fold_aucs))
    std_auc = float(np.nanstd(fold_aucs))
    avg_ep = max(10, int(np.median(best_epochs)))

    valid = np.isfinite(oof_scores)
    mic = marginal_ic_test(oof_scores[valid], lgbm_conf[valid], y[valid])
    gate_pass = mic["ic"] >= MIN_MARGINAL_IC and abs(mic["t_stat"]) >= MIN_T_STAT

    logger.info(
        f"Marginal IC: {mic['ic']:+.4f} t={mic['t_stat']:+.2f} "
        f"n={mic['n']} -> {'PASS' if gate_pass else 'FAIL'}"
    )

    base_wr = float(y[valid].mean())
    print(f"\n  Threshold sweep (OOF SHORT, n={valid.sum():,}, base WR={base_wr*100:.1f}%):")
    print(f"  {'thr':>6} {'sel':>8} {'cover%':>7} {'WR_sel':>7}")
    sweep_rows = []
    for thr in [0.45, 0.50, 0.55, 0.60, 0.65]:
        sel = oof_scores[valid] >= thr
        if sel.sum() == 0:
            continue
        wr_sel = float(y[valid][sel].mean())
        sweep_rows.append({
            "threshold": thr,
            "selected": int(sel.sum()),
            "coverage_pct": round(float(sel.mean() * 100), 2),
            "wr_selected": round(wr_sel, 4),
            "wr_base": round(base_wr, 4),
        })
        print(
            f"  {thr:>6.2f} {sel.sum():>8,} {sel.mean()*100:>6.1f}% {wr_sel*100:>6.1f}%"
        )

    # Final retrain
    nf = X.shape[2]
    final_scaler = RobustScaler()
    final_scaler.fit(X.reshape(-1, nf))
    X_sc = final_scaler.transform(X.reshape(-1, nf)).reshape(n_total, SEQ_LEN, nf).astype(np.float32)

    final_model = BinaryLSTMMeta(nf, HIDDEN, N_LAYERS, DROPOUT).to(DEVICE)
    opt = torch.optim.Adam(final_model.parameters(), lr=LR, weight_decay=1e-4)
    pw = torch.tensor([pos_weight], dtype=torch.float32, device=DEVICE)
    ds = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_sc).to(DEVICE),
        torch.FloatTensor(y).to(DEVICE),
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=True)
    for _ in range(avg_ep):
        final_model.train()
        for xb, yb in loader:
            opt.zero_grad()
            pred = final_model(xb).squeeze(1)
            weights = torch.where(yb == 1, pw, torch.ones_like(yb))
            loss = (
                weights * nn.functional.binary_cross_entropy(pred, yb, reduction="none")
            ).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
            opt.step()

    torch.save(final_model.state_dict(), RUN_DIR / "lstm_binary_meta.pt")
    joblib.dump(final_scaler, RUN_DIR / "lstm_binary_meta_scaler.pkl")
    with open(RUN_DIR / f"{RUN_NAME}_features.json", "w", encoding="utf-8") as f:
        json.dump(feat_cols, f, indent=2)

    oof_out = pd.DataFrame({
        "coin": coins,
        "direction": -1,
        "win": y.astype(np.int8),
        "confidence": lgbm_conf,
        "lstm_score": oof_scores,
        "has_oof": valid,
    }, index=ts_index)
    oof_out.to_parquet(RUN_DIR / "oof_meta_short_scores.parquet")

    meta = {
        "run_name": RUN_NAME,
        "target": "SHORT trade WIN=1 / LOSS=0",
        "question": "If LGBM wants SHORT now, will this trade win?",
        "oof_source": str(OOF_PATH),
        "lgbm_partner": LGBM_RUN,
        "n_samples": n_total,
        "base_win_rate": round(float(y.mean()), 4),
        "n_features": len(feat_cols),
        "feat_cols": feat_cols,
        "excluded_lgbm_overlap": sorted(lgbm_feats & set(META_FEAT_CANDIDATES)),
        "seq_len": SEQ_LEN,
        "hidden": HIDDEN,
        "cv_mean_auc": round(mean_auc, 4),
        "cv_std_auc": round(std_auc, 4),
        "marginal_ic": mic,
        "gate_pass": gate_pass,
        "gate_criteria": {"min_marginal_ic": MIN_MARGINAL_IC, "min_t_stat": MIN_T_STAT},
        "threshold_sweep": sweep_rows,
        "fold_aucs": [round(a, 4) for a in fold_aucs],
        "deploy_note": "Do NOT deploy live until gate PASS + OOF trade eval vs conditional_momentum",
        "created": datetime.now().isoformat(),
        "train_cutoff": str(TRAIN_CUTOFF_DATE.date()),
    }
    with open(RUN_DIR / f"{RUN_NAME}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*66}")
    print(f"  {RUN_NAME} COMPLETE")
    print(f"  CV Mean AUC : {mean_auc:.4f} +/- {std_auc:.4f}")
    print(f"  Simon Gate  : {'PASS' if gate_pass else 'FAIL'}")
    print(f"  Marginal IC : {mic['ic']:+.4f}  t={mic['t_stat']:+.2f}")
    print(f"  Artefak     : {RUN_DIR}")
    print(f"{'='*66}\n")


if __name__ == "__main__":
    main()