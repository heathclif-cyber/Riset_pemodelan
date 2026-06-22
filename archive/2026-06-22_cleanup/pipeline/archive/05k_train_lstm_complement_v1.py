"""
pipeline/05k_train_lstm_complement_v1.py

Retrain LSTM khusus complement: LGBM FLAT + pump/dump gate bars.

Perbedaan vs tb_lstm_genuine_v2:
  - Sample: gate bar AND LGBM FLAT (HMM Config B thresholds, OOF LGBM)
  - Loss: FocalLoss + alpha boost BEAR/BULL (probe asym B)
  - Eval utama: complement precision asymmetric (BULL>=0.38, BEAR>=0.50)
  - Simpan: oof_lstm_predictions.parquet + lstm_momentum.pt + scaler

Genuine: purged CV OOF, scaler per fold, TRAIN_CUTOFF, holdout NOT touched.

Usage:
  python pipeline/05k_train_lstm_complement_v1.py --all
"""
import argparse
import gc
import importlib.util
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
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, LABEL_DIR, TRAIN_CUTOFF_DATE, MODEL_DIR,
    N_FOLDS, TB_PURGE_GAP_BARS, LSTM_SEQ_LEN, LSTM_BATCH_SIZE,
    LSTM_EPOCHS, LSTM_PATIENCE, LSTM_V2_HIDDEN, LSTM_V2_LAYERS,
    LSTM_V2_DROPOUT, LSTM_V2_WEIGHT_DECAY, LSTM_V2_LR,
)
from core.models import save_lstm
from core.utils import setup_logger, get_lstm_device

logger = setup_logger("05k_lstm_complement_v1")
DEVICE = get_lstm_device()

RUN_NAME = "tb_lstm_complement_v1"
LGBM_RUN = "tb_lgbm_genuine_v2"
LSTM_FEAT_RUN = MODEL_DIR / "runs" / "tb_lstm_genuine_v2"

HMM_THR_CFG = {
    0: (0.55, 0.55), 1: (0.55, 0.55), 2: (0.50, 0.50),
    3: (0.45, 0.50), -1: (0.45, 0.45),
}

COMPLEMENT_CFG = {
    "bull_thr": 0.38,
    "bear_thr": 0.50,
    "vol_spike_thr": 2.0,
    "focal_gamma": 2.0,
    "alpha_bear_boost": 1.6,
    "alpha_bull_boost": 1.3,
    "alpha_neu_scale": 0.9,
}

_spec = importlib.util.spec_from_file_location(
    "lstm_base", ROOT / "pipeline" / "05_train_lstm_genuine_v2.py"
)
_base = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_base)


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
    boosts = {0: COMPLEMENT_CFG["alpha_bear_boost"], 1: COMPLEMENT_CFG["alpha_neu_scale"],
              2: COMPLEMENT_CFG["alpha_bull_boost"]}
    return torch.tensor(
        [base.get(i, 1.0) * boosts.get(i, 1.0) for i in range(3)],
        dtype=torch.float32,
    ).to(DEVICE)


def load_lgbm_oof_indexed() -> pd.DataFrame:
    path = MODEL_DIR / "runs" / LGBM_RUN / "oof_predictions.parquet"
    df = pd.read_parquet(path)
    df = df.loc[df["has_oof"]].copy()
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def load_data_complement(coins: list[str], feat_cols: list[str], lgbm_oof: pd.DataFrame):
    """Gate bars where LGBM is FLAT on same timestamp (complement pool)."""
    X_seqs, y_seqs, ts_seqs, meta_rows = [], [], [], []
    skipped, total_gate, total_flat = 0, 0, 0

    for coin in coins:
        fp = LABEL_DIR / f"{coin}_features_v3.parquet"
        lp = LABEL_DIR / f"{coin}_momentum_v4_labels.parquet"
        rp = LABEL_DIR / f"{coin}_regime_h1.parquet"
        if not fp.exists() or not lp.exists():
            skipped.append(coin)
            continue

        df = pd.read_parquet(fp).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]
        lbl = pd.read_parquet(lp).sort_index()
        df = df.join(lbl[["momentum_v4_label", "is_pump_dump_bar"]], how="inner")
        df = df.dropna(subset=["momentum_v4_label"])

        if "hmm_regime_enc" not in df.columns:
            if rp.exists():
                reg = pd.read_parquet(rp).sort_index()
                df = df.join(reg[["hmm_regime_enc"]], how="left")
            else:
                df["hmm_regime_enc"] = -1
        df["hmm_regime_enc"] = df["hmm_regime_enc"].fillna(-1).astype(np.int8)

        sym_lgbm = lgbm_oof[lgbm_oof["coin"] == coin][["p0", "p2"]].rename(
            columns={"p0": "p0_lgbm", "p2": "p2_lgbm"}
        )
        df = df.join(sym_lgbm, how="left")
        if df["p0_lgbm"].notna().sum() < 100:
            skipped.append(coin)
            continue

        avail = [c for c in feat_cols if c in df.columns]
        if len(df) < LSTM_SEQ_LEN + 10:
            skipped.append(coin)
            continue

        feat_vals = {}
        for c in avail:
            vals = df[c].ffill().fillna(0).values.astype(np.float32)
            if c in _base._PERCOIN_ZSCORE_FEATS:
                vals = _base._percoin_z(vals.astype(np.float64)).astype(np.float32)
            feat_vals[c] = vals

        X_c = np.column_stack([feat_vals[c] for c in avail])
        y_c = df["momentum_v4_label"].values.astype(np.int64)
        gate = df["is_pump_dump_bar"].values.astype(bool)
        p0_lgbm = df["p0_lgbm"].ffill().values.astype(np.float32)
        p2_lgbm = df["p2_lgbm"].ffill().values.astype(np.float32)
        hmm_enc = df["hmm_regime_enc"].values.astype(np.int8)
        flat = _base._lgbm_flat_mask(p0_lgbm, p2_lgbm, hmm_enc)
        vol_spike = (
            df["vol_spike_zscore"].values.astype(np.float32)
            if "vol_spike_zscore" in df.columns else np.zeros(len(df), np.float32)
        )
        ts_c = df.index

        n_gate = n_flat = 0
        for i in range(LSTM_SEQ_LEN - 1, len(X_c)):
            if not gate[i]:
                continue
            n_gate += 1
            if not flat[i]:
                continue
            if np.isnan(p0_lgbm[i]) or np.isnan(p2_lgbm[i]):
                continue
            n_flat += 1
            X_seqs.append(X_c[i - LSTM_SEQ_LEN + 1:i + 1])
            y_seqs.append(y_c[i])
            ts_seqs.append(ts_c[i])
            meta_rows.append({
                "coin": coin,
                "vol_spike": float(vol_spike[i]),
                "hmm_enc": int(hmm_enc[i]),
                "p0_lgbm": float(p0_lgbm[i]),
                "p2_lgbm": float(p2_lgbm[i]),
                "is_gate": 1,
                "lgbm_flat": 1,
            })

        sub = y_c[gate & flat]
        total_gate += n_gate
        total_flat += n_flat
        if n_flat > 0:
            logger.info(
                f"  [{coin}] gate={n_gate:,} flat={n_flat:,} | "
                f"BULL={(sub == 2).mean()*100:.0f}% NEU={(sub == 1).mean()*100:.0f}% "
                f"BEAR={(sub == 0).mean()*100:.0f}%"
            )

    if skipped:
        logger.warning(f"Skipped: {skipped}")
    if not X_seqs:
        raise ValueError("No complement sequences. Check LGBM OOF + momentum labels.")

    X = np.stack(X_seqs)
    y = np.array(y_seqs, dtype=np.int64)
    ts = np.array(ts_seqs)
    meta_df = pd.DataFrame(meta_rows)
    meta_df["ts"] = ts
    order = np.argsort(ts)
    logger.info(f"Complement pool: gate={total_gate:,} -> flat={total_flat:,} sequences")
    return X[order], y[order], ts[order], meta_df.iloc[order].reset_index(drop=True), avail


def complement_asymmetric(frame: pd.DataFrame) -> dict:
    p0g, p2g = frame["p0_lgbm"].values, frame["p2_lgbm"].values
    hmm = frame["hmm_enc"].values.astype(np.int8)
    flat = _base._lgbm_flat_mask(p0g, p2g, hmm)
    vol = frame["vol_spike"].values >= COMPLEMENT_CFG["vol_spike_thr"]
    pool = flat & vol
    p0, p2 = frame["p0_lstm"].values, frame["p2_lstm"].values
    labels = frame["label"].values
    bull_thr, bear_thr = COMPLEMENT_CFG["bull_thr"], COMPLEMENT_CFG["bear_thr"]

    bull_fire = pool & (p2 > p0) & (p2 >= bull_thr)
    bear_fire = pool & (p0 > p2) & (p0 >= bear_thr)
    comp = bull_fire | bear_fire

    def prec(mask, cls):
        return float((labels[mask] == cls).mean()) if mask.any() else 0.0

    dirs = np.where(bull_fire, 2, np.where(bear_fire, 0, -1))
    active = comp
    correct = ((dirs == 2) & (labels == 2)) | ((dirs == 0) & (labels == 0))
    dir_lbl = labels[active] != 1
    prec_dir = float(correct[active][dir_lbl].mean()) if dir_lbl.any() else 0.0

    return {
        "n_pool": int(pool.sum()),
        "n_complement": int(comp.sum()),
        "n_bull": int(bull_fire.sum()),
        "n_bear": int(bear_fire.sum()),
        "bull_precision": round(prec(bull_fire, 2), 4),
        "bear_precision": round(prec(bear_fire, 0), 4),
        "mixed_precision_dir": round(prec_dir, 4),
        "bull_thr": bull_thr,
        "bear_thr": bear_thr,
    }


def train_one_fold(X_tr, y_tr, X_te, y_te, fold_num):
    n_features = X_tr.shape[2]
    fold_scaler = _base.fit_scaler(X_tr)
    X_tr_s = _base.scale_X(X_tr, fold_scaler)
    del X_tr
    gc.collect()
    X_te_s = _base.scale_X(X_te, fold_scaler)
    del X_te
    gc.collect()

    tr_ds = _base.PrebuiltSeqDataset(X_tr_s, y_tr)
    te_ds = _base.PrebuiltSeqDataset(X_te_s, y_te)
    tr_ld = DataLoader(tr_ds, batch_size=LSTM_BATCH_SIZE, shuffle=True, num_workers=0)
    te_ld = DataLoader(te_ds, batch_size=LSTM_BATCH_SIZE, shuffle=False, num_workers=0)

    model = _base.TradingLSTM(n_features, LSTM_V2_HIDDEN, LSTM_V2_LAYERS, LSTM_V2_DROPOUT).to(DEVICE)
    criterion = FocalLoss(alpha=focal_alpha(y_tr), gamma=COMPLEMENT_CFG["focal_gamma"])
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
            best_f1 = f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_count = 0
            best_epoch = epoch
        else:
            patience_count += 1
            if patience_count >= LSTM_PATIENCE:
                break
        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"  Fold {fold_num} Epoch {epoch:>3} | F1={f1:.4f} | Best={best_f1:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    oof_proba = []
    with torch.no_grad():
        for xb, _ in te_ld:
            oof_proba.append(torch.softmax(model(xb.to(DEVICE)), dim=1).cpu().numpy())
    oof_proba = np.vstack(oof_proba)

    pv = oof_proba.argmax(axis=1)
    val_f1 = float(f1_score(y_te, pv, average="macro", zero_division=0))
    f1p = f1_score(y_te, pv, average=None, zero_division=0, labels=[0, 1, 2])
    metrics = {
        "fold": fold_num, "val_f1": round(val_f1, 4),
        "f1_BEARISH": round(float(f1p[0]), 4),
        "f1_NEUTRAL": round(float(f1p[1]), 4),
        "f1_BULLISH": round(float(f1p[2]), 4),
        "best_epoch": best_epoch,
    }
    return model, fold_scaler, metrics, oof_proba


def retrain_final_focal(X_all, y_all, n_epochs):
    n_features = X_all.shape[2]
    final_scaler = _base.fit_scaler(X_all)
    X_sc = _base.scale_X(X_all, final_scaler)
    del X_all
    gc.collect()
    ds = _base.PrebuiltSeqDataset(X_sc, y_all)
    loader = DataLoader(ds, batch_size=LSTM_BATCH_SIZE, shuffle=True, num_workers=0)
    model = _base.TradingLSTM(n_features, LSTM_V2_HIDDEN, LSTM_V2_LAYERS, LSTM_V2_DROPOUT).to(DEVICE)
    criterion = FocalLoss(alpha=focal_alpha(y_all), gamma=COMPLEMENT_CFG["focal_gamma"])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LSTM_V2_LR,
        weight_decay=LSTM_V2_WEIGHT_DECAY, foreach=False,
    )
    model.train()
    for epoch in range(1, n_epochs + 1):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            criterion(model(xb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"  Final epoch {epoch}/{n_epochs}")
    model.eval()
    return model, final_scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    coins = TRAINING_COINS if args.all else TRAINING_COINS[:5]
    run_dir = MODEL_DIR / "runs" / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)
    feat_cols = _base.load_feature_list(LSTM_FEAT_RUN)

    print(f"\n{'='*70}")
    print(f"  LSTM COMPLEMENT RETRAIN -- {RUN_NAME}")
    print(f"  Sample: pump/dump gate + LGBM FLAT only")
    print(f"  Loss: Focal gamma={COMPLEMENT_CFG['focal_gamma']}")
    print(f"  Eval: complement BULL>={COMPLEMENT_CFG['bull_thr']} BEAR>={COMPLEMENT_CFG['bear_thr']}")
    print(f"  Folds: {N_FOLDS} | Purge: {TB_PURGE_GAP_BARS} | Coins: {len(coins)}")
    print(f"  Device: {DEVICE}")
    print(f"{'='*70}\n")

    torch.manual_seed(42)
    np.random.seed(42)

    lgbm_oof = load_lgbm_oof_indexed()
    X, y, ts, meta_df, feat_cols_used = load_data_complement(coins, feat_cols, lgbm_oof)
    logger.info(f"Sequences: {X.shape[0]:,} | seq={LSTM_SEQ_LEN} | feat={X.shape[2]}")

    with open(run_dir / f"{RUN_NAME}_features.json", "w") as f:
        json.dump(feat_cols_used, f, indent=2)

    ts_index = pd.to_datetime(ts, utc=True)
    folds = _base.build_purged_folds(ts_index, n_folds=N_FOLDS, purge=TB_PURGE_GAP_BARS)

    all_metrics = []
    oof_proba_all = np.full((len(y), 3), np.nan, dtype=np.float64)
    oof_has = np.zeros(len(y), dtype=bool)

    for fi, (tr_idx, te_idx) in enumerate(folds):
        _, _, m, oof_proba = train_one_fold(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], fi + 1)
        all_metrics.append(m)
        oof_proba_all[te_idx] = oof_proba
        oof_has[te_idx] = True

    oof_df = pd.DataFrame({
        "coin": meta_df["coin"].values,
        "p0": oof_proba_all[:, 0],
        "p1": oof_proba_all[:, 1],
        "p2": oof_proba_all[:, 2],
        "has_oof": oof_has,
        "momentum_v4_label": y.astype(np.int8),
        "is_gate": np.ones(len(y), dtype=np.int8),
        "lgbm_flat": np.ones(len(y), dtype=np.int8),
        "vol_spike": meta_df["vol_spike"].values,
        "hmm_enc": meta_df["hmm_enc"].values.astype(np.int8),
    }, index=pd.to_datetime(ts, utc=True))
    oof_df.to_parquet(run_dir / "oof_lstm_predictions.parquet")

    comp_frame = meta_df.copy()
    comp_frame["p0_lstm"] = oof_proba_all[:, 0]
    comp_frame["p2_lstm"] = oof_proba_all[:, 2]
    comp_frame["label"] = y
    comp_asym = complement_asymmetric(comp_frame[oof_has])

    val_f1s = [m["val_f1"] for m in all_metrics]
    mean_f1, std_f1 = float(np.mean(val_f1s)), float(np.std(val_f1s))

    avg_epochs = int(np.median([m.get("best_epoch", 30) for m in all_metrics]))
    final_epochs = max(20, min(avg_epochs + 5, LSTM_EPOCHS))
    logger.info(f"Retrain final model ({final_epochs} epochs)...")
    final_model, final_scaler = retrain_final_focal(X, y, final_epochs)
    save_lstm(final_model, run_dir / "lstm_momentum.pt")
    joblib.dump(final_scaler, run_dir / "lstm_momentum_scaler.pkl")

    meta = {
        "run_name": RUN_NAME,
        "model_type": "lstm_momentum_complement_flat_only",
        "lgbm_partner": LGBM_RUN,
        "sample_filter": "is_pump_dump_bar==1 AND lgbm_flat==1",
        "loss": "focal_loss",
        "complement_cfg": COMPLEMENT_CFG,
        "n_features": len(feat_cols_used),
        "features": feat_cols_used,
        "seq_len": LSTM_SEQ_LEN,
        "purge_gap": TB_PURGE_GAP_BARS,
        "n_folds": N_FOLDS,
        "n_samples": int(X.shape[0]),
        "n_coins": len(coins),
        "mean_f1_macro": round(mean_f1, 4),
        "std_f1_macro": round(std_f1, 4),
        "folds": all_metrics,
        "complement_asymmetric_oof": comp_asym,
        "baseline_compare": "tb_lstm_genuine_v2 (all gate bars)",
        "created": datetime.now().isoformat(),
        "train_cutoff": str(TRAIN_CUTOFF_DATE.date()),
        "methodology": "purged CV OOF, scaler per fold, complement flat-only, holdout not used",
    }
    with open(run_dir / f"{RUN_NAME}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(run_dir / "best_lstm_complement.json", "w") as f:
        json.dump({"mode": "asymmetric_flat_only", "oof_metrics": comp_asym}, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  COMPLETE: {RUN_NAME}")
    print(f"  CV F1 macro: {mean_f1:.4f} +/- {std_f1:.4f}")
    print(f"  Complement pool (OOF): n={comp_asym['n_pool']:,}")
    print(f"  Fires: n={comp_asym['n_complement']:,} (bull={comp_asym['n_bull']:,} bear={comp_asym['n_bear']:,})")
    print(f"  Precision: bull={comp_asym['bull_precision']:.3f} bear={comp_asym['bear_precision']:.3f} "
          f"mixed_dir={comp_asym['mixed_precision_dir']:.3f}")
    print(f"  Model: {run_dir}/lstm_momentum.pt")
    print(f"  OOF:   {run_dir}/oof_lstm_predictions.parquet")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()