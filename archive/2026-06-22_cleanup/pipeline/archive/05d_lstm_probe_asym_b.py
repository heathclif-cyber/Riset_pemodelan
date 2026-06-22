"""
pipeline/05d_lstm_probe_asym_b.py -- Probe B: focal loss + complement asimetris 0.40/0.50.

vs Probe A: baseline architecture (no extra reg), focal loss untuk boost BEAR/BULL.
  - hidden=96, layers=2, dropout=0.45, wd=2e-4 (sama config.py)
  - FocalLoss gamma=2.0, alpha BEAR x1.6 / BULL x1.3 / NEU x0.9
  - 4-fold CV, complement: BULL>=0.40, BEAR>=0.50

Output: models/runs/tb_lstm_probe_asym_b/

Usage:
  python pipeline/05d_lstm_probe_asym_b.py --all
"""
import argparse
import gc
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
from torch.utils.data import DataLoader

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    MODEL_DIR, TB_PURGE_GAP_BARS,
    LSTM_V2_HIDDEN, LSTM_V2_LAYERS, LSTM_V2_DROPOUT,
    LSTM_V2_WEIGHT_DECAY, LSTM_V2_LR, LSTM_BATCH_SIZE, LSTM_EPOCHS,
)

PROBE = {
    "run_name": "tb_lstm_probe_asym_b",
    "n_folds": 4,
    "epochs": min(60, LSTM_EPOCHS),
    "patience": 10,
    "bull_thr": 0.40,
    "bear_thr": 0.50,
    "vol_spike_thr": 2.0,
    "focal_gamma": 2.0,
    "alpha_bear_boost": 1.6,
    "alpha_bull_boost": 1.3,
    "alpha_neu_scale": 0.9,
}

_spec = importlib.util.spec_from_file_location(
    "lstm_train", ROOT / "pipeline" / "05_train_lstm_genuine_v2.py"
)
_lstm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lstm)


class FocalLoss(nn.Module):
    """DirectML-safe focal loss (weights on CPU)."""

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


def focal_alpha(y_train, device):
    """Class weights x directional boost."""
    classes, counts = np.unique(y_train, return_counts=True)
    total = len(y_train)
    base = {c: total / (len(classes) * cnt) for c, cnt in zip(classes, counts)}
    boosts = {0: PROBE["alpha_bear_boost"], 1: PROBE["alpha_neu_scale"], 2: PROBE["alpha_bull_boost"]}
    alpha = torch.tensor(
        [base.get(i, 1.0) * boosts.get(i, 1.0) for i in range(3)],
        dtype=torch.float32,
    )
    return alpha.to(device)


def complement_asymmetric(frame, bull_thr, bear_thr, vol_thr):
    p0g = frame["p0_lgbm"].values
    p2g = frame["p2_lgbm"].values
    hmm = frame["hmm_enc"].values.astype(np.int8)
    flat = _lstm._lgbm_flat_mask(p0g, p2g, hmm)
    vol = frame["vol_spike"].values >= vol_thr
    pool = flat & vol
    p0, p2 = frame["p0_lstm"].values, frame["p2_lstm"].values
    labels = frame["label"].values

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
        "bull_thr": bull_thr, "bear_thr": bear_thr,
        "n_complement": int(comp.sum()),
        "n_bull": int(bull_fire.sum()), "n_bear": int(bear_fire.sum()),
        "n_pool": int(pool.sum()),
        "bull_precision": round(prec(bull_fire, 2), 4),
        "bear_precision": round(prec(bear_fire, 0), 4),
        "mixed_precision_dir": round(prec_dir, 4),
    }


def train_one_fold(X_tr, y_tr, X_te, y_te, fold_num):
    n_features = X_tr.shape[2]
    fold_scaler = _lstm.fit_scaler(X_tr)
    X_tr_s = _lstm.scale_X(X_tr, fold_scaler)
    del X_tr
    gc.collect()
    X_te_s = _lstm.scale_X(X_te, fold_scaler)
    del X_te
    gc.collect()

    tr_ds = _lstm.PrebuiltSeqDataset(X_tr_s, y_tr)
    te_ds = _lstm.PrebuiltSeqDataset(X_te_s, y_te)
    tr_ld = DataLoader(tr_ds, batch_size=LSTM_BATCH_SIZE, shuffle=True, num_workers=0)
    te_ld = DataLoader(te_ds, batch_size=LSTM_BATCH_SIZE, shuffle=False, num_workers=0)

    model = _lstm.TradingLSTM(
        n_features, LSTM_V2_HIDDEN, LSTM_V2_LAYERS, LSTM_V2_DROPOUT
    ).to(_lstm.DEVICE)
    alpha = focal_alpha(y_tr, _lstm.DEVICE)
    criterion = FocalLoss(alpha=alpha, gamma=PROBE["focal_gamma"])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LSTM_V2_LR,
        weight_decay=LSTM_V2_WEIGHT_DECAY, foreach=False,
    )

    best_f1, best_state, patience_count, best_epoch = -1.0, None, 0, 0
    for epoch in range(1, PROBE["epochs"] + 1):
        model.train()
        for xb, yb in tr_ld:
            xb, yb = xb.to(_lstm.DEVICE), yb.to(_lstm.DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        model.eval()
        pv, lv = [], []
        with torch.no_grad():
            for xb, yb in te_ld:
                pv.extend(model(xb.to(_lstm.DEVICE)).argmax(dim=1).cpu().numpy())
                lv.extend(yb.numpy())
        f1 = float(f1_score(lv, pv, average="macro", zero_division=0))
        if f1 > best_f1:
            best_f1 = f1
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience_count = 0
            best_epoch = epoch
        else:
            patience_count += 1
            if patience_count >= PROBE["patience"]:
                break

    model.load_state_dict(best_state)
    model.eval()
    oof_proba = []
    with torch.no_grad():
        for xb, _ in te_ld:
            oof_proba.append(torch.softmax(model(xb.to(_lstm.DEVICE)), dim=1).cpu().numpy())
    oof_proba = np.vstack(oof_proba)

    pv = oof_proba.argmax(1)
    val_f1 = float(f1_score(y_te, pv, average="macro", zero_division=0))
    f1p = f1_score(y_te, pv, average=None, zero_division=0, labels=[0, 1, 2])
    metrics = {
        "fold": fold_num, "val_f1": round(val_f1, 4),
        "f1_BEARISH": round(float(f1p[0]), 4),
        "f1_NEUTRAL": round(float(f1p[1]), 4),
        "f1_BULLISH": round(float(f1p[2]), 4),
        "best_epoch": best_epoch,
    }
    _lstm.logger.info(
        f"  [probe-B] Fold {fold_num} F1={val_f1:.4f} "
        f"BEAR={f1p[0]:.3f} NEU={f1p[1]:.3f} BULL={f1p[2]:.3f}"
    )
    return metrics, oof_proba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    run_dir = MODEL_DIR / "runs" / PROBE["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    feat_cols = _lstm.load_feature_list(MODEL_DIR / "runs" / "tb_lstm_genuine_v2")
    coins = _lstm.TRAINING_COINS if args.all else _lstm.TRAINING_COINS[:5]

    print(f"\n{'='*66}")
    print(f"  PROBE B -- focal loss + asymmetric 0.40/0.50")
    print(f"  arch: hidden={LSTM_V2_HIDDEN} layers={LSTM_V2_LAYERS} dropout={LSTM_V2_DROPOUT}")
    print(f"  focal gamma={PROBE['focal_gamma']} alpha_boost BEAR={PROBE['alpha_bear_boost']}")
    print(f"  folds={PROBE['n_folds']} complement BULL>={PROBE['bull_thr']} BEAR>={PROBE['bear_thr']}")
    print(f"{'='*66}\n")

    torch.manual_seed(42)
    np.random.seed(42)

    X, y, ts, meta_df, feat_cols_used = _lstm.load_data(coins, feat_cols)
    import pandas as pd
    ts_index = pd.to_datetime(ts, utc=True)
    folds = _lstm.build_purged_folds(ts_index, n_folds=PROBE["n_folds"], purge=TB_PURGE_GAP_BARS)

    all_metrics = []
    oof_proba_all = np.full((len(y), 3), np.nan)
    oof_has = np.zeros(len(y), dtype=bool)

    for fi, (tr_idx, te_idx) in enumerate(folds):
        m, oof_proba = train_one_fold(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], fi + 1)
        all_metrics.append(m)
        oof_proba_all[te_idx] = oof_proba
        oof_has[te_idx] = True

    oof_df = pd.DataFrame({
        "coin": meta_df["coin"].values,
        "p0": oof_proba_all[:, 0], "p1": oof_proba_all[:, 1], "p2": oof_proba_all[:, 2],
        "has_oof": oof_has,
        "momentum_v4_label": y.astype(np.int8),
        "is_gate": np.ones(len(y), dtype=np.int8),
        "vol_spike": meta_df["vol_spike"].values,
        "hmm_enc": meta_df["hmm_enc"].values.astype(np.int8),
    }, index=pd.to_datetime(ts, utc=True))
    oof_df.to_parquet(run_dir / "oof_lstm_predictions.parquet")

    frame = _lstm.build_complement_frame(
        oof_proba_all[oof_has],
        meta_df.iloc[np.where(oof_has)[0]].reset_index(drop=True),
        y[oof_has],
    )
    asym = complement_asymmetric(frame, PROBE["bull_thr"], PROBE["bear_thr"], PROBE["vol_spike_thr"])

    val_f1s = [m["val_f1"] for m in all_metrics]
    mean_f1, std_f1 = float(np.mean(val_f1s)), float(np.std(val_f1s))
    mean_bear = float(np.mean([m["f1_BEARISH"] for m in all_metrics]))

    meta = {
        "run_name": PROBE["run_name"],
        "probe": "B_focal_loss_asymmetric_040_050",
        "hyperparams": {**PROBE, "hidden": LSTM_V2_HIDDEN, "layers": LSTM_V2_LAYERS, "dropout": LSTM_V2_DROPOUT},
        "mean_f1_macro": round(mean_f1, 4),
        "std_f1_macro": round(std_f1, 4),
        "mean_f1_bear": round(mean_bear, 4),
        "folds": all_metrics,
        "complement_asymmetric": asym,
        "features": feat_cols_used,
        "created": datetime.now().isoformat(),
    }
    with open(run_dir / "probe_b_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(run_dir / "best_lstm_complement.json", "w") as f:
        json.dump({
            "mode": "asymmetric",
            "lstm_bull_thr": PROBE["bull_thr"],
            "lstm_bear_thr": PROBE["bear_thr"],
            "oof_metrics": asym,
        }, f, indent=2)

    print(f"\n  CV F1: {mean_f1:.4f} +/- {std_f1:.4f}  |  mean F1 BEAR: {mean_bear:.3f}")
    print(f"  Asymmetric complement:")
    print(f"    n={asym['n_complement']:,} (bull={asym['n_bull']:,} bear={asym['n_bear']:,})")
    print(f"    bull_prec={asym['bull_precision']:.3f}  bear_prec={asym['bear_precision']:.3f}")
    print(f"    mixed_prec_dir={asym['mixed_precision_dir']:.3f}")
    print(f"  Compare baseline F1~0.402 | probe-A F1~0.382 bear_n=0")
    print(f"  Saved: {run_dir}/probe_b_meta.json")
    print(f"{'='*66}\n")


if __name__ == "__main__":
    main()