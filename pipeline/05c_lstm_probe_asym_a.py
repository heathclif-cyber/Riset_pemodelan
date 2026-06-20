"""
pipeline/05c_lstm_probe_asym_a.py -- Probe A: regularisasi kuat + complement asimetris.

Perubahan vs 05_train_lstm_genuine_v2.py:
  - dropout 0.55, weight_decay 5e-4, hidden 64, layers 1
  - 4-fold CV (probe cepat), max 60 epoch, patience 8
  - Complement asimetris: BULL thr=0.40, BEAR thr=0.55 (frozen, dari OOF sweep)

Output: models/runs/tb_lstm_probe_asym_a/

Usage:
  python pipeline/05c_lstm_probe_asym_a.py --all
"""
import argparse
import gc
import importlib.util
import json
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import torch

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import MODEL_DIR, N_FOLDS, TB_PURGE_GAP_BARS, TRAINING_COINS

# Probe A hyperparams (tidak ubah config.py)
PROBE = {
    "run_name": "tb_lstm_probe_asym_a",
    "hidden": 64,
    "layers": 1,
    "dropout": 0.55,
    "weight_decay": 5e-4,
    "lr": 0.0005,
    "n_folds": 4,
    "epochs": 60,
    "patience": 8,
    "bull_thr": 0.40,
    "bear_thr": 0.55,
    "vol_spike_thr": 2.0,
}

# Load training module functions without running main
_spec = importlib.util.spec_from_file_location(
    "lstm_train", ROOT / "pipeline" / "05_train_lstm_genuine_v2.py"
)
_lstm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lstm)


def complement_asymmetric(frame, bull_thr, bear_thr, vol_thr):
    p0g = frame["p0_lgbm"].values
    p2g = frame["p2_lgbm"].values
    hmm = frame["hmm_enc"].values.astype(np.int8)
    flat = _lstm._lgbm_flat_mask(p0g, p2g, hmm)
    vol = frame["vol_spike"].values >= vol_thr
    pool = flat & vol

    p0 = frame["p0_lstm"].values
    p2 = frame["p2_lstm"].values
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
        "bull_thr": bull_thr,
        "bear_thr": bear_thr,
        "n_complement": int(comp.sum()),
        "n_bull": int(bull_fire.sum()),
        "n_bear": int(bear_fire.sum()),
        "n_pool": int(pool.sum()),
        "bull_precision": round(prec(bull_fire, 2), 4),
        "bear_precision": round(prec(bear_fire, 0), 4),
        "mixed_precision_dir": round(prec_dir, 4),
    }


def train_one_fold_probe(X_tr, y_tr, X_te, y_te, fold_num):
    """Override arch hyperparams for probe."""
    orig_h, orig_l, orig_d, orig_wd, orig_lr = (
        _lstm.LSTM_V2_HIDDEN, _lstm.LSTM_V2_LAYERS, _lstm.LSTM_V2_DROPOUT,
        _lstm.LSTM_V2_WEIGHT_DECAY, _lstm.LSTM_V2_LR,
    )
    _lstm.LSTM_V2_HIDDEN = PROBE["hidden"]
    _lstm.LSTM_V2_LAYERS = PROBE["layers"]
    _lstm.LSTM_V2_DROPOUT = PROBE["dropout"]
    _lstm.LSTM_V2_WEIGHT_DECAY = PROBE["weight_decay"]
    _lstm.LSTM_V2_LR = PROBE["lr"]

    import torch.nn as nn
    from sklearn.metrics import f1_score
    from torch.utils.data import DataLoader

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
    tr_ld = DataLoader(tr_ds, batch_size=_lstm.LSTM_BATCH_SIZE, shuffle=True, num_workers=0)
    te_ld = DataLoader(te_ds, batch_size=_lstm.LSTM_BATCH_SIZE, shuffle=False, num_workers=0)

    model = _lstm.TradingLSTM(
        n_features, PROBE["hidden"], PROBE["layers"], PROBE["dropout"]
    ).to(_lstm.DEVICE)
    cw = _lstm.compute_class_weights(y_tr)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=PROBE["lr"],
        weight_decay=PROBE["weight_decay"], foreach=False,
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
            best_f1, best_state, patience_count, best_epoch = f1, {
                k: v.cpu() for k, v in model.state_dict().items()
            }, 0, epoch
        else:
            patience_count += 1
            if patience_count >= PROBE["patience"]:
                break

    model.load_state_dict(best_state)
    model.eval()
    oof_proba = []
    with torch.no_grad():
        for xb, _ in te_ld:
            logits = model(xb.to(_lstm.DEVICE))
            oof_proba.append(torch.softmax(logits, dim=1).cpu().numpy())
    oof_proba = np.vstack(oof_proba)

    pv = oof_proba.argmax(axis=1)
    val_f1 = float(f1_score(y_te, pv, average="macro", zero_division=0))
    val_f1_p = f1_score(y_te, pv, average=None, zero_division=0, labels=[0, 1, 2])
    metrics = {
        "fold": fold_num, "val_f1": round(val_f1, 4),
        "f1_BEARISH": round(float(val_f1_p[0]), 4),
        "f1_NEUTRAL": round(float(val_f1_p[1]), 4),
        "f1_BULLISH": round(float(val_f1_p[2]), 4),
        "best_epoch": best_epoch,
    }
    _lstm.logger.info(
        f"  [probe] Fold {fold_num} F1={val_f1:.4f} "
        f"BEAR={val_f1_p[0]:.3f} NEU={val_f1_p[1]:.3f} BULL={val_f1_p[2]:.3f}"
    )

    _lstm.LSTM_V2_HIDDEN, _lstm.LSTM_V2_LAYERS = orig_h, orig_l
    _lstm.LSTM_V2_DROPOUT, _lstm.LSTM_V2_WEIGHT_DECAY, _lstm.LSTM_V2_LR = orig_d, orig_wd, orig_lr
    return model, fold_scaler, metrics, oof_proba


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    run_dir = MODEL_DIR / "runs" / PROBE["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    base_run = MODEL_DIR / "runs" / "tb_lstm_genuine_v2"
    feat_cols = _lstm.load_feature_list(base_run)

    coins = _lstm.TRAINING_COINS if args.all else _lstm.TRAINING_COINS[:5]

    print(f"\n{'='*66}")
    print(f"  PROBE A -- asymmetric complement + strong regularization")
    print(f"  hidden={PROBE['hidden']} layers={PROBE['layers']} dropout={PROBE['dropout']}")
    print(f"  wd={PROBE['weight_decay']} folds={PROBE['n_folds']} epochs<={PROBE['epochs']}")
    print(f"  complement: BULL>={PROBE['bull_thr']} BEAR>={PROBE['bear_thr']}")
    print(f"{'='*66}\n")

    torch.manual_seed(42)
    np.random.seed(42)

    X, y, ts, meta_df, feat_cols_used = _lstm.load_data(coins, feat_cols)
    ts_index = __import__("pandas").to_datetime(ts, utc=True)
    folds = _lstm.build_purged_folds(ts_index, n_folds=PROBE["n_folds"], purge=TB_PURGE_GAP_BARS)

    all_metrics = []
    oof_proba_all = np.full((len(y), 3), np.nan)
    oof_has = np.zeros(len(y), dtype=bool)

    for fi, (tr_idx, te_idx) in enumerate(folds):
        _, _, m, oof_proba = train_one_fold_probe(
            X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], fi + 1
        )
        all_metrics.append(m)
        oof_proba_all[te_idx] = oof_proba
        oof_has[te_idx] = True

    import pandas as pd
    oof_df = pd.DataFrame({
        "coin": meta_df["coin"].values,
        "p0": oof_proba_all[:, 0],
        "p1": oof_proba_all[:, 1],
        "p2": oof_proba_all[:, 2],
        "has_oof": oof_has,
        "momentum_v4_label": y.astype("int8"),
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
    asym = complement_asymmetric(
        frame, PROBE["bull_thr"], PROBE["bear_thr"], PROBE["vol_spike_thr"]
    )

    val_f1s = [m["val_f1"] for m in all_metrics]
    mean_f1 = float(np.mean(val_f1s))
    std_f1 = float(np.std(val_f1s))

    meta = {
        "run_name": PROBE["run_name"],
        "probe": "A_asymmetric_complement_strong_reg",
        "hyperparams": PROBE,
        "mean_f1_macro": round(mean_f1, 4),
        "std_f1_macro": round(std_f1, 4),
        "folds": all_metrics,
        "complement_asymmetric": asym,
        "features": feat_cols_used,
        "created": datetime.now().isoformat(),
    }
    with open(run_dir / "probe_a_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(run_dir / "best_lstm_complement.json", "w") as f:
        json.dump({
            "mode": "asymmetric",
            "lstm_bull_thr": PROBE["bull_thr"],
            "lstm_bear_thr": PROBE["bear_thr"],
            "vol_spike_thr": PROBE["vol_spike_thr"],
            "oof_metrics": asym,
            "sweep_method": "frozen_probe_a",
        }, f, indent=2)

    print(f"\n  CV F1: {mean_f1:.4f} +/- {std_f1:.4f}")
    print(f"  Asymmetric complement OOF:")
    print(f"    n={asym['n_complement']:,} (bull={asym['n_bull']:,} bear={asym['n_bear']:,})")
    print(f"    bull_prec={asym['bull_precision']:.3f}  bear_prec={asym['bear_precision']:.3f}")
    print(f"    mixed_prec_dir={asym['mixed_precision_dir']:.3f}")
    print(f"  Saved: {run_dir}/probe_a_meta.json")
    print(f"{'='*66}\n")


if __name__ == "__main__":
    main()