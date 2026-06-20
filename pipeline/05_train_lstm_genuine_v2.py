"""
pipeline/05_train_lstm_genuine_v2.py
LSTM Momentum Complement untuk tb_lgbm_genuine_v2.

Peran LSTM : deteksi momentum (pump/dump) saat LGBM FLAT + vol spike tinggi.
Label      : momentum_v4 (continuation, gate bars only)
Features   : lstm_v4_selected_features.json (IC CV-fold)
Eval       : Purged CV OOF F1 + complement gate sweep (BUKAN holdout)

Methodology:
  - TB_PURGE_GAP_BARS = 36 (purge gap)
  - RobustScaler per fold (Aturan 3)
  - Train/eval hanya pada bar gate (is_pump_dump_bar == 1)
  - Simpan oof_lstm_predictions.parquet
  - Sweep LSTM_COMP_THR via OOF complement simulation
  - Holdout tidak disentuh

Usage:
  python pipeline/05_train_lstm_genuine_v2.py --all
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
    N_FOLDS, TB_PURGE_GAP_BARS, LSTM_SEQ_LEN, LSTM_BATCH_SIZE,
    LSTM_EPOCHS, LSTM_PATIENCE, LSTM_V2_HIDDEN, LSTM_V2_LAYERS,
    LSTM_V2_DROPOUT, LSTM_V2_WEIGHT_DECAY, LSTM_V2_LR,
)
from core.models import TradingLSTM, save_lstm
from core.utils import setup_logger, get_lstm_device
from pipeline.shared import build_purged_folds

logger = setup_logger("05_lstm_genuine_v2")
DEVICE = get_lstm_device()

RUN_NAME = "tb_lstm_genuine_v2"
LGBM_RUN = "tb_lgbm_genuine_v2"

# HMM Config B -- sama dengan 07_holdout_genuine_v2 (frozen stack partner)
HMM_THR_CFG = {
    0:  (0.55, 0.55),
    1:  (0.55, 0.55),
    2:  (0.50, 0.50),
    3:  (0.45, 0.50),
    -1: (0.45, 0.45),
}

VOL_SPIKE_THR = 2.0
LSTM_THR_SWEEP = [0.40, 0.45, 0.50, 0.55, 0.60]

_PERCOIN_ZSCORE_FEATS = {"cvd", "volume_delta", "buy_volume", "sell_volume"}
_ZSCORE_WINDOW = 500

MOMENTUM_LABEL_MAP = {0: "BEARISH", 1: "NEUTRAL", 2: "BULLISH"}


class PrebuiltSeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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


def load_feature_list(run_dir: Path) -> list[str]:
    path = run_dir / "lstm_v4_selected_features.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run pipeline/05b_lstm_feature_ic_v3.py first."
        )
    with open(path) as f:
        return json.load(f)


def load_data(coins: list[str], feat_cols: list[str]):
    X_seqs, y_seqs, ts_seqs, meta_rows = [], [], [], []
    skipped = []

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

        avail = [c for c in feat_cols if c in df.columns]
        if len(avail) < len(feat_cols):
            missing = set(feat_cols) - set(avail)
            logger.warning(f"  [{coin}] missing feats: {missing}")

        if len(df) < LSTM_SEQ_LEN + 10:
            skipped.append(coin)
            continue

        feat_vals = {}
        for c in avail:
            vals = df[c].ffill().fillna(0).values.astype(np.float32)
            if c in _PERCOIN_ZSCORE_FEATS:
                vals = _percoin_z(vals.astype(np.float64)).astype(np.float32)
            feat_vals[c] = vals

        X_c = np.column_stack([feat_vals[c] for c in avail])
        y_c = df["momentum_v4_label"].values.astype(np.int64)
        gate = df["is_pump_dump_bar"].values.astype(bool)
        ts_c = df.index
        vol_spike = (
            df["vol_spike_zscore"].values
            if "vol_spike_zscore" in df.columns
            else np.zeros(len(df))
        )
        hmm_enc = df["hmm_regime_enc"].values.astype(np.int8)

        n_gate = 0
        for i in range(LSTM_SEQ_LEN - 1, len(X_c)):
            if not gate[i]:
                continue
            n_gate += 1
            X_seqs.append(X_c[i - LSTM_SEQ_LEN + 1:i + 1])
            y_seqs.append(y_c[i])
            ts_seqs.append(ts_c[i])
            meta_rows.append({
                "coin": coin,
                "vol_spike": float(vol_spike[i]),
                "hmm_enc": int(hmm_enc[i]),
                "is_gate": 1,
            })

        sub = y_c[gate]
        logger.info(
            f"  [{coin}] gate={n_gate:,} | "
            f"BULL={(sub == 2).mean()*100:.0f}% "
            f"NEU={(sub == 1).mean()*100:.0f}% "
            f"BEAR={(sub == 0).mean()*100:.0f}%"
        )

    if skipped:
        logger.warning(f"Skipped coins: {skipped}")
    if not X_seqs:
        raise ValueError("No gate-bar sequences. Run 05a_momentum_labels_v4.py --all first.")

    X = np.stack(X_seqs)
    y = np.array(y_seqs, dtype=np.int64)
    ts = np.array(ts_seqs)
    meta_df = pd.DataFrame(meta_rows)
    meta_df["ts"] = ts
    order = np.argsort(ts)
    return X[order], y[order], ts[order], meta_df.iloc[order].reset_index(drop=True), avail


def compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    total = len(y)
    weights = {c: total / (len(classes) * count) for c, count in zip(classes, counts)}
    return torch.tensor([weights.get(i, 1.0) for i in range(3)], dtype=torch.float32).to(DEVICE)


def train_one_fold(X_tr, y_tr, X_te, y_te, fold_num):
    n_features = X_tr.shape[2]
    fold_scaler = fit_scaler(X_tr)
    X_tr_s = scale_X(X_tr, fold_scaler)
    del X_tr
    gc.collect()
    X_te_s = scale_X(X_te, fold_scaler)
    del X_te
    gc.collect()

    tr_ds = PrebuiltSeqDataset(X_tr_s, y_tr)
    te_ds = PrebuiltSeqDataset(X_te_s, y_te)
    tr_ld = DataLoader(tr_ds, batch_size=LSTM_BATCH_SIZE, shuffle=True, num_workers=0)
    te_ld = DataLoader(te_ds, batch_size=LSTM_BATCH_SIZE, shuffle=False, num_workers=0)

    model = TradingLSTM(n_features, LSTM_V2_HIDDEN, LSTM_V2_LAYERS, LSTM_V2_DROPOUT).to(DEVICE)
    cw = compute_class_weights(y_tr)
    criterion = nn.CrossEntropyLoss(weight=cw)
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
            logits = model(xb.to(DEVICE))
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
    logger.info(
        f"  Fold {fold_num} DONE: F1={val_f1:.4f} "
        f"BEAR={val_f1_p[0]:.3f} NEU={val_f1_p[1]:.3f} BULL={val_f1_p[2]:.3f}"
    )
    return model, fold_scaler, metrics, oof_proba


def _lgbm_flat_mask(p0, p2, hmm_enc):
    n = len(p0)
    tl_arr = np.full(n, HMM_THR_CFG[-1][0], dtype=np.float32)
    ts_arr = np.full(n, HMM_THR_CFG[-1][1], dtype=np.float32)
    for state, (tl, ts) in HMM_THR_CFG.items():
        if state == -1:
            continue
        mask = hmm_enc == state
        tl_arr[mask] = tl
        ts_arr[mask] = ts
    long_sig = p2 >= tl_arr
    short_sig = (p0 >= ts_arr) & ~long_sig
    return ~(long_sig | short_sig)


def build_complement_frame(
    oof_proba: np.ndarray,
    meta_df: pd.DataFrame,
    y_true: np.ndarray,
) -> pd.DataFrame | None:
    oof_path = MODEL_DIR / "runs" / LGBM_RUN / "oof_predictions.parquet"
    if not oof_path.exists():
        logger.warning("LGBM OOF not found -- skip complement analysis")
        return None

    oof_lgbm = pd.read_parquet(oof_path)
    oof_lgbm = oof_lgbm.loc[oof_lgbm["has_oof"]].copy()
    oof_lgbm.index = pd.to_datetime(oof_lgbm.index, utc=True)
    lgbm_df = oof_lgbm.reset_index().rename(columns={"index": "ts"})
    if "ts" not in lgbm_df.columns:
        lgbm_df = lgbm_df.rename(columns={lgbm_df.columns[0]: "ts"})
    lgbm_df = lgbm_df.rename(columns={"p0": "p0_lgbm", "p2": "p2_lgbm"})

    lstm_df = meta_df.reset_index(drop=True).copy()
    lstm_df["ts"] = pd.to_datetime(lstm_df["ts"], utc=True)
    lstm_df["p0_lstm"] = oof_proba[:, 0]
    lstm_df["p1_lstm"] = oof_proba[:, 1]
    lstm_df["p2_lstm"] = oof_proba[:, 2]
    lstm_df["label"] = y_true

    merged = lstm_df.merge(
        lgbm_df[["coin", "ts", "p0_lgbm", "p2_lgbm"]],
        on=["coin", "ts"],
        how="inner",
    )
    return merged if len(merged) else None


def complement_metrics(frame: pd.DataFrame, lstm_thr: float) -> dict:
    p0 = frame["p0_lgbm"].values
    p2 = frame["p2_lgbm"].values
    hmm = frame["hmm_enc"].values.astype(np.int8)

    lstm_dom = np.maximum(frame["p0_lstm"].values, frame["p2_lstm"].values)
    lstm_dir = np.where(
        frame["p2_lstm"].values > frame["p0_lstm"].values, 2,
        np.where(frame["p0_lstm"].values > frame["p2_lstm"].values, 0, 1),
    )
    lstm_confident = (lstm_dir != 1) & (lstm_dom >= lstm_thr)

    lgbm_flat = _lgbm_flat_mask(p0, p2, hmm)
    vol_hi = frame["vol_spike"].values >= VOL_SPIKE_THR
    complement = lgbm_flat & vol_hi & lstm_confident

    n_comp = int(complement.sum())
    if n_comp == 0:
        return {
            "lstm_thr": lstm_thr, "n_complement": 0,
            "n_candidates": int((lgbm_flat & vol_hi).sum()),
        }

    labels = frame["label"].values[complement]
    dirs = lstm_dir[complement]
    correct = ((dirs == 2) & (labels == 2)) | ((dirs == 0) & (labels == 0))
    directional_label = labels != 1
    prec_dir = float(correct[directional_label].mean()) if directional_label.any() else 0.0

    return {
        "lstm_thr": lstm_thr,
        "n_complement": n_comp,
        "n_candidates": int((lgbm_flat & vol_hi).sum()),
        "precision_all": round(float(correct.mean()), 4),
        "precision_directional": round(prec_dir, 4),
        "coverage_pct": round(n_comp / max((lgbm_flat & vol_hi).sum(), 1) * 100, 2),
        "bull_pct": round(float((dirs == 2).mean()), 4),
        "bear_pct": round(float((dirs == 0).mean()), 4),
    }


def sweep_complement_thr(oof_proba_all: np.ndarray, meta_df: pd.DataFrame, y_all: np.ndarray):
    frame = build_complement_frame(oof_proba_all, meta_df, y_all)
    if frame is None or frame.empty:
        return [], 0.45, frame

    results = [complement_metrics(frame, thr) for thr in LSTM_THR_SWEEP]
    viable = [r for r in results if r.get("n_complement", 0) >= 50]
    if not viable:
        viable = results

    best = max(
        viable,
        key=lambda r: (r.get("precision_directional", 0), r.get("n_complement", 0)),
    )
    return results, best["lstm_thr"], frame


def retrain_final(X_all, y_all, n_epochs):
    n_features = X_all.shape[2]
    final_scaler = fit_scaler(X_all)
    X_sc = scale_X(X_all, final_scaler)
    del X_all
    gc.collect()

    ds = PrebuiltSeqDataset(X_sc, y_all)
    loader = DataLoader(ds, batch_size=LSTM_BATCH_SIZE, shuffle=True, num_workers=0)

    model = TradingLSTM(n_features, LSTM_V2_HIDDEN, LSTM_V2_LAYERS, LSTM_V2_DROPOUT).to(DEVICE)
    cw = compute_class_weights(y_all)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LSTM_V2_LR,
        weight_decay=LSTM_V2_WEIGHT_DECAY, foreach=False,
    )

    model.train()
    for epoch in range(1, n_epochs + 1):
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
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

    feat_cols = load_feature_list(run_dir)

    print(f"\n{'='*66}")
    print(f"  LSTM Momentum Complement -- {RUN_NAME}")
    print(f"  Partner : {LGBM_RUN}")
    print(f"  Label   : momentum_v4 (gate bars only)")
    print(f"  Features: {len(feat_cols)} from lstm_v4_selected_features.json")
    print(f"  Purge   : {TB_PURGE_GAP_BARS} | Folds: {N_FOLDS}")
    print(f"  Device  : {DEVICE}")
    print(f"{'='*66}\n")

    torch.manual_seed(42)
    np.random.seed(42)

    X, y, ts, meta_df, feat_cols_used = load_data(coins, feat_cols)
    logger.info(f"Gate sequences: {X.shape[0]:,} | seq={LSTM_SEQ_LEN} | feat={X.shape[2]}")

    for lbl_int, lbl_str in MOMENTUM_LABEL_MAP.items():
        cnt = (y == lbl_int).sum()
        logger.info(f"  {lbl_str}: {cnt:,} ({cnt / len(y) * 100:.1f}%)")

    with open(run_dir / f"{RUN_NAME}_features.json", "w") as f:
        json.dump(feat_cols_used, f, indent=2)

    ts_index = pd.to_datetime(ts, utc=True)
    folds = build_purged_folds(ts_index, n_folds=N_FOLDS, purge=TB_PURGE_GAP_BARS)

    all_metrics = []
    oof_proba_all = np.full((len(y), 3), np.nan, dtype=np.float64)
    oof_has = np.zeros(len(y), dtype=bool)

    for fi, (tr_idx, te_idx) in enumerate(folds):
        _, _, m, oof_proba = train_one_fold(
            X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], fi + 1
        )
        all_metrics.append(m)
        oof_proba_all[te_idx] = oof_proba
        oof_has[te_idx] = True

    val_f1s = [m["val_f1"] for m in all_metrics]
    mean_f1 = float(np.mean(val_f1s))
    std_f1 = float(np.std(val_f1s))

    oof_df = pd.DataFrame({
        "coin": meta_df["coin"].values,
        "p0": oof_proba_all[:, 0],
        "p1": oof_proba_all[:, 1],
        "p2": oof_proba_all[:, 2],
        "has_oof": oof_has,
        "momentum_v4_label": y.astype(np.int8),
        "is_gate": np.ones(len(y), dtype=np.int8),
        "vol_spike": meta_df["vol_spike"].values,
        "hmm_enc": meta_df["hmm_enc"].values.astype(np.int8),
    }, index=pd.to_datetime(ts, utc=True))
    oof_df.to_parquet(run_dir / "oof_lstm_predictions.parquet")
    logger.info(f"Saved oof_lstm_predictions.parquet ({oof_has.sum():,} bars)")

    sweep_results, best_thr, _ = sweep_complement_thr(
        oof_proba_all[oof_has], meta_df.iloc[np.where(oof_has)[0]].reset_index(drop=True),
        y[oof_has],
    )

    print(f"\n-- Complement Gate Sweep (OOF, LGBM flat + vol_spike>={VOL_SPIKE_THR}) --")
    print(f"  {'thr':>5} {'n':>7} {'prec_dir':>9} {'prec_all':>9} {'cov%':>6}")
    for r in sweep_results:
        print(
            f"  {r['lstm_thr']:>5.2f} {r.get('n_complement', 0):>7,} "
            f"{r.get('precision_directional', 0):>9.3f} "
            f"{r.get('precision_all', 0):>9.3f} "
            f"{r.get('coverage_pct', 0):>6.1f}"
        )
    print(f"  BEST lstm_thr = {best_thr}")

    with open(run_dir / "best_lstm_complement.json", "w") as f:
        json.dump({
            "lstm_comp_thr": best_thr,
            "vol_spike_thr": VOL_SPIKE_THR,
            "hmm_thr_cfg": {str(k): v for k, v in HMM_THR_CFG.items()},
            "sweep_method": "OOF_complement_simulation",
            "sweep_all": sweep_results,
            "created": datetime.now().isoformat(),
        }, f, indent=2)

    avg_epochs = int(np.median([m.get("best_epoch", 30) for m in all_metrics]))
    final_epochs = max(20, min(avg_epochs + 5, LSTM_EPOCHS))

    logger.info(f"\nRetraining final model ({final_epochs} epochs)...")
    final_model, final_scaler = retrain_final(X, y, final_epochs)

    save_lstm(final_model, run_dir / "lstm_momentum.pt")
    joblib.dump(final_scaler, run_dir / "lstm_momentum_scaler.pkl")

    best_comp = next((r for r in sweep_results if r["lstm_thr"] == best_thr), {})

    meta = {
        "run_name": RUN_NAME,
        "model_type": "lstm_momentum_complement",
        "lgbm_partner": LGBM_RUN,
        "role": "momentum continuation on pump/dump gate bars",
        "label_type": "momentum_v4_continuation_option_A",
        "label_thresholds": {"bull_fwd": 0.010, "bear_fwd": -0.010, "flow_bull": 0.0},
        "sample_filter": "is_pump_dump_bar == 1",
        "n_features": len(feat_cols_used),
        "features": feat_cols_used,
        "seq_len": LSTM_SEQ_LEN,
        "purge_gap": TB_PURGE_GAP_BARS,
        "n_folds": N_FOLDS,
        "hidden": LSTM_V2_HIDDEN,
        "layers": LSTM_V2_LAYERS,
        "dropout": LSTM_V2_DROPOUT,
        "n_samples": int(X.shape[0]),
        "n_coins": len(coins),
        "mean_f1_macro": round(mean_f1, 4),
        "std_f1_macro": round(std_f1, 4),
        "folds": all_metrics,
        "complement_gate": {
            "best_lstm_thr": best_thr,
            "vol_spike_thr": VOL_SPIKE_THR,
            "oof_sweep": sweep_results,
            "best_precision_directional": best_comp.get("precision_directional"),
            "best_n_complement": best_comp.get("n_complement"),
        },
        "integration_mode": "FLAT_review_when_vol_spike_high",
        "created": datetime.now().isoformat(),
        "train_cutoff": str(TRAIN_CUTOFF_DATE.date()),
        "methodology": "purged CV OOF, scaler per fold, gate bars only, holdout not used",
    }
    with open(run_dir / f"{RUN_NAME}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*66}")
    print(f"  {RUN_NAME} COMPLETE")
    print(f"  CV Mean F1     : {mean_f1:.4f} +/- {std_f1:.4f}")
    print(f"  Complement thr : {best_thr} (OOF sweep)")
    print(f"  OOF saved      : {run_dir}/oof_lstm_predictions.parquet")
    print(f"  Model          : {run_dir}/lstm_momentum.pt")
    print(f"{'='*66}\n")


if __name__ == "__main__":
    main()