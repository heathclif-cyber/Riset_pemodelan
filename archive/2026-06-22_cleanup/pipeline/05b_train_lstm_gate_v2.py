"""
pipeline/05_train_lstm_ic32_regime_v2.py
LSTM Swing Complement untuk ic32_regime_v2.

Peran LSTM : survival filter — konfirmasi atau veto sinyal LGBM.
Label      : swing label (label column dari features_v3.parquet)
Gate       : pump/dump spike AND LGBM OOF FLAT
Features   : lstm_v4_selected_features.json (11 temporal features)
Loss       : FocalLoss (sesuai archive ic32 swing complement)
Eval       : Purged CV OOF F1 + complement asymmetric (BUKAN holdout)

Methodology:
  - PURGE_GAP_BARS = 20 (swing labels)
  - RobustScaler per fold (Aturan 3)
  - Train hanya pada gate bars: pump/dump AND LGBM FLAT
  - Simpan oof_lstm_predictions.parquet
  - Holdout tidak disentuh

Usage:
  python pipeline/05_train_lstm_ic32_regime_v2.py --all
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
    N_FOLDS, PURGE_GAP_BARS, LSTM_SEQ_LEN, LSTM_BATCH_SIZE,
    LSTM_EPOCHS, LSTM_PATIENCE, LSTM_V2_HIDDEN, LSTM_V2_LAYERS,
    LSTM_V2_DROPOUT, LSTM_V2_WEIGHT_DECAY, LSTM_V2_LR,
)

# LSTM purge harus >= SEQ_LEN agar input window tidak overlap dengan training period
# purge=20 (LGBM) << SEQ_LEN=72 → inflate CV F1; override ke SEQ_LEN
PURGE_GAP_BARS = LSTM_SEQ_LEN  # 72
from core.models import TradingLSTM, save_lstm
from core.utils import setup_logger, get_lstm_device, ensure_utc_index
from pipeline.shared import build_purged_folds

logger = setup_logger("05b_train_lstm_gate_v2")
DEVICE = get_lstm_device()

RUN_NAME = "ic32_lstm_gate_v2"
LGBM_RUN = "ic32_regime_v2"

VOL_SPIKE_THR = 2.0
RANGE_EXP_THR = 1.5

COMPLEMENT_CFG = {
    "bull_thr": 0.38,
    "bear_thr": 0.50,
    "vol_spike_thr": 2.0,
    "focal_gamma": 2.0,
    "alpha_bear_boost": 1.6,
    "alpha_bull_boost": 1.3,
    "alpha_neu_scale": 0.9,
}

_PERCOIN_ZSCORE_FEATS = {"cvd", "volume_delta", "buy_volume", "sell_volume"}
_ZSCORE_WINDOW = 500


class PrebuiltSeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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
        raise FileNotFoundError(f"{path} not found.")
    with open(path) as f:
        return json.load(f)


def load_lgbm_thresholds() -> tuple[float, float]:
    path = MODEL_DIR / "runs" / LGBM_RUN / "best_thresholds.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} — run 04_train_lgbm_ic32_regime_v2.py first")
    with open(path) as f:
        d = json.load(f)
    return float(d["thr_long"]), float(d["thr_short"])


def load_lgbm_oof() -> pd.DataFrame:
    path = MODEL_DIR / "runs" / LGBM_RUN / "oof_predictions.parquet"
    df = pd.read_parquet(path)
    df = df.loc[df["has_oof"]].copy()
    df.index = pd.to_datetime(df.index, utc=True)
    return df


def _ic32_flat_mask(p0: np.ndarray, p2: np.ndarray,
                    thr_long: float, thr_short: float) -> np.ndarray:
    long_sig = p2 >= thr_long
    short_sig = (p0 >= thr_short) & ~long_sig
    return ~(long_sig | short_sig)


def _pump_dump_gate(df: pd.DataFrame) -> np.ndarray:
    vs = df["vol_spike_zscore"].values if "vol_spike_zscore" in df.columns else np.zeros(len(df))
    re = df["range_expansion_h4"].values if "range_expansion_h4" in df.columns else np.zeros(len(df))
    return (vs >= VOL_SPIKE_THR) | (re >= RANGE_EXP_THR)


def load_data_complement(coins: list[str], feat_cols: list[str],
                         lgbm_oof: pd.DataFrame, thr_long: float, thr_short: float):
    X_seqs, y_seqs, ts_seqs, meta_rows = [], [], [], []
    skipped, total_gate, total_flat = [], 0, 0

    # Pre-load BTC returns untuk cross-asset feature btc_h1_return
    btc_ret = None
    if "btc_h1_return" in feat_cols:
        btc_fp = LABEL_DIR / "BTCUSDT_features_v3.parquet"
        if btc_fp.exists():
            btc_df = pd.read_parquet(btc_fp, columns=["log_ret_1"])
            btc_df = ensure_utc_index(btc_df).sort_index()
            btc_df = btc_df[btc_df.index < TRAIN_CUTOFF_DATE]
            btc_ret = btc_df["log_ret_1"].rename("btc_h1_return")
            logger.info(f"  Loaded BTC returns for cross-asset feature ({len(btc_ret):,} bars)")

    for coin in coins:
        fp = LABEL_DIR / f"{coin}_features_v3.parquet"
        if not fp.exists():
            skipped.append(coin)
            continue

        df = pd.read_parquet(fp).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]
        df = df[df["label"].astype(str).isin(LABEL_MAP)].copy()
        df = df.dropna(subset=["label"])
        if len(df) < LSTM_SEQ_LEN + 10:
            skipped.append(coin)
            continue

        sym_lgbm = lgbm_oof[lgbm_oof["coin"] == coin][["p0", "p2"]].rename(
            columns={"p0": "p0_lgbm", "p2": "p2_lgbm"}
        )
        df = df.join(sym_lgbm, how="left")
        if df["p0_lgbm"].notna().sum() < 100:
            skipped.append(coin)
            continue

        # Join BTC 1h return sebagai cross-asset feature
        if btc_ret is not None and "btc_h1_return" not in df.columns:
            df = df.join(btc_ret, how="left")
            df["btc_h1_return"] = df["btc_h1_return"].ffill().fillna(0).astype(np.float32)

        avail = [c for c in feat_cols if c in df.columns]
        if len(avail) < len(feat_cols):
            missing = [c for c in feat_cols if c not in avail]
            logger.warning(f"  [{coin}] missing LSTM feats: {missing}")
            skipped.append(coin)
            continue

        feat_vals = {}
        for c in avail:
            vals = df[c].ffill().fillna(0).values.astype(np.float32)
            if c in _PERCOIN_ZSCORE_FEATS:
                vals = _percoin_z(vals.astype(np.float64)).astype(np.float32)
            feat_vals[c] = vals

        X_c = np.column_stack([feat_vals[c] for c in avail])
        y_c = df["label"].map(LABEL_MAP).values.astype(np.int64)
        gate = _pump_dump_gate(df)
        p0_lgbm = df["p0_lgbm"].ffill().values.astype(np.float32)
        p2_lgbm = df["p2_lgbm"].ffill().values.astype(np.float32)
        flat = _ic32_flat_mask(p0_lgbm, p2_lgbm, thr_long, thr_short)
        vol_spike = (
            df["vol_spike_zscore"].values.astype(np.float32)
            if "vol_spike_zscore" in df.columns else np.zeros(len(df), np.float32)
        )
        ts_c = df.index

        # Exp B (Gate domain): gate pass saja, tanpa filter LGBM FLAT
        # LSTM belajar dari semua high-vol bars, tidak hanya LGBM uncertain
        n_gate = n_incl = 0
        for i in range(LSTM_SEQ_LEN - 1, len(X_c)):
            if not gate[i]:
                continue
            n_gate += 1
            if np.isnan(p0_lgbm[i]) or np.isnan(p2_lgbm[i]):
                continue
            n_incl += 1
            X_seqs.append(X_c[i - LSTM_SEQ_LEN + 1:i + 1])
            y_seqs.append(y_c[i])
            ts_seqs.append(ts_c[i])
            meta_rows.append({
                "coin": coin,
                "vol_spike": float(vol_spike[i]),
                "p0_lgbm": float(p0_lgbm[i]),
                "p2_lgbm": float(p2_lgbm[i]),
            })

        sub = y_c[gate]
        total_gate += n_gate
        total_flat += n_incl
        if n_incl > 0:
            logger.info(
                f"  [{coin}] gate={n_gate:,} incl={n_incl:,} | "
                f"LONG={(sub == 2).mean()*100:.0f}% "
                f"FLAT={(sub == 1).mean()*100:.0f}% "
                f"SHORT={(sub == 0).mean()*100:.0f}%"
            )

    if skipped:
        logger.warning(f"Skipped: {skipped}")
    if not X_seqs:
        raise ValueError("No complement sequences. Check LGBM OOF + swing labels.")

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
    bull_thr = COMPLEMENT_CFG["bull_thr"]
    bear_thr = COMPLEMENT_CFG["bear_thr"]
    vol = frame["vol_spike"].values >= COMPLEMENT_CFG["vol_spike_thr"]

    p0, p2 = frame["p0_lstm"].values, frame["p2_lstm"].values
    labels = frame["label"].values

    bull_fire = vol & (p2 > p0) & (p2 >= bull_thr)
    bear_fire = vol & (p0 > p2) & (p0 >= bear_thr)
    comp = bull_fire | bear_fire

    def prec(mask, cls):
        return float((labels[mask] == cls).mean()) if mask.any() else 0.0

    dirs = np.where(bull_fire, 2, np.where(bear_fire, 0, -1))
    active = comp
    correct = ((dirs == 2) & (labels == 2)) | ((dirs == 0) & (labels == 0))
    dir_lbl = labels[active] != 1
    prec_dir = float(correct[active][dir_lbl].mean()) if dir_lbl.any() else 0.0

    return {
        "n_pool": int(vol.sum()),
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
            model.eval()
            pt, lt = [], []
            with torch.no_grad():
                for xb, yb in tr_ld:
                    pt.extend(model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy())
                    lt.extend(yb.numpy())
            f1_tr = float(f1_score(lt, pt, average="macro", zero_division=0))
            gap = f1_tr - f1
            logger.info(
                f"  Fold {fold_num} Epoch {epoch:>3} | val={f1:.4f} tr={f1_tr:.4f} gap={gap:+.3f} | Best={best_f1:.4f}"
            )

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
        "f1_SHORT": round(float(f1p[0]), 4),
        "f1_FLAT": round(float(f1p[1]), 4),
        "f1_LONG": round(float(f1p[2]), 4),
        "best_epoch": best_epoch,
    }
    logger.info(
        f"  Fold {fold_num} DONE: F1={val_f1:.4f} "
        f"S={f1p[0]:.3f} F={f1p[1]:.3f} L={f1p[2]:.3f}"
    )
    return model, fold_scaler, metrics, oof_proba


def retrain_final_focal(X_all, y_all, n_epochs):
    n_features = X_all.shape[2]
    final_scaler = fit_scaler(X_all)
    X_sc = scale_X(X_all, final_scaler)
    del X_all
    gc.collect()

    ds = PrebuiltSeqDataset(X_sc, y_all)
    batch = max(32, LSTM_BATCH_SIZE // 4)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)
    model = TradingLSTM(n_features, LSTM_V2_HIDDEN, LSTM_V2_LAYERS, LSTM_V2_DROPOUT).to(DEVICE)
    criterion = FocalLoss(alpha=focal_alpha(y_all), gamma=COMPLEMENT_CFG["focal_gamma"])
    optimizer = torch.optim.Adam(
        model.parameters(), lr=LSTM_V2_LR,
        weight_decay=LSTM_V2_WEIGHT_DECAY, foreach=False,
    )

    model.train()
    try:
        for epoch in range(1, n_epochs + 1):
            for xb, yb in loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                optimizer.zero_grad()
                criterion(model(xb), yb).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            if epoch % 10 == 0 or epoch == 1:
                logger.info(f"  Final epoch {epoch}/{n_epochs}")
    except RuntimeError as exc:
        if "memory" not in str(exc).lower():
            raise
        logger.warning(f"GPU OOM on final retrain, falling back to CPU: {exc}")
        model = model.cpu()
        criterion = FocalLoss(alpha=focal_alpha(y_all).cpu(), gamma=COMPLEMENT_CFG["focal_gamma"])
        optimizer = torch.optim.Adam(
            model.parameters(), lr=LSTM_V2_LR,
            weight_decay=LSTM_V2_WEIGHT_DECAY, foreach=False,
        )
        model.train()
        for epoch in range(1, n_epochs + 1):
            for xb, yb in loader:
                optimizer.zero_grad()
                criterion(model(xb), yb).backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            if epoch % 10 == 0 or epoch == 1:
                logger.info(f"  Final epoch {epoch}/{n_epochs} (cpu)")

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
    thr_long, thr_short = load_lgbm_thresholds()
    lgbm_oof = load_lgbm_oof()

    print(f"\n{'='*70}")
    print(f"  LSTM IC32 SWING COMPLEMENT -- {RUN_NAME}")
    print(f"  Partner : {LGBM_RUN} | Label: swing H4")
    print(f"  Features: {len(feat_cols)} temporal features")
    print(f"  Sample  : pump/dump gate + LGBM FLAT (thr {thr_long}/{thr_short})")
    print(f"  Loss    : FocalLoss gamma={COMPLEMENT_CFG['focal_gamma']}")
    print(f"  Folds   : {N_FOLDS} | Purge: {PURGE_GAP_BARS} | Coins: {len(coins)}")
    print(f"  Device  : {DEVICE}")
    print(f"{'='*70}\n")

    torch.manual_seed(42)
    np.random.seed(42)

    X, y, ts, meta_df, feat_cols_used = load_data_complement(
        coins, feat_cols, lgbm_oof, thr_long, thr_short
    )
    logger.info(f"Sequences: {X.shape[0]:,} | seq={LSTM_SEQ_LEN} | feat={X.shape[2]}")
    for lbl_int, lbl_str in {0: "SHORT", 1: "FLAT", 2: "LONG"}.items():
        cnt = (y == lbl_int).sum()
        logger.info(f"  {lbl_str}: {cnt:,} ({cnt/len(y)*100:.1f}%)")

    with open(run_dir / f"{RUN_NAME}_features.json", "w") as f:
        json.dump(feat_cols_used, f, indent=2)

    ts_index = pd.to_datetime(ts, utc=True)
    folds = build_purged_folds(ts_index, n_folds=N_FOLDS, purge=PURGE_GAP_BARS)

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
        "coin":        meta_df["coin"].values,
        "p0":          oof_proba_all[:, 0],
        "p1":          oof_proba_all[:, 1],
        "p2":          oof_proba_all[:, 2],
        "has_oof":     oof_has,
        "swing_label": y.astype(np.int8),
        "vol_spike":   meta_df["vol_spike"].values,
        "p0_lgbm":     meta_df["p0_lgbm"].values,
        "p2_lgbm":     meta_df["p2_lgbm"].values,
    }, index=pd.to_datetime(ts, utc=True))
    oof_df.to_parquet(run_dir / "oof_lstm_predictions.parquet")
    logger.info(f"Saved oof_lstm_predictions.parquet ({oof_has.sum():,} bars with OOF)")

    comp_frame = meta_df.copy()
    comp_frame["p0_lstm"] = oof_proba_all[:, 0]
    comp_frame["p2_lstm"] = oof_proba_all[:, 2]
    comp_frame["label"] = y
    comp_asym = complement_asymmetric(comp_frame[oof_has])

    avg_epochs = int(np.median([m.get("best_epoch", 30) for m in all_metrics]))
    final_epochs = max(20, min(avg_epochs + 5, LSTM_EPOCHS))
    logger.info(f"Retraining final model ({final_epochs} epochs)...")
    final_model, final_scaler = retrain_final_focal(X, y, final_epochs)

    save_lstm(final_model, run_dir / "lstm_momentum.pt")
    joblib.dump(final_scaler, run_dir / "lstm_momentum_scaler.pkl")

    meta = {
        "run_name": RUN_NAME,
        "model_type": "lstm_swing_complement_flat_only",
        "lgbm_partner": LGBM_RUN,
        "label_source": "swing_h4 (features_v3 label column)",
        "sample_filter": "pump_dump_gate AND lgbm_flat",
        "lgbm_thr_long": thr_long,
        "lgbm_thr_short": thr_short,
        "loss": "focal_loss",
        "complement_cfg": COMPLEMENT_CFG,
        "n_features": len(feat_cols_used),
        "features": feat_cols_used,
        "seq_len": LSTM_SEQ_LEN,
        "purge_gap": PURGE_GAP_BARS,
        "n_folds": N_FOLDS,
        "n_samples": int(X.shape[0]),
        "mean_f1_macro": round(mean_f1, 4),
        "std_f1_macro": round(std_f1, 4),
        "folds": all_metrics,
        "complement_asymmetric_oof": comp_asym,
        "created": datetime.now().isoformat(),
        "train_cutoff": str(TRAIN_CUTOFF_DATE.date()),
        "methodology": "purged CV OOF, scaler per fold, swing labels, holdout not used",
    }
    with open(run_dir / f"{RUN_NAME}_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n{'='*70}")
    print(f"  COMPLETE: {RUN_NAME}")
    print(f"  CV F1 macro: {mean_f1:.4f} +/- {std_f1:.4f}")
    print(f"  Complement pool (OOF): n={comp_asym['n_pool']:,}")
    print(f"  Fires: n={comp_asym['n_complement']:,} (bull={comp_asym['n_bull']:,} bear={comp_asym['n_bear']:,})")
    print(f"  Precision: bull={comp_asym['bull_precision']:.3f} bear={comp_asym['bear_precision']:.3f} "
          f"mixed_dir={comp_asym['mixed_precision_dir']:.3f}")
    print(f"  Model  : {run_dir}/lstm_momentum.pt")
    print(f"  Scaler : {run_dir}/lstm_momentum_scaler.pkl")
    print(f"\n  LANGKAH BERIKUTNYA:")
    print(f"  python pipeline/06_train_guardian_ic32_regime_v2.py --all")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
