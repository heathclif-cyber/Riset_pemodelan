"""
pipeline/05d_train_lstm_momentum_v4.py — LSTM Momentum V4 (seq_len=32 + MACRO features)

Upgrade dari V3 (seq_len=16, F1=0.407):
- seq_len: 32 (8 candle H4) — capture momentum cycle penuh
- MACRO features: Fear & Greed + ETF flow (daily, di-join ke hourly bars)
- Target: F1 > 0.44 dengan informasi macro

Usage:
  python pipeline/05d_train_lstm_momentum_v4.py --all
  python pipeline/05d_train_lstm_momentum_v4.py --run-id lstm_momentum_v4
"""

import argparse, gc, json, sys, warnings, numpy as np, pandas as pd
from pathlib import Path

ROOT = Path(__file__).parent.parent; sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")

import torch, torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import f1_score, accuracy_score
import joblib

from config import (
    TRAINING_COINS, LABEL_DIR, TRAIN_CUTOFF_DATE, MODEL_DIR,
    N_FOLDS, PURGE_GAP_BARS,
    LSTM_SEQ_LEN, LSTM_BATCH_SIZE, LSTM_EPOCHS, LSTM_PATIENCE,
    LSTM_V2_HIDDEN, LSTM_V2_LAYERS, LSTM_V2_DROPOUT,
    LSTM_V2_WEIGHT_DECAY, LSTM_V2_LR,
)
from core.models import TradingLSTM, save_lstm
from core.utils import setup_logger, get_lstm_device
from pipeline.shared import build_purged_folds

logger = setup_logger("05d_lstm_momentum_v4")
DEVICE = get_lstm_device()

# ─── V4 Features: 16 OHLCV + 5 MACRO = 21 features ────────────────────────
# OHLCV core (16 — same as V3)
OHLCV_FEATURES = [
    # Price trajectory
    "log_ret_1", "log_ret_5", "log_ret_20",
    # Oscillators
    "rsi_6", "rsi_h4",
    # Trend strength
    "h4_trend", "trend_strength",
    "ema_21_slope_h4", "ema_50_slope_h4",
    "price_vs_ema_50_h4",
    # Flow
    "cvd", "cvd_momentum_adv", "volume_delta",
    # Liquidation
    "Buy_Liq", "Sell_Liq",
    # Cross-market
    "whale_retail_divergence",
]

# MACRO features (5 — engineered from backfilled data)
# These are daily-frequency, forward-filled to hourly bars
MACRO_FEATURES = [
    "fear_greed_value",         # Raw F&G (0-100)
    "fear_greed_momentum_5d",   # 5-day delta in F&G
    "etf_volume_zscore_20d",     # Unusual ETF volume (z-score over 20d)
    "etf_volume_momentum_7d",    # 7-day ETF volume delta
    "etf_price_momentum_7d",     # 7-day avg ETF price change
]

LSTM_MOMENTUM_V4_FEATURES = OHLCV_FEATURES + MACRO_FEATURES

# Features needing per-coin z-score
_PERCOIN_ZSCORE_FEATS = {"cvd", "volume_delta", "Buy_Liq", "Sell_Liq"}
_ZSCORE_WINDOW = 500

MOMENTUM_LABEL_MAP = {"BEARISH": 0, "NEUTRAL": 1, "BULLISH": 2}
MOMENTUM_LABEL_MAP_INV = {0: "BEARISH", 1: "NEUTRAL", 2: "BULLISH"}


class PrebuiltSeqDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]


def _percoin_z(series, window=_ZSCORE_WINDOW):
    s = pd.Series(series)
    mean = s.rolling(window=window, min_periods=50).mean()
    std = s.rolling(window=window, min_periods=50).std().clip(lower=1e-8)
    return ((s - mean) / std).clip(-4, 4).fillna(0).values.astype(np.float32)


def fit_scaler(X):
    n, s, f = X.shape; scl = RobustScaler(); scl.fit(X.reshape(-1, f)); return scl


def scale_X(X, scl):
    n, s, f = X.shape; return scl.transform(X.reshape(-1, f)).reshape(n, s, f).astype(np.float32)


def load_macro_data():
    """Load and engineer macro features from backfilled data."""
    macro_dir = ROOT / "data" / "macro"
    fng_path = macro_dir / "fear_greed.parquet"
    etf_path = macro_dir / "etf_btc_combined.parquet"

    macro_df = None

    # Fear & Greed
    if fng_path.exists():
        fng = pd.read_parquet(fng_path)
        fng = fng[["fear_greed_value"]].copy()
        fng["fear_greed_momentum_5d"] = fng["fear_greed_value"].diff(5)
        if macro_df is None:
            macro_df = fng
        else:
            macro_df = macro_df.join(fng, how="outer")

    # ETF combined data
    if etf_path.exists():
        etf = pd.read_parquet(etf_path)
        # Engineer ETF features
        etf["etf_volume_zscore_20d"] = (
            (etf["btc_etf_volume_usd"] - etf["btc_etf_volume_usd"].rolling(20).mean())
            / etf["btc_etf_volume_usd"].rolling(20).std().clip(lower=1e-8)
        ).clip(-4, 4).fillna(0)
        etf["etf_volume_momentum_7d"] = etf["btc_etf_volume_usd"].diff(7) / etf["btc_etf_volume_usd"].rolling(7).mean().clip(lower=1e-8)
        etf["etf_price_momentum_7d"] = etf["btc_etf_avg_price"].pct_change(7)
        etf_keep = ["etf_volume_zscore_20d", "etf_volume_momentum_7d", "etf_price_momentum_7d"]
        if macro_df is None:
            macro_df = etf[etf_keep].copy()
        else:
            macro_df = macro_df.join(etf[etf_keep], how="outer")

    if macro_df is not None:
        macro_df = macro_df.sort_index()
        logger.info(f"Macro data loaded: {len(macro_df)} daily rows | {macro_df.index[0].date()} -> {macro_df.index[-1].date()} | cols: {list(macro_df.columns)}")

    return macro_df


def join_macro_to_hourly(df_hourly, macro_df):
    """Join daily macro features to hourly bars (forward-fill within day)."""
    if macro_df is None:
        return df_hourly

    # Add date column for join
    df = df_hourly.copy()
    df["_date"] = pd.to_datetime(df.index.date, utc=True)

    # Join: each hourly bar gets the macro value for that day
    for col in MACRO_FEATURES:
        if col in macro_df.columns:
            # Resample macro to match hourly index date
            macro_daily = macro_df[[col]].copy()
            macro_daily.index = pd.to_datetime(macro_daily.index.date, utc=True)
            macro_daily = macro_daily[~macro_daily.index.duplicated(keep="last")]
            df = df.join(macro_daily[col], on="_date", how="left")
            # Forward-fill missing (weekends/holidays)
            df[col] = df[col].ffill().fillna(0)
        else:
            df[col] = 0.0

    df = df.drop(columns=["_date"])
    return df


def load_data(coins, macro_df):
    available_feat = LSTM_MOMENTUM_V4_FEATURES
    X_seqs, y_seqs, ts_seqs = [], [], []
    skipped = []
    total_bars = 0

    for coin in coins:
        fp = LABEL_DIR / f"{coin}_features_v3.parquet"
        lp = LABEL_DIR / f"{coin}_momentum_v2_labels.parquet"
        if not fp.exists() or not lp.exists():
            skipped.append(coin); continue

        df = pd.read_parquet(fp).sort_index()
        df = df[df.index < TRAIN_CUTOFF_DATE]
        lbl = pd.read_parquet(lp).sort_index()
        df = df.join(lbl["momentum_v2_label"], how="inner")
        df = df.dropna(subset=["momentum_v2_label"])

        if len(df) < LSTM_SEQ_LEN + 10:
            skipped.append(coin); continue

        # Join macro features
        df = join_macro_to_hourly(df, macro_df)

        # Build feature matrix
        feat_vals = {}
        missing = []
        for c in available_feat:
            if c in df.columns:
                vals = df[c].ffill().fillna(0).values.astype(np.float32)
                if c in _PERCOIN_ZSCORE_FEATS:
                    vals = _percoin_z(vals.astype(np.float64)).astype(np.float32)
                feat_vals[c] = vals
            else:
                feat_vals[c] = np.zeros(len(df), dtype=np.float32)
                missing.append(c)

        if missing:
            logger.info(f"{coin}: missing features (filled 0): {missing}")

        X_c = np.column_stack([feat_vals[c] for c in available_feat])
        y_c = df["momentum_v2_label"].values.astype(np.int64)
        ts_c = df.index.astype(np.int64).values

        n_coin_seqs = 0
        for i in range(LSTM_SEQ_LEN - 1, len(X_c)):
            X_seqs.append(X_c[i - LSTM_SEQ_LEN + 1:i + 1])
            y_seqs.append(y_c[i])
            ts_seqs.append(ts_c[i])
            n_coin_seqs += 1
            total_bars += 1

        n_bull = int((y_c == 2).sum()); n_neu = int((y_c == 1).sum()); n_bear = int((y_c == 0).sum())
        n_total = len(y_c)
        logger.info(f"{coin}: {len(df):,} bars | BULL={n_bull/n_total*100:.0f}% NEU={n_neu/n_total*100:.0f}% BEAR={n_bear/n_total*100:.0f}% | seqs={n_coin_seqs:,}")

    if skipped:
        logger.warning(f"Skipped: {skipped}")
    if not X_seqs:
        raise ValueError("No sequences built. Check momentum_v2_labels exist.")
    X = np.stack(X_seqs); y = np.array(y_seqs, dtype=np.int64); ts = np.array(ts_seqs, dtype=np.int64)
    order = np.argsort(ts)
    logger.info(f"Total sequences: {len(X_seqs):,} | bars processed: {total_bars:,}")
    return X[order], y[order], ts[order], available_feat


def compute_class_weights(y):
    classes, counts = np.unique(y, return_counts=True)
    total = len(y); weights = {c: total / (len(classes) * count) for c, count in zip(classes, counts)}
    return torch.tensor([weights.get(i, 1.0) for i in range(3)], dtype=torch.float32).to(DEVICE)


def train_one_fold(X_tr, y_tr, X_te, y_te, fold_num):
    n_features = X_tr.shape[2]
    fold_scaler = fit_scaler(X_tr)
    X_tr_s = scale_X(X_tr, fold_scaler); del X_tr; gc.collect()
    X_te_s = scale_X(X_te, fold_scaler); del X_te; gc.collect()

    tr_ds = PrebuiltSeqDataset(X_tr_s, y_tr); te_ds = PrebuiltSeqDataset(X_te_s, y_te)
    tr_ld = DataLoader(tr_ds, batch_size=LSTM_BATCH_SIZE, shuffle=True, num_workers=0)
    te_ld = DataLoader(te_ds, batch_size=LSTM_BATCH_SIZE, shuffle=False, num_workers=0)

    model = TradingLSTM(n_features, LSTM_V2_HIDDEN, LSTM_V2_LAYERS, LSTM_V2_DROPOUT).to(DEVICE)
    cw = compute_class_weights(y_tr)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_V2_LR,
                                  weight_decay=LSTM_V2_WEIGHT_DECAY, foreach=False)

    best_f1, best_state, patience_count = -1.0, None, 0
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
            best_f1, best_state, patience_count = f1, {k: v.cpu() for k, v in model.state_dict().items()}, 0
        else:
            patience_count += 1
            if patience_count >= LSTM_PATIENCE: break

        if epoch % 5 == 0 or epoch == 1:
            logger.info(f"[Fold {fold_num}] Epoch {epoch:>3} | F1={f1:.4f} | Best={best_f1:.4f}")

    model.load_state_dict(best_state); model.eval()
    pv, lv = [], []
    with torch.no_grad():
        for xb, yb in te_ld:
            pv.extend(model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy())
            lv.extend(yb.numpy())
    val_f1 = float(f1_score(lv, pv, average="macro", zero_division=0))
    val_f1_p = f1_score(lv, pv, average=None, zero_division=0, labels=[0, 1, 2])

    tr_pv, tr_lv = [], []
    with torch.no_grad():
        for xb, yb in tr_ld:
            tr_pv.extend(model(xb.to(DEVICE)).argmax(dim=1).cpu().numpy())
            tr_lv.extend(yb.numpy())
    train_f1 = float(f1_score(tr_lv, tr_pv, average="macro", zero_division=0))

    metrics = {"fold": fold_num, "train_f1": round(train_f1, 4), "val_f1": round(val_f1, 4),
               "f1_BEARISH": round(float(val_f1_p[0]), 4),
               "f1_NEUTRAL": round(float(val_f1_p[1]), 4),
               "f1_BULLISH": round(float(val_f1_p[2]), 4)}
    logger.info(f"[Fold {fold_num}] Train={train_f1:.4f} Val={val_f1:.4f} Gap={train_f1-val_f1:+.4f}")
    return model, fold_scaler, metrics


def retrain_final(X_all, y_all, n_epochs):
    n_features = X_all.shape[2]
    final_scaler = fit_scaler(X_all)
    X_sc = scale_X(X_all, final_scaler); del X_all; gc.collect()

    ds = PrebuiltSeqDataset(X_sc, y_all)
    loader = DataLoader(ds, batch_size=LSTM_BATCH_SIZE, shuffle=True, num_workers=0)

    model = TradingLSTM(n_features, LSTM_V2_HIDDEN, LSTM_V2_LAYERS, LSTM_V2_DROPOUT).to(DEVICE)
    cw = compute_class_weights(y_all)
    criterion = nn.CrossEntropyLoss(weight=cw)
    optimizer = torch.optim.Adam(model.parameters(), lr=LSTM_V2_LR,
                                  weight_decay=LSTM_V2_WEIGHT_DECAY, foreach=False)

    model.train()
    for epoch in range(1, n_epochs + 1):
        total_loss = 0.0
        for xb, yb in loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        if epoch % 10 == 0 or epoch == 1:
            logger.info(f"[Final] Epoch {epoch:>3}/{n_epochs} | loss={total_loss/len(loader):.4f}")

    model.eval()
    return model, final_scaler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="lstm_momentum_v4")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--coins", type=int, default=5)
    args = parser.parse_args()

    coins = TRAINING_COINS if args.all else TRAINING_COINS[:args.coins]

    print(f"\n{'='*65}")
    print(f"  LSTM MOMENTUM V4 | run_id={args.run_id}")
    print(f"  OHLCV features: {len(OHLCV_FEATURES)} | MACRO features: {len(MACRO_FEATURES)}")
    print(f"  Total features: {len(LSTM_MOMENTUM_V4_FEATURES)} | Seq len: {LSTM_SEQ_LEN}")
    print(f"  Coins: {len(coins)} | Labels: momentum_v2 (flow-based)")
    print(f"{'='*65}\n")

    print("Loading macro data...")
    macro_df = load_macro_data()
    if macro_df is not None:
        coverage_start = macro_df.index[0]
        coverage_end = macro_df.index[-1]
        macro_cols = [c for c in MACRO_FEATURES if c in macro_df.columns]
        print(f"  Available: {len(macro_cols)}/{len(MACRO_FEATURES)} macro features")
        print(f"  Coverage: {coverage_start.date()} -> {coverage_end.date()}")
        for c in macro_cols:
            print(f"    {c}: mean={macro_df[c].mean():.4f}  std={macro_df[c].std():.4f}  non-null={macro_df[c].notna().sum()}")
    print()

    torch.manual_seed(42); np.random.seed(42)
    X, y, ts, feat_cols = load_data(coins, macro_df)
    logger.info(f"Dataset: X={X.shape} | y={y.shape}")
    for lbl_int, lbl_str in MOMENTUM_LABEL_MAP_INV.items():
        logger.info(f"  {lbl_str}: {(y==lbl_int).sum():,} ({(y==lbl_int).mean()*100:.1f}%)")

    run_dir = MODEL_DIR / "runs" / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "lstm_momentum_v4_feature_cols.json", "w") as f:
        json.dump(feat_cols, f, indent=2)

    ts_index = pd.to_datetime(ts, unit="ns", utc=True)
    folds = build_purged_folds(ts_index, n_folds=N_FOLDS, purge=PURGE_GAP_BARS)

    all_metrics = []
    for fi, (tr_idx, te_idx) in enumerate(folds):
        _, _, m = train_one_fold(X[tr_idx], y[tr_idx], X[te_idx], y[te_idx], fi + 1)
        all_metrics.append(m)

    avg_best_epoch = int(np.median([m.get("best_epoch", 30) for m in all_metrics])) if all_metrics else 30
    final_epochs = max(20, min(avg_best_epoch + 5, LSTM_EPOCHS))

    logger.info(f"Retraining final LSTM on 100% data for {final_epochs} epochs...")
    final_model, final_scaler = retrain_final(X, y, final_epochs)

    save_lstm(final_model, run_dir / "lstm_momentum_v4.pt")
    joblib.dump(final_scaler, run_dir / "lstm_momentum_v4_scaler.pkl")

    meta = {
        "run_id": args.run_id, "model_type": "lstm_momentum_v4",
        "label_type": "momentum_flow_v2",
        "n_features": len(feat_cols), "features": feat_cols,
        "ohlcv_features": len(OHLCV_FEATURES),
        "macro_features": len([c for c in MACRO_FEATURES if c in feat_cols]),
        "percoin_z_features": sorted(_PERCOIN_ZSCORE_FEATS),
        "seq_len": LSTM_SEQ_LEN, "hidden": LSTM_V2_HIDDEN,
        "layers": LSTM_V2_LAYERS, "dropout": LSTM_V2_DROPOUT,
        "cv_folds": len(all_metrics),
        "note": "V4: seq_len=32 (8 H4 candles) + macro features (Fear & Greed, ETF flow). Target: break OHLCV ceiling F1 > 0.44."
    }
    with open(run_dir / "lstm_momentum_v4_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    with open(run_dir / "lstm_momentum_v4_cv_results.json", "w") as f:
        json.dump({"run_id": args.run_id, "metrics": all_metrics}, f, indent=2)

    val_f1s = [m["val_f1"] for m in all_metrics]
    train_f1s = [m["train_f1"] for m in all_metrics]
    gaps = [t - v for t, v in zip(train_f1s, val_f1s)]

    print(f"\n{'='*65}")
    print(f"  LSTM MOMENTUM V4 COMPLETE — {args.run_id}")
    print(f"  OHLCV features: {len(OHLCV_FEATURES)} | MACRO features: {len([c for c in MACRO_FEATURES if c in feat_cols])}")
    print(f"  Seq len: {LSTM_SEQ_LEN}")
    print(f"  Mean Val F1: {np.mean(val_f1s):.4f} +/- {np.std(val_f1s):.4f}")
    print(f"  Random baseline: 0.333")
    print(f"  Gain vs random: {np.mean(val_f1s) - 0.333:+.4f}")
    print(f"  Mean Train F1: {np.mean(train_f1s):.4f}")
    print(f"  Mean Gap: {np.mean(gaps):+.4f} (target: < +0.08)")
    print(f"\n  Fold results:")
    for m in all_metrics:
        g = m["train_f1"] - m["val_f1"]
        print(f"  Fold {m['fold']}: Train={m['train_f1']:.4f} Val={m['val_f1']:.4f} Gap={g:+.4f}")
    print(f"\n  V3 baseline (seq=16, 16 OHLCV):   0.407 ± 0.007")
    print(f"  V4 target  (seq=32, 16 OHLCV+5 MACRO): > 0.44")
    print(f"\n  Model: {run_dir / 'lstm_momentum_v4.pt'}")


if __name__ == "__main__":
    main()
