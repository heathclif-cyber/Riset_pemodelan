"""
pipeline/03e_lstm_feature_selection.py — Feature Selection khusus LSTM

Metode:
  1. Spearman IC (Standalone + Marginal via Gram-Schmidt)
  2. Mutual Information — non-linear dependency dengan target 3-class
  3. Permutation Importance (MDA) — OOF drop F1 via LSTM terlatih
  4. Ablation — backward elimination berbasis F1 validation

Metode 1-2 untuk screening cepat (tanpa training LSTM).
Metode 3-4 membutuhkan training LSTM — lebih akurat tapi mahal.

Usage:
  python pipeline/03e_lstm_feature_selection.py --run-id ic32_hybrid_lstm --stage 1    # IC+MI screening
  python pipeline/03e_lstm_feature_selection.py --run-id ic32_hybrid_lstm --stage 2    # Permutation MDA
  python pipeline/03e_lstm_feature_selection.py --run-id ic32_hybrid_lstm --stage all  # Full pipeline
"""

import argparse, json, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.preprocessing import RobustScaler
import torch, torch.nn as nn
from torch.utils.data import DataLoader
import joblib

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from config import (
    TRAINING_COINS, PROC_DIR, LABEL_DIR, TRAIN_CUTOFF_DATE,
    N_FOLDS, PURGE_GAP_BARS, MODEL_DIR,
    LSTM_SEQ_LEN, LSTM_BATCH_SIZE, LSTM_EPOCHS, LSTM_PATIENCE,
    LSTM_V2_HIDDEN, LSTM_V2_LAYERS, LSTM_V2_DROPOUT,
    LSTM_V2_WEIGHT_DECAY, LSTM_V2_LR,
    FEATURE_COLS_V3,
)
from core.models import TradingLSTM
from core.utils import setup_logger, get_lstm_device
from pipeline.shared import build_purged_folds

warnings.filterwarnings("ignore")
logger = setup_logger("03e_lstm_feat_sel")


def load_training_data(feature_cols):
    """Load training data for all coins, return X (flat) and y."""
    frames = []
    for coin in TRAINING_COINS:
        path = LABEL_DIR / f"{coin}_features_v3.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = df[df.index <= TRAIN_CUTOFF_DATE]
        available = [c for c in feature_cols if c in df.columns]
        if len(available) < len(feature_cols):
            missing = set(feature_cols) - set(available)
            if len(missing) > 5:
                logger.warning(f"[{coin}] Missing {len(missing)} features: {list(missing)[:5]}...")
        df = df[available + ["label"]]
        etf_cols = [c for c in ["etf_total_change_usd", "etf_gbtc_change_usd"] if c in df.columns]
        for c in etf_cols:
            df[c] = df[c].fillna(0.0)
        df = df.dropna()
        frames.append(df)

    data = pd.concat(frames, axis=0)
    X = data[available].values.astype(np.float32)
    label_map = {"SHORT": 0, "FLAT": 1, "LONG": 2}
    y_raw = data["label"].map(label_map)
    y = y_raw.values.astype(np.int64)
    # Drop rows with unmapped labels
    valid = ~np.isnan(y_raw.values.astype(float))
    X, y = X[valid], y[valid]

    # Subsampling FLAT untuk balancing
    rng = np.random.RandomState(42)
    flat_idx = np.where(y == 1)[0]
    n_target = len(flat_idx) // 3
    drop_flat = rng.choice(flat_idx, size=len(flat_idx) - n_target, replace=False)
    keep = np.ones(len(y), dtype=bool)
    keep[drop_flat] = False
    X, y = X[keep], y[keep]

    logger.info(f"Data: {len(X)} rows x {len(available)} features | Label: {np.bincount(y)}")
    return X, y, available


def calc_spearman_ic(X, y, feature_names):
    """Standalone Spearman IC."""
    results = []
    for i, name in enumerate(feature_names):
        col = X[:, i]
        mask = np.isfinite(col)
        if mask.sum() < 100:
            results.append({"feature": name, "sa_ic": 0, "t_stat": 0})
            continue
        ic, pval = spearmanr(col[mask], y[mask])
        t_stat = ic * np.sqrt((mask.sum() - 2) / (1 - ic**2 + 1e-10))
        results.append({"feature": name, "sa_ic": ic, "t_stat": t_stat})
    return pd.DataFrame(results).sort_values("sa_ic", key=abs, ascending=False)


def calc_mutual_info(X, y, feature_names):
    """Mutual Information dengan target."""
    mi = mutual_info_classif(X, y, random_state=42, n_neighbors=5)
    df = pd.DataFrame({"feature": feature_names, "mutual_info": mi})
    return df.sort_values("mutual_info", ascending=False)


def calc_marginal_ic(X, y, feature_names):
    """Marginal IC via simplified Gram-Schmidt orthogonalization."""
    n = X.shape[1]
    order = []
    used = np.zeros(len(y), dtype=np.float32)
    results = []

    # Sort by standalone IC
    sa_ic = []
    for i in range(n):
        ic, _ = spearmanr(X[:, i], y)
        sa_ic.append((abs(ic), i))
    sa_ic.sort(reverse=True)

    for _, idx in sa_ic:
        col = X[:, idx].copy()
        if len(order) > 0:
            # Residualize against already-selected features
            for j in order:
                beta = np.dot(col, X[:, j]) / (np.dot(X[:, j], X[:, j]) + 1e-10)
                col = col - beta * X[:, j]
        marg_ic, _ = spearmanr(col, y)
        results.append({"feature": feature_names[idx], "marg_ic": marg_ic})
        order.append(idx)

    return pd.DataFrame(results).sort_values("marg_ic", key=abs, ascending=False)


def build_lstm_sequences(X_coin, y_coin, seq_len=32, step=1):
    """Build sequences for LSTM from per-coin data."""
    n = len(X_coin)
    if n <= seq_len:
        return np.array([]), np.array([])
    idx = np.arange(seq_len - 1, n, step)
    X_seq = np.stack([X_coin[i - seq_len + 1 : i + 1] for i in idx])
    y_seq = y_coin[idx]
    return X_seq, y_seq


def calc_permutation_mda(feature_cols, run_id="lstm_mda_temp"):
    """
    Train LSTM once, then measure permutation importance per feature.
    Only runs on 1 fold + 5 coins for speed.
    """
    n_coins = min(5, len(TRAINING_COINS))
    coins = TRAINING_COINS[:n_coins]

    # Load data
    X_list, y_list = [], []
    for coin in coins:
        path = LABEL_DIR / f"{coin}_features_v3.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        df = df[df.index <= TRAIN_CUTOFF_DATE]
        available = [c for c in feature_cols if c in df.columns]
        df = df[available + ["label"]]
        etf_cols = [c for c in ["etf_total_change_usd", "etf_gbtc_change_usd"] if c in df.columns]
        for c in etf_cols:
            df[c] = df[c].fillna(0.0)
        df = df.dropna()
        X_coin = RobustScaler().fit_transform(df[available].values.astype(np.float32))
        label_map = {"SHORT": 0, "FLAT": 1, "LONG": 2}
        y_coin = df["label"].map(label_map).values.astype(np.int64)
        X_seq, y_seq = build_lstm_sequences(X_coin, y_coin, LSTM_SEQ_LEN, step=3)
        if len(X_seq) > 0:
            X_list.append(X_seq)
            y_list.append(y_seq)

    if not X_list:
        logger.error("No sequences built!")
        return pd.DataFrame()

    X_all = np.concatenate(X_list)
    y_all = np.concatenate(y_list)

    # Single train/val split (80/20)
    n_train = int(len(X_all) * 0.8)
    X_train, X_val = X_all[:n_train], X_all[n_train:]
    y_train, y_val = y_all[:n_train], y_all[n_train:]

    # Train LSTM
    n_feat = X_train.shape[2]
    model = TradingLSTM(n_feat, LSTM_V2_HIDDEN, LSTM_V2_LAYERS, LSTM_V2_DROPOUT, 3)
    device = torch.device("cpu")
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LSTM_V2_LR, weight_decay=LSTM_V2_WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()

    # Quick training (10 epochs)
    X_t = torch.from_numpy(X_train.astype(np.float32))
    y_t = torch.from_numpy(y_train.astype(np.int64))
    model.train()
    for epoch in range(10):
        perm = torch.randperm(len(X_t))
        for i in range(0, len(X_t), LSTM_BATCH_SIZE):
            idx = perm[i:i+LSTM_BATCH_SIZE]
            xb, yb = X_t[idx].to(device), y_t[idx].to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    # Evaluate baseline
    model.eval()
    X_v = torch.from_numpy(X_val.astype(np.float32))
    y_v = torch.from_numpy(y_val.astype(np.int64))
    with torch.no_grad():
        base_pred = model(X_v.to(device)).argmax(-1).cpu().numpy()
    base_f1 = f1_score(y_v, base_pred, average="macro")

    # Permutation importance
    results = []
    for fi in range(n_feat):
        X_perm = X_val.copy()
        # Permute feature fi across all sequences
        orig = X_perm[:, :, fi].copy()
        perm_idx = np.random.permutation(len(X_perm))
        X_perm[:, :, fi] = orig[perm_idx]
        X_p = torch.from_numpy(X_perm.astype(np.float32))
        with torch.no_grad():
            perm_pred = model(X_p.to(device)).argmax(-1).cpu().numpy()
        perm_f1 = f1_score(y_v, perm_pred, average="macro")
        f1_drop = base_f1 - perm_f1
        results.append({"feature": feature_cols[fi], "mda_drop": f1_drop, "base_f1": base_f1})

    df = pd.DataFrame(results).sort_values("mda_drop", ascending=False)
    logger.info(f"Baseline F1: {base_f1:.4f} | Top MDA: {df.iloc[0]['feature']} ({df.iloc[0]['mda_drop']:.4f})")
    return df


def main():
    parser = argparse.ArgumentParser(description="LSTM Feature Selection")
    parser.add_argument("--run-id", default="lstm_feat_sel", help="Run ID")
    parser.add_argument("--stage", default="1", choices=["1", "2", "all"], help="Stage")
    parser.add_argument("--min-ic", type=float, default=0.01, help="Min standalone IC")
    parser.add_argument("--min-mi", type=float, default=0.002, help="Min mutual info")
    parser.add_argument("--min-marginal", type=float, default=0.005, help="Min marginal IC")
    args = parser.parse_args()

    # Candidate features for LSTM v2 — full 27 + extras
    feature_candidates = [
        "rsi_6", "stochrsi_k", "stochrsi_d", "rsi_slope_h4", "rsi_h4",
        "ema_21_slope_h4", "cvd_slope_h4", "ofi_h4_delta", "cvd_momentum_adv",
        "swing_momentum", "dist_from_8h_high", "price_in_range",
        "long_short_ratio", "dist_liq_50x_long", "hmm_regime_enc",
        "vol_price_confirm", "ofi_z_score", "absorption_at_swing",
        "ema_7_h1", "volume_delta", "trend_accel_4h", "whale_retail_divergence",
        "dist_liq_50x_short", "Buy_Liq", "Sell_Liq",
        "etf_total_change_usd", "etf_gbtc_change_usd",
        "log_ret_5", "log_ret_20", "ofi_raw", "h4_trend", "vol_regime",
    ]
    logger.info(f"Candidate features: {len(feature_candidates)}")

    print("=" * 70)
    print(f"  LSTM FEATURE SELECTION | run_id={args.run_id} | Stage {args.stage}")
    print(f"  Candidates: {len(feature_candidates)} features")
    print(f"  Thresholds: IC>={args.min_ic}, MI>={args.min_mi}, Marg>={args.min_marginal}")
    print("=" * 70)

    # ─── Stage 1: IC + MI Screening ─────────────────────────────────────────
    if args.stage in ("1", "all"):
        print("\n[Stage 1] Spearman IC + Mutual Information screening...")
        X, y, available = load_training_data(feature_candidates)

        ic_df = calc_spearman_ic(X, y, available)
        mi_df = calc_mutual_info(X, y, available)
        marg_df = calc_marginal_ic(X, y, available)

        # Merge results
        results = ic_df.merge(mi_df, on="feature").merge(marg_df, on="feature")

        # Classification
        results["verdict"] = "WEAK"
        results.loc[
            (results["sa_ic"].abs() >= args.min_ic) &
            (results["mutual_info"] >= args.min_mi) &
            (results["marg_ic"].abs() >= args.min_marginal),
            "verdict"
        ] = "KEEP"
        # Strong IC (> 0.03) keeps even with low MI
        results.loc[results["sa_ic"].abs() >= 0.03, "verdict"] = "KEEP"
        results.loc[
            (results["sa_ic"].abs() >= args.min_ic) &
            (results["marg_ic"].abs() < args.min_marginal),
            "verdict"
        ] = "REDUNDANT"

        keep = results[results["verdict"] == "KEEP"]["feature"].tolist()
        redundant = results[results["verdict"] == "REDUNDANT"]["feature"].tolist()
        weak = results[results["verdict"] == "WEAK"]["feature"].tolist()

        print(f"\n{'='*70}")
        print(f"{'Feature':<35s} {'SA_IC':>7s} {'MI':>7s} {'Marg_IC':>7s} {'Verdict':>12s}")
        print(f"{'-'*70}")
        for _, r in results.iterrows():
            print(f"{r['feature']:<35s} {r['sa_ic']:>+7.4f} {r['mutual_info']:>7.4f} {r['marg_ic']:>+7.4f} {r['verdict']:>12s}")
        print(f"{'-'*70}")
        print(f"KEEP={len(keep)} | REDUNDANT={len(redundant)} | WEAK={len(weak)}")

        # Save stage 1
        selected = keep
        out_path = MODEL_DIR / "runs" / args.run_id
        out_path.mkdir(parents=True, exist_ok=True)
        feat_path = out_path / "lstm_selected_feats_stage1.json"
        json.dump(selected, open(feat_path, "w"), indent=2)
        print(f"\nStage 1 results -> {feat_path} ({len(selected)} features)")

        # Full report
        report = {
            "run_id": args.run_id,
            "stage": 1,
            "candidates": len(feature_candidates),
            "thresholds": {"min_ic": args.min_ic, "min_mi": args.min_mi, "min_marginal": args.min_marginal},
            "keep": keep, "redundant": redundant, "weak": weak,
            "details": results.to_dict("records"),
        }
        json.dump(report, open(out_path / "lstm_feat_sel_report.json", "w"), indent=2, default=str)

    # ─── Stage 2: Permutation MDA ───────────────────────────────────────────
    if args.stage in ("2", "all"):
        print("\n[Stage 2] Permutation Importance (MDA)...")
        # Load stage 1 results if available
        stage1_path = MODEL_DIR / "runs" / args.run_id / "lstm_selected_feats_stage1.json"
        if stage1_path.exists():
            candidates = json.load(open(stage1_path))
            print(f"Using {len(candidates)} KEEP features from Stage 1")
        else:
            candidates = feature_candidates

        mda_df = calc_permutation_mda(candidates, run_id=args.run_id)

        if not mda_df.empty:
            print(f"\n{'='*60}")
            print(f"  Permutation Importance (F1 drop on permutation)")
            print(f"{'='*60}")
            print(f"{'Feature':<35s} {'MDA_F1_drop':>12s}")
            print(f"{'-'*60}")
            for _, r in mda_df.iterrows():
                bar = "#" * max(1, int(r["mda_drop"] * 500))
                print(f"{r['feature']:<35s} {r['mda_drop']:>+12.4f}  {bar}")
            print(f"{'-'*60}")

            # Select features with positive MDA contribution
            selected = mda_df[mda_df["mda_drop"] > 0]["feature"].tolist()
            feat_path = MODEL_DIR / "runs" / args.run_id / "lstm_selected_feats_final.json"
            json.dump(selected, open(feat_path, "w"), indent=2)
            print(f"\nFinal selected features -> {feat_path} ({len(selected)} features)")

    print("\nLSTM Feature Selection complete.")


if __name__ == "__main__":
    main()
