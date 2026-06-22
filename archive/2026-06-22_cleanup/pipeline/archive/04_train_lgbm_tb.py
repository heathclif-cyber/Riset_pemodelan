"""
pipeline/04_train_lgbm_tb.py — Triple Barrier LGBM widyawardhana_v1

Stage 1: Triple Barrier labeling (on-the-fly dari OHLCV)
Stage 2: Multistage Feature Selection (IC standalone + Marginal + Decay + Temporal)
Stage 3: LGBM purged CV training
Stage 4: Save model + metadata

Jalankan:
  python pipeline/04_train_lgbm_tb.py
  python pipeline/04_train_lgbm_tb.py --all
  python pipeline/04_train_lgbm_tb.py --tp 2.5 --sl 1.5 --max-hold 36
"""

import argparse
import json
import sys
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.shared import build_purged_folds
from core.features import triple_barrier_labeling
from core.utils import setup_logger

from config import (
    TRAINING_COINS, ALL_COINS,
    LABEL_DIR, MODEL_DIR,
    LGBM_PARAMS, LGBM_EARLY_STOPPING,
    N_FOLDS, PURGE_GAP_BARS,
    LGBM_CLASS_WEIGHTS,
    FEATURE_COLS_V3, TRAIN_CUTOFF_DATE,
    MAX_HOLDING_BARS, TP_SL_FALLBACK_TP, TP_SL_FALLBACK_SL,
)

logger = setup_logger("04_train_lgbm_tb")

# ─── Constants ─────────────────────────────────────────────────────────────────
RUN_NAME = "tb_lgbm_widyawardhana_v1"
LABEL_ORDINAL = {"SHORT": -1, "FLAT": 0, "LONG": 1}
TB_LABEL_MAP = {"SHORT": 0, "FLAT": 1, "LONG": 2}
AUTOCORR_FACTOR = 24  # H1 data — effective N = N / 24
META_COLS = {"label", "coin", "symbol", "h4_swing_high", "h4_swing_low",
             "hmm_regime", "hmm_regime_enc", "tb_label"}

# IC thresholds
MIN_STANDALONE_IC = 0.02
MIN_TSTAT = 2.0
MIN_MARGINAL_IC = 0.01

# Temporal decay windows (years)
DECAY_WINDOWS = [
    ("2020", "2020-01-01", "2020-12-31"),
    ("2021", "2021-01-01", "2021-12-31"),
    ("2022", "2022-01-01", "2022-12-31"),
    ("2023", "2023-01-01", "2023-12-31"),
    ("2024", "2024-01-01", "2024-12-31"),
    ("2025", "2025-01-01", "2025-10-31"),
]

# Temporal IC lags
TEMPORAL_LAGS = [0, 1, 2, 4, 8, 12, 16, 24, 32]


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 1 — DATA LOADING + TRIPLE BARRIER LABELING
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_label(coins: list[str], tp_atr: float, sl_atr: float, max_hold: int) -> pd.DataFrame:
    """Load training data, generate TB labels on-the-fly."""
    frames = []
    for sym in coins:
        path = LABEL_DIR / f"{sym}_features_v3.parquet"
        if not path.exists():
            logger.warning(f"[{sym}] File tidak ditemukan, skip")
            continue

        df = pd.read_parquet(path)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index, utc=True)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        df = df[df.index < TRAIN_CUTOFF_DATE]
        if df.empty:
            continue

        # Verify required columns
        required = ["close", "high", "low", "atr_14_h1"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            logger.warning(f"[{sym}] Missing {missing}, skip")
            continue

        # Generate TB labels
        tb_labels = triple_barrier_labeling(
            close=df["close"],
            high=df["high"],
            low=df["low"],
            atr_base=df["atr_14_h1"],
            tp_atr_mult=tp_atr,
            sl_atr_mult=sl_atr,
            max_hold=max_hold,
        )
        df["tb_label"] = tb_labels.map(TB_LABEL_MAP)

        # Keep only rows with valid TB labels
        df = df.dropna(subset=["tb_label"])
        mask = df["tb_label"].astype(int).isin([0, 1, 2])
        df = df[mask].copy()
        if len(df) < 100:
            logger.warning(f"[{sym}] Too few rows after TB labeling: {len(df)}, skip")
            continue

        df["coin"] = sym
        frames.append(df)

        dist = df["tb_label"].value_counts().to_dict()
        n = len(df)
        logger.info(
            f"  [{sym}] {n:,} bars | "
            f"LONG={dist.get(2,0)/n*100:.1f}% SHORT={dist.get(0,0)/n*100:.1f}% FLAT={dist.get(1,0)/n*100:.1f}%"
        )

    if not frames:
        raise RuntimeError("Tidak ada data training!")

    combined = pd.concat(frames).sort_index()
    logger.info(f"Total: {len(combined):,} rows x {len(combined.columns)} cols")
    return combined


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 2 — MULTISTAGE FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

# ── 2a: Standalone IC ──────────────────────────────────────────────────────────

def standalone_ic(x: np.ndarray, y: np.ndarray) -> float:
    mask = ~(np.isnan(x) | np.isnan(y))
    if mask.sum() < 100:
        return 0.0
    corr, _ = stats.spearmanr(x[mask], y[mask])
    return float(corr) if not np.isnan(corr) else 0.0


def tstat_ic(ic: float, n: int) -> float:
    n_eff = max(n // AUTOCORR_FACTOR, 10)
    denom = np.sqrt(max(1.0 - ic ** 2, 1e-10))
    return ic * np.sqrt(n_eff) / denom


# ── 2a: Marginal IC via Gram-Schmidt ───────────────────────────────────────────

def _rank_norm(x: np.ndarray) -> np.ndarray:
    r = stats.rankdata(x).astype(np.float64)
    r -= r.mean()
    std = r.std()
    return r / std if std > 1e-10 else np.zeros_like(r)


def _project_out(vec: np.ndarray, pivot: np.ndarray) -> np.ndarray:
    norm_sq = np.dot(pivot, pivot)
    if norm_sq < 1e-10:
        return vec.copy()
    return vec - (np.dot(vec, pivot) / norm_sq) * pivot


def gram_schmidt_marginal(X_raw: np.ndarray, y_raw: np.ndarray,
                          feature_names: list, max_samples: int = 50000) -> dict:
    n, p = X_raw.shape

    # Downsample jika terlalu besar (Gram-Schmidt O(p^2 * n))
    if n > max_samples:
        rng = np.random.RandomState(42)
        idx = rng.choice(n, max_samples, replace=False)
        X_raw = X_raw[idx]
        y_raw = y_raw[idx]
        n = max_samples
        logger.info(f"  Gram-Schmidt: downsampled to {max_samples:,} rows")

    X = X_raw.copy().astype(np.float64)
    for j in range(p):
        col = X[:, j]
        nan_mask = np.isnan(col)
        if nan_mask.any():
            med = np.nanmedian(col)
            col[nan_mask] = med if not np.isnan(med) else 0.0
            X[:, j] = col

    X_r = np.column_stack([_rank_norm(X[:, j]) for j in range(p)])
    y_r = _rank_norm(y_raw.astype(np.float64))

    remaining = list(range(p))
    marginal = {}

    for _ in range(p):
        if not remaining:
            break
        corrs = np.zeros(len(remaining))
        for k, j in enumerate(remaining):
            xj = X_r[:, j]
            nx = np.sqrt(np.dot(xj, xj))
            ny = np.sqrt(np.dot(y_r, y_r))
            if nx < 1e-10 or ny < 1e-10:
                corrs[k] = 0.0
            else:
                corrs[k] = np.dot(xj, y_r) / (nx * ny)

        best_k = int(np.argmax(np.abs(corrs)))
        best_j = remaining[best_k]
        marginal[feature_names[best_j]] = float(corrs[best_k])

        pivot = X_r[:, best_j].copy()
        for j in remaining:
            if j != best_j:
                X_r[:, j] = _project_out(X_r[:, j], pivot)
        y_r = _project_out(y_r, pivot)
        remaining.remove(best_j)

    return marginal


def make_verdict(sa: float, ts: float, mg: float) -> str:
    sa_pass = abs(sa) >= MIN_STANDALONE_IC and abs(ts) >= MIN_TSTAT
    mg_pass = abs(mg) >= MIN_MARGINAL_IC
    if sa_pass and mg_pass:
        return "KEEP"
    elif sa_pass and not mg_pass:
        return "REDUNDANT"
    elif not sa_pass and mg_pass:
        return "WEAK"
    else:
        return "DROP"


# ── 2b: IC Decay Stability ─────────────────────────────────────────────────────

def run_ic_decay_test(df: pd.DataFrame, feature_cols: list,
                      target_col: str = "tb_label_ord") -> dict:
    """Hitung IC per temporal window untuk setiap fitur."""
    results = {}
    for feat in feature_cols:
        if feat not in df.columns:
            continue
        window_ics = []
        for w_name, w_start, w_end in DECAY_WINDOWS:
            mask = (df.index >= w_start) & (df.index <= w_end)
            w_df = df.loc[mask]
            if len(w_df) < 500:
                window_ics.append({"window": w_name, "ic": float("nan"), "n": len(w_df)})
                continue
            ic = standalone_ic(w_df[feat].values, w_df[target_col].values)
            window_ics.append({"window": w_name, "ic": round(ic, 4), "n": len(w_df)})

        valid_ics = [w["ic"] for w in window_ics if not np.isnan(w["ic"])]
        if len(valid_ics) >= 4:
            ic_mean = np.mean(valid_ics)
            ic_std = np.std(valid_ics)
            ic_ir = abs(ic_mean) / ic_std if ic_std > 1e-10 else 0.0
            # sign consistency: berapa window yang IC-nya searah dengan mean
            sign_cons = sum(1 for v in valid_ics if np.sign(v) == np.sign(ic_mean))
            is_stable = ic_ir >= 0.5 and sign_cons >= len(valid_ics) - 1
        else:
            ic_mean = valid_ics[0] if valid_ics else 0.0
            ic_std = 0.0
            ic_ir = 0.0
            sign_cons = len(valid_ics)
            is_stable = False

        results[feat] = {
            "ic_mean": round(float(ic_mean), 4),
            "ic_std": round(float(ic_std), 4),
            "ic_ir": round(float(ic_ir), 2),
            "sign_consistency": sign_cons,
            "n_windows_valid": len(valid_ics),
            "is_stable": is_stable,
            "windows": window_ics,
        }
    return results


# ── 2c: Temporal IC (Half-Life) ────────────────────────────────────────────────

def run_temporal_ic(df: pd.DataFrame, feature_cols: list,
                    target_col: str = "tb_label_ord") -> dict:
    """IC(feat_{t-k}, label_t) untuk k in TEMPORAL_LAGS."""
    results = {}
    for feat in feature_cols:
        if feat not in df.columns:
            continue
        x = df[feat]
        y = df[target_col]
        lag_ics = {}
        for k in TEMPORAL_LAGS:
            x_lag = x.shift(k)
            mask = ~(x_lag.isna() | y.isna())
            if mask.sum() < 100:
                lag_ics[str(k)] = float("nan")
                continue
            corr, _ = stats.spearmanr(x_lag[mask], y[mask])
            lag_ics[str(k)] = round(float(corr) if not np.isnan(corr) else float("nan"), 4)

        # Hitung half-life: lag dimana |IC| turun ke 50% dari |IC_0|
        ic0 = abs(lag_ics.get("0", 0.0) or 0.0)
        half_life = None
        if ic0 >= 0.01:
            for k in TEMPORAL_LAGS:
                ic_k = abs(lag_ics.get(str(k), 0.0) or 0.0)
                if ic_k <= ic0 * 0.5:
                    half_life = k
                    break
            if half_life is None:
                half_life = TEMPORAL_LAGS[-1]  # never decayed to 50%

        if half_life and half_life >= 4:
            category = "STRONG"
        elif half_life and half_life >= 2:
            category = "MODERATE"
        else:
            category = "SHORT"

        results[feat] = {
            "ic_0": lag_ics.get("0", float("nan")),
            "half_life": half_life,
            "category": category,
            "lag_ics": lag_ics,
        }
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# STAGE 3 — LGBM TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def walk_forward_cv(X: pd.DataFrame, y: pd.Series, params: dict,
                    n_splits: int = N_FOLDS, purge: int = PURGE_GAP_BARS):
    """Walk-forward purged CV, same pattern as 04_train_lgbm.py."""
    logger.info(f"Walk-Forward CV (n_splits={n_splits}, purge={purge} bars)...")

    folds = build_purged_folds(X.index, n_folds=n_splits, purge=purge)
    results = []

    for fold, (train_idx, val_idx) in enumerate(folds, 1):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        sample_w = np.array([LGBM_CLASS_WEIGHTS[int(label)] for label in y_tr],
                           dtype=np.float32)

        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            sample_weight=sample_w,
            eval_set=[(X_val, y_val)],
            callbacks=[
                lgb.early_stopping(LGBM_EARLY_STOPPING, verbose=False),
                lgb.log_evaluation(period=-1),
            ],
        )

        y_pred = model.predict(X_val)
        f1_macro = float(f1_score(y_val, y_pred, average="macro", zero_division=0))
        f1_per = f1_score(y_val, y_pred, average=None, zero_division=0,
                          labels=[0, 1, 2])
        acc = float(accuracy_score(y_val, y_pred))

        y_train_pred = model.predict(X_tr)
        train_f1_macro = float(f1_score(y_tr, y_train_pred, average="macro",
                                       zero_division=0))
        train_f1_per = f1_score(y_tr, y_train_pred, average=None, zero_division=0,
                                labels=[0, 1, 2])

        metrics = {
            "fold": fold,
            "n_train": len(X_tr), "n_val": len(X_val),
            "best_iteration": model.best_iteration_,
            "train_f1_macro": round(train_f1_macro, 4),
            "train_f1_SHORT": round(float(train_f1_per[0]), 4),
            "train_f1_FLAT": round(float(train_f1_per[1]), 4),
            "train_f1_LONG": round(float(train_f1_per[2]), 4),
            "accuracy": round(acc, 4),
            "f1_macro": round(f1_macro, 4),
            "f1_SHORT": round(float(f1_per[0]), 4),
            "f1_FLAT": round(float(f1_per[1]), 4),
            "f1_LONG": round(float(f1_per[2]), 4),
            "confusion_matrix": confusion_matrix(y_val, y_pred, labels=[0, 1, 2]).tolist(),
        }

        gap = train_f1_macro - f1_macro
        logger.info(
            f"  Fold {fold}: Train F1={train_f1_macro:.4f} | Val F1={f1_macro:.4f} | "
            f"Gap={gap:+.4f} | Acc={acc:.4f} | "
            f"LONG={f1_per[2]:.4f} SHORT={f1_per[0]:.4f} FLAT={f1_per[1]:.4f}"
        )
        results.append({"metrics": metrics, "model": model})

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="TB LGBM widyawardhana_v1")
    parser.add_argument("--all", action="store_true",
                        help="Pakai semua koin")
    parser.add_argument("--coins", nargs="+", default=None,
                        help="Koin spesifik")
    parser.add_argument("--tp", type=float, default=TP_SL_FALLBACK_TP,
                        help="TP multiplier (ATR)")
    parser.add_argument("--sl", type=float, default=TP_SL_FALLBACK_SL,
                        help="SL multiplier (ATR)")
    parser.add_argument("--max-hold", type=int, default=MAX_HOLDING_BARS,
                        help="Max holding bars")
    parser.add_argument("--skip-ic", action="store_true",
                        help="Skip IC test, use all features")
    args = parser.parse_args()

    if args.coins:
        coins = args.coins
    elif args.all:
        coins = ALL_COINS
    else:
        coins = TRAINING_COINS

    tp_atr = args.tp
    sl_atr = args.sl
    max_hold = args.max_hold
    rr = tp_atr / sl_atr

    run_dir = MODEL_DIR / "runs" / RUN_NAME
    run_dir.mkdir(parents=True, exist_ok=True)

    sep = "=" * 70
    print(f"\n{sep}")
    print(f"  TRIPLE BARRIER LGBM — {RUN_NAME}")
    print(f"{sep}")
    print(f"  TP={tp_atr}xATR  SL={sl_atr}xATR  RR={rr:.2f}  MaxHold={max_hold}")
    print(f"  Coins: {len(coins)} | Cutoff: {TRAIN_CUTOFF_DATE.date()}")
    print(f"  Output: {run_dir}")
    print(f"{sep}\n")

    # ── Stage 1: Load + Label ──────────────────────────────────────────────────
    print("STAGE 1: Triple Barrier Labeling...")
    print("-" * 50)
    df = load_and_label(coins, tp_atr, sl_atr, max_hold)

    # Encode TB label to ordinal for IC test
    df["tb_label_ord"] = df["tb_label"].map({0: -1, 1: 0, 2: 1})
    y_full = df["tb_label"].astype(np.int32)

    # Determine feature columns
    avail_features = [c for c in FEATURE_COLS_V3
                      if c in df.columns and c not in META_COLS]
    logger.info(f"Features available: {len(avail_features)}")

    # ── Stage 2: Feature Selection ─────────────────────────────────────────────
    if not args.skip_ic:
        print(f"\nSTAGE 2a: IC Test (Standalone + Marginal)...")
        print("-" * 50)
        target = df["tb_label_ord"].values
        n_total = len(df)
        n_eff = n_total // AUTOCORR_FACTOR

        label_dist = df["tb_label"].value_counts().to_dict()
        label_names = {0: "SHORT", 1: "FLAT", 2: "LONG"}
        print(f"  Rows: {n_total:,} | Effective N: {n_eff:,}")
        print(f"  Label: {{{', '.join(f'{label_names[k]}: {v:,}' for k, v in sorted(label_dist.items()))}}}")
        print(f"  Features tested: {len(avail_features)}")
        print(f"  Thresholds: |SA|>={MIN_STANDALONE_IC}, |t|>={MIN_TSTAT}, |MG|>={MIN_MARGINAL_IC}")

        # Standalone IC
        standalone = {}
        tstats = {}
        for feat in avail_features:
            ic = standalone_ic(df[feat].values, target)
            standalone[feat] = ic
            tstats[feat] = tstat_ic(ic, n_total)

        # Marginal IC
        X_mat = df[avail_features].values
        marginal = gram_schmidt_marginal(X_mat, target, avail_features)

        # Verdict
        ic_results = []
        for feat in avail_features:
            sa = standalone[feat]
            ts = tstats[feat]
            mg = marginal.get(feat, 0.0)
            v = make_verdict(sa, ts, mg)
            ic_results.append({
                "feature": feat,
                "standalone_ic": round(sa, 4),
                "tstat": round(ts, 2),
                "marginal_ic": round(mg, 4),
                "verdict": v,
            })

        verdict_order = {"KEEP": 0, "REDUNDANT": 1, "WEAK": 2, "DROP": 3}
        ic_results.sort(key=lambda x: (verdict_order[x["verdict"]],
                                        -abs(x["standalone_ic"])))

        verdicts = [r["verdict"] for r in ic_results]
        summary = {v: verdicts.count(v) for v in ["KEEP", "REDUNDANT", "WEAK", "DROP"]}
        keep_features = [r["feature"] for r in ic_results if r["verdict"] == "KEEP"]

        print(f"\n  Summary: KEEP={summary['KEEP']} | REDUNDANT={summary['REDUNDANT']} | "
              f"WEAK={summary['WEAK']} | DROP={summary['DROP']}")
        print(f"\n  KEEP features ({summary['KEEP']}):")
        for r in ic_results:
            if r["verdict"] == "KEEP":
                print(f"    {r['standalone_ic']:+.4f}  {r['feature']}")

        # ── 2b: IC Decay Stability ─────────────────────────────────────────────
        print(f"\nSTAGE 2b: IC Decay Stability ({len(DECAY_WINDOWS)} windows)...")
        print("-" * 50)
        decay_results = run_ic_decay_test(df, avail_features)
        n_stable = sum(1 for v in decay_results.values() if v["is_stable"])
        print(f"  Stable (IC_IR>=0.5, sign_cons>=5/6): {n_stable}/{len(decay_results)}")

        # Mark unstable features
        for feat in avail_features:
            if feat not in decay_results:
                continue
            if not decay_results[feat]["is_stable"] and feat in keep_features:
                print(f"  [DROP-unstable] {feat}: IC_IR={decay_results[feat]['ic_ir']:.2f} "
                      f"sign_cons={decay_results[feat]['sign_consistency']}/{decay_results[feat]['n_windows_valid']}")

        # ── 2c: Temporal IC ────────────────────────────────────────────────────
        print(f"\nSTAGE 2c: Temporal IC Decay (half-life)...")
        print("-" * 50)
        temporal_results = run_temporal_ic(df, avail_features)
        strong_count = sum(1 for v in temporal_results.values() if v["category"] == "STRONG")
        moderate_count = sum(1 for v in temporal_results.values() if v["category"] == "MODERATE")
        short_count = sum(1 for v in temporal_results.values() if v["category"] == "SHORT")
        print(f"  STRONG (hl>=4): {strong_count} | MODERATE (hl=2-4): {moderate_count} | "
              f"SHORT (hl<2): {short_count}")

        # Top STRONG features
        strong_feats = [(f, d["half_life"]) for f, d in temporal_results.items()
                        if d["category"] == "STRONG"]
        strong_feats.sort(key=lambda x: x[1], reverse=True)
        if strong_feats:
            print(f"\n  Top STRONG temporal features:")
            for f, hl in strong_feats[:10]:
                print(f"    hl={hl:>2}  {f}")

        # ── Final selection: KEEP + STABLE ─────────────────────────────────────
        final_features = []
        for feat in keep_features:
            if feat in decay_results and decay_results[feat]["is_stable"]:
                final_features.append(feat)

        if not final_features:
            logger.warning("Tidak ada KEEP+STABLE features! Fallback ke KEEP only.")
            final_features = keep_features

        print(f"\n  FINAL training features: {len(final_features)} (KEEP + STABLE)")
        print(f"  Dropped from KEEP (unstable): {len(keep_features) - len(final_features)}")

    else:
        # Skip IC — use all features
        print("\nSTAGE 2: SKIPPED (--skip-ic). Using all features.")
        ic_results = []
        keep_features = avail_features
        final_features = avail_features
        decay_results = {}
        temporal_results = {}
        summary = {"KEEP": len(avail_features), "REDUNDANT": 0, "WEAK": 0, "DROP": 0}

    # ── Stage 3: LGBM Training ─────────────────────────────────────────────────
    print(f"\nSTAGE 3: LGBM Training with {len(final_features)} features...")
    print("-" * 50)

    X_train = df[final_features].ffill().fillna(0)
    y_train = y_full.loc[X_train.index]

    cv_results = walk_forward_cv(X_train, y_train, LGBM_PARAMS,
                                 n_splits=N_FOLDS, purge=PURGE_GAP_BARS)

    # Find best model
    best_model, best_f1, best_fold = None, -1.0, -1
    all_metrics = []
    for res in cv_results:
        m = res["metrics"]
        all_metrics.append(m)
        if m["f1_macro"] > best_f1:
            best_f1, best_model, best_fold = m["f1_macro"], res["model"], m["fold"]

    avg_best_iter = int(np.mean([m["best_iteration"] for m in all_metrics]))
    logger.info(f"CV complete. Avg best_iteration: {avg_best_iter} | "
                f"Best Fold: {best_fold} (F1={best_f1:.4f})")
    logger.info(f"Retraining final model on 100% training data ({len(X_train):,} samples)...")

    # Positional indexing — avoid .loc[] because multi-coin df has duplicate DatetimeIndex
    y_train_aligned = df["tb_label"].values.astype(np.int32)
    assert len(X_train) == len(y_train_aligned), \
        f"Length mismatch: X={len(X_train)} y={len(y_train_aligned)}"

    final_params = LGBM_PARAMS.copy()
    final_params["n_estimators"] = avg_best_iter

    full_sample_w = np.array([LGBM_CLASS_WEIGHTS[int(l)] for l in y_train_aligned],
                             dtype=np.float32)
    final_model = lgb.LGBMClassifier(**final_params)
    final_model.fit(X_train, y_train_aligned, sample_weight=full_sample_w)

    # ── Stage 4: Save ──────────────────────────────────────────────────────────
    print(f"\nSTAGE 4: Saving outputs to {run_dir}...")
    print("-" * 50)

    # Model
    model_path = run_dir / "lgbm.pkl"
    joblib.dump(final_model, model_path)
    logger.info(f"Model -> {model_path}")

    # Feature list
    feat_path = run_dir / f"{RUN_NAME}_keep_features.json"
    with open(feat_path, "w") as f:
        json.dump(final_features, f, indent=2)

    # IC results
    if ic_results:
        ic_path = run_dir / f"{RUN_NAME}_ic_results.json"
        with open(ic_path, "w") as f:
            json.dump({
                "run_name": RUN_NAME,
                "n_rows": n_total,
                "n_eff": n_eff,
                "thresholds": {
                    "min_standalone_ic": MIN_STANDALONE_IC,
                    "min_tstat": MIN_TSTAT,
                    "min_marginal_ic": MIN_MARGINAL_IC,
                },
                "summary": summary,
                "keep_features": keep_features,
                "results": ic_results,
            }, f, indent=2, default=str)
        logger.info(f"IC results -> {ic_path}")

    # IC Decay
    if decay_results:
        decay_path = run_dir / f"{RUN_NAME}_ic_decay.json"
        with open(decay_path, "w") as f:
            json.dump(decay_results, f, indent=2, default=str)
        logger.info(f"IC decay -> {decay_path}")

    # Temporal IC
    if temporal_results:
        temporal_path = run_dir / f"{RUN_NAME}_temporal_ic.json"
        with open(temporal_path, "w") as f:
            json.dump(temporal_results, f, indent=2, default=str)
        logger.info(f"Temporal IC -> {temporal_path}")

    # CV results
    f1s = [m["f1_macro"] for m in all_metrics]
    accs = [m["accuracy"] for m in all_metrics]
    cv_summary = {
        "run_name": RUN_NAME,
        "coins": coins,
        "tp_atr_mult": tp_atr,
        "sl_atr_mult": sl_atr,
        "max_hold": max_hold,
        "rr_ratio": round(rr, 2),
        "n_features_training": len(final_features),
        "n_folds": N_FOLDS,
        "gap_bars": PURGE_GAP_BARS,
        "best_fold": best_fold,
        "best_f1_macro": round(best_f1, 4),
        "mean_f1_macro": round(float(np.mean(f1s)), 4),
        "std_f1_macro": round(float(np.std(f1s)), 4),
        "mean_accuracy": round(float(np.mean(accs)), 4),
        "lgbm_params": LGBM_PARAMS,
        "final_retrain_n_estimators": avg_best_iter,
        "class_weights": LGBM_CLASS_WEIGHTS,
        "folds": all_metrics,
    }
    cv_path = run_dir / f"{RUN_NAME}_cv_results.json"
    with open(cv_path, "w") as f:
        json.dump(cv_summary, f, indent=2, default=str)
    logger.info(f"CV results -> {cv_path}")

    # Metadata
    meta = {
        "run_name": RUN_NAME,
        "created": datetime.now().isoformat(),
        "labeling": "Triple Barrier",
        "tp_atr_mult": tp_atr,
        "sl_atr_mult": sl_atr,
        "max_hold": max_hold,
        "rr_ratio": round(rr, 2),
        "label_distribution": {label_names.get(k, str(k)): int(v)
                              for k, v in label_dist.items()},
        "ic_summary": summary,
        "n_keep_features": len(keep_features),
        "n_final_features": len(final_features),
        "cv_mean_f1_macro": round(float(np.mean(f1s)), 4),
        "cv_std_f1_macro": round(float(np.std(f1s)), 4),
        "best_fold_f1": round(best_f1, 4),
        "n_samples": len(df),
        "n_coins": len(coins),
    }
    meta_path = run_dir / f"{RUN_NAME}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    logger.info(f"Metadata -> {meta_path}")

    # ── Print Summary ──────────────────────────────────────────────────────────
    print(f"\n{sep}")
    print(f"  TRIPLE BARRIER LGBM COMPLETE — {RUN_NAME}")
    print(f"{sep}")
    print(f"  TP={tp_atr}xATR  SL={sl_atr}xATR  RR={rr:.2f}  MaxHold={max_hold}")
    print(f"  IC: KEEP={summary.get('KEEP',0)} | Final (KEEP+STABLE): {len(final_features)}")
    print(f"  CV Mean F1: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")
    print(f"  Best Fold: {best_fold} (F1={best_f1:.4f})")
    print(f"  Final model: {model_path} (n_estimators={avg_best_iter})")
    print(f"{sep}\n")

    print(f"  {'Fold':>4}  {'Acc':>7}  {'F1-mac':>7}  {'LONG':>7}  {'SHORT':>7}  {'FLAT':>7}  {'Iter':>6}")
    print("  " + "-" * 52)
    for m in all_metrics:
        print(f"  {m['fold']:>4}  {m['accuracy']:>7.4f}  {m['f1_macro']:>7.4f}  "
              f"{m['f1_LONG']:>7.4f}  {m['f1_SHORT']:>7.4f}  {m['f1_FLAT']:>7.4f}  "
              f"{m['best_iteration']:>6}")


if __name__ == "__main__":
    main()
